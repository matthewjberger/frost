use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::build_cache::{
    BuildCache, ModuleRecord, digest, fnv1a, interface_fingerprint,
    module_fingerprint, stamp_file,
};
use crate::import_visibility::{FileNames, unimported_names};
use crate::interface::ModuleInterface;
use crate::lexer::Lexer;
use crate::lexer::Token;
use crate::parser::Parser;
use crate::parser::{
    Block, Expression, Parameter, Pattern, ReturnKind, ReturnSignature,
    Spanned, Statement, SwitchCase, TEST_PREFIX,
};
use crate::types::Type;

// One place an import may be found, and the name that place gives a module.
//
// A module's identity has to be a property of the module, not of where the
// machine happens to keep it, because private symbol names and the build cache
// are both keyed on it. So identity is the path relative to the root it was
// found under, with that root's label in front. A project file stays
// `lib/slab.frost` exactly as before, and a standard library module is
// `std/option.frost` wherever the standard library is installed.
#[derive(Debug, Clone)]
pub struct SearchRoot {
    pub label: String,
    pub directory: PathBuf,
}

impl SearchRoot {
    pub fn project(directory: PathBuf) -> Self {
        SearchRoot {
            label: String::new(),
            directory,
        }
    }

    pub fn named(label: &str, directory: PathBuf) -> Self {
        SearchRoot {
            label: label.to_string(),
            directory,
        }
    }
}

// A test block lowers to a function, and a test's body calls `assert`, which
// only exists when the test harness declares it. So an imported module's tests
// have to be dropped rather than spliced. Otherwise a library with tests breaks
// every program that imports it, and `--test` on a program would run its
// dependencies' tests as well as its own.
//
// `--test <file>` runs that file's tests. `--test <directory>` runs everything
// under it, one file at a time.
fn without_tests(
    statements: Vec<Spanned<Statement>>,
) -> Vec<Spanned<Statement>> {
    statements
        .into_iter()
        .filter(|statement| {
            !matches!(
                &statement.node,
                Statement::Constant(name, _) if name.contains(TEST_PREFIX)
            )
        })
        .collect()
}

// Where an import was found: the file itself, and the identity to give it.
struct Found {
    path: PathBuf,
    module: String,
}

// Resolve `import "x.frost"` written in a file in `importing_dir`.
//
// The importing file's own directory is tried first and always, because a
// file's neighbours are the most specific thing it could mean and because that
// is what every existing program relies on. Only then the search roots, in the
// order the driver assembled them.
fn find_import(
    importing_dir: &Path,
    path: &str,
    roots: &[SearchRoot],
    project_root: &Path,
) -> Option<Found> {
    let neighbour = importing_dir.join(path);
    if neighbour.exists() {
        let key = neighbour
            .canonicalize()
            .unwrap_or_else(|_| neighbour.clone());
        return Some(Found {
            module: relative_module_name(&key, project_root),
            path: neighbour,
        });
    }
    for root in roots {
        let candidate = root.directory.join(path);
        if !candidate.exists() {
            continue;
        }
        let key = candidate
            .canonicalize()
            .unwrap_or_else(|_| candidate.clone());
        let directory = root
            .directory
            .canonicalize()
            .unwrap_or_else(|_| root.directory.clone());
        let relative = relative_module_name(&key, &directory);
        let module = if root.label.is_empty() {
            relative
        } else {
            format!("{}/{relative}", root.label)
        };
        return Some(Found {
            path: candidate,
            module,
        });
    }
    None
}

pub struct Resolved {
    pub statements: Vec<Spanned<Statement>>,
    pub linear_types: HashSet<String>,
    pub tests: Vec<(String, String)>,
    // One per imported module, and empty unless interface checking is on. The
    // compiler does not build from these yet.
    pub interfaces: Vec<crate::interface::ModuleInterface>,
    // One per imported module, and empty without a build cache. What the driver
    // needs to link a module's cached object instead of compiling it, and to
    // write back what it did compile.
    pub modules: Vec<ModulePlan>,
}

// What was decided about one module before the program was built: whether it
// still has to be compiled, and where its object is either way.
pub struct ModulePlan {
    pub module: String,
    pub tag: String,
    pub file: u32,
    pub object: PathBuf,
    pub reused: bool,
    pub record: ModuleRecord,
}

// The file named on the command line, registered so a diagnostic from it names
// a file like every other one does rather than a bare line number.
pub fn register_entry_file(path: &Path, base_dir: &Path) -> u32 {
    let root = base_dir.canonicalize().unwrap_or_else(|_| base_dir.into());
    let key = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    crate::source_map::register_at(
        &relative_module_name(&key, &root),
        &key.to_string_lossy(),
    )
}

pub fn resolve_imports(
    statements: Vec<Spanned<Statement>>,
    base_dir: &Path,
    linear_types: HashSet<String>,
    tests: Vec<(String, String)>,
) -> Result<Resolved> {
    resolve_imports_cached(
        statements,
        base_dir,
        linear_types,
        tests,
        Resolution::default(),
    )
}

// What resolution is allowed to use beyond the source it is handed: a build
// cache to answer for unchanged modules, and the directories to search for an
// import that is not beside the file that wrote it.
#[derive(Default, Clone, Copy)]
pub struct Resolution<'a> {
    pub cache: Option<&'a BuildCache>,
    pub roots: &'a [SearchRoot],
}

// With a cache, a module whose own source and whose imported interfaces are
// all unchanged is not read past its
// first line. It contributes the interface the cache already holds, and its
// object is linked rather than built.
pub fn resolve_imports_cached(
    statements: Vec<Spanned<Statement>>,
    base_dir: &Path,
    linear_types: HashSet<String>,
    tests: Vec<(String, String)>,
    options: Resolution<'_>,
) -> Result<Resolved> {
    let Resolution { cache, roots } = options;
    let resolved = Resolved {
        statements: Vec::new(),
        linear_types,
        tests,
        interfaces: Vec::new(),
        modules: Vec::new(),
    };
    // The directory of the file named on the command line is the project root,
    // and a module's identity is its path relative to that, which is the
    // smallest thing that can answer what a project root is.
    let root = base_dir.canonicalize().unwrap_or_else(|_| base_dir.into());

    // Deciding whether a module can be skipped needs the interfaces of
    // everything below it, so the graph is walked bottom up before anything is
    // spliced. The walk parses only the modules it cannot answer for from the
    // cache, and hands those parses to the splice below rather than repeating
    // them.
    let mut plans: Plans = BTreeMap::new();
    if let Some(cache) = cache {
        let mut stack = HashSet::new();
        for statement in &statements {
            if let Statement::Import(path, _) = &statement.node {
                let Some(found) = find_import(base_dir, path, roots, &root)
                else {
                    continue;
                };
                plan_module(
                    &found,
                    &Planning {
                        root: &root,
                        cache,
                        roots,
                    },
                    &mut plans,
                    &mut stack,
                )?;
            }
        }
    }

    let mut walk = Walk {
        root: &root,
        roots,
        seen: HashSet::new(),
        resolved,
        plans,
        files: Vec::new(),
        module_exports: HashMap::new(),
        module_view: HashMap::new(),
    };
    // The entry file is a file like any other, and the one most likely to have
    // been leaning on a name it never imported.
    walk.files.push(FileNames::of(
        "the entry file",
        &statements,
        &import_identities(&statements, base_dir, roots, &root),
    ));
    walk.resolve_into(statements, base_dir, "the entry file")?;
    let reports = unimported_names(&walk.files, &walk.module_exports);
    if !reports.is_empty() {
        bail!(
            "an import says what a file may name, and these name what they did not import:\n{}",
            reports.join("\n")
        );
    }
    let mut resolved = walk.resolved;

    resolved.modules = walk
        .plans
        .into_values()
        .map(|planned| {
            let mut interface = planned.interface;
            // A file id is handed out in registration order, so an interface
            // written down with one in it would mean something different in the
            // process that reads it back.
            stamp_file(&mut interface.declarations, 0);
            ModulePlan {
                file: planned.file,
                object: planned.object,
                reused: planned.reused,
                tag: planned.tag,
                module: planned.module.clone(),
                record: ModuleRecord {
                    module: planned.module,
                    source_hash: planned.source_hash,
                    imports: planned.imports,
                    interface,
                    emits_object: planned.emits_object,
                },
            }
        })
        .collect();
    Ok(resolved)
}

struct ParsedModule {
    statements: Vec<Spanned<Statement>>,
    exports: Vec<String>,
    linear_types: HashSet<String>,
    tests: Vec<(String, String)>,
}

struct Planned {
    module: String,
    tag: String,
    file: u32,
    source_hash: String,
    imports: Vec<String>,
    // Absent when the module was answered for from the cache. Its source was
    // never parsed.
    parsed: Option<ParsedModule>,
    interface: ModuleInterface,
    interface_hash: String,
    // The interface hash of every module reachable through this one's imports.
    // Transitive, because a generic this module instantiates can instantiate one
    // from further down.
    closure: BTreeMap<String, String>,
    object: PathBuf,
    reused: bool,
    emits_object: bool,
}

type Plans = BTreeMap<PathBuf, Planned>;

fn parse_module(
    source: &str,
    file: u32,
    path: &Path,
    generics: HashSet<String>,
) -> Result<Box<ParsedModule>> {
    let mut lexer = Lexer::new(source);
    let tokens = lexer
        .tokenize()
        .with_context(|| format!("lexing {}", path.display()))?;
    // Every position the lexer produced for this file belongs to this file, and
    // stamping them here is the only place that knows which file it is.
    let positions: Vec<_> = lexer
        .positions()
        .iter()
        .map(|position| crate::lexer::Position { file, ..*position })
        .collect();
    let mut parser = Parser::with_positions(&tokens, &positions);
    parser.also_generic(generics);
    let statements = parser
        .parse()
        .with_context(|| format!("parsing {}", path.display()))?;
    Ok(Box::new(ParsedModule {
        statements,
        exports: parser.exports().to_vec(),
        linear_types: parser.linear_types().iter().cloned().collect(),
        tests: parser.tests().to_vec(),
    }))
}

// The planner's fixed inputs, so the recursive walk carries one reference
// rather than four positional arguments.
struct Planning<'a> {
    root: &'a Path,
    cache: &'a BuildCache,
    roots: &'a [SearchRoot],
}

fn plan_module(
    found: &Found,
    context: &Planning<'_>,
    plans: &mut Plans,
    stack: &mut HashSet<PathBuf>,
) -> Result<PathBuf> {
    let Planning { root, cache, roots } = context;
    let full = found.path.as_path();
    let key = full.canonicalize().unwrap_or_else(|_| full.to_path_buf());
    if plans.contains_key(&key) || !stack.insert(key.clone()) {
        return Ok(key);
    }

    let source = fs::read_to_string(full).with_context(|| {
        format!("failed to read imported file: {}", full.display())
    })?;
    let module = found.module.clone();
    let file = crate::source_map::register_at(&module, &full.to_string_lossy());
    let tag = module_tag_of(&module);
    let source_hash = digest(&source);
    let record = cache.load(&tag, &source_hash);

    let mut parsed: Option<Box<ParsedModule>> = None;
    let mut interface = match &record {
        Some(record) => {
            let mut interface = record.interface.clone();
            stamp_file(&mut interface.declarations, file);
            interface
        }
        None => {
            let fresh = parse_module(
                &source,
                file,
                full,
                imported_generic_types(
                    &source,
                    &directory_of(full),
                    roots,
                    root,
                ),
            )?;
            let interface = ModuleInterface::of(
                &module,
                &fresh.statements,
                &fresh.exports,
                &fresh.linear_types,
            );
            parsed = Some(fresh);
            interface
        }
    };
    let imports: Vec<String> = match (&record, &parsed) {
        (Some(record), _) => record.imports.clone(),
        (None, Some(fresh)) => import_paths(&fresh.statements),
        (None, None) => Vec::new(),
    };

    let directory = full.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut closure: BTreeMap<String, String> = BTreeMap::new();
    for import in &imports {
        let Some(child_found) = find_import(&directory, import, roots, root)
        else {
            continue;
        };
        let child = plan_module(&child_found, context, plans, stack)?;
        if let Some(planned) = plans.get(&child) {
            closure
                .insert(planned.module.clone(), planned.interface_hash.clone());
            for (name, hash) in &planned.closure {
                closure.insert(name.clone(), hash.clone());
            }
        }
    }

    let interface_hash = interface_fingerprint(&interface)?;
    let fingerprint = module_fingerprint(&source_hash, &closure);
    let object = cache.object_path(&tag, &fingerprint);
    // A record answers for the module only while the object it describes is
    // still there. Deleting the build directory has to mean a full rebuild
    // rather than a link against nothing.
    let reused = record
        .as_ref()
        .is_some_and(|record| !record.emits_object || object.exists());
    if !reused && parsed.is_none() {
        let fresh = parse_module(
            &source,
            file,
            full,
            imported_generic_types(&source, &directory, roots, root),
        )?;
        interface = ModuleInterface::of(
            &module,
            &fresh.statements,
            &fresh.exports,
            &fresh.linear_types,
        );
        parsed = Some(fresh);
    }

    stack.remove(&key);
    plans.insert(
        key.clone(),
        Planned {
            module,
            tag,
            file,
            source_hash,
            imports,
            parsed: parsed.map(|parsed| *parsed),
            interface,
            interface_hash,
            closure,
            object,
            reused,
            emits_object: record
                .as_ref()
                .is_some_and(|record| record.emits_object),
        },
    );
    Ok(key)
}

fn import_identities(
    statements: &[Spanned<Statement>],
    base_dir: &Path,
    roots: &[SearchRoot],
    root: &Path,
) -> Vec<PathBuf> {
    import_paths(statements)
        .iter()
        .filter_map(|path| {
            find_import(base_dir, path, roots, root).map(|found| {
                found
                    .path
                    .canonicalize()
                    .unwrap_or_else(|_| found.path.clone())
            })
        })
        .collect()
}

fn import_paths(statements: &[Spanned<Statement>]) -> Vec<String> {
    statements
        .iter()
        .filter_map(|statement| match &statement.node {
            Statement::Import(path, _) => Some(path.clone()),
            _ => None,
        })
        .collect()
}

// The tag that distinguishes one module's private names from another's. It has
// to be a property of the module and nothing else. It used to be a counter
// handed out in import traversal order, which meant the same file's private
// `helper` was `__m3_helper` in one program and `__m7_helper` in another, and
// adding an unrelated import renamed every symbol downstream of it. Separate
// compilation cannot work on top of that, since a module compiled once has to
// produce the symbols every other module expects to link against.
// The tag for a module identity. Taken from the identity rather than from a
// path, so a module found through a search root is tagged by what it is called
// (`std/option.frost`) rather than by where the machine keeps it.
fn module_tag_of(module: &str) -> String {
    format!("{:016x}", fnv1a(module.as_bytes()))
}

// A module's identity: its path relative to the project root, with separators
// normalized, because the identity must not vary by platform.
fn relative_module_name(path: &Path, root: &Path) -> String {
    let relative = path.strip_prefix(root).unwrap_or(path);
    let joined: Vec<String> = relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect();
    joined.join("/")
}

struct Walk<'a> {
    root: &'a Path,
    roots: &'a [SearchRoot],
    seen: HashSet<PathBuf>,
    resolved: Resolved,
    plans: Plans,
    // What each file declared, imported and used, and what each module exports,
    // for the check that a file only names what it imported.
    files: Vec<FileNames>,
    module_exports: HashMap<PathBuf, (String, HashSet<String>)>,
    // Each module's tag and exported names, for building the view of every
    // file that imports it. A module read once is imported many times.
    module_view: HashMap<PathBuf, ModuleView>,
}

type Contribution = (Vec<Spanned<Statement>>, HashSet<String>, String);

// What an importer needs to know about a module: what it is called, what it
// exports, and what each of its names became in the spliced program. An
// `extern` keeps its own name, since that is the symbol a C library defines, so
// the map is what says which of the two a given export is rather than the tag.
struct ModuleView {
    module: String,
    exports: HashSet<String>,
    symbols: HashMap<String, String>,
}

impl Walk<'_> {
    // A file's imports are resolved before its own statements are spliced,
    // because what its statements may name comes from them. The imports go in
    // first either way, since a module has to be declared before it is used.
    fn resolve_into(
        &mut self,
        statements: Vec<Spanned<Statement>>,
        base_dir: &Path,
        module: &str,
    ) -> Result<()> {
        let mut view: HashMap<String, String> = HashMap::new();
        // A local name two imports both offer. Naming it is the error, since
        // importing two modules that happen to share a name you never write is
        // not.
        let mut ambiguous: HashMap<String, (String, String)> = HashMap::new();
        // Which module each symbol in the view came from, so an ambiguity names
        // the two modules rather than the two symbols.
        let mut owner_of: HashMap<String, String> = HashMap::new();
        let mut body = Vec::new();

        for statement in statements {
            let Statement::Import(path, renames) = &statement.node else {
                body.push(statement);
                continue;
            };
            let renames = renames.clone();

            let Some(found) =
                find_import(base_dir, path, self.roots, self.root)
            else {
                bail!(
                    "failed to read imported file: '{path}' is not beside {} and is not on any library path",
                    base_dir.display()
                );
            };
            let full = found.path.clone();
            let key = full.canonicalize().unwrap_or_else(|_| full.clone());
            if self.seen.insert(key.clone()) {
                self.splice_module(&full, &found.module, &key)?;
            }

            let Some(offered) = self.module_view.get(&key) else {
                continue;
            };
            let owner = offered.module.clone();
            let exports = offered.exports.clone();
            let symbols = offered.symbols.clone();

            // Everything the module exports arrives under its own name, except
            // the names this import renamed.
            let local_of: HashMap<&str, &str> = renames
                .iter()
                .map(|held| (held.exported.as_str(), held.local.as_str()))
                .collect();
            for name in &renames {
                if !exports.contains(&name.exported) {
                    bail!(
                        "'{}' does not export '{}', so there is nothing to read as '{}'",
                        owner,
                        name.exported,
                        name.local
                    );
                }
            }
            for name in &exports {
                let local = local_of
                    .get(name.as_str())
                    .map(|held| (*held).to_string())
                    .unwrap_or_else(|| name.clone());
                // An `extern` keeps the name the C library defines, so it is
                // offered as itself rather than as a symbol nothing declares.
                let symbol =
                    symbols.get(name).cloned().unwrap_or_else(|| name.clone());
                if let Some(first) = view.insert(local.clone(), symbol.clone())
                    && first != symbol
                {
                    ambiguous.insert(
                        local,
                        (
                            owner_of.get(&first).cloned().unwrap_or(first),
                            owner.clone(),
                        ),
                    );
                }
                owner_of.insert(symbol, owner.clone());
            }
        }

        // A file's own declarations win over anything it imported, so the view
        // only reaches the names it did not declare itself.
        for statement in &body {
            if let Some(name) = top_level_name(&statement.node) {
                view.remove(name);
                ambiguous.remove(name);
            }
        }

        if !ambiguous.is_empty() {
            let used = FileNames::of(module, &body, &[]);
            for name in &used.used {
                if let Some((first, second)) = ambiguous.get(name) {
                    bail!(
                        "'{name}' is exported by two modules {module} imports, {first} and {second}; read one of them under another name with `import \"...\" ({name} as ...)`"
                    );
                }
            }
        }

        if !view.is_empty() {
            let renamer = Renamer { renames: view };
            renamer.block(&mut body, &mut Vec::new());
        }
        self.resolved.statements.extend(body);
        Ok(())
    }

    // Read a module, mangle its names, and splice it and everything it imports.
    fn splice_module(
        &mut self,
        full: &Path,
        module: &str,
        key: &Path,
    ) -> Result<()> {
        let (mut imported, exports, tag) = if self.plans.contains_key(key) {
            self.planned_module(key)?
        } else {
            self.read_module(full, module)?
        };

        let child_dir =
            full.parent().map(Path::to_path_buf).unwrap_or_default();
        self.files.push(FileNames::of(
            module,
            &imported,
            &import_identities(&imported, &child_dir, self.roots, self.root),
        ));
        self.module_exports
            .insert(key.to_path_buf(), (module.to_string(), exports.clone()));
        let renames = module_renames(&imported, &tag);
        self.module_view.insert(
            key.to_path_buf(),
            ModuleView {
                module: module.to_string(),
                exports: exports.clone(),
                symbols: renames.clone(),
            },
        );

        // A linear type's name is what the ownership check keys on, so it
        // follows its declaration.
        for name in exports.iter().chain(renames.keys()) {
            if self.resolved.linear_types.remove(name) {
                self.resolved.linear_types.insert(mangled_name(&tag, name));
            }
        }
        if !renames.is_empty() {
            let renamer = Renamer { renames };
            renamer.block(&mut imported, &mut Vec::new());
        }

        self.resolve_into(imported, &child_dir, module)
    }

    // What a planned module contributes. A module the plan could answer for
    // contributes its interface and its own import lines, which is exactly what
    // the `FROST_BUILD_FROM_INTERFACES` oracle has been checking on every commit
    // since step 4. The difference is that here its object is not rebuilt
    // either.
    fn planned_module(&mut self, key: &Path) -> Result<Contribution> {
        let planned = self
            .plans
            .get_mut(key)
            .expect("a planned module the walk just found");
        let tag = planned.tag.clone();
        let interface = planned.interface.clone();
        let Some(parsed) = planned.parsed.take() else {
            let imports = planned.imports.clone();
            self.resolved
                .linear_types
                .extend(interface.linear_types.iter().cloned());
            let mut statements: Vec<Spanned<Statement>> = imports
                .into_iter()
                .map(|path| Spanned::from(Statement::Import(path, Vec::new())))
                .collect();
            // The module's object is being linked rather than rebuilt, so it
            // contributes signatures where it can and bodies only where a
            // caller needs one. See `as_declaration`.
            statements.extend(interface.declarations.into_iter().map(
                |statement| match crate::build_cache::as_declaration(
                    &statement.node,
                ) {
                    Some(declared) => Spanned {
                        node: declared,
                        position: statement.position,
                    },
                    None => statement,
                },
            ));
            let exports = interface.exports.into_iter().collect();
            return Ok((statements, exports, tag));
        };

        self.resolved
            .linear_types
            .extend(parsed.linear_types.iter().cloned());
        self.resolved.tests.extend(parsed.tests.iter().cloned());
        let exports: HashSet<String> = parsed.exports.into_iter().collect();
        let mut statements = parsed.statements;
        self.check_and_reduce(&interface, &mut statements)?;
        Ok((statements, exports, tag))
    }

    fn read_module(
        &mut self,
        full: &Path,
        module_name: &str,
    ) -> Result<Contribution> {
        let source = fs::read_to_string(full).with_context(|| {
            format!("failed to read imported file: {}", full.display())
        })?;
        let file = crate::source_map::register_at(
            module_name,
            &full.to_string_lossy(),
        );
        let parsed = parse_module(
            &source,
            file,
            full,
            imported_generic_types(
                &source,
                &directory_of(full),
                self.roots,
                self.root,
            ),
        )?;
        self.resolved
            .linear_types
            .extend(parsed.linear_types.iter().cloned());

        let exports: HashSet<String> = parsed.exports.iter().cloned().collect();
        let tag = module_tag_of(module_name);
        let mut statements = without_tests(parsed.statements);

        // The interface is derived at the one place a module is parsed, which is what keeps it
        // from drifting out of step with the source it describes.
        if crate::interface::interfaces_are_checked()
            || crate::interface::built_from_interfaces()
        {
            let interface = ModuleInterface::of(
                module_name,
                &statements,
                &parsed.exports,
                &parsed.linear_types,
            );
            self.check_and_reduce(&interface, &mut statements)?;
        }
        Ok((statements, exports, tag))
    }

    fn check_and_reduce(
        &mut self,
        interface: &ModuleInterface,
        statements: &mut Vec<Spanned<Statement>>,
    ) -> Result<()> {
        check_and_reduce(interface, statements, &mut self.resolved.interfaces)
    }
}

fn check_and_reduce(
    interface: &ModuleInterface,
    statements: &mut Vec<Spanned<Statement>>,
    interfaces: &mut Vec<ModuleInterface>,
) -> Result<()> {
    if !crate::interface::interfaces_are_checked()
        && !crate::interface::built_from_interfaces()
    {
        return Ok(());
    }
    crate::interface::check_interface_round_trip(interface)?;
    crate::interface::check_interface_covers_exports(interface)?;
    crate::interface::check_interface_is_closed(interface, statements)?;

    // The oracle for step 4: build the program from what the interface says
    // rather than from the module's source, and require the result to be the
    // same program. An interface that is missing something a caller needs fails
    // here, loudly, rather than once the compiler is trusting interfaces for
    // real.
    //
    // The module's own `import` lines are kept, because an interface carries
    // declarations and not dependencies, and the modules behind them still have
    // to be reached. Everything else the module declared is replaced by the
    // interface's view of it, so anything it kept private and nothing reaches
    // is gone.
    if crate::interface::built_from_interfaces() {
        let mut rebuilt: Vec<Spanned<Statement>> = statements
            .iter()
            .filter(|statement| matches!(statement.node, Statement::Import(..)))
            .cloned()
            .collect();
        rebuilt.extend(interface.declarations.iter().cloned());
        *statements = rebuilt;
    }
    interfaces.push(interface.clone());
    Ok(())
}

// Every generic type declared by a file this one imports, transitively. Which
// names can start a literal is settled before the parse (a `Pair<i64, bool> {`
// would otherwise read as two comparisons), and a file that imports a generic
// type writes a literal of it exactly as the file declaring it does. Only the
// declarations are read, by lexing each file and looking for the shape, since
// this runs before anything is parsed.
// The directory a file is in, which is where an import written in it is looked
// for first.
fn directory_of(file: &Path) -> PathBuf {
    file.parent().map(Path::to_path_buf).unwrap_or_default()
}

pub fn imported_generic_types(
    source: &str,
    base_dir: &Path,
    roots: &[SearchRoot],
    project_root: &Path,
) -> HashSet<String> {
    let mut names = HashSet::new();
    let mut seen = HashSet::new();
    let scan = Scan {
        roots,
        project_root,
    };
    for path in import_paths_in_source(source) {
        collect_generic_types(base_dir, &path, &scan, &mut names, &mut seen);
    }
    names
}

// What the walk above needs everywhere and reads nowhere, so the recursion
// carries one reference rather than two positional arguments.
struct Scan<'a> {
    roots: &'a [SearchRoot],
    project_root: &'a Path,
}

fn collect_generic_types(
    importing_dir: &Path,
    path: &str,
    scan: &Scan<'_>,
    names: &mut HashSet<String>,
    seen: &mut HashSet<PathBuf>,
) {
    let Some(found) =
        find_import(importing_dir, path, scan.roots, scan.project_root)
    else {
        return;
    };
    let key = found
        .path
        .canonicalize()
        .unwrap_or_else(|_| found.path.clone());
    if !seen.insert(key.clone()) {
        return;
    }
    let Ok(source) = fs::read_to_string(&found.path) else {
        return;
    };
    let mut lexer = Lexer::new(&source);
    if let Ok(tokens) = lexer.tokenize() {
        names.extend(crate::parser::scan_generic_types(&tokens));
    }
    let directory = directory_of(&found.path);
    for next in import_paths_in_source(&source) {
        collect_generic_types(&directory, &next, scan, names, seen);
    }
}

// The paths a file imports, read off its tokens rather than its parse, because
// this is what runs before the parse.
fn import_paths_in_source(source: &str) -> Vec<String> {
    let mut lexer = Lexer::new(source);
    let Ok(tokens) = lexer.tokenize() else {
        return Vec::new();
    };
    let mut paths = Vec::new();
    for index in 0..tokens.len() {
        if !matches!(tokens[index], Token::Import) {
            continue;
        }
        if let Some(Token::StringLiteral(path)) = tokens.get(index + 1) {
            paths.push(path.clone());
        }
    }
    paths
}

fn top_level_name(statement: &Statement) -> Option<&str> {
    match statement {
        Statement::Constant(name, _)
        | Statement::Struct(name, _, _)
        | Statement::Enum(name, _, _)
        | Statement::Flags(name, _, _)
        | Statement::TypeAlias(name, _)
        | Statement::Extern { name, .. }
        | Statement::Declared { name, .. } => Some(name),
        _ => None,
    }
}

// Turns `__m<tag>_helper` back into `helper` for a diagnostic. A reader did not
// write the mangled name and should not have to recognize it. Kept next to
// `private_renames`, which is the only thing that produces the shape, so the
// two cannot drift apart.
pub fn demangle_private_names(text: &str) -> String {
    const PREFIX: &str = "__m";
    const TAG: usize = 16;
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(PREFIX) {
        out.push_str(&rest[..start]);
        let after = &rest[start + PREFIX.len()..];
        let is_tag = after.len() > TAG
            && after.as_bytes()[..TAG].iter().all(u8::is_ascii_hexdigit)
            && after.as_bytes()[TAG] == b'_';
        if is_tag {
            rest = &after[TAG + 1..];
        } else {
            out.push_str(PREFIX);
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

fn module_renames(
    statements: &[Spanned<Statement>],
    tag: &str,
) -> HashMap<String, String> {
    let mut renames = HashMap::new();
    for statement in statements {
        // An `extern` name is not this module's to rename. It is the symbol a
        // C library defines, so mangling it produces a link against a
        // name nothing exports. That only showed up once a module other than
        // the entry file declared one, which is what a standard library doing
        // its own IO is.
        if matches!(statement.node, Statement::Extern { .. }) {
            continue;
        }
        if let Some(name) = top_level_name(&statement.node) {
            renames.insert(name.to_string(), mangled_name(tag, name));
        }
    }
    renames
}

// What a module's name is called in the spliced program. Exported names are
// mangled the same way private ones are, so two modules exporting `insert` are
// two symbols rather than a collision, and each importer binds the one it
// imported. The tag is a property of the module identity, so the symbol is the
// same in every program the module appears in, which is what separate
// compilation links against.
fn mangled_name(tag: &str, name: &str) -> String {
    format!("__m{tag}_{name}")
}

struct Renamer {
    renames: HashMap<String, String>,
}

type Scope = Vec<HashSet<String>>;

impl Renamer {
    fn mapped(&self, name: &str, scope: &Scope) -> Option<String> {
        if scope.iter().any(|frame| frame.contains(name)) {
            return None;
        }
        self.renames.get(name).cloned()
    }

    fn block(&self, block: &mut Block, scope: &mut Scope) {
        scope.push(HashSet::new());
        for statement in block.iter_mut() {
            self.statement(&mut statement.node, scope);
        }
        scope.pop();
    }

    fn bind(&self, scope: &mut Scope, name: &str) {
        if let Some(frame) = scope.last_mut() {
            frame.insert(name.to_string());
        }
    }

    fn statement(&self, statement: &mut Statement, scope: &mut Scope) {
        match statement {
            Statement::Constant(name, value) => {
                if let Some(mangled) = self.renames.get(name.as_str()) {
                    *name = mangled.clone();
                }
                self.expression(value, scope);
            }
            Statement::Struct(name, _, fields) => {
                if let Some(mangled) = self.renames.get(name.as_str()) {
                    *name = mangled.clone();
                }
                for field in fields {
                    self.ty(&mut field.field_type);
                }
            }
            Statement::Enum(name, _, variants) => {
                if let Some(mangled) = self.renames.get(name.as_str()) {
                    *name = mangled.clone();
                }
                for variant in variants {
                    if let Some(fields) = &mut variant.fields {
                        for field in fields {
                            self.ty(&mut field.field_type);
                        }
                    }
                }
            }
            Statement::TypeAlias(name, ty) | Statement::Flags(name, ty, _) => {
                if let Some(mangled) = self.renames.get(name.as_str()) {
                    *name = mangled.clone();
                }
                self.ty(ty);
            }
            Statement::Extern {
                name,
                params,
                return_type,
                ..
            } => {
                if let Some(mangled) = self.renames.get(name.as_str()) {
                    *name = mangled.clone();
                }
                for param in params.iter_mut() {
                    if let Some(ty) = &mut param.type_annotation {
                        self.ty(ty);
                    }
                    if let Some(ty) = &mut param.compile_time_signature {
                        self.ty(ty);
                    }
                }
                if let Some(ty) = return_type {
                    self.ty(ty);
                }
            }
            Statement::Declared {
                name,
                params,
                return_sig,
            } => {
                if let Some(mangled) = self.renames.get(name.as_str()) {
                    *name = mangled.clone();
                }
                self.parameters(params, &mut Vec::new());
                self.return_signature(return_sig);
            }
            Statement::Let {
                name,
                type_annotation,
                value,
                ..
            } => {
                self.expression(value, scope);
                if let Some(ty) = type_annotation {
                    self.ty(ty);
                }
                self.bind(scope, name);
            }
            Statement::LetMultiple(bindings, value) => {
                self.expression(value, scope);
                for binding in bindings.iter() {
                    self.bind(scope, &binding.name);
                }
            }
            Statement::Return(value) => self.expression(value, scope),
            Statement::Expression(value) => self.expression(value, scope),
            Statement::Print(value, arguments) => {
                self.expression(value, scope);
                for argument in arguments {
                    self.expression(argument, scope);
                }
            }
            Statement::Assignment(target, value) => {
                self.expression(target, scope);
                self.expression(value, scope);
            }
            Statement::Defer(inner) => self.statement(inner, scope),
            Statement::For(variable, _, range, body) => {
                self.expression(range, scope);
                scope.push(HashSet::new());
                self.bind(scope, variable);
                for statement in body.iter_mut() {
                    self.statement(&mut statement.node, scope);
                }
                scope.pop();
            }
            Statement::While(condition, body) => {
                self.expression(condition, scope);
                self.block(body, scope);
            }
            Statement::With(capability, body) => {
                if let Some(mangled) = self.mapped(capability, scope) {
                    *capability = mangled;
                }
                self.block(body, scope);
            }
            Statement::Break | Statement::Continue | Statement::Import(..) => {}
        }
    }

    // A private type is just as nameable in a return position as in a
    // parameter, and this used to be skipped, so an exported function returning
    // an unexported struct kept the un-renamed name and the importer could not
    // resolve it.
    // `Base<A, B>` with the base and every argument renamed, or `None` when the
    // name is not an instance or nothing in it is private.
    fn generic_instance(&self, name: &str) -> Option<String> {
        let open = name.find('<')?;
        let inner = name.strip_suffix('>')?.get(open + 1..)?;
        let base = &name[..open];
        let mut changed = false;
        let renamed_base = match self.renames.get(base) {
            Some(mangled) => {
                changed = true;
                mangled.clone()
            }
            None => base.to_string(),
        };
        // Arguments are split at the top level only, so a nested instance stays
        // whole and is renamed by the recursive call.
        let mut arguments: Vec<String> = Vec::new();
        let mut depth = 0usize;
        let mut current = String::new();
        for character in inner.chars() {
            match character {
                '<' => {
                    depth += 1;
                    current.push(character);
                }
                '>' => {
                    depth -= 1;
                    current.push(character);
                }
                ',' if depth == 0 => {
                    arguments.push(current.trim().to_string());
                    current = String::new();
                }
                _ => current.push(character),
            }
        }
        arguments.push(current.trim().to_string());
        let arguments: Vec<String> = arguments
            .into_iter()
            .map(|argument| match self.renames.get(argument.as_str()) {
                Some(mangled) => {
                    changed = true;
                    mangled.clone()
                }
                None => match self.generic_instance(&argument) {
                    Some(renamed) => {
                        changed = true;
                        renamed
                    }
                    None => argument,
                },
            })
            .collect();
        changed.then(|| format!("{renamed_base}<{}>", arguments.join(", ")))
    }

    fn return_signature(&self, signature: &mut ReturnSignature) {
        match &mut signature.kind {
            ReturnKind::None => {}
            ReturnKind::Single(ty) => self.ty(ty),
            ReturnKind::Multiple(values) => {
                for held in values.iter_mut() {
                    self.ty(&mut held.value_type);
                }
            }
            ReturnKind::Fallible(value, failure) => {
                self.ty(value);
                self.ty(failure);
            }
        }
        for capability in signature.uses.iter_mut() {
            self.ty(capability);
        }
    }

    fn parameters(&self, params: &mut [Parameter], scope: &mut Scope) {
        for param in params.iter_mut() {
            if let Some(ty) = &mut param.type_annotation {
                self.ty(ty);
            }
            // What a compile-time parameter is declared to take is a type like
            // any other, and a bundle parameter names one this module imported.
            if let Some(ty) = &mut param.compile_time_signature {
                self.ty(ty);
            }
            self.bind(scope, &param.name);
        }
    }

    fn expression(&self, expression: &mut Expression, scope: &mut Scope) {
        match expression {
            Expression::Identifier(name) => {
                if let Some(mangled) = self.mapped(name, scope) {
                    *name = mangled;
                }
            }
            // An array literal holds expressions, so a name written inside one
            // is a name this module may have imported. Treating every literal
            // as a leaf left those unmapped, and a call to an imported function
            // written inside an array literal reached the backend under a name
            // nothing had defined.
            Expression::Literal(crate::parser::Literal::Array(elements)) => {
                for element in elements.iter_mut() {
                    self.expression(element, scope);
                }
            }
            Expression::Literal(_) | Expression::Boolean(_) => {}
            Expression::PackMap(operand, _, _)
            | Expression::Prefix(_, operand)
            | Expression::AddressOf(operand)
            | Expression::Borrow(operand)
            | Expression::BorrowMut(operand)
            | Expression::Try(operand)
            | Expression::ArrayRepeat(operand, _)
            | Expression::Dereference(operand) => {
                self.expression(operand, scope)
            }
            Expression::Infix(left, _, right)
            | Expression::Index(left, right) => {
                self.expression(left, scope);
                self.expression(right, scope);
            }
            Expression::Range(start, end, _) => {
                self.expression(start, scope);
                self.expression(end, scope);
            }
            Expression::If(condition, consequence, alternative) => {
                self.expression(condition, scope);
                self.block(consequence, scope);
                if let Some(block) = alternative {
                    self.block(block, scope);
                }
            }
            Expression::Function(params, return_sig, body)
            | Expression::Proc(params, return_sig, body) => {
                scope.push(HashSet::new());
                self.parameters(params, scope);
                self.return_signature(return_sig);
                for statement in body.iter_mut() {
                    self.statement(&mut statement.node, scope);
                }
                scope.pop();
            }
            Expression::Call(callee, arguments) => {
                self.expression(callee, scope);
                for argument in arguments.iter_mut() {
                    self.expression(argument, scope);
                }
            }
            Expression::FieldAccess(base, _) => self.expression(base, scope),
            Expression::StructInit(name, fields) => {
                if let Some(mangled) = self.renames.get(name.as_str()) {
                    *name = mangled.clone();
                } else if let Some(renamed) = self.generic_instance(name) {
                    // A literal that says which instance it is names it as one
                    // string, `Ordering<i64>`, so looking the whole thing up
                    // finds nothing. Both halves are renamed the way a type
                    // annotation naming the same instance is.
                    *name = renamed;
                }
                for (_, value) in fields.iter_mut() {
                    self.expression(value, scope);
                }
            }
            Expression::EnumVariantInit(enum_name, _, fields) => {
                if let Some(mangled) = self.renames.get(enum_name.as_str()) {
                    *enum_name = mangled.clone();
                }
                for (_, value) in fields.iter_mut() {
                    self.expression(value, scope);
                }
            }
            Expression::Sizeof(ty)
            | Expression::TypeId(ty)
            | Expression::TypeName(ty)
            | Expression::TypeValue(ty) => self.ty(ty),
            Expression::Tuple(elements) => {
                for element in elements.iter_mut() {
                    self.expression(element, scope);
                }
            }
            Expression::Switch(scrutinee, cases) => {
                self.expression(scrutinee, scope);
                for case in cases.iter_mut() {
                    self.switch_case(case, scope);
                }
            }
            Expression::Unsafe(body) => self.block(body, scope),
            Expression::UnsafeFn(inner) => self.expression(inner, scope),
        }
    }

    fn switch_case(&self, case: &mut SwitchCase, scope: &mut Scope) {
        scope.push(HashSet::new());
        self.pattern(&mut case.pattern, scope);
        for statement in case.body.iter_mut() {
            self.statement(&mut statement.node, scope);
        }
        scope.pop();
    }

    fn pattern(&self, pattern: &mut Pattern, scope: &mut Scope) {
        match pattern {
            Pattern::EnumVariant {
                enum_name,
                bindings,
                ..
            } => {
                if let Some(name) = enum_name
                    && let Some(mangled) = self.renames.get(name.as_str())
                {
                    *name = mangled.clone();
                }
                for (_, binding) in bindings {
                    self.bind(scope, binding);
                }
            }
            Pattern::Tuple(patterns) => {
                for pattern in patterns {
                    self.pattern(pattern, scope);
                }
            }
            Pattern::Identifier(name) => self.bind(scope, name),
            Pattern::Wildcard | Pattern::Literal(_) => {}
        }
    }

    fn ty(&self, ty: &mut Type) {
        match ty {
            Type::Struct(name) | Type::Enum(name) => {
                if let Some(mangled) = self.renames.get(name.as_str()) {
                    *name = mangled.clone();
                    return;
                }
                // A generic instance is one name, `Boxed<i64>`, so looking the
                // whole thing up finds nothing and a private generic type kept
                // its un-renamed name. Both the base and the arguments can name
                // private types, and both are renamed here.
                if let Some(renamed) = self.generic_instance(name) {
                    *name = renamed;
                }
            }
            Type::Ptr(inner)
            | Type::Ref(inner)
            | Type::RefMut(inner)
            | Type::Slice(inner)
            | Type::Array(inner, _)
            | Type::Distinct(_, inner)
            | Type::Handle(inner) => self.ty(inner),
            Type::Proc(params, ret) => {
                for param in params.iter_mut() {
                    self.ty(param);
                }
                self.ty(ret);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demangling_leaves_alone_what_it_did_not_mangle() {
        // A real mangled name, the same shape `private_renames` produces.
        assert_eq!(
            demangle_private_names(
                "struct '__m6999e911a6ca1ff4_Dot' has no field 'x'"
            ),
            "struct 'Dot' has no field 'x'"
        );
        // A name that merely starts the same way is not a tag.
        assert_eq!(demangle_private_names("__mixer"), "__mixer");
        assert_eq!(demangle_private_names("__m123_short"), "__m123_short");
        assert_eq!(demangle_private_names("nothing here"), "nothing here");
        // Two in one message, and the tail after the last one is kept.
        assert_eq!(
            demangle_private_names(
                "__m0000000000000001_a calls __m0000000000000002_b twice"
            ),
            "a calls b twice"
        );
    }

    #[test]
    fn a_module_tag_is_the_same_for_the_same_relative_path() {
        let root = Path::new("/project");
        assert_eq!(
            module_tag_of(&relative_module_name(
                Path::new("/project/lib/a.frost"),
                root
            )),
            module_tag_of(&relative_module_name(
                Path::new("/project/lib/a.frost"),
                root
            ))
        );
        assert_ne!(
            module_tag_of(&relative_module_name(
                Path::new("/project/lib/a.frost"),
                root
            )),
            module_tag_of(&relative_module_name(
                Path::new("/project/lib/b.frost"),
                root
            ))
        );
    }
}
