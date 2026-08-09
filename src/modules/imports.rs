use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::ast::{
    Ast, ExprId, Expression, Module, Pattern, PatternId, Range32, ReturnKind,
    SignatureId, Splicer, Statement, StmtId, TokenSpan, splice_positions,
};
use crate::lexer::Lexer;
use crate::lexer::Token;
use crate::modules::build_cache::{
    BuildCache, ModuleRecord, digest, fnv1a, interface_fingerprint,
    module_fingerprint, stamp_file,
};
use crate::modules::import_visibility::{
    FileNames, declared_compiler_names, declared_twice, shadowed_imports,
    unimported_names,
};
use crate::modules::interface::ModuleInterface;
use crate::modules::layers::Layer;
use crate::parser::Parser;
use crate::parser::TEST_PREFIX;
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
fn without_tests(module: &mut Module) {
    let ast = &module.ast;
    module.roots.retain(|statement| {
        !matches!(
            ast.stmt(*statement),
            Statement::Constant(name, _)
                if ast.name(*name).contains(TEST_PREFIX)
        )
    });
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
    layers: &[Layer],
    project_root: &Path,
) -> Result<Option<Found>> {
    let Some(found) = locate_import(importing_dir, path, roots, project_root)
    else {
        return Ok(None);
    };
    // The layer rule is about the file an import names rather than the text of
    // the import, so it is asked here, where resolution has settled which file
    // that is. A path spelling its way back up and down again resolves to the
    // same file and is weighed the same.
    if let Some(complaint) = crate::modules::layers::reaching_upward(
        layers,
        importing_dir,
        &found.path,
    ) {
        bail!("{complaint}");
    }
    Ok(Some(found))
}

fn locate_import(
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

// A file's bytes, spliced in where `include_str("path")` was written. The
// four tokens become one string literal before the parser runs, so nothing
// downstream of the lexer knows the feature exists: the type checker sees a
// `str`, the region walk sees a literal, and both backends emit it the way
// they emit any other one.
//
// The path is relative to the file the call is written in and nowhere else,
// which is Rust's rule and the one that keeps a library's shader beside the
// module that uses it wherever the library is installed. Carriage returns are
// dropped, so a file checked out with CRLF line endings reads the same bytes
// as the same file checked out with LF.
/// Which of the three this build is for. Both compilers answer alike because
/// both compile for the machine they run on, and the constants a `when` chooses
/// on are these and nothing else.
pub fn target_constant(name: &str) -> Option<bool> {
    match name {
        "TARGET_WINDOWS" => Some(cfg!(windows)),
        "TARGET_MACOS" => Some(cfg!(target_os = "macos")),
        "TARGET_LINUX" => Some(!cfg!(windows) && !cfg!(target_os = "macos")),
        _ => None,
    }
}

/// What a `when` leaves behind: the stream with the untaken branches gone, the
/// positions beside it, and the lines each kept branch came from.
pub type WhenResolved =
    (Vec<Token>, Vec<crate::lexer::Position>, Vec<(usize, usize)>);

fn is_when(token: &Token) -> bool {
    matches!(token, Token::Identifier(name) if name == "when")
}

// Where a `when` opens one, or nothing for a word that is only a name. The
// shape is what tells them apart: `when` is not reserved, so a program may
// still call a function by that name, and a call is not followed by a block.
fn opens_when(tokens: &[Token], at: usize) -> bool {
    is_when(&tokens[at])
        && matches!(tokens.get(at + 1), Some(Token::LeftParentheses))
}

// The token past the one that closes a run opened at `at`.
fn past_match(
    tokens: &[Token],
    at: usize,
    open: &Token,
    close: &Token,
) -> usize {
    let mut depth = 0usize;
    let mut index = at;
    while index < tokens.len() {
        if &tokens[index] == open {
            depth += 1;
        } else if &tokens[index] == close {
            depth -= 1;
            if depth == 0 {
                return index + 1;
            }
        }
        index += 1;
    }
    tokens.len()
}

// The token past the whole `when` construct opening at `at`, however many
// `else when` arms follow it.
fn past_when(tokens: &[Token], at: usize) -> usize {
    let after_condition = past_match(
        tokens,
        at + 1,
        &Token::LeftParentheses,
        &Token::RightParentheses,
    );
    let after_block = past_match(
        tokens,
        after_condition,
        &Token::LeftBrace,
        &Token::RightBrace,
    );
    if matches!(tokens.get(after_block), Some(Token::Else)) {
        if opens_when(tokens, after_block + 1) {
            return past_when(tokens, after_block + 1);
        }
        return past_match(
            tokens,
            after_block + 1,
            &Token::LeftBrace,
            &Token::RightBrace,
        );
    }
    after_block
}

// The three target constants, wherever a value is read. They are the compiler's
// own and a program may not declare one, so the spelling means this and nothing
// else, and answering it here is what makes it an ordinary boolean outside a
// `when` as well as inside one.
//
// After the conditionals, since a condition is read as the names it was written
// with. A name being declared or naming a field is left alone: the declaration
// is refused by name further on, and telling the reader that beats a parse that
// fails on a `true` they did not write.
fn answer_targets(tokens: &mut [Token]) {
    for index in 0..tokens.len() {
        let Token::Identifier(name) = &tokens[index] else {
            continue;
        };
        let Some(holds) = target_constant(name) else {
            continue;
        };
        if index > 0 && matches!(tokens[index - 1], Token::Dot) {
            continue;
        }
        if matches!(
            tokens.get(index + 1),
            Some(Token::Colon | Token::DoubleColon)
        ) {
            continue;
        }
        tokens[index] = Token::Identifier(holds.to_string());
    }
}

/// A compile-time conditional, decided while the tokens are still tokens.
///
/// The branch that is not taken is removed from the stream, so nothing after
/// this reads it: no name is interned, no type is laid out, and the emitted
/// output does not depend on it having been there. What the taken branch holds
/// stands where the `when` stood, which is what lets one choose between
/// declarations as well as between statements, and is why the block opens no
/// scope of its own: a name bound inside one is bound for what follows it, the
/// way a name written without the `when` would be.
pub fn resolve_when(
    tokens: Vec<Token>,
    positions: Vec<crate::lexer::Position>,
) -> Result<WhenResolved> {
    if !tokens.iter().any(is_when) {
        let mut tokens = tokens;
        answer_targets(&mut tokens);
        return Ok((tokens, positions, Vec::new()));
    }
    let mut held_tokens = tokens;
    let mut held_positions = positions;
    // The lines a kept branch came from. Every statement of a block begins at
    // the same column, and these begin one level in from the block they now
    // belong to, so the rule that reads a deeper line as continuing the one
    // above it is not asked about them.
    let mut lifted: Vec<(usize, usize)> = Vec::new();
    // A branch may hold another, so what is kept is read again. The stream
    // shrinks every round, which is what ends this.
    loop {
        let mut depth = 0usize;
        let mut found = None;
        for index in 0..held_tokens.len() {
            if opens_when(&held_tokens, index) {
                found = Some((index, depth));
                break;
            }
            match held_tokens[index] {
                Token::LeftBrace => depth += 1,
                Token::RightBrace => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
        let Some((at, depth)) = found else {
            answer_targets(&mut held_tokens);
            return Ok((held_tokens, held_positions, lifted));
        };
        let _ = depth;
        let after_condition = past_match(
            &held_tokens,
            at + 1,
            &Token::LeftParentheses,
            &Token::RightParentheses,
        );
        let holds = when_condition(
            &held_tokens[at + 2..after_condition - 1],
            &held_positions[at],
        )?;
        let after_block = past_match(
            &held_tokens,
            after_condition,
            &Token::LeftBrace,
            &Token::RightBrace,
        );
        let mut kept = if holds {
            Some((after_condition + 1, after_block - 1))
        } else {
            None
        };
        let past = if matches!(held_tokens.get(after_block), Some(Token::Else))
        {
            let opens = after_block + 1;
            if opens_when(&held_tokens, opens) {
                let end = past_when(&held_tokens, opens);
                if !holds {
                    kept = Some((opens, end));
                }
                end
            } else {
                let end = past_match(
                    &held_tokens,
                    opens,
                    &Token::LeftBrace,
                    &Token::RightBrace,
                );
                if !holds {
                    kept = Some((opens + 1, end - 1));
                }
                end
            }
        } else {
            after_block
        };
        let (from, to) = kept.unwrap_or((at, at));
        if to > from {
            lifted
                .push((held_positions[from].line, held_positions[to - 1].line));
        }
        let mut rebuilt_tokens = Vec::with_capacity(held_tokens.len());
        let mut rebuilt_positions = Vec::with_capacity(held_positions.len());
        for index in (0..at).chain(from..to).chain(past..held_tokens.len()) {
            rebuilt_tokens.push(held_tokens[index].clone());
            rebuilt_positions.push(held_positions[index]);
        }
        held_tokens = rebuilt_tokens;
        held_positions = rebuilt_positions;
    }
}

// `TARGET_WINDOWS`, `!`, `&&`, `||` and parentheses. A `when` chooses on what
// the build is for, which is known before anything is read, so the vocabulary
// is that and nothing else: a condition a reader has to run the program to
// settle is not one a compile-time conditional can be written over.
fn when_condition(
    tokens: &[Token],
    position: &crate::lexer::Position,
) -> Result<bool> {
    let mut at = 0usize;
    let held = when_or(tokens, &mut at, position)?;
    if at != tokens.len() {
        return Err(crate::diagnostic::LocatedError {
            position: *position,
            message: "a `when` chooses on the target, so its condition is the target constants joined by `&&`, `||` and `!`".to_string(),
        }
        .into());
    }
    Ok(held)
}

fn when_or(
    tokens: &[Token],
    at: &mut usize,
    position: &crate::lexer::Position,
) -> Result<bool> {
    let mut held = when_and(tokens, at, position)?;
    while matches!(tokens.get(*at), Some(Token::Or)) {
        *at += 1;
        held = when_and(tokens, at, position)? || held;
    }
    Ok(held)
}

fn when_and(
    tokens: &[Token],
    at: &mut usize,
    position: &crate::lexer::Position,
) -> Result<bool> {
    let mut held = when_term(tokens, at, position)?;
    while matches!(tokens.get(*at), Some(Token::And)) {
        *at += 1;
        held = when_term(tokens, at, position)? && held;
    }
    Ok(held)
}

fn when_term(
    tokens: &[Token],
    at: &mut usize,
    position: &crate::lexer::Position,
) -> Result<bool> {
    match tokens.get(*at) {
        Some(Token::Bang) => {
            *at += 1;
            Ok(!when_term(tokens, at, position)?)
        }
        Some(Token::LeftParentheses) => {
            *at += 1;
            let held = when_or(tokens, at, position)?;
            if matches!(tokens.get(*at), Some(Token::RightParentheses)) {
                *at += 1;
            }
            Ok(held)
        }
        Some(Token::Identifier(name)) => {
            let named = name.clone();
            *at += 1;
            match target_constant(&named) {
                Some(held) => Ok(held),
                None => Err(crate::diagnostic::LocatedError {
                    position: *position,
                    message: format!(
                        "'{named}' is not one of the targets a `when` chooses on, which are TARGET_WINDOWS, TARGET_LINUX and TARGET_MACOS"
                    ),
                }
                .into()),
            }
        }
        _ => Err(crate::diagnostic::LocatedError {
            position: *position,
            message: "a `when` chooses on the target, so its condition is the target constants joined by `&&`, `||` and `!`".to_string(),
        }
        .into()),
    }
}

pub fn expand_includes(
    tokens: Vec<Token>,
    positions: Vec<crate::lexer::Position>,
    directory: &Path,
) -> Result<(Vec<Token>, Vec<crate::lexer::Position>)> {
    if !tokens.iter().any(is_include_name) {
        return Ok((tokens, positions));
    }
    let mut spliced_tokens = Vec::with_capacity(tokens.len());
    let mut spliced_positions = Vec::with_capacity(positions.len());
    let mut index = 0;
    while index < tokens.len() {
        if !is_include_name(&tokens[index]) {
            spliced_tokens.push(tokens[index].clone());
            spliced_positions.push(positions[index]);
            index += 1;
            continue;
        }
        let path = match (
            tokens.get(index + 1),
            tokens.get(index + 2),
            tokens.get(index + 3),
        ) {
            (
                Some(Token::LeftParentheses),
                Some(Token::StringLiteral(path)),
                Some(Token::RightParentheses),
            ) => path.clone(),
            _ => bail!(
                "at {}: include_str takes one string literal naming a file, so the path is known while the program is being compiled",
                positions[index].describe()
            ),
        };
        let full = directory.join(&path);
        let content = fs::read_to_string(&full).map_err(|_| {
            // Written with `/` throughout, which is how a path reads in every
            // other report and how the same path reads on the other platform.
            anyhow::anyhow!(
                "at {}: include_str: cannot read '{}'",
                positions[index].describe(),
                full.display().to_string().replace('\\', "/")
            )
        })?;
        spliced_tokens.push(Token::StringLiteral(content.replace('\r', "")));
        spliced_positions.push(positions[index]);
        index += 4;
    }
    Ok((spliced_tokens, spliced_positions))
}

fn is_include_name(token: &Token) -> bool {
    matches!(token, Token::Identifier(name) if name == "include_str")
}

// The files a module includes, read off its tokens the way its imports are,
// because whether a cached module is stale is answered before it is parsed.
fn include_paths_in_source(source: &str) -> Vec<String> {
    let mut lexer = Lexer::new(source);
    let Ok(tokens) = lexer.tokenize() else {
        return Vec::new();
    };
    let mut paths = Vec::new();
    for index in 0..tokens.len() {
        if !is_include_name(&tokens[index]) {
            continue;
        }
        if let (
            Some(Token::LeftParentheses),
            Some(Token::StringLiteral(path)),
        ) = (tokens.get(index + 1), tokens.get(index + 2))
        {
            paths.push(path.clone());
        }
    }
    paths
}

// A module's hash covers the files it includes, because their bytes are in the
// object the way its own source is: an edit to a shader has to rebuild the
// module that spliced it in. A module including nothing hashes exactly as it
// always has, so nothing rebuilds for this existing.
fn digest_with_includes(source: &str, directory: &Path) -> String {
    let included = include_paths_in_source(source);
    if included.is_empty() {
        return digest(source);
    }
    let mut text = String::from(source);
    for path in included {
        text.push('\n');
        text.push_str(&path);
        text.push('\n');
        if let Ok(content) = fs::read_to_string(directory.join(&path)) {
            text.push_str(&content);
        }
    }
    digest(&text)
}

pub struct Resolved {
    pub program: Module,
    pub linear_types: HashSet<String>,
    pub tests: Vec<(String, String)>,
    // One per imported module, and empty unless interface checking is on. The
    // compiler does not build from these yet.
    pub interfaces: Vec<crate::modules::interface::ModuleInterface>,
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
    entry: Module,
    base_dir: &Path,
    linear_types: HashSet<String>,
    tests: Vec<(String, String)>,
) -> Result<Resolved> {
    resolve_imports_cached(
        entry,
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
    pub layers: &'a [Layer],
}

// With a cache, a module whose own source and whose imported interfaces are
// all unchanged is not read past its
// first line. It contributes the interface the cache already holds, and its
// object is linked rather than built.
pub fn resolve_imports_cached(
    entry: Module,
    base_dir: &Path,
    linear_types: HashSet<String>,
    tests: Vec<(String, String)>,
    options: Resolution<'_>,
) -> Result<Resolved> {
    let Resolution {
        cache,
        roots,
        layers,
    } = options;
    let resolved = Resolved {
        program: Module::default(),
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
        for statement in &entry.roots {
            if let Statement::Import(path, _) = entry.ast.stmt(*statement) {
                let Some(found) =
                    find_import(base_dir, path, roots, layers, &root)?
                else {
                    continue;
                };
                plan_module(
                    &found,
                    &Planning {
                        root: &root,
                        cache,
                        roots,
                        layers,
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
        layers,
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
        &entry.ast,
        &entry.roots,
        &import_identities(&entry, base_dir, roots, layers, &root),
    ));
    walk.resolve_into(entry, base_dir, "the entry file")?;
    let reports = unimported_names(&walk.files, &walk.module_exports);
    if !reports.is_empty() {
        bail!(
            "an import says what a file may name, and these name what they did not import:\n{}",
            reports.join("\n")
        );
    }
    let reports = declared_compiler_names(&walk.files);
    if !reports.is_empty() {
        bail!(
            "{}",
            reports.join(
                "
"
            )
        );
    }
    let reports = declared_twice(&walk.files);
    if !reports.is_empty() {
        bail!(
            "{}",
            reports.join(
                "
"
            )
        );
    }
    let reports = shadowed_imports(&walk.files, &walk.module_exports);
    if !reports.is_empty() {
        bail!(
            "{}",
            reports.join(
                "
"
            )
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
            stamp_file(&mut interface, 0);
            ModulePlan {
                file: planned.file,
                object: planned.object,
                reused: planned.reused,
                tag: planned.tag,
                module: planned.module.clone(),
                record: ModuleRecord {
                    format_version: crate::modules::build_cache::CACHE_FORMAT,
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
    module: Module,
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
    imported: Imported,
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
    let (tokens, positions) =
        expand_includes(tokens, positions, &directory_of(path))
            .with_context(|| format!("in {}", path.display()))?;
    let (tokens, positions, lifted) = resolve_when(tokens, positions)
        .with_context(|| format!("in {}", path.display()))?;
    let mut parser = Parser::with_positions(&tokens, &positions);
    parser.also_lifted_lines(lifted);
    parser.also_generic(imported.generic_types);
    parser.also_const_functions(imported.const_functions);
    parser.preload_diagnostics(lexer.diagnostics_in_file(file));
    let module = parser
        .parse()
        .with_context(|| format!("parsing {}", path.display()))?;
    Ok(Box::new(ParsedModule {
        module,
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
    layers: &'a [Layer],
}

fn plan_module(
    found: &Found,
    context: &Planning<'_>,
    plans: &mut Plans,
    stack: &mut HashSet<PathBuf>,
) -> Result<PathBuf> {
    let Planning {
        root,
        cache,
        roots,
        layers,
    } = context;
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
    let source_hash = digest_with_includes(&source, &directory_of(full));
    let record = cache.load(&tag, &source_hash);

    let mut parsed: Option<Box<ParsedModule>> = None;
    let mut interface = match &record {
        Some(record) => {
            let mut interface = record.interface.clone();
            stamp_file(&mut interface, file);
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
                    layers,
                    root,
                ),
            )?;
            let interface = ModuleInterface::of(
                &module,
                &fresh.module.ast,
                &fresh.module.roots,
                &fresh.exports,
                &fresh.linear_types,
            );
            parsed = Some(fresh);
            interface
        }
    };
    let imports: Vec<String> = match (&record, &parsed) {
        (Some(record), _) => record.imports.clone(),
        (None, Some(fresh)) => import_paths(&fresh.module),
        (None, None) => Vec::new(),
    };

    let directory = full.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut closure: BTreeMap<String, String> = BTreeMap::new();
    for import in &imports {
        let Some(child_found) =
            find_import(&directory, import, roots, context.layers, root)?
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
            imported_generic_types(&source, &directory, roots, layers, root),
        )?;
        interface = ModuleInterface::of(
            &module,
            &fresh.module.ast,
            &fresh.module.roots,
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
    module: &Module,
    base_dir: &Path,
    roots: &[SearchRoot],
    layers: &[Layer],
    root: &Path,
) -> Vec<PathBuf> {
    import_paths(module)
        .iter()
        .filter_map(|path| {
            find_import(base_dir, path, roots, layers, root)
                .ok()
                .flatten()
                .map(|found| {
                    found
                        .path
                        .canonicalize()
                        .unwrap_or_else(|_| found.path.clone())
                })
        })
        .collect()
}

fn import_paths(module: &Module) -> Vec<String> {
    module
        .roots
        .iter()
        .filter_map(|statement| match module.ast.stmt(*statement) {
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
    layers: &'a [Layer],
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

type Contribution = (Module, HashSet<String>, String);

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
        mut source: Module,
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

        for statement in std::mem::take(&mut source.roots) {
            let Statement::Import(path, renames) = source.ast.stmt(statement)
            else {
                body.push(statement);
                continue;
            };
            let path = path.clone();
            let renames: Vec<(String, String)> = source
                .ast
                .renames_in(*renames)
                .iter()
                .map(|held| {
                    (
                        source.ast.name(held.exported).to_string(),
                        source.ast.name(held.local).to_string(),
                    )
                })
                .collect();
            let path = &path;

            let Some(found) = find_import(
                base_dir,
                path,
                self.roots,
                self.layers,
                self.root,
            )?
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
                .map(|held| (held.0.as_str(), held.1.as_str()))
                .collect();
            for (exported, local) in &renames {
                if !exports.contains(exported) {
                    bail!(
                        "'{}' does not export '{}', so there is nothing to read as '{}'",
                        owner,
                        exported,
                        local
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
            if let Some(name) = top_level_name(&source.ast, *statement) {
                let name = name.to_string();
                view.remove(&name);
                ambiguous.remove(&name);
            }
        }

        if !ambiguous.is_empty() {
            let used = FileNames::of(module, &source.ast, &body, &[]);
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
            renamer.run(&mut source.ast, &body);
        }
        let offset =
            splice_positions(&mut self.resolved.program.ast, &source.ast);
        let splicer = Splicer::new(&source.ast, offset);
        for statement in body {
            let copied = splicer.statement(
                &mut self.resolved.program.ast,
                statement,
                &mut |name| name.to_string(),
            );
            self.resolved.program.roots.push(copied);
        }
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
            &imported.ast,
            &imported.roots,
            &import_identities(
                &imported,
                &child_dir,
                self.roots,
                self.layers,
                self.root,
            ),
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
            let roots = imported.roots.clone();
            let renamer = Renamer { renames };
            renamer.run(&mut imported.ast, &roots);
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
            let mut contribution = Module::default();
            for path in imports {
                let id = contribution.ast.push_stmt(
                    Statement::Import(path, Range32::EMPTY),
                    TokenSpan::NONE,
                );
                contribution.roots.push(id);
            }
            // The module's object is being linked rather than rebuilt, so it
            // contributes signatures where it can and bodies only where a
            // caller needs one. See `as_declaration`.
            let held = &interface.declarations;
            let offset = splice_positions(&mut contribution.ast, &held.ast);
            let splicer = Splicer::new(&held.ast, offset);
            for statement in &held.roots {
                let copied =
                    match crate::modules::build_cache::push_as_declaration(
                        &mut contribution.ast,
                        &splicer,
                        *statement,
                    ) {
                        Some(declared) => declared,
                        None => splicer.statement(
                            &mut contribution.ast,
                            *statement,
                            &mut |name| name.to_string(),
                        ),
                    };
                contribution.roots.push(copied);
            }
            let exports = interface.exports.into_iter().collect();
            return Ok((contribution, exports, tag));
        };

        self.resolved
            .linear_types
            .extend(parsed.linear_types.iter().cloned());
        self.resolved.tests.extend(parsed.tests.iter().cloned());
        let exports: HashSet<String> = parsed.exports.into_iter().collect();
        let mut contribution = parsed.module;
        self.check_and_reduce(&interface, &mut contribution)?;
        Ok((contribution, exports, tag))
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
                self.layers,
                self.root,
            ),
        )?;
        self.resolved
            .linear_types
            .extend(parsed.linear_types.iter().cloned());

        let exports: HashSet<String> = parsed.exports.iter().cloned().collect();
        let tag = module_tag_of(module_name);
        let mut module = parsed.module;
        without_tests(&mut module);

        // The interface is derived at the one place a module is parsed, which is what keeps it
        // from drifting out of step with the source it describes.
        if crate::modules::interface::interfaces_are_checked()
            || crate::modules::interface::built_from_interfaces()
        {
            let interface = ModuleInterface::of(
                module_name,
                &module.ast,
                &module.roots,
                &parsed.exports,
                &parsed.linear_types,
            );
            self.check_and_reduce(&interface, &mut module)?;
        }
        Ok((module, exports, tag))
    }

    fn check_and_reduce(
        &mut self,
        interface: &ModuleInterface,
        module: &mut Module,
    ) -> Result<()> {
        check_and_reduce(interface, module, &mut self.resolved.interfaces)
    }
}

fn check_and_reduce(
    interface: &ModuleInterface,
    module: &mut Module,
    interfaces: &mut Vec<ModuleInterface>,
) -> Result<()> {
    if !crate::modules::interface::interfaces_are_checked()
        && !crate::modules::interface::built_from_interfaces()
    {
        return Ok(());
    }
    crate::modules::interface::check_interface_round_trip(interface)?;
    crate::modules::interface::check_interface_covers_exports(interface)?;
    crate::modules::interface::check_interface_is_closed(
        interface,
        &module.ast,
        &module.roots,
    )?;

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
    if crate::modules::interface::built_from_interfaces() {
        let mut rebuilt = Module::default();
        for statement in &module.roots {
            if matches!(module.ast.stmt(*statement), Statement::Import(..)) {
                let offset = rebuilt.ast.token_positions.len() as u32;
                let source_span = module.ast.stmt_span(*statement);
                rebuilt
                    .ast
                    .token_positions
                    .push(module.ast.position_of(source_span));
                let splicer = Splicer::new(&module.ast, 0);
                let copied = splicer.statement(
                    &mut rebuilt.ast,
                    *statement,
                    &mut |name| name.to_string(),
                );
                let span = TokenSpan {
                    first: offset,
                    last: offset,
                };
                let index = copied.0 as usize;
                rebuilt.ast.stmt_spans[index] = span;
                rebuilt.roots.push(copied);
            }
        }
        let held = &interface.declarations;
        let offset = splice_positions(&mut rebuilt.ast, &held.ast);
        let splicer = Splicer::new(&held.ast, offset);
        for statement in &held.roots {
            let copied =
                splicer.statement(&mut rebuilt.ast, *statement, &mut |name| {
                    name.to_string()
                });
            rebuilt.roots.push(copied);
        }
        *module = rebuilt;
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
    layers: &[Layer],
    project_root: &Path,
) -> Imported {
    let mut found = Imported::default();
    let mut seen = HashSet::new();
    let scan = Scan {
        roots,
        layers,
        project_root,
    };
    for path in import_paths_in_source(source) {
        collect_generic_types(base_dir, &path, &scan, &mut found, &mut seen);
        // A compile-time call names a function this file can name, and an
        // `import` is what says which those are. A file two imports away
        // exports nothing to this one, so it is read for generic type names
        // and for nothing else.
        collect_const_functions(base_dir, &path, &scan, &mut found);
    }
    found
}

/// What a file's imports offer the parse that has not happened yet: which
/// names are generic types, and which functions a compile-time call may read.
/// Both are read off the imported files' tokens, in the one walk over them.
#[derive(Default)]
pub struct Imported {
    pub generic_types: crate::parser::GenericDefaults,
    pub const_functions: HashMap<String, std::rc::Rc<Vec<Token>>>,
}

// What the walk above needs everywhere and reads nowhere, so the recursion
// carries one reference rather than two positional arguments.
struct Scan<'a> {
    roots: &'a [SearchRoot],
    layers: &'a [Layer],
    project_root: &'a Path,
}

fn collect_generic_types(
    importing_dir: &Path,
    path: &str,
    scan: &Scan<'_>,
    found: &mut Imported,
    seen: &mut HashSet<PathBuf>,
) {
    let Some(located) = find_import(
        importing_dir,
        path,
        scan.roots,
        scan.layers,
        scan.project_root,
    )
    .ok()
    .flatten() else {
        return;
    };
    let key = located
        .path
        .canonicalize()
        .unwrap_or_else(|_| located.path.clone());
    if !seen.insert(key.clone()) {
        return;
    }
    let Ok(source) = fs::read_to_string(&located.path) else {
        return;
    };
    let mut lexer = Lexer::new(&source);
    if let Ok(tokens) = lexer.tokenize() {
        found
            .generic_types
            .extend(crate::parser::scan_generic_types(&tokens));
    }
    let directory = directory_of(&located.path);
    for next in import_paths_in_source(&source) {
        collect_generic_types(&directory, &next, scan, found, seen);
    }
}

// What one imported file exports as a function body a compile-time call may
// read. Only the file named on the `import` line, since that is what a program
// may name.
fn collect_const_functions(
    importing_dir: &Path,
    path: &str,
    scan: &Scan<'_>,
    found: &mut Imported,
) {
    let Some(located) = find_import(
        importing_dir,
        path,
        scan.roots,
        scan.layers,
        scan.project_root,
    )
    .ok()
    .flatten() else {
        return;
    };
    let Ok(source) = fs::read_to_string(&located.path) else {
        return;
    };
    let mut lexer = Lexer::new(&source);
    if let Ok(tokens) = lexer.tokenize() {
        found
            .const_functions
            .extend(crate::parser::exported_function_bodies(&tokens));
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

fn top_level_name(ast: &Ast, statement: StmtId) -> Option<&str> {
    match ast.stmt(statement) {
        Statement::Constant(name, _)
        | Statement::Struct(name, _, _)
        | Statement::Enum(name, _, _)
        | Statement::Flags(name, _, _)
        | Statement::TypeAlias(name, _)
        | Statement::Extern { name, .. }
        | Statement::Declared { name, .. } => Some(ast.name(*name)),
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

fn module_renames(module: &Module, tag: &str) -> HashMap<String, String> {
    let mut renames = HashMap::new();
    for statement in &module.roots {
        // An `extern` name is not this module's to rename. It is the symbol a
        // C library defines, so mangling it produces a link against a
        // name nothing exports. That only showed up once a module other than
        // the entry file declared one, which is what a standard library doing
        // its own IO is.
        if matches!(module.ast.stmt(*statement), Statement::Extern { .. }) {
            continue;
        }
        if let Some(name) = top_level_name(&module.ast, *statement) {
            // A function written as `extern fn` with a body is named by a C
            // caller, so it keeps what it was written as for the same reason
            // the declaration above does.
            if module.ast.is_exported_symbol(name) {
                continue;
            }
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

type Scope = Vec<HashSet<crate::ast::Symbol>>;

impl Renamer {
    fn run(&self, ast: &mut Ast, roots: &[StmtId]) {
        let mut scope: Scope = vec![HashSet::new()];
        for statement in roots {
            self.statement(ast, *statement, &mut scope);
        }
    }

    // The renamed symbol for a name, unless a local binding shadows it. The
    // symbol table is injective, so a frame holding the symbol is a frame
    // holding the spelling.
    fn mapped(
        &self,
        ast: &mut Ast,
        name: crate::ast::Symbol,
        scope: &Scope,
    ) -> Option<crate::ast::Symbol> {
        if scope.iter().any(|frame| frame.contains(&name)) {
            return None;
        }
        let mangled = self.renames.get(ast.name(name))?.clone();
        Some(ast.intern(&mangled))
    }

    fn plain(
        &self,
        ast: &mut Ast,
        name: crate::ast::Symbol,
    ) -> Option<crate::ast::Symbol> {
        let mangled = self.renames.get(ast.name(name))?.clone();
        Some(ast.intern(&mangled))
    }

    fn block(&self, ast: &mut Ast, block: Range32, scope: &mut Scope) {
        scope.push(HashSet::new());
        for index in block.indices() {
            let statement = ast.stmt_list[index];
            self.statement(ast, statement, scope);
        }
        scope.pop();
    }

    fn bind(&self, scope: &mut Scope, name: crate::ast::Symbol) {
        if let Some(frame) = scope.last_mut() {
            frame.insert(name);
        }
    }

    fn statement(&self, ast: &mut Ast, id: StmtId, scope: &mut Scope) {
        match ast.stmt(id).clone() {
            Statement::Constant(name, value) => {
                if let Some(mangled) = self.plain(ast, name) {
                    let Statement::Constant(held, _) =
                        &mut ast.statements[id.0 as usize]
                    else {
                        return;
                    };
                    *held = mangled;
                }
                self.expression(ast, value, scope);
            }
            Statement::Struct(name, _, fields) => {
                if let Some(mangled) = self.plain(ast, name) {
                    // Packing is recorded against the name, so the record has
                    // to move with it. Left behind, a `packed struct` from a
                    // module is laid out as an ordinary one wherever it is
                    // read, and nothing says the two disagree.
                    if ast.packed_structs.contains(&name) {
                        ast.packed_structs.push(mangled);
                    }
                    let Statement::Struct(held, _, _) =
                        &mut ast.statements[id.0 as usize]
                    else {
                        return;
                    };
                    *held = mangled;
                }
                for index in fields.indices() {
                    self.ty(&mut ast.struct_fields[index].field_type);
                }
            }
            Statement::Enum(name, _, variants) => {
                if let Some(mangled) = self.plain(ast, name) {
                    let Statement::Enum(held, _, _) =
                        &mut ast.statements[id.0 as usize]
                    else {
                        return;
                    };
                    *held = mangled;
                }
                for variant_index in variants.indices() {
                    if let Some(fields) =
                        ast.enum_variants[variant_index].fields
                    {
                        for index in fields.indices() {
                            self.ty(&mut ast.struct_fields[index].field_type);
                        }
                    }
                }
            }
            Statement::TypeAlias(name, _) | Statement::Flags(name, _, _) => {
                if let Some(mangled) = self.plain(ast, name) {
                    let (Statement::TypeAlias(held, _)
                    | Statement::Flags(held, _, _)) =
                        &mut ast.statements[id.0 as usize]
                    else {
                        return;
                    };
                    *held = mangled;
                }
                let (Statement::TypeAlias(_, ty) | Statement::Flags(_, ty, _)) =
                    &mut ast.statements[id.0 as usize]
                else {
                    return;
                };
                self.ty(ty);
            }
            Statement::Extern {
                name,
                params,
                return_type,
                ..
            } => {
                if let Some(mangled) = self.plain(ast, name) {
                    let Statement::Extern { name: held, .. } =
                        &mut ast.statements[id.0 as usize]
                    else {
                        return;
                    };
                    *held = mangled;
                }
                for index in params.indices() {
                    if let Some(ty) = &mut ast.parameters[index].type_annotation
                    {
                        self.ty(ty);
                    }
                    if let Some(ty) =
                        &mut ast.parameters[index].compile_time_signature
                    {
                        self.ty(ty);
                    }
                }
                if return_type.is_some() {
                    let Statement::Extern {
                        return_type: Some(ty),
                        ..
                    } = &mut ast.statements[id.0 as usize]
                    else {
                        return;
                    };
                    self.ty(ty);
                }
            }
            Statement::Declared {
                name,
                params,
                return_sig,
            } => {
                if let Some(mangled) = self.plain(ast, name) {
                    let Statement::Declared { name: held, .. } =
                        &mut ast.statements[id.0 as usize]
                    else {
                        return;
                    };
                    *held = mangled;
                }
                self.parameters(ast, params, &mut vec![HashSet::new()]);
                self.return_signature(ast, return_sig);
            }
            Statement::Let {
                name,
                type_annotation,
                value,
                ..
            } => {
                self.expression(ast, value, scope);
                if type_annotation.is_some() {
                    let Statement::Let {
                        type_annotation: Some(ty),
                        ..
                    } = &mut ast.statements[id.0 as usize]
                    else {
                        return;
                    };
                    self.ty(ty);
                }
                self.bind(scope, name);
            }
            Statement::LetMultiple(bindings, value) => {
                self.expression(ast, value, scope);
                for binding in bindings.indices() {
                    let name = ast.bindings[binding].name;
                    self.bind(scope, name);
                }
            }
            Statement::Return(value) => self.expression(ast, value, scope),
            Statement::Expression(value) => self.expression(ast, value, scope),
            Statement::Assignment(target, value) => {
                self.expression(ast, target, scope);
                self.expression(ast, value, scope);
            }
            Statement::Defer(inner) | Statement::ErrDefer(inner) => {
                self.statement(ast, inner, scope)
            }
            Statement::For(variable, _, range, body) => {
                self.expression(ast, range, scope);
                scope.push(HashSet::new());
                self.bind(scope, variable);
                for index in body.indices() {
                    let statement = ast.stmt_list[index];
                    self.statement(ast, statement, scope);
                }
                scope.pop();
            }
            Statement::While(condition, body) => {
                self.expression(ast, condition, scope);
                self.block(ast, body, scope);
            }
            Statement::With(capability, body) => {
                if let Some(mangled) = self.mapped(ast, capability, scope) {
                    let Statement::With(held, _) =
                        &mut ast.statements[id.0 as usize]
                    else {
                        return;
                    };
                    *held = mangled;
                }
                self.block(ast, body, scope);
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

    fn return_signature(&self, ast: &mut Ast, signature: SignatureId) {
        let values = {
            let held = &mut ast.signatures[signature.0 as usize];
            let mut values = None;
            match &mut held.kind {
                ReturnKind::None => {}
                ReturnKind::Single(ty) => self.ty(ty),
                ReturnKind::Multiple(range) => values = Some(*range),
                ReturnKind::Fallible(value, failure) => {
                    self.ty(value);
                    self.ty(failure);
                }
            }
            for capability in held.uses.iter_mut() {
                self.ty(capability);
            }
            values
        };
        if let Some(values) = values {
            for index in values.indices() {
                self.ty(&mut ast.return_values[index].value_type);
            }
        }
    }

    fn parameters(&self, ast: &mut Ast, params: Range32, scope: &mut Scope) {
        for index in params.indices() {
            if let Some(ty) = &mut ast.parameters[index].type_annotation {
                self.ty(ty);
            }
            // What a compile-time parameter is declared to take is a type like
            // any other, and a bundle parameter names one this module imported.
            if let Some(ty) = &mut ast.parameters[index].compile_time_signature
            {
                self.ty(ty);
            }
            let name = ast.parameters[index].name;
            self.bind(scope, name);
        }
    }

    fn expression(&self, ast: &mut Ast, id: ExprId, scope: &mut Scope) {
        match ast.expr(id).clone() {
            Expression::Identifier(name) => {
                if let Some(mangled) = self.mapped(ast, name, scope) {
                    let Expression::Identifier(held) =
                        &mut ast.expressions[id.0 as usize]
                    else {
                        return;
                    };
                    *held = mangled;
                }
            }
            // An array literal holds expressions, so a name written inside one
            // is a name this module may have imported. Treating every literal
            // as a leaf left those unmapped, and a call to an imported function
            // written inside an array literal reached the backend under a name
            // nothing had defined.
            Expression::Literal(crate::ast::Literal::Array(elements)) => {
                for index in elements.indices() {
                    let element = ast.expr_list[index];
                    self.expression(ast, element, scope);
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
                self.expression(ast, operand, scope)
            }
            Expression::Infix(left, _, right)
            | Expression::Index(left, right) => {
                self.expression(ast, left, scope);
                self.expression(ast, right, scope);
            }
            Expression::Range(start, end, _) => {
                self.expression(ast, start, scope);
                self.expression(ast, end, scope);
            }
            Expression::If(condition, consequence, alternative) => {
                self.expression(ast, condition, scope);
                self.block(ast, consequence, scope);
                if let Some(block) = alternative {
                    self.block(ast, block, scope);
                }
            }
            Expression::Function(params, return_sig, body)
            | Expression::Proc(params, return_sig, body) => {
                scope.push(HashSet::new());
                self.parameters(ast, params, scope);
                self.return_signature(ast, return_sig);
                for index in body.indices() {
                    let statement = ast.stmt_list[index];
                    self.statement(ast, statement, scope);
                }
                scope.pop();
            }
            Expression::Call(callee, arguments) => {
                self.expression(ast, callee, scope);
                for index in arguments.indices() {
                    let argument = ast.expr_list[index];
                    self.expression(ast, argument, scope);
                }
            }
            Expression::FieldAccess(base, _) => {
                self.expression(ast, base, scope)
            }
            Expression::StructInit(name, fields) => {
                let renamed = match self.plain(ast, name) {
                    Some(mangled) => Some(mangled),
                    None => {
                        // A literal that says which instance it is names it as
                        // one string, `Ordering<i64>`, so looking the whole
                        // thing up finds nothing. Both halves are renamed the
                        // way a type annotation naming the same instance is.
                        let spelled = ast.name(name).to_string();
                        self.generic_instance(&spelled)
                            .map(|held| ast.intern(&held))
                    }
                };
                if let Some(mangled) = renamed {
                    let Expression::StructInit(held, _) =
                        &mut ast.expressions[id.0 as usize]
                    else {
                        return;
                    };
                    *held = mangled;
                }
                for index in fields.indices() {
                    let value = ast.named_exprs[index].value;
                    self.expression(ast, value, scope);
                }
            }
            Expression::EnumVariantInit(enum_name, _, fields) => {
                if let Some(mangled) = self.plain(ast, enum_name) {
                    let Expression::EnumVariantInit(held, _, _) =
                        &mut ast.expressions[id.0 as usize]
                    else {
                        return;
                    };
                    *held = mangled;
                }
                for index in fields.indices() {
                    let value = ast.named_exprs[index].value;
                    self.expression(ast, value, scope);
                }
            }
            Expression::TypeValue(..) => {
                let Expression::TypeValue(ty) =
                    &mut ast.expressions[id.0 as usize]
                else {
                    return;
                };
                self.ty(ty);
            }
            Expression::Tuple(elements) => {
                for index in elements.indices() {
                    let element = ast.expr_list[index];
                    self.expression(ast, element, scope);
                }
            }
            Expression::Switch(scrutinee, cases) => {
                self.expression(ast, scrutinee, scope);
                for index in cases.indices() {
                    let case = ast.cases[index];
                    self.switch_case(ast, case, scope);
                }
            }
            Expression::Unsafe(body) => self.block(ast, body, scope),
            Expression::UnsafeFn(inner) => self.expression(ast, inner, scope),
        }
    }

    fn switch_case(
        &self,
        ast: &mut Ast,
        case: crate::ast::SwitchCase,
        scope: &mut Scope,
    ) {
        scope.push(HashSet::new());
        self.pattern(ast, case.pattern, scope);
        for index in case.body.indices() {
            let statement = ast.stmt_list[index];
            self.statement(ast, statement, scope);
        }
        scope.pop();
    }

    fn pattern(&self, ast: &mut Ast, id: PatternId, scope: &mut Scope) {
        match ast.pattern(id).clone() {
            Pattern::EnumVariant {
                enum_name,
                bindings,
                ..
            } => {
                if let Some(name) = enum_name
                    && let Some(mangled) = self.plain(ast, name)
                {
                    let Pattern::EnumVariant {
                        enum_name: Some(held),
                        ..
                    } = &mut ast.patterns[id.0 as usize]
                    else {
                        return;
                    };
                    *held = mangled;
                }
                for index in bindings.indices() {
                    let binding = ast.pattern_bindings[index].binding;
                    self.bind(scope, binding);
                }
            }
            Pattern::Tuple(patterns) | Pattern::Or(patterns) => {
                for index in patterns.indices() {
                    let pattern = ast.pattern_list[index];
                    self.pattern(ast, pattern, scope);
                }
            }
            Pattern::Wildcard | Pattern::Literal(_) | Pattern::Range { .. } => {
            }
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

    // The build cache answers for a module by its source hash, and a module
    // that includes a file carries that file's bytes in its object. An edit to
    // the included file has to change the hash or `--incremental` links the
    // old bytes; a module including nothing has to hash exactly as it always
    // did or every cached module rebuilds once for this feature existing.
    #[test]
    fn an_included_file_is_part_of_the_module_hash() {
        let directory = std::env::temp_dir().join("frost_include_hash_test");
        std::fs::create_dir_all(&directory).unwrap();
        let shader = directory.join("shape.wgsl");
        let source = "SHADER :: include_str(\"shape.wgsl\")\n";
        std::fs::write(&shader, "one").unwrap();
        let first = digest_with_includes(source, &directory);
        std::fs::write(&shader, "two").unwrap();
        let second = digest_with_includes(source, &directory);
        assert_ne!(first, second);
        let plain = "ANSWER :: 42\n";
        assert_eq!(digest_with_includes(plain, &directory), digest(plain));
        let _ = std::fs::remove_dir_all(&directory);
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
