use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::build_cache::{
    BuildCache, ModuleRecord, digest, fnv1a, interface_fingerprint,
    module_fingerprint, stamp_file,
};
use crate::interface::ModuleInterface;
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::parser::{
    Block, Expression, Parameter, Pattern, ReturnKind, ReturnSignature,
    Spanned, Statement, SwitchCase,
};
use crate::types::Type;

pub struct Resolved {
    pub statements: Vec<Spanned<Statement>>,
    pub linear_types: HashSet<String>,
    pub tests: Vec<(String, String)>,
    // One per imported module, and empty unless interface checking is on. The
    // compiler does not build from these yet; see step 2 of
    // docs/separate-compilation.md.
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

pub fn resolve_imports(
    statements: Vec<Spanned<Statement>>,
    base_dir: &Path,
    linear_types: HashSet<String>,
    tests: Vec<(String, String)>,
) -> Result<Resolved> {
    resolve_imports_cached(statements, base_dir, linear_types, tests, None)
}

// Step 5 of docs/separate-compilation.md. With a cache, a module whose own
// source and whose imported interfaces are all unchanged is not read past its
// first line: it contributes the interface the cache already holds, and its
// object is linked rather than built.
pub fn resolve_imports_cached(
    statements: Vec<Spanned<Statement>>,
    base_dir: &Path,
    linear_types: HashSet<String>,
    tests: Vec<(String, String)>,
    cache: Option<&BuildCache>,
) -> Result<Resolved> {
    let resolved = Resolved {
        statements: Vec::new(),
        linear_types,
        tests,
        interfaces: Vec::new(),
        modules: Vec::new(),
    };
    // The directory of the file named on the command line is the project root,
    // and a module's identity is its path relative to that. See the "what is a
    // project root" question in docs/separate-compilation.md, which this is the
    // smallest answer to.
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
            if let Statement::Import(path) = &statement.node {
                plan_module(
                    &base_dir.join(path),
                    &root,
                    cache,
                    &mut plans,
                    &mut stack,
                )?;
            }
        }
    }

    let mut walk = Walk {
        root: &root,
        seen: HashSet::new(),
        resolved,
        plans,
    };
    walk.resolve_into(statements, base_dir)?;
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
    // Absent when the module was answered for from the cache, which is the
    // whole point: its source was never parsed.
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

fn plan_module(
    full: &Path,
    root: &Path,
    cache: &BuildCache,
    plans: &mut Plans,
    stack: &mut HashSet<PathBuf>,
) -> Result<PathBuf> {
    let key = full.canonicalize().unwrap_or_else(|_| full.to_path_buf());
    if plans.contains_key(&key) || !stack.insert(key.clone()) {
        return Ok(key);
    }

    let source = fs::read_to_string(full).with_context(|| {
        format!("failed to read imported file: {}", full.display())
    })?;
    let module = relative_module_name(&key, root);
    let file = crate::source_map::register(&module);
    let tag = module_tag(&key, root);
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
            let fresh = parse_module(&source, file, full)?;
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
        let child =
            plan_module(&directory.join(import), root, cache, plans, stack)?;
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
        let fresh = parse_module(&source, file, full)?;
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

fn import_paths(statements: &[Spanned<Statement>]) -> Vec<String> {
    statements
        .iter()
        .filter_map(|statement| match &statement.node {
            Statement::Import(path) => Some(path.clone()),
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
fn module_tag(path: &Path, root: &Path) -> String {
    format!(
        "{:016x}",
        fnv1a(relative_module_name(path, root).as_bytes())
    )
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
    seen: HashSet<PathBuf>,
    resolved: Resolved,
    plans: Plans,
}

type Contribution = (Vec<Spanned<Statement>>, HashSet<String>, String);

impl Walk<'_> {
    fn resolve_into(
        &mut self,
        statements: Vec<Spanned<Statement>>,
        base_dir: &Path,
    ) -> Result<()> {
        for statement in statements {
            let Statement::Import(path) = &statement.node else {
                self.resolved.statements.push(statement);
                continue;
            };

            let full = base_dir.join(path);
            let key = full.canonicalize().unwrap_or_else(|_| full.clone());
            if !self.seen.insert(key.clone()) {
                continue;
            }

            let (mut imported, exports, tag) = if self.plans.contains_key(&key)
            {
                self.planned_module(&key)?
            } else {
                self.read_module(&full, &key)?
            };

            let renames = private_renames(&imported, &exports, &tag);
            if !renames.is_empty() {
                let renamer = Renamer { renames };
                renamer.block(&mut imported, &mut Vec::new());
            }

            let child_dir =
                full.parent().map(Path::to_path_buf).unwrap_or_default();
            self.resolve_into(imported, &child_dir)?;
        }
        Ok(())
    }

    // What a planned module contributes. A module the plan could answer for
    // contributes its interface and its own import lines, which is exactly what
    // the `FROST_BUILD_FROM_INTERFACES` oracle has been checking on every commit
    // since step 4; the difference is that here its object is not rebuilt
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
                .map(|path| Spanned::from(Statement::Import(path)))
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

    fn read_module(&mut self, full: &Path, key: &Path) -> Result<Contribution> {
        let source = fs::read_to_string(full).with_context(|| {
            format!("failed to read imported file: {}", full.display())
        })?;
        let module_name = relative_module_name(key, self.root);
        let file = crate::source_map::register(&module_name);
        let parsed = parse_module(&source, file, full)?;
        self.resolved
            .linear_types
            .extend(parsed.linear_types.iter().cloned());
        self.resolved.tests.extend(parsed.tests.iter().cloned());

        let exports: HashSet<String> = parsed.exports.iter().cloned().collect();
        let tag = module_tag(key, self.root);
        let mut statements = parsed.statements;

        // Steps 2 and 4 of docs/separate-compilation.md. The interface is
        // derived at the one place a module is parsed, which is what keeps it
        // from drifting out of step with the source it describes.
        if crate::interface::interfaces_are_checked()
            || crate::interface::built_from_interfaces()
        {
            let interface = ModuleInterface::of(
                &module_name,
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
    // is simply gone.
    if crate::interface::built_from_interfaces() {
        let mut rebuilt: Vec<Spanned<Statement>> = statements
            .iter()
            .filter(|statement| matches!(statement.node, Statement::Import(_)))
            .cloned()
            .collect();
        rebuilt.extend(interface.declarations.iter().cloned());
        *statements = rebuilt;
    }
    interfaces.push(interface.clone());
    Ok(())
}

fn top_level_name(statement: &Statement) -> Option<&str> {
    match statement {
        Statement::Constant(name, _)
        | Statement::Struct(name, _, _)
        | Statement::Enum(name, _)
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

fn private_renames(
    statements: &[Spanned<Statement>],
    exports: &HashSet<String>,
    tag: &str,
) -> HashMap<String, String> {
    let mut renames = HashMap::new();
    for statement in statements {
        if let Some(name) = top_level_name(&statement.node)
            && !exports.contains(name)
        {
            renames.insert(name.to_string(), format!("__m{tag}_{name}"));
        }
    }
    renames
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
            Statement::Enum(name, variants) => {
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
            Statement::TypeAlias(name, ty) => {
                if let Some(mangled) = self.renames.get(name.as_str()) {
                    *name = mangled.clone();
                }
                self.ty(ty);
            }
            Statement::Extern {
                name,
                params,
                return_type,
            } => {
                if let Some(mangled) = self.renames.get(name.as_str()) {
                    *name = mangled.clone();
                }
                for param in params.iter_mut() {
                    if let Some(ty) = &mut param.type_annotation {
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
            Statement::Return(value) => self.expression(value, scope),
            Statement::Expression(value) => self.expression(value, scope),
            Statement::Assignment(target, value) => {
                self.expression(target, scope);
                self.expression(value, scope);
            }
            Statement::Defer(inner) => self.statement(inner, scope),
            Statement::For(variable, range, body) => {
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
            Statement::Break | Statement::Continue | Statement::Import(_) => {}
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
            ReturnKind::Named(params) => {
                for param in params.iter_mut() {
                    self.ty(&mut param.param_type);
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
            Expression::Literal(_) | Expression::Boolean(_) => {}
            Expression::Prefix(_, operand)
            | Expression::AddressOf(operand)
            | Expression::Borrow(operand)
            | Expression::BorrowMut(operand)
            | Expression::Try(operand)
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
            Expression::Sizeof(ty) | Expression::TypeValue(ty) => self.ty(ty),
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
            | Type::Distinct(inner)
            | Type::Handle(inner)
            | Type::Optional(inner) => self.ty(inner),
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
            module_tag(Path::new("/project/lib/a.frost"), root),
            module_tag(Path::new("/project/lib/a.frost"), root)
        );
        assert_ne!(
            module_tag(Path::new("/project/lib/a.frost"), root),
            module_tag(Path::new("/project/lib/b.frost"), root)
        );
    }
}
