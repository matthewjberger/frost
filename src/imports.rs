use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

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
}

pub fn resolve_imports(
    statements: Vec<Spanned<Statement>>,
    base_dir: &Path,
    linear_types: HashSet<String>,
    tests: Vec<(String, String)>,
) -> Result<Resolved> {
    let mut seen = HashSet::new();
    let mut resolved = Resolved {
        statements: Vec::new(),
        linear_types,
        tests,
        interfaces: Vec::new(),
    };
    // The directory of the file named on the command line is the project root,
    // and a module's identity is its path relative to that. See the "what is a
    // project root" question in docs/separate-compilation.md, which this is the
    // smallest answer to.
    let root = base_dir.canonicalize().unwrap_or_else(|_| base_dir.into());
    resolve_into(statements, base_dir, &root, &mut seen, &mut resolved)?;
    Ok(resolved)
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

// FNV-1a, written out rather than taken from the standard library because the
// hash has to mean the same thing in every build of the compiler, and
// `DefaultHasher` promises only that it is consistent within one version.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn resolve_into(
    statements: Vec<Spanned<Statement>>,
    base_dir: &Path,
    root: &Path,
    seen: &mut HashSet<PathBuf>,
    resolved: &mut Resolved,
) -> Result<()> {
    for statement in statements {
        let Statement::Import(path) = &statement.node else {
            resolved.statements.push(statement);
            continue;
        };

        let full = base_dir.join(path);
        let key = full.canonicalize().unwrap_or_else(|_| full.clone());
        if !seen.insert(key.clone()) {
            continue;
        }

        let source = fs::read_to_string(&full).with_context(|| {
            format!("failed to read imported file: {}", full.display())
        })?;
        let mut lexer = Lexer::new(&source);
        let tokens = lexer
            .tokenize()
            .with_context(|| format!("lexing {}", full.display()))?;
        let module_name = relative_module_name(&key, root);
        // Every position the lexer produced for this file belongs to this file,
        // and stamping them here is the only place that knows which file it is.
        let file = crate::source_map::register(&module_name);
        let positions: Vec<_> = lexer
            .positions()
            .iter()
            .map(|position| crate::lexer::Position { file, ..*position })
            .collect();
        let mut parser = Parser::with_positions(&tokens, &positions);
        let mut imported = parser
            .parse()
            .with_context(|| format!("parsing {}", full.display()))?;
        resolved
            .linear_types
            .extend(parser.linear_types().iter().cloned());
        resolved.tests.extend(parser.tests().iter().cloned());

        let exports: HashSet<String> =
            parser.exports().iter().cloned().collect();
        let tag = module_tag(&key, root);

        // Steps 2 and 4 of docs/separate-compilation.md. The interface is
        // derived at the one place a module is parsed, which is what keeps it
        // from drifting out of step with the source it describes.
        if crate::interface::interfaces_are_checked()
            || crate::interface::built_from_interfaces()
        {
            let interface = crate::interface::ModuleInterface::of(
                &module_name,
                &imported,
                parser.exports(),
                &parser.linear_types().iter().cloned().collect(),
            );
            crate::interface::check_interface_round_trip(&interface)?;
            crate::interface::check_interface_covers_exports(&interface)?;
            crate::interface::check_interface_is_closed(&interface, &imported)?;

            // The oracle for step 4: build the program from what the interface
            // says rather than from the module's source, and require the result
            // to be the same program. An interface that is missing something a
            // caller needs fails here, loudly, instead of at step 5 when the
            // compiler has started trusting interfaces for real.
            //
            // The module's own `import` lines are kept, because an interface
            // carries declarations and not dependencies, and the modules behind
            // them still have to be reached. Everything else the module
            // declared is replaced by the interface's view of it, so anything
            // it kept private and nothing reaches is simply gone.
            if crate::interface::built_from_interfaces() {
                let mut rebuilt: Vec<Spanned<Statement>> = imported
                    .iter()
                    .filter(|statement| {
                        matches!(statement.node, Statement::Import(_))
                    })
                    .cloned()
                    .collect();
                rebuilt.extend(interface.declarations.iter().cloned());
                imported = rebuilt;
            }
            resolved.interfaces.push(interface);
        }

        let renames = private_renames(&imported, &exports, &tag);
        if !renames.is_empty() {
            let renamer = Renamer { renames };
            renamer.block(&mut imported, &mut Vec::new());
        }

        let child_dir =
            full.parent().map(Path::to_path_buf).unwrap_or_default();
        resolve_into(imported, &child_dir, root, seen, resolved)?;
    }
    Ok(())
}

fn top_level_name(statement: &Statement) -> Option<&str> {
    match statement {
        Statement::Constant(name, _)
        | Statement::Struct(name, _, _)
        | Statement::Enum(name, _)
        | Statement::TypeAlias(name, _)
        | Statement::Extern { name, .. } => Some(name),
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
