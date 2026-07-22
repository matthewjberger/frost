use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::parser::{
    Block, Expression, Parameter, Pattern, Spanned, Statement, SwitchCase,
};
use crate::types::Type;

pub struct Resolved {
    pub statements: Vec<Spanned<Statement>>,
    pub linear_types: HashSet<String>,
    pub tests: Vec<(String, String)>,
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
    let relative = path.strip_prefix(root).unwrap_or(path);
    // Path separators differ by platform and the identity must not, so the
    // components are joined rather than the string taken as it is.
    let joined: Vec<String> = relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect();
    format!("{:016x}", fnv1a(joined.join("/").as_bytes()))
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
        let positions = lexer.positions().to_vec();
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
            Expression::Function(params, _, body)
            | Expression::Proc(params, _, body) => {
                scope.push(HashSet::new());
                self.parameters(params, scope);
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
