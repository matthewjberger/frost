use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::interface::ModuleInterface;
use crate::parser::{Expression, Parameter, Spanned, Statement};
use crate::types::Type;

// What the compiler remembers about a module between builds. Step 5 of
// docs/separate-compilation.md: a module is rebuilt only when its own source or
// an imported interface changes, and this is the thing that answers that
// question without reading the module.
//
// The import list is here rather than in the interface because an interface
// carries declarations and not dependencies, and yet deciding whether to skip a
// module requires knowing what it imports before it has been parsed.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct ModuleRecord {
    pub module: String,
    pub source_hash: String,
    pub imports: Vec<String>,
    pub interface: ModuleInterface,
    // A module that lowers to no functions at all produces no object, and
    // without this the next build would look for one, not find it, and rebuild
    // a module that has nothing to build.
    pub emits_object: bool,
}

pub struct BuildCache {
    directory: PathBuf,
}

impl BuildCache {
    pub fn open(directory: &Path) -> Result<Self> {
        std::fs::create_dir_all(directory).with_context(|| {
            format!("creating the build directory {}", directory.display())
        })?;
        Ok(Self {
            directory: directory.to_path_buf(),
        })
    }

    fn record_path(&self, tag: &str) -> PathBuf {
        self.directory.join(format!("{tag}.json"))
    }

    // The fingerprint is in the name, so a module that changes and changes back
    // finds its old object still there, and an object never has to be checked
    // against the record that describes it.
    pub fn object_path(&self, tag: &str, fingerprint: &str) -> PathBuf {
        self.directory.join(format!("{tag}.{fingerprint}.o"))
    }

    pub fn load(&self, tag: &str, source_hash: &str) -> Option<ModuleRecord> {
        let text = std::fs::read_to_string(self.record_path(tag)).ok()?;
        let record: ModuleRecord = serde_json::from_str(&text).ok()?;
        (record.source_hash == source_hash && record.module_tag() == tag)
            .then_some(record)
    }

    pub fn store(&self, tag: &str, record: &ModuleRecord) -> Result<()> {
        let text = serde_json::to_string_pretty(record)
            .context("serializing a build record")?;
        let path = self.record_path(tag);
        std::fs::write(&path, text)
            .with_context(|| format!("writing {}", path.display()))
    }

    // An object named for a fingerprint the module no longer has can never be
    // used again, so the directory would otherwise grow by one object per edit.
    pub fn discard_other_objects(&self, tag: &str, keep: &Path) {
        let Ok(entries) = std::fs::read_dir(&self.directory) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path == keep {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(tag) && name.ends_with(".o") {
                std::fs::remove_file(&path).ok();
            }
        }
    }
}

impl ModuleRecord {
    fn module_tag(&self) -> String {
        format!("{:016x}", fnv1a(self.module.as_bytes()))
    }
}

// FNV-1a, written out rather than taken from the standard library because a
// build record outlives the compiler that wrote it, and `DefaultHasher`
// promises only that it is consistent within one version of Rust.
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub(crate) fn digest(text: &str) -> String {
    format!("{:016x}", fnv1a(text.as_bytes()))
}

// What a caller of this module has to be rebuilt for. A generic's body is part
// of its interface, because the caller chooses the type arguments and so the
// caller is what stamps out the template. An ordinary body is not, and hashing
// it would rebuild every dependent for an edit that cannot reach them, which is
// most of the edits anyone makes.
//
// The bodies stay in the interface itself. They are how the module's own object
// gets built when it is the module being rebuilt; it is only the fingerprint
// that looks past them.
pub fn interface_fingerprint(interface: &ModuleInterface) -> Result<String> {
    let mut view = interface.clone();
    // A file id is registration order, so leaving it in would make the hash
    // depend on which other modules the program happened to reach first.
    stamp_file(&mut view.declarations, 0);
    for statement in &mut view.declarations {
        blank_ordinary_body(&mut statement.node);
    }
    Ok(digest(&view.to_json()?))
}

fn blank_ordinary_body(statement: &mut Statement) {
    let Statement::Constant(_, value) = statement else {
        return;
    };
    let (Expression::Function(params, _, body)
    | Expression::Proc(params, _, body)) = value
    else {
        return;
    };
    if params.iter().any(is_compile_time) {
        return;
    }
    body.clear();
}

fn is_compile_time(parameter: &Parameter) -> bool {
    parameter.compile_time_signature.is_some()
        || matches!(
            &parameter.type_annotation,
            Some(Type::TypeParam(name)) if name == &parameter.name
        )
}

// A module's own source, plus the interface of everything it can reach through
// its imports. Transitive because a generic this module instantiates can
// instantiate one from a module further down, so a change down there changes
// what this module emits.
pub fn module_fingerprint(
    source_hash: &str,
    closure: &BTreeMap<String, String>,
) -> String {
    let mut text = String::from(source_hash);
    for (module, hash) in closure {
        text.push('\n');
        text.push_str(module);
        text.push(' ');
        text.push_str(hash);
    }
    digest(&text)
}

// A file id is handed out in registration order, so the same module is not
// necessarily the same id in the process that wrote an interface and the one
// that reads it back. Module attribution reads the top-level position's file
// id, so an interface loaded from a record has to be restamped or its
// declarations land in another module's object.
pub fn stamp_file(statements: &mut [Spanned<Statement>], file: u32) {
    for statement in statements.iter_mut() {
        statement.position.file = file;
        stamp_statement(&mut statement.node, file);
    }
}

fn stamp_statement(statement: &mut Statement, file: u32) {
    match statement {
        Statement::Constant(_, value) => stamp_expression(value, file),
        Statement::Let { value, .. } => stamp_expression(value, file),
        Statement::Return(value) | Statement::Expression(value) => {
            stamp_expression(value, file)
        }
        Statement::Assignment(target, value) => {
            stamp_expression(target, file);
            stamp_expression(value, file);
        }
        Statement::Defer(inner) => stamp_statement(inner, file),
        Statement::For(_, range, body) => {
            stamp_expression(range, file);
            stamp_file(body, file);
        }
        Statement::While(condition, body) => {
            stamp_expression(condition, file);
            stamp_file(body, file);
        }
        Statement::With(_, body) => stamp_file(body, file),
        Statement::Struct(..)
        | Statement::Enum(..)
        | Statement::TypeAlias(..)
        | Statement::Extern { .. }
        | Statement::Break
        | Statement::Continue
        | Statement::Import(_) => {}
    }
}

fn stamp_expression(expression: &mut Expression, file: u32) {
    match expression {
        Expression::Prefix(_, operand)
        | Expression::AddressOf(operand)
        | Expression::Borrow(operand)
        | Expression::BorrowMut(operand)
        | Expression::Try(operand)
        | Expression::Dereference(operand)
        | Expression::FieldAccess(operand, _) => {
            stamp_expression(operand, file)
        }
        Expression::Infix(left, _, right)
        | Expression::Index(left, right)
        | Expression::Range(left, right, _) => {
            stamp_expression(left, file);
            stamp_expression(right, file);
        }
        Expression::If(condition, consequence, alternative) => {
            stamp_expression(condition, file);
            stamp_file(consequence, file);
            if let Some(block) = alternative {
                stamp_file(block, file);
            }
        }
        Expression::Function(_, _, body) | Expression::Proc(_, _, body) => {
            stamp_file(body, file)
        }
        Expression::Call(callee, arguments) => {
            stamp_expression(callee, file);
            for argument in arguments.iter_mut() {
                stamp_expression(argument, file);
            }
        }
        Expression::StructInit(_, fields)
        | Expression::EnumVariantInit(_, _, fields) => {
            for (_, value) in fields.iter_mut() {
                stamp_expression(value, file);
            }
        }
        Expression::Tuple(elements) => {
            for element in elements.iter_mut() {
                stamp_expression(element, file);
            }
        }
        Expression::Switch(scrutinee, cases) => {
            stamp_expression(scrutinee, file);
            for case in cases.iter_mut() {
                stamp_file(&mut case.body, file);
            }
        }
        Expression::Unsafe(body) => stamp_file(body, file),
        Expression::Identifier(_)
        | Expression::Literal(_)
        | Expression::Boolean(_)
        | Expression::Sizeof(_)
        | Expression::TypeValue(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn interface_of(source: &str) -> ModuleInterface {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let statements = parser.parse().unwrap();
        ModuleInterface::of(
            "lib/test.frost",
            &statements,
            parser.exports(),
            &parser.linear_types().iter().cloned().collect(),
        )
    }

    // The whole distinction step 5 rests on: an ordinary body is the module's
    // own business, a generic body is its callers' business too.
    #[test]
    fn an_ordinary_body_is_not_part_of_the_fingerprint() {
        let before = interface_of(
            "export twice\n\
             twice :: fn(x: i64) -> i64 { x + x }\n",
        );
        let after = interface_of(
            "export twice\n\
             twice :: fn(x: i64) -> i64 { x * 2 }\n",
        );
        assert_eq!(
            interface_fingerprint(&before).unwrap(),
            interface_fingerprint(&after).unwrap()
        );
    }

    #[test]
    fn a_signature_change_is_part_of_the_fingerprint() {
        let before = interface_of(
            "export twice\n\
             twice :: fn(x: i64) -> i64 { x + x }\n",
        );
        let after = interface_of(
            "export twice\n\
             twice :: fn(x: i32) -> i64 { 0 }\n",
        );
        assert_ne!(
            interface_fingerprint(&before).unwrap(),
            interface_fingerprint(&after).unwrap()
        );
    }

    #[test]
    fn a_generic_body_is_part_of_the_fingerprint() {
        let before = interface_of(
            "export pick\n\
             pick :: fn($T: Type, move x: $T, move y: $T) -> $T { x }\n",
        );
        let after = interface_of(
            "export pick\n\
             pick :: fn($T: Type, move x: $T, move y: $T) -> $T { y }\n",
        );
        assert_ne!(
            interface_fingerprint(&before).unwrap(),
            interface_fingerprint(&after).unwrap()
        );
    }

    // A struct's layout is what a caller lays out its own frame with, so it is
    // in whatever shape it is written.
    #[test]
    fn a_field_change_is_part_of_the_fingerprint() {
        let before = interface_of(
            "export Point\n\
             Point :: struct { x: i64 }\n",
        );
        let after = interface_of(
            "export Point\n\
             Point :: struct { x: i64, y: i64 }\n",
        );
        assert_ne!(
            interface_fingerprint(&before).unwrap(),
            interface_fingerprint(&after).unwrap()
        );
    }

    #[test]
    fn a_module_fingerprint_follows_what_it_imports() {
        let mut closure = BTreeMap::new();
        closure.insert("lib/a.frost".to_string(), "1".to_string());
        let first = module_fingerprint("source", &closure);
        closure.insert("lib/a.frost".to_string(), "2".to_string());
        let second = module_fingerprint("source", &closure);
        assert_ne!(first, second);
        assert_ne!(first, module_fingerprint("other", &closure));
    }
}
