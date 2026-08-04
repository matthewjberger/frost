use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::ast::{
    Ast, Expression, Module, Parameter, Range32, Splicer, Statement, StmtId,
    splice_positions,
};
use crate::interface::ModuleInterface;
use crate::types::Type;

// The shape of a record on disk. Bumped when the serialized AST changes form,
// so a record written by an older compiler misses cleanly instead of
// deserializing into the wrong meaning. The field has no serde default on
// purpose: a record without one is from before the arena AST and must miss.
pub const CACHE_FORMAT: u32 = 3;

// What the compiler remembers about a module between builds. A module is
// rebuilt only when its own source or an imported interface changes, and this is the thing that answers that
// question without reading the module.
//
// The import list is here rather than in the interface because an interface
// carries declarations and not dependencies, and yet deciding whether to skip a
// module requires knowing what it imports before it has been parsed.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct ModuleRecord {
    pub format_version: u32,
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
        (record.format_version == CACHE_FORMAT
            && record.source_hash == source_hash
            && record.module_tag() == tag)
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
// gets built when it is the module being rebuilt. It is only the fingerprint
// that looks past them.
//
// Blanking a body severs its nodes from the tree but leaves them in the
// arena, and the arena is what serializes, so the hashed view is rebuilt by
// copying only what the blanked declarations still reach. The copy re-interns
// symbols in walk order, so the view is deterministic whatever order the
// original arena grew in.
pub fn interface_fingerprint(interface: &ModuleInterface) -> Result<String> {
    let mut view = interface.clone();
    for statement in &view.declarations.roots.clone() {
        blank_ordinary_body(&mut view.declarations.ast, *statement);
    }
    let mut compact = Module::default();
    {
        let held = &view.declarations;
        splice_positions(&mut compact.ast, &held.ast);
        let splicer = Splicer::new(&held.ast, 0);
        for statement in &held.roots {
            let copied =
                splicer.statement(&mut compact.ast, *statement, &mut |name| {
                    name.to_string()
                });
            compact.roots.push(copied);
        }
    }
    view.declarations = compact;
    // A file id is registration order, so leaving it in would make the hash
    // depend on which other modules the program happened to reach first.
    stamp_file(&mut view, 0);
    Ok(digest(&view.to_json()?))
}

fn blank_ordinary_body(ast: &mut Ast, statement: StmtId) {
    let Statement::Constant(_, value) = ast.stmt(statement) else {
        return;
    };
    let value = *value;
    let (Expression::Function(params, _, _) | Expression::Proc(params, _, _)) =
        ast.expr(value)
    else {
        return;
    };
    let params = *params;
    if ast
        .params_in(params)
        .iter()
        .any(|parameter| is_compile_time(ast, parameter))
    {
        return;
    }
    let (Expression::Function(_, _, body) | Expression::Proc(_, _, body)) =
        &mut ast.expressions[value.0 as usize]
    else {
        return;
    };
    *body = Range32::EMPTY;
}

// What a module contributes when it is not being rebuilt. An ordinary function
// becomes its signature and nothing else, because its body is already compiled
// into the object about to be linked and walking it again is the last piece of
// the front end that is still whole-program. A generic keeps its body, because
// the caller is what stamps out the template. Everything else is carried as it
// stands. A type is layout the caller lays out its own frame with, and a
// constant is a value.
//
// Answers the reduced form pushed into `dest`, or `None` when the declaration
// is carried as it stands and the caller splices it whole.
pub fn push_as_declaration(
    dest: &mut Ast,
    splicer: &Splicer<'_>,
    statement: StmtId,
) -> Option<StmtId> {
    let source = splicer.source;
    let Statement::Constant(name, value) = source.stmt(statement) else {
        return None;
    };
    let (Expression::Function(params, return_sig, _)
    | Expression::Proc(params, return_sig, _)) = source.expr(*value)
    else {
        return None;
    };
    if source
        .params_in(*params)
        .iter()
        .any(|parameter| is_compile_time(source, parameter))
    {
        return None;
    }
    let name = *name;
    let params = *params;
    let return_sig = *return_sig;
    let span = source.stmt_span(statement);
    let copied_params = splicer.copy_parameters(dest, params);
    let copied_signature = splicer.copy_signature(dest, return_sig);
    let copied_name = dest.intern(source.name(name));
    Some(dest.push_stmt(
        Statement::Declared {
            name: copied_name,
            params: copied_params,
            return_sig: copied_signature,
        },
        splicer.shifted(span),
    ))
}

fn is_compile_time(ast: &Ast, parameter: &Parameter) -> bool {
    parameter.compile_time_signature.is_some()
        || matches!(
            &parameter.type_annotation,
            Some(Type::TypeParam(name)) if name == ast.name(parameter.name)
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
// declarations land in another module's object. Positions live in the
// arena's one table, so restamping is a pass over it rather than a walk.
pub fn stamp_file(interface: &mut ModuleInterface, file: u32) {
    for position in &mut interface.declarations.ast.token_positions {
        position.file = file;
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
        let module = parser.parse().unwrap();
        ModuleInterface::of(
            "lib/test.frost",
            &module.ast,
            &module.roots,
            parser.exports(),
            &parser.linear_types().iter().cloned().collect(),
        )
    }

    // The whole distinction step 5 rests on. An ordinary body is the module's
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
