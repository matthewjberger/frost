mod compiler;
pub mod ffi;
mod ir;
mod ir_build;
mod ir_c;
mod ir_codegen;
mod ir_interp;
mod ir_typecheck;
mod lexer;
mod ownership;
mod parser;
mod typechecker;
mod typed_vm;
mod types;
mod value;

pub use self::{
    compiler::*, ffi::*, ir::*, ir_build::*, ir_c::*, ir_codegen::*,
    ir_interp::*, ir_typecheck::*, lexer::*, ownership::*, parser::*,
    typechecker::*, typed_vm::*, types::*, value::*,
};

use std::{
    collections::hash_map::DefaultHasher,
    fmt::Display,
    hash::{Hash, Hasher},
};

fn flatten(items: &[impl Display], separator: &str) -> String {
    let strings = items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    strings.join(separator)
}

fn hash<T>(t: &T) -> u64
where
    T: Hash,
    T: ?Sized,
{
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}
