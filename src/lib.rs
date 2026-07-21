mod compiler;
pub mod ffi;
mod imports;
mod ir;
mod ir_build;
mod ir_c;
mod ir_codegen;
mod ir_interp;
mod ir_typecheck;
mod lexer;
mod ownership;
mod parser;
mod typed_vm;
mod types;
mod value;

pub use self::{
    compiler::*, ffi::*, imports::*, ir::*, ir_build::*, ir_c::*,
    ir_codegen::*, ir_interp::*, ir_typecheck::*, lexer::*, ownership::*,
    parser::*, typed_vm::*, types::*, value::*,
};

use std::fmt::Display;

fn flatten(items: &[impl Display], separator: &str) -> String {
    let strings = items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    strings.join(separator)
}
