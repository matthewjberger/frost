mod imports;
mod ir;
mod ir_build;
mod ir_c;
mod ir_codegen;
mod ir_interp;
mod ir_ownership;
mod ir_typecheck;
mod lexer;
mod ownership;
mod param_modes;
mod parser;
mod types;

pub use self::{
    imports::*, ir::*, ir_build::*, ir_c::*, ir_codegen::*, ir_interp::*,
    ir_ownership::*, ir_typecheck::*, lexer::*, ownership::*, param_modes::*,
    parser::*, types::*,
};

use std::fmt::Display;

fn flatten(items: &[impl Display], separator: &str) -> String {
    let strings = items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    strings.join(separator)
}
