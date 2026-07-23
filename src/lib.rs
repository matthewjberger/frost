mod allocation_sources;
mod build_cache;
mod c_abi;
mod callbacks;
mod failure_sets;
mod imports;
mod interface;
mod interface_names;
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
mod regions;
mod source_map;
mod types;

pub use self::{
    allocation_sources::*, build_cache::*, c_abi::*, callbacks::*,
    failure_sets::*, imports::*, interface::*, ir::*, ir_build::*, ir_c::*,
    ir_codegen::*, ir_interp::*, ir_ownership::*, ir_typecheck::*, lexer::*,
    ownership::*, param_modes::*, parser::*, regions::*, types::*,
};

use std::fmt::Display;

fn flatten(items: &[impl Display], separator: &str) -> String {
    let strings = items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    strings.join(separator)
}
