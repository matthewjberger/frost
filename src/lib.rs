mod allocation_sources;
mod arith_prelude;
mod ast;
mod ast_display;
mod build_cache;
mod c_abi;
mod callbacks;
mod diagnostic;
pub use diagnostic::{
    Diagnostic, Place, Replacement, Report, as_json as diagnostics_as_json,
    as_report, grouped as grouped_diagnostics, render as render_diagnostic,
};
mod fixes;
pub use fixes::{Edit, byte_offset, edit_for};
mod declared_types;
pub use declared_types::check_declared_types;
mod format;
mod lint;
pub use format::{format as format_source, formatted, tokens_and_gaps};
pub use lint::lint;
mod distinct_types;
mod failure_sets;
mod import_visibility;
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
mod layers;
mod lexer;
mod linear_instances;
mod manifest;
mod multi_returns;
mod ownership;
mod param_modes;
mod parser;
mod query;
mod regions;
mod source_map;
mod types;
mod unsafety;

pub use self::{
    allocation_sources::*, ast::*, ast_display::*, build_cache::*, c_abi::*,
    callbacks::*, distinct_types::*, failure_sets::*, import_visibility::*,
    imports::*, interface::*, ir::*, ir_build::*, ir_c::*, ir_codegen::*,
    ir_interp::*, ir_ownership::*, ir_typecheck::*, layers::Layer, lexer::*,
    manifest::*, multi_returns::*, ownership::*, param_modes::*, parser::*,
    query::*, regions::*, types::*, unsafety::*,
};

use std::fmt::Display;

fn flatten(items: &[impl Display], separator: &str) -> String {
    let strings = items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    strings.join(separator)
}
