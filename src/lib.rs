mod ast;
mod ast_display;
mod c_abi;
mod const_eval;
mod diagnostic;
pub use check::declared_types::check_declared_types;
pub use check::linear_instances::pooled_instance_names;
pub use diagnostic::{
    Diagnostic, Place, Replacement, Report, as_json as diagnostics_as_json,
    as_report, grouped as grouped_diagnostics, render as render_diagnostic,
    render_warnings,
};
pub use tools::api::{Exported, exported, nearest, sources};
pub use tools::fixes::{Edit, byte_offset, edit_for};
pub use tools::format::{format as format_source, formatted, tokens_and_gaps};
pub use tools::lint::lint;
mod check;
mod ir;
mod lexer;
mod lower;
mod modules;
mod parser;
mod source_map;
mod tools;
mod types;

pub use self::{
    ast::*, ast_display::*, c_abi::*, check::ownership::*, check::regions::*,
    check::unsafety::*, ir::build::*, ir::c::*, ir::codegen::*, ir::interp::*,
    ir::ownership::*, ir::typecheck::*, ir::*, lexer::*,
    lower::allocation_sources::*, lower::callbacks::*,
    lower::distinct_types::*, lower::failure_sets::*, lower::multi_returns::*,
    lower::param_modes::*, modules::build_cache::*,
    modules::import_visibility::*, modules::imports::*, modules::interface::*,
    modules::layers::Layer, modules::manifest::*, parser::*, tools::query::*,
    types::*,
};

use std::fmt::Display;

fn flatten(items: &[impl Display], separator: &str) -> String {
    let strings = items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    strings.join(separator)
}
