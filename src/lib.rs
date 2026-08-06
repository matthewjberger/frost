mod ast;
mod ast_display;
mod c_abi;
mod const_eval;
mod diagnostic;
pub use diagnostic::{
    Diagnostic, Place, Replacement, Report, as_json as diagnostics_as_json,
    as_report, grouped as grouped_diagnostics, render as render_diagnostic,
};
pub use tools::fixes::{Edit, byte_offset, edit_for};
pub use tools::api::{Exported, exported, nearest, sources};
pub use check::declared_types::check_declared_types;
pub use tools::format::{format as format_source, formatted, tokens_and_gaps};
pub use tools::lint::lint;
mod check;
mod ir;
mod lower;
mod modules;
mod tools;
mod lexer;
mod parser;
mod source_map;
mod types;

pub use self::{
    lower::allocation_sources::*, ast::*, ast_display::*, modules::build_cache::*, c_abi::*,
    lower::callbacks::*, lower::distinct_types::*, lower::failure_sets::*, modules::import_visibility::*,
    modules::imports::*, modules::interface::*, ir::*, ir::build::*, ir::c::*, ir::codegen::*,
    ir::interp::*, ir::ownership::*, ir::typecheck::*, modules::layers::Layer, lexer::*,
    modules::manifest::*, lower::multi_returns::*, check::ownership::*, lower::param_modes::*, parser::*,
    tools::query::*, check::regions::*, types::*, check::unsafety::*,
};

use std::fmt::Display;

fn flatten(items: &[impl Display], separator: &str) -> String {
    let strings = items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    strings.join(separator)
}
