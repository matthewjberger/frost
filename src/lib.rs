mod ast;
mod ast_display;
mod c_abi;
mod const_eval;
mod diagnostic;
pub use check::constant_arithmetic::check_constant_arithmetic;
pub use check::constant_names::{NOT_A_CONSTANT, check_constant_names};
pub use check::regions::REGION_ESCAPE;
pub use check::declared_types::{UNDECLARED_TYPE, check_declared_types};
pub use check::entry::check_entry_point;
pub use check::linear_instances::pooled_instance_names;
pub use check::nested_functions::{NESTED_FUNCTION, check_nested_functions};
pub use check::recursive_structs::{RECURSIVE_STRUCT, check_recursive_structs};
pub use check::template_calls::check_template_calls;
pub use diagnostic::{
    Diagnostic, LocatedError, Place, Replacement, Report,
    as_json as diagnostics_as_json, as_report, grouped as grouped_diagnostics,
    in_source_order, render as render_diagnostic, render_warnings,
};
pub use ir::build::UNDEFINED_CALL;
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
