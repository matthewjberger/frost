// The passes that rewrite the tree before the IR is built. Each turns a surface
// form into one the rest of the pipeline already handles, so nothing after this
// point knows the form existed.
pub(crate) mod allocation_sources;
pub(crate) mod arith_prelude;
pub(crate) mod callbacks;
pub(crate) mod distinct_types;
pub(crate) mod failure_sets;
pub(crate) mod multi_returns;
pub(crate) mod param_modes;
