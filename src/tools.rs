// What the compiler answers that is not a compile: the formatter, the linter,
// the edits a diagnostic offers, the questions an editor asks, and the exported
// surface of a project.
pub(crate) mod api;
pub(crate) mod fixes;
pub(crate) mod format;
pub(crate) mod lint;
pub(crate) mod query;
