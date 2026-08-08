// The passes that read a whole program and refuse one. None of them rewrites
// anything: what each answers is a list of diagnostics, so a program that
// passes them all is the program the front end parsed.
pub(crate) mod declared_types;
pub(crate) mod entry;
pub(crate) mod linear_instances;
pub(crate) mod ownership;
pub(crate) mod regions;
pub(crate) mod unsafety;
