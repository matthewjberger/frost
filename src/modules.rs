// What an  means: finding the file, reading what it exports, mangling
// what it declares, and reusing what an earlier build already compiled.
pub(crate) mod build_cache;
pub(crate) mod import_visibility;
pub(crate) mod imports;
pub(crate) mod interface;
pub(crate) mod interface_names;
pub(crate) mod layers;
pub(crate) mod manifest;
