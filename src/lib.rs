mod compiler;
mod evaluator;
mod lexer;
mod parser;
mod vm;

pub use self::{compiler::*, evaluator::*, lexer::*, parser::*, vm::*};

use std::{
    collections::hash_map::DefaultHasher,
    fmt::Display,
    hash::{Hash, Hasher},
};

fn flatten(items: &[impl Display], separator: &str) -> String {
    let strings = items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    strings.join(separator)
}

fn hash<T>(t: &T) -> u64
where
    T: Hash,
    T: ?Sized,
{
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}
