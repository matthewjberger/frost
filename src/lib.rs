mod evaluator;
mod lexer;
mod parser;

pub use self::{evaluator::*, lexer::*, parser::*};

use std::fmt::Display;

fn flatten(items: &[impl Display], separator: &str) -> String {
    let strings = items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    format!("{}", strings.join(separator))
}
