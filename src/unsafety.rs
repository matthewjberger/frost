use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::parser::{Block, Expression, Statement, StructField};
use crate::{Position, Spanned};
use crate::types::Type;

// Where the compiler's guarantees stop.
//
// Every other check in this compiler proves something about a program it can
// see all of: a value is not used after it moves, a linear resource is consumed
// exactly once, an arena pointer does not outlive its block. Three operations
// reach memory none of that covers, and an `unsafe` block is where they are
// allowed:
//
//   - reading or writing through a raw pointer, `p^` and `p[i]`
//   - `ptr_cast`, which says the bytes at an address are a different type
//   - calling an `extern fn`, which is arbitrary C
//
// Nothing else in the language can touch memory it has not been shown to own.
// So the point of the block is not that it enables anything: it is that the
// three are refused outside one, which makes `unsafe` the complete list of
// places to look when something has corrupted memory. Without the refusal the
// block would be a comment.
//
// What this cannot see: a raw pointer whose type this pass could not work out.
// It resolves a name's type from a parameter's annotation, a `let` annotation,
// a `ptr_cast`, and a struct field, which is how a raw pointer is actually
// held. A pointer arriving somewhere none of those describe is indexed without
// complaint. Dereference, `ptr_cast` and extern calls are exact, since none of
// them needs a type to be recognized.
pub fn check_unsafety(statements: &[Spanned<Statement>]) -> Result<()> {
    let mut checker = Checker {
        externs: HashSet::new(),
        fields: HashMap::new(),
        depth: 0,
        scope: Vec::new(),
    };
    for statement in statements {
        match &statement.node {
            Statement::Extern { name, .. } => {
                checker.externs.insert(name.clone());
            }
            Statement::Struct(name, _, declared) => {
                checker.fields.insert(name.clone(), declared.clone());
            }
            _ => {}
        }
    }
    for statement in statements {
        checker.statement(statement)?;
    }
    Ok(())
}

struct Checker {
    externs: HashSet<String>,
    fields: HashMap<String, Vec<StructField>>,
    // How many `unsafe` blocks enclose what is being walked. Nesting one inside
    // another is allowed and means nothing extra, the same as in Rust.
    depth: usize,
    scope: Vec<HashMap<String, Type>>,
}

impl Checker {
    fn refuse(&self, what: &str, position: Position) -> Result<()> {
        if self.depth > 0 {
            return Ok(());
        }
        bail!(
            "{}:{}: {what} is unchecked, so it belongs in an `unsafe` block",
            position.line,
            position.column
        )
    }

    fn bind(&mut self, name: &str, ty: Option<Type>) {
        if let (Some(ty), Some(top)) = (ty, self.scope.last_mut()) {
            top.insert(name.to_string(), ty);
        }
    }

    fn lookup(&self, name: &str) -> Option<&Type> {
        self.scope.iter().rev().find_map(|frame| frame.get(name))
    }

    // What a place expression holds, where this pass can tell. `None` means
    // unknown rather than "not a pointer", which is why the index rule below
    // only fires on a definite raw pointer.
    fn type_of(&self, expression: &Expression) -> Option<Type> {
        match expression {
            Expression::Identifier(name) => self.lookup(name).cloned(),
            Expression::FieldAccess(base, field) => {
                let base_type = self.type_of(base)?;
                let name = match base_type {
                    Type::Struct(name) => name,
                    Type::Ptr(inner) | Type::Ref(inner) | Type::RefMut(inner) => {
                        match *inner {
                            Type::Struct(name) => name,
                            _ => return None,
                        }
                    }
                    _ => return None,
                };
                self.fields
                    .get(&name)?
                    .iter()
                    .find(|declared| &declared.name == field)
                    .map(|declared| declared.field_type.clone())
            }
            _ => None,
        }
    }

    fn block(&mut self, block: &Block) -> Result<()> {
        self.scope.push(HashMap::new());
        let outcome = (|| {
            for statement in block {
                self.statement(statement)?;
            }
            Ok(())
        })();
        self.scope.pop();
        outcome
    }

    fn statement(&mut self, statement: &Spanned<Statement>) -> Result<()> {
        let at = statement.position;
        match &statement.node {
            Statement::Let {
                name,
                type_annotation,
                value,
                ..
            } => {
                self.expression(value, at)?;
                let known = type_annotation
                    .clone()
                    .or_else(|| self.produced_type(value));
                self.bind(name, known);
            }
            Statement::Constant(_, value) | Statement::Return(value) => {
                self.expression(value, at)?;
            }
            Statement::Expression(value) => self.expression(value, at)?,
            Statement::Assignment(place, value) => {
                self.expression(place, at)?;
                self.expression(value, at)?;
            }
            Statement::Defer(inner) => {
                self.statement(&Spanned {
                    node: (**inner).clone(),
                    position: at,
                })?;
            }
            Statement::For(name, range, body) => {
                self.expression(range, at)?;
                self.scope.push(HashMap::new());
                self.bind(name, Some(Type::I64));
                let outcome = self.block(body);
                self.scope.pop();
                outcome?;
            }
            Statement::While(condition, body) => {
                self.expression(condition, at)?;
                self.block(body)?;
            }
            Statement::With(_, body) => self.block(body)?,
            _ => {}
        }
        Ok(())
    }

    // The type an expression hands back, for the few forms that say so plainly.
    fn produced_type(&self, value: &Expression) -> Option<Type> {
        match value {
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(name) = &**callee else {
                    return None;
                };
                if name != "ptr_cast" {
                    return None;
                }
                match arguments.first() {
                    Some(Expression::TypeValue(inner)) => {
                        Some(Type::Ptr(Box::new(inner.clone())))
                    }
                    _ => None,
                }
            }
            Expression::AddressOf(inner) => {
                Some(Type::Ptr(Box::new(self.type_of(inner)?)))
            }
            _ => self.type_of(value),
        }
    }

    fn expression(&mut self, value: &Expression, at: Position) -> Result<()> {
        match value {
            Expression::Unsafe(body) => {
                self.depth += 1;
                let outcome = self.block(body);
                self.depth -= 1;
                return outcome;
            }
            Expression::Dereference(inner) => {
                self.refuse("reading through a raw pointer", at)?;
                self.expression(inner, at)?;
            }
            Expression::Index(base, index) => {
                if matches!(self.type_of(base), Some(Type::Ptr(_))) {
                    self.refuse("indexing a raw pointer", at)?;
                }
                self.expression(base, at)?;
                self.expression(index, at)?;
            }
            Expression::Call(callee, arguments) => {
                if let Expression::Identifier(name) = &**callee {
                    if name == "ptr_cast" {
                        self.refuse("ptr_cast", at)?;
                    } else if self.externs.contains(name) {
                        self.refuse(
                            &format!("calling the C function '{name}'"),
                            at,
                        )?;
                    }
                }
                self.expression(callee, at)?;
                for argument in arguments {
                    self.expression(argument, at)?;
                }
            }
            Expression::Function(parameters, _, body)
            | Expression::Proc(parameters, _, body) => {
                self.scope.push(HashMap::new());
                for parameter in parameters {
                    let annotation = parameter.type_annotation.clone();
                    self.bind(&parameter.name, annotation);
                }
                let outcome = self.block(body);
                self.scope.pop();
                return outcome;
            }
            Expression::If(condition, consequence, alternative) => {
                self.expression(condition, at)?;
                self.block(consequence)?;
                if let Some(alternative) = alternative {
                    self.block(alternative)?;
                }
            }
            Expression::Switch(subject, cases) => {
                self.expression(subject, at)?;
                for case in cases {
                    self.block(&case.body)?;
                }
            }
            Expression::Prefix(_, inner)
            | Expression::AddressOf(inner)
            | Expression::Borrow(inner)
            | Expression::BorrowMut(inner)
            | Expression::Try(inner)
            | Expression::FieldAccess(inner, _) => {
                self.expression(inner, at)?;
            }
            Expression::Infix(left, _, right)
            | Expression::Range(left, right, _) => {
                self.expression(left, at)?;
                self.expression(right, at)?;
            }
            Expression::Tuple(parts) => {
                for part in parts {
                    self.expression(part, at)?;
                }
            }
            Expression::StructInit(_, initializers)
            | Expression::EnumVariantInit(_, _, initializers) => {
                for (_, initializer) in initializers {
                    self.expression(initializer, at)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}
