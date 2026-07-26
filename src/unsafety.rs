use std::collections::{HashMap, HashSet};

use anyhow::Result;

use crate::parser::{Block, Diagnostic, Expression, Statement, StructField};
use crate::types::Type;
use crate::{Position, Spanned};

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
// So the point of the block is not that it enables anything. It is that the
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
    let diagnostics = check_unsafety_recovering(statements);
    if diagnostics.is_empty() {
        return Ok(());
    }
    Err(anyhow::anyhow!(crate::flatten(&diagnostics, "\n")))
}

/// Walk the whole program, reporting every operation that belongs in an
/// `unsafe` block rather than stopping at the first. Each refusal is
/// independent of the others, so one does not colour what follows it.
pub fn check_unsafety_recovering(
    statements: &[Spanned<Statement>],
) -> Vec<Diagnostic> {
    walk_unsafety(statements, false)
}

/// The same walk, also reporting a block that vouches for nothing: one holding
/// no unchecked operation, and one written inside another, which already
/// covers it. This is off unless asked for, because a build should pay for the
/// checks that keep a program correct and not for the ones that keep it tidy.
pub fn audit_unsafe_blocks(
    statements: &[Spanned<Statement>],
) -> Vec<Diagnostic> {
    walk_unsafety(statements, true)
        .into_iter()
        .filter(|d| d.message.starts_with("this `unsafe`"))
        .collect()
}

fn walk_unsafety(
    statements: &[Spanned<Statement>],
    audit: bool,
) -> Vec<Diagnostic> {
    let mut checker = Checker {
        externs: HashSet::new(),
        unsafe_fns: HashSet::new(),
        fields: HashMap::new(),
        depth: 0,
        audit,
        vouched: Vec::new(),
        scope: Vec::new(),
        diagnostics: Vec::new(),
    };
    for statement in statements {
        match &statement.node {
            // An extern is the built-in unsafe function. Calling C is
            // unchecked, so a call to one is gated exactly as a call to a
            // user's `unsafe fn` is. Unless it is declared `safe extern fn`,
            // which is the author saying this one was audited and cannot
            // corrupt memory. Putting that assertion on the declaration keeps
            // `unsafe` blocks to what can actually go wrong. A call that only
            // writes bytes and returns is not a place to look for corruption,
            // and listing it there makes the list worth less.
            Statement::Extern { name, safe, .. } => {
                if !safe {
                    checker.externs.insert(name.clone());
                }
            }
            Statement::Constant(name, Expression::UnsafeFn(_)) => {
                checker.unsafe_fns.insert(name.clone());
            }
            Statement::Struct(name, _, declared) => {
                checker.fields.insert(name.clone(), declared.clone());
            }
            _ => {}
        }
    }
    for statement in statements {
        checker.statement(statement);
    }
    checker.diagnostics
}

// Replace every `unsafe fn(...)` with the plain function it wraps, once the
// unsafety check has read which names were unsafe. No pass after that point
// needs to know, so this keeps the marker out of lowering and the backends.
pub fn strip_unsafe_fns(statements: &mut [Spanned<Statement>]) {
    for statement in statements {
        strip_statement(&mut statement.node);
    }
}

fn strip_statement(statement: &mut Statement) {
    match statement {
        Statement::Let { value, .. }
        | Statement::Constant(_, value)
        | Statement::Return(value)
        | Statement::Expression(value) => strip_expression(value),
        Statement::Assignment(place, value) => {
            strip_expression(place);
            strip_expression(value);
        }
        Statement::For(_, _, range, body) => {
            strip_expression(range);
            strip_block(body);
        }
        Statement::While(condition, body) => {
            strip_expression(condition);
            strip_block(body);
        }
        Statement::With(_, body) => strip_block(body),
        Statement::Defer(inner) => strip_statement(inner),
        _ => {}
    }
}

fn strip_block(block: &mut Block) {
    for statement in block {
        strip_statement(&mut statement.node);
    }
}

fn strip_expression(expression: &mut Expression) {
    if let Expression::UnsafeFn(inner) = expression {
        let mut taken =
            std::mem::replace(inner.as_mut(), Expression::Boolean(false));
        strip_expression(&mut taken);
        *expression = taken;
        return;
    }
    match expression {
        Expression::Function(_, _, body)
        | Expression::Proc(_, _, body)
        | Expression::Unsafe(body) => strip_block(body),
        Expression::If(condition, consequence, alternative) => {
            strip_expression(condition);
            strip_block(consequence);
            if let Some(alternative) = alternative {
                strip_block(alternative);
            }
        }
        Expression::Switch(subject, cases) => {
            strip_expression(subject);
            for case in cases {
                strip_block(&mut case.body);
            }
        }
        Expression::Call(callee, arguments) => {
            strip_expression(callee);
            for argument in arguments {
                strip_expression(argument);
            }
        }
        Expression::PackMap(inner, _, _)
        | Expression::Prefix(_, inner)
        | Expression::AddressOf(inner)
        | Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::Dereference(inner)
        | Expression::Try(inner)
        | Expression::FieldAccess(inner, _) => strip_expression(inner),
        Expression::Infix(left, _, right)
        | Expression::Index(left, right)
        | Expression::Range(left, right, _) => {
            strip_expression(left);
            strip_expression(right);
        }
        Expression::Tuple(parts) => {
            for part in parts {
                strip_expression(part);
            }
        }
        Expression::StructInit(_, initializers)
        | Expression::EnumVariantInit(_, _, initializers) => {
            for (_, initializer) in initializers {
                strip_expression(initializer);
            }
        }
        _ => {}
    }
}

struct Checker {
    externs: HashSet<String>,
    unsafe_fns: HashSet<String>,
    fields: HashMap<String, Vec<StructField>>,
    // How many `unsafe` blocks enclose what is being walked. Nesting one inside
    // another is allowed and means nothing extra, the same as in Rust.
    depth: usize,
    // Whether to report a block that vouches for nothing. Off for an ordinary
    // build, so nothing below costs anything there.
    audit: bool,
    // One entry per open `unsafe` block: whether anything inside it needed one.
    vouched: Vec<bool>,
    scope: Vec<HashMap<String, Type>>,
    diagnostics: Vec<Diagnostic>,
}

impl Checker {
    fn refuse(&mut self, what: &str, position: Position) {
        if self.depth > 0 {
            if self.audit
                && let Some(top) = self.vouched.last_mut()
            {
                *top = true;
            }
            return;
        }
        self.diagnostics.push(Diagnostic {
            position,
            message: format!(
                "{what} is unchecked, so it belongs in an `unsafe` block"
            ),
        });
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
                    Type::Ptr(inner)
                    | Type::Ref(inner)
                    | Type::RefMut(inner) => match *inner {
                        Type::Struct(name) => name,
                        _ => return None,
                    },
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

    fn block(&mut self, block: &Block) {
        self.scope.push(HashMap::new());
        for statement in block {
            self.statement(statement);
        }
        self.scope.pop();
    }

    fn statement(&mut self, statement: &Spanned<Statement>) {
        let at = statement.position;
        match &statement.node {
            Statement::Let {
                name,
                type_annotation,
                value,
                ..
            } => {
                self.expression(value, at);
                let known = type_annotation
                    .clone()
                    .or_else(|| self.produced_type(value));
                self.bind(name, known);
            }
            Statement::Constant(_, value) | Statement::Return(value) => {
                self.expression(value, at);
            }
            Statement::Expression(value) => self.expression(value, at),
            Statement::Assignment(place, value) => {
                self.expression(place, at);
                self.expression(value, at);
            }
            Statement::Defer(inner) => {
                self.statement(&Spanned {
                    node: (**inner).clone(),
                    position: at,
                });
            }
            Statement::For(name, _, range, body) => {
                self.expression(range, at);
                self.scope.push(HashMap::new());
                self.bind(name, Some(Type::I64));
                self.block(body);
                self.scope.pop();
            }
            Statement::While(condition, body) => {
                self.expression(condition, at);
                self.block(body);
            }
            Statement::With(_, body) => self.block(body),
            _ => {}
        }
    }

    // The type an expression hands back, for the few forms that say so plainly.
    fn produced_type(&self, value: &Expression) -> Option<Type> {
        match value {
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(name) = &**callee else {
                    return None;
                };
                match name.as_str() {
                    "ptr_cast" => match arguments.first() {
                        Some(Expression::TypeValue(inner)) => {
                            Some(Type::Ptr(Box::new(inner.clone())))
                        }
                        _ => None,
                    },
                    // `ptr_to(place)` is the surface address-of. It always hands
                    // back a raw pointer, so a value bound from it is one whether
                    // or not this pass can name the pointee. Without this the
                    // index gate below never learns the binding is a pointer and
                    // `p[i]` slips out of the `unsafe` block it belongs in.
                    "ptr_to" => Some(Type::Ptr(Box::new(
                        arguments
                            .first()
                            .and_then(|argument| self.type_of(argument))
                            .unwrap_or(Type::Void),
                    ))),
                    _ => None,
                }
            }
            Expression::AddressOf(inner) => {
                Some(Type::Ptr(Box::new(self.type_of(inner)?)))
            }
            _ => self.type_of(value),
        }
    }

    fn expression(&mut self, value: &Expression, at: Position) {
        match value {
            Expression::Unsafe(body) => {
                if self.audit {
                    if self.depth > 0 {
                        self.diagnostics.push(Diagnostic {
                            position: at,
                            message: "this `unsafe` block is inside another one, which already vouches for what is in it".to_string(),
                        });
                    }
                    self.vouched.push(false);
                }
                self.depth += 1;
                self.block(body);
                self.depth -= 1;
                if self.audit
                    && let Some(used) = self.vouched.pop()
                    && !used
                {
                    self.diagnostics.push(Diagnostic {
                        position: at,
                        message: "this `unsafe` block holds no unchecked operation, so it vouches for nothing".to_string(),
                    });
                }
            }
            // An `unsafe fn`'s body is an implicit unsafe block. The whole
            // function is the dangerous region, so the gated operations are
            // allowed throughout it without a nested block.
            Expression::UnsafeFn(inner) => {
                self.depth += 1;
                self.expression(inner, at);
                self.depth -= 1;
            }
            Expression::Dereference(inner) => {
                self.refuse("reading through a raw pointer", at);
                self.expression(inner, at);
            }
            Expression::Index(base, index) => {
                if matches!(self.type_of(base), Some(Type::Ptr(_))) {
                    self.refuse("indexing a raw pointer", at);
                }
                self.expression(base, at);
                self.expression(index, at);
            }
            Expression::Call(callee, arguments) => {
                if let Expression::Identifier(name) = &**callee {
                    if name == "ptr_cast" {
                        self.refuse("ptr_cast", at);
                    } else if name == "slice_from" {
                        self.refuse("forming a slice from a raw pointer", at);
                    } else if self.externs.contains(name) {
                        let what = format!("calling the C function '{name}'");
                        self.refuse(&what, at);
                    } else if self.unsafe_fns.contains(name) {
                        let what =
                            format!("calling the unsafe function '{name}'");
                        self.refuse(&what, at);
                    }
                }
                self.expression(callee, at);
                for argument in arguments {
                    self.expression(argument, at);
                }
            }
            Expression::Function(parameters, _, body)
            | Expression::Proc(parameters, _, body) => {
                self.scope.push(HashMap::new());
                for parameter in parameters {
                    let annotation = parameter.type_annotation.clone();
                    self.bind(&parameter.name, annotation);
                }
                self.block(body);
                self.scope.pop();
            }
            Expression::If(condition, consequence, alternative) => {
                self.expression(condition, at);
                self.block(consequence);
                if let Some(alternative) = alternative {
                    self.block(alternative);
                }
            }
            Expression::Switch(subject, cases) => {
                self.expression(subject, at);
                for case in cases {
                    self.block(&case.body);
                }
            }
            Expression::PackMap(inner, _, _)
            | Expression::Prefix(_, inner)
            | Expression::AddressOf(inner)
            | Expression::Borrow(inner)
            | Expression::BorrowMut(inner)
            | Expression::Try(inner)
            | Expression::ArrayRepeat(inner, _)
            | Expression::FieldAccess(inner, _) => {
                self.expression(inner, at);
            }
            Expression::Infix(left, _, right)
            | Expression::Range(left, right, _) => {
                self.expression(left, at);
                self.expression(right, at);
            }
            Expression::Tuple(parts) => {
                for part in parts {
                    self.expression(part, at);
                }
            }
            Expression::StructInit(_, initializers)
            | Expression::EnumVariantInit(_, _, initializers) => {
                for (_, initializer) in initializers {
                    self.expression(initializer, at);
                }
            }
            // Listed rather than caught by `_`, so a new expression form is a
            // compile error here instead of walking past whatever it holds.
            Expression::Identifier(_)
            | Expression::Literal(_)
            | Expression::Boolean(_)
            | Expression::Sizeof(_)
            | Expression::TypeValue(_) => {}
        }
    }
}
