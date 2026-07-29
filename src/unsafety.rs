use std::collections::{HashMap, HashSet};

use anyhow::Result;

use crate::parser::{
    Block, Diagnostic, Expression, Literal, Statement, StructField,
};
use crate::types::Type;
use crate::{Position, Spanned};

// Where the compiler's guarantees stop.
//
// Every other check in this compiler proves something about a program it can
// see all of: a value is not used after it moves, a linear resource is consumed
// exactly once, an arena pointer does not outlive its block. Four operations
// reach memory none of that covers, and an `unsafe` block is where they are
// allowed:
//
//   - reading or writing through a raw pointer, `p^` and `p[i]`
//   - `ptr_cast`, which says the bytes at an address are a different type
//   - `slice_from`, which says how many elements live at an address
//   - calling an `extern fn`, which is arbitrary C
//
// Nothing else in the language can touch memory it has not been shown to own.
// So the point of the block is not that it enables anything. It is that the
// four are refused outside one, which makes `unsafe` the complete list of
// places to look when something has corrupted memory. Without the refusal the
// block would be a comment.
//
// Three of the four are recognized by shape and need no type. The index rule is
// the one that has to know whether the base is a raw pointer, and it refuses a
// base whose type it cannot name rather than allowing it: a gate that lets the
// unknown through reports what it happened to recognize, and the list of blocks
// is then worth nothing. What keeps that from refusing ordinary code is that
// `type_of` reads a call's return type off the declaration, an element's off its
// array or slice, and a field's off the struct, so a base is named in the shapes
// programs actually write.
//
// The walks below list every statement and expression form rather than ending in
// a wildcard. A form nobody handled is a compile error here instead of a hole
// nobody sees until a program reaches through it. `print` was that hole.
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
        returns: HashMap::new(),
        generics: HashMap::new(),
        depth: 0,
        audit,
        vouched: Vec::new(),
        scope: Vec::new(),
        diagnostics: Vec::new(),
    };
    let mut top_level: HashMap<String, Type> = HashMap::new();
    for statement in statements {
        // What each function answers with. The index rule below refuses a base
        // whose type it cannot name, and a binding is most often given its type
        // by the call that produced it, so without this the rule would fall to
        // the refusal on ordinary code rather than on a raw pointer.
        match &statement.node {
            Statement::Constant(name, value) => {
                if let Some(ty) = declared_return(value) {
                    checker.returns.insert(name.clone(), ty);
                }
                let parameters = type_parameters(value);
                if !parameters.is_empty() {
                    checker.generics.insert(name.clone(), parameters);
                }
                // A constant is named from inside every function, so its type
                // belongs to the walk before any of them start. `ROW :: [1, 2]`
                // then `ROW[i]` is an index into an array, and without this the
                // base has no type and the rule refuses it.
                if let Some(ty) = constant_type(value) {
                    top_level.insert(name.clone(), ty);
                }
            }
            Statement::Extern {
                name,
                return_type: Some(return_type),
                ..
            } => {
                checker.returns.insert(name.clone(), return_type.clone());
            }
            Statement::Declared {
                name, return_sig, ..
            } => {
                if let Some(ty) = return_sig.to_type() {
                    checker.returns.insert(name.clone(), ty);
                }
            }
            _ => {}
        }
        match &statement.node {
            // An extern is the built-in unsafe function. Calling C is
            // unchecked, so a call to one is gated exactly as a call to a
            // user's `unsafe fn` is. Unless it is declared `safe extern fn`,
            // which is the author saying this one was audited and cannot
            // corrupt memory. Putting that assertion on the declaration keeps
            // `unsafe` blocks to what can go wrong. A call that only
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
    checker.scope.push(top_level);
    for statement in statements {
        checker.statement(statement);
    }
    checker.diagnostics
}

/// The type a top-level constant holds, for the shapes that say so plainly. Only
/// enough to tell an index into one from an index into a raw pointer.
fn constant_type(value: &Expression) -> Option<Type> {
    match value {
        Expression::Literal(Literal::Array(elements)) => {
            Some(Type::Array(Box::new(Type::Unknown), elements.len()))
        }
        Expression::ArrayRepeat(_, _) => {
            Some(Type::Array(Box::new(Type::Unknown), 0))
        }
        Expression::Literal(Literal::String(_)) => Some(Type::Str),
        _ => None,
    }
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

/// What a named constant answers with, where the constant is a function. An
/// `unsafe fn` wraps the function it marks, so the signature is one level in.
fn declared_return(value: &Expression) -> Option<Type> {
    match value {
        Expression::Function(_, return_sig, _)
        | Expression::Proc(_, return_sig, _) => return_sig.to_type(),
        Expression::UnsafeFn(inner) => declared_return(inner),
        _ => None,
    }
}

/// The type a borrow names, or the type itself where it is not one.
fn without_borrow(ty: Type) -> Type {
    match ty {
        Type::Ref(inner) | Type::RefMut(inner) => *inner,
        other => other,
    }
}

/// Which of a function's parameters are types, and what each is called. A
/// signature written `vec_slice($T: Type, v: Vec<T>) -> []T` answers with `T`,
/// and the argument at that position at a call site is what `T` stands for
/// there. Without it the return type names the parameter rather than the
/// argument, so a field of the element has no declaration to look up.
fn type_parameters(value: &Expression) -> Vec<(usize, String)> {
    let params = match value {
        Expression::Function(params, _, _) | Expression::Proc(params, _, _) => {
            params
        }
        Expression::UnsafeFn(inner) => return type_parameters(inner),
        _ => return Vec::new(),
    };
    params
        .iter()
        .enumerate()
        .filter_map(|(position, param)| match &param.type_annotation {
            Some(Type::TypeParam(name)) => Some((position, name.clone())),
            _ => None,
        })
        .collect()
}

struct Checker {
    externs: HashSet<String>,
    unsafe_fns: HashSet<String>,
    fields: HashMap<String, Vec<StructField>>,
    // What each named function answers with, so a binding takes its type from
    // the call that produced it.
    returns: HashMap<String, Type>,
    // The type parameters of each named function, by argument position.
    generics: HashMap<String, Vec<(usize, String)>>,
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
                // A `columns<T, N>` is laid out by reflecting T's fields, which
                // happens after this pass, so the field a body names has no
                // declaration to look up yet. Every one of them is an array,
                // which is all this rule needs: what it refuses is a raw
                // pointer, and a column is not one.
                if let Type::Struct(name) = &base_type
                    && name.starts_with("columns<")
                {
                    return Some(Type::Array(Box::new(Type::Unknown), 0));
                }
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
                // A generic's fields are declared once, on the template, so a
                // parameter written `Box<$T>` or a local of `Box<i64>` finds
                // them under `Box`. Without this the gate cannot see that
                // `b.data` is a `^T`, and indexing a raw pointer held by any
                // generic container is allowed outside an `unsafe` block.
                self.fields
                    .get(Type::template_of(&name))?
                    .iter()
                    .find(|declared| &declared.name == field)
                    .map(|declared| declared.field_type.clone())
            }
            // An element of an array or a slice, so `rows[i][j]` names the inner
            // element rather than nothing. A `str` indexes to a byte, and a
            // borrow indexes as the place it names.
            Expression::Index(base, _) => match self.type_of(base)? {
                Type::Array(inner, _) | Type::Slice(inner) => Some(*inner),
                Type::Str => Some(Type::U8),
                Type::Ptr(inner) => Some(*inner),
                Type::Ref(inner) | Type::RefMut(inner) => match *inner {
                    Type::Array(element, _) | Type::Slice(element) => {
                        Some(*element)
                    }
                    Type::Str => Some(Type::U8),
                    _ => None,
                },
                _ => None,
            },
            // `ref name := place` binds a borrow, so what it holds is the type
            // of the place it names. Without this a `ref` local has no type at
            // all and every field and element reached through one falls to the
            // refusal.
            Expression::Borrow(place) => {
                Some(Type::Ref(Box::new(self.type_of(place)?)))
            }
            Expression::BorrowMut(place) => {
                Some(Type::RefMut(Box::new(self.type_of(place)?)))
            }
            Expression::Dereference(inner) => match self.type_of(inner)? {
                Type::Ptr(pointee)
                | Type::Ref(pointee)
                | Type::RefMut(pointee) => Some(*pointee),
                _ => None,
            },
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(name) = &**callee else {
                    return None;
                };
                let declared = self.returns.get(name)?;
                let Some(parameters) = self.generics.get(name) else {
                    return Some(declared.clone());
                };
                let mut bound = HashMap::new();
                for (position, parameter) in parameters {
                    if let Some(Expression::TypeValue(argument)) =
                        arguments.get(*position)
                    {
                        bound.insert(parameter.clone(), argument.clone());
                    }
                }
                Some(crate::ir_build::substitute_type(declared, &bound))
            }
            // A block's value is what its last statement answers with, and
            // `ptr_cast` is written inside one, so the type of what comes out is
            // read through it rather than lost at the boundary.
            Expression::Unsafe(body) => match &body.last()?.node {
                Statement::Expression(value) => self.produced_type(value),
                _ => None,
            },
            Expression::If(_, consequence, alternative) => {
                [Some(consequence), alternative.as_ref()]
                    .into_iter()
                    .flatten()
                    .find_map(|block| match &block.last()?.node {
                        Statement::Expression(value) => {
                            self.produced_type(value)
                        }
                        _ => None,
                    })
            }
            Expression::Switch(_, cases) => {
                cases.iter().find_map(|case| match &case.body.last()?.node {
                    Statement::Expression(value) => self.produced_type(value),
                    _ => None,
                })
            }
            Expression::StructInit(name, _) => Some(Type::Struct(name.clone())),
            Expression::EnumVariantInit(name, _, _) => {
                Some(Type::Enum(name.clone()))
            }
            // A written-out array. Its length is how many elements it holds, and
            // its element type is whatever the first one is, which is the whole
            // of what the index rule needs: a written array is not a pointer.
            Expression::Literal(Literal::Array(elements)) => {
                let element = elements
                    .first()
                    .and_then(|first| self.produced_type(first))
                    .unwrap_or(Type::Unknown);
                Some(Type::Array(Box::new(element), elements.len()))
            }
            // `[value; N]` where `N` is a generic's value parameter. How many is
            // not known until the specialization, and the index rule does not
            // ask how many.
            Expression::ArrayRepeat(value, count) => {
                let element =
                    self.produced_type(value).unwrap_or(Type::Unknown);
                Some(Type::ArrayGeneric(Box::new(element), count.clone()))
            }
            Expression::Literal(Literal::String(_)) => Some(Type::Str),
            Expression::Literal(Literal::Integer(_)) => Some(Type::I64),
            Expression::Literal(Literal::Float(_)) => Some(Type::F64),
            Expression::Literal(Literal::Float32(_)) => Some(Type::F32),
            Expression::Literal(Literal::Boolean(_))
            | Expression::Boolean(_) => Some(Type::Bool),
            Expression::Sizeof(_) | Expression::TypeId(_) => Some(Type::I64),
            Expression::TypeName(_) => Some(Type::Str),
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
            // The multiple-return lowering runs after this pass, so a call
            // bound to several names is still written this way here. Walking
            // past it left an unchecked call with no block around it.
            Statement::LetMultiple(bindings, value) => {
                self.expression(value, at);
                for binding in bindings {
                    self.bind(&binding.name, None);
                }
            }
            Statement::Constant(_, value) | Statement::Return(value) => {
                self.expression(value, at);
            }
            Statement::Expression(value) => self.expression(value, at),
            // `print` holds expressions the way any other statement does, and
            // the gated operations are expression forms. A walk that stopped
            // here let a raw-pointer read out of the block it belongs in.
            Statement::Print(value, arguments) => {
                self.expression(value, at);
                for argument in arguments {
                    self.expression(argument, at);
                }
            }
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
            // Two names bind the index and then the element, so the first is an
            // integer either way. One name over a range is an integer too, and
            // one over a sequence is an element whose type this pass cannot
            // name, which is left unknown rather than assumed.
            Statement::For(name, element, range, body) => {
                self.expression(range, at);
                self.scope.push(HashMap::new());
                let counts =
                    element.is_some() || matches!(range, Expression::Range(..));
                self.bind(name, counts.then_some(Type::I64));
                if let Some(element) = element {
                    self.bind(element, None);
                }
                self.block(body);
                self.scope.pop();
            }
            Statement::While(condition, body) => {
                self.expression(condition, at);
                self.block(body);
            }
            Statement::With(_, body) => self.block(body),
            // A declaration holds no expression to walk, and neither does a
            // control transfer. Listed rather than caught by `_`, so a new
            // statement form is a compile error here instead of a hole in the
            // gate that nobody sees until a program reaches through it.
            Statement::Struct(..)
            | Statement::Enum(..)
            | Statement::Flags(..)
            | Statement::TypeAlias(..)
            | Statement::Break
            | Statement::Continue
            | Statement::Import(..)
            | Statement::Extern { .. }
            | Statement::Declared { .. } => {}
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
                    // Anything else answers with whatever its signature says,
                    // which `type_of` reads off the declaration.
                    _ => self.type_of(value),
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
            // An array, a slice and a `str` each carry a length and are checked,
            // so indexing one is not gated. A raw pointer carries neither, so
            // indexing one is, and a borrow of one is the same pointer under
            // another name. A base this pass cannot name might be either, and a
            // gate that lets the unknown through reports what it happened to
            // recognize rather than what a program can reach. Refusing here is
            // what makes the list of blocks the whole list.
            Expression::Index(base, index) => {
                match self.type_of(base).map(without_borrow) {
                    Some(Type::Ptr(_)) => {
                        self.refuse("indexing a raw pointer", at);
                    }
                    Some(_) => {}
                    None => self.refuse(
                        "indexing a value whose type is not known here",
                        at,
                    ),
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
            | Expression::TypeName(_)
            | Expression::TypeId(_)
            | Expression::TypeValue(_) => {}
        }
    }
}
