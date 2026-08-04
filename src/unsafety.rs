use std::collections::{HashMap, HashSet};

use crate::ast::{
    Ast, ExprId, Expression, Literal, Range32, ReturnKind, Statement, StmtId,
};
use crate::lexer::Position;
use crate::parser::Diagnostic;
use crate::types::Type;

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
// nobody sees until a program reaches through it. The old `print` statement
// was that hole.
/// One walk, both answers: the operations that must sit inside an `unsafe`
/// block, and the blocks holding none. Marking a block as vouched-for costs a
/// bool per block and changes no refusal, so a build gets the second answer for
/// what the first already costs.
///
/// A block that vouches for nothing is a warning rather than a refusal. The
/// program is correct; the block is a place a reader will look for a danger
/// that is not there, and the list of blocks is worth what it is because
/// everything on it earns its place.
pub fn check_unsafety_and_audit(
    ast: &Ast,
    roots: &[StmtId],
) -> (Vec<Diagnostic>, Vec<Diagnostic>) {
    walk_unsafety(ast, roots, true)
        .into_iter()
        .partition(|d| !d.message.starts_with("this `unsafe`"))
}

fn walk_unsafety(ast: &Ast, roots: &[StmtId], audit: bool) -> Vec<Diagnostic> {
    let mut checker = Checker {
        ast,
        externs: HashSet::new(),
        unsafe_fns: HashSet::new(),
        fields: HashMap::new(),
        returns: HashMap::new(),
        multi_returns: HashMap::new(),
        generics: HashMap::new(),
        depth: 0,
        audit,
        vouched: Vec::new(),
        scope: Vec::new(),
        diagnostics: Vec::new(),
    };
    let mut top_level: HashMap<String, Type> = HashMap::new();
    for statement in roots {
        // What each function answers with. The index rule below refuses a base
        // whose type it cannot name, and a binding is most often given its type
        // by the call that produced it, so without this the rule would fall to
        // the refusal on ordinary code rather than on a raw pointer.
        match ast.stmt(*statement) {
            Statement::Constant(name, value) => {
                if let Some(held) = declared_returns(ast, *value) {
                    checker
                        .multi_returns
                        .insert(ast.name(*name).to_string(), held);
                }
                if let Some(ty) = declared_return(ast, *value) {
                    checker.returns.insert(ast.name(*name).to_string(), ty);
                }
                let parameters = type_parameters(ast, *value);
                if !parameters.is_empty() {
                    checker
                        .generics
                        .insert(ast.name(*name).to_string(), parameters);
                }
                // A constant is named from inside every function, so its type
                // belongs to the walk before any of them start. `ROW :: [1, 2]`
                // then `ROW[i]` is an index into an array, and without this the
                // base has no type and the rule refuses it.
                if let Some(ty) = constant_type(ast, *value) {
                    top_level.insert(ast.name(*name).to_string(), ty);
                }
            }
            Statement::Extern {
                name,
                return_type: Some(return_type),
                ..
            } => {
                checker
                    .returns
                    .insert(ast.name(*name).to_string(), return_type.clone());
            }
            Statement::Declared {
                name, return_sig, ..
            } => {
                if let Some(ty) =
                    ast.signature_to_type(ast.signature(*return_sig))
                {
                    checker.returns.insert(ast.name(*name).to_string(), ty);
                }
            }
            _ => {}
        }
        match ast.stmt(*statement) {
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
                    checker.externs.insert(ast.name(*name).to_string());
                }
            }
            Statement::Constant(name, value)
                if matches!(ast.expr(*value), Expression::UnsafeFn(_)) =>
            {
                checker.unsafe_fns.insert(ast.name(*name).to_string());
            }
            Statement::Struct(name, _, declared) => {
                checker
                    .fields
                    .insert(ast.name(*name).to_string(), *declared);
            }
            _ => {}
        }
    }
    checker.scope.push(top_level);
    for statement in roots {
        checker.statement(*statement);
    }
    checker.diagnostics
}

/// The type a top-level constant holds, for the shapes that say so plainly. Only
/// enough to tell an index into one from an index into a raw pointer.
fn constant_type(ast: &Ast, value: ExprId) -> Option<Type> {
    match ast.expr(value) {
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
pub fn strip_unsafe_fns(ast: &mut Ast, roots: &[StmtId]) {
    for statement in roots {
        strip_statement(ast, *statement);
    }
}

fn strip_statement(ast: &mut Ast, statement: StmtId) {
    match ast.stmt(statement).clone() {
        Statement::Let { value, .. }
        | Statement::Constant(_, value)
        | Statement::Return(value)
        | Statement::Expression(value) => strip_expression(ast, value),
        Statement::Assignment(place, value) => {
            strip_expression(ast, place);
            strip_expression(ast, value);
        }
        Statement::For(_, _, range, body) => {
            strip_expression(ast, range);
            strip_block(ast, body);
        }
        Statement::While(condition, body) => {
            strip_expression(ast, condition);
            strip_block(ast, body);
        }
        Statement::With(_, body) => strip_block(ast, body),
        Statement::Defer(inner) => strip_statement(ast, inner),
        _ => {}
    }
}

fn strip_block(ast: &mut Ast, block: Range32) {
    for index in block.indices() {
        let statement = ast.stmt_list[index];
        strip_statement(ast, statement);
    }
}

fn strip_expression(ast: &mut Ast, expression: ExprId) {
    while let Expression::UnsafeFn(inner) = ast.expr(expression) {
        let inner = *inner;
        ast.expressions[expression.0 as usize] = ast.expr(inner).clone();
    }
    match ast.expr(expression).clone() {
        Expression::Function(_, _, body)
        | Expression::Proc(_, _, body)
        | Expression::Unsafe(body) => strip_block(ast, body),
        Expression::If(condition, consequence, alternative) => {
            strip_expression(ast, condition);
            strip_block(ast, consequence);
            if let Some(alternative) = alternative {
                strip_block(ast, alternative);
            }
        }
        Expression::Switch(subject, cases) => {
            strip_expression(ast, subject);
            for index in cases.indices() {
                let body = ast.cases[index].body;
                strip_block(ast, body);
            }
        }
        Expression::Call(callee, arguments) => {
            strip_expression(ast, callee);
            for index in arguments.indices() {
                let argument = ast.expr_list[index];
                strip_expression(ast, argument);
            }
        }
        Expression::PackMap(inner, _, _)
        | Expression::Prefix(_, inner)
        | Expression::AddressOf(inner)
        | Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::Dereference(inner)
        | Expression::Try(inner)
        | Expression::FieldAccess(inner, _) => strip_expression(ast, inner),
        Expression::Infix(left, _, right)
        | Expression::Index(left, right)
        | Expression::Range(left, right, _) => {
            strip_expression(ast, left);
            strip_expression(ast, right);
        }
        Expression::Tuple(parts) => {
            for index in parts.indices() {
                let part = ast.expr_list[index];
                strip_expression(ast, part);
            }
        }
        Expression::StructInit(_, initializers)
        | Expression::EnumVariantInit(_, _, initializers) => {
            for index in initializers.indices() {
                let initializer = ast.named_exprs[index].value;
                strip_expression(ast, initializer);
            }
        }
        _ => {}
    }
}

/// What a named constant answers with, where the constant is a function. An
/// `unsafe fn` wraps the function it marks, so the signature is one level in.
/// The types a function answering with several values hands back, in order.
///
/// The multiple-return lowering runs after this pass, so the struct those
/// values become does not exist yet and the list is read off the signature.
fn declared_returns(ast: &Ast, value: ExprId) -> Option<Vec<Type>> {
    match ast.expr(value) {
        Expression::Function(_, return_sig, _)
        | Expression::Proc(_, return_sig, _) => {
            match &ast.signature(*return_sig).kind {
                ReturnKind::Multiple(values) => Some(
                    ast.return_values_in(*values)
                        .iter()
                        .map(|one| one.value_type.clone())
                        .collect(),
                ),
                _ => None,
            }
        }
        Expression::UnsafeFn(inner) => declared_returns(ast, *inner),
        _ => None,
    }
}

fn declared_return(ast: &Ast, value: ExprId) -> Option<Type> {
    match ast.expr(value) {
        Expression::Function(_, return_sig, _)
        | Expression::Proc(_, return_sig, _) => {
            ast.signature_to_type(ast.signature(*return_sig))
        }
        Expression::UnsafeFn(inner) => declared_return(ast, *inner),
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
fn type_parameters(ast: &Ast, value: ExprId) -> Vec<(usize, String)> {
    let params = match ast.expr(value) {
        Expression::Function(params, _, _) | Expression::Proc(params, _, _) => {
            *params
        }
        Expression::UnsafeFn(inner) => return type_parameters(ast, *inner),
        _ => return Vec::new(),
    };
    ast.params_in(params)
        .iter()
        .enumerate()
        .filter_map(|(position, param)| match &param.type_annotation {
            Some(Type::TypeParam(name)) => Some((position, name.clone())),
            _ => None,
        })
        .collect()
}

struct Checker<'walk> {
    ast: &'walk Ast,
    externs: HashSet<String>,
    unsafe_fns: HashSet<String>,
    fields: HashMap<String, Range32>,
    // What each named function answers with, so a binding takes its type from
    // the call that produced it.
    returns: HashMap<String, Type>,
    // What each function answering with several values hands back, in order, so
    // a binding taken from one has the type of the value it was given.
    multi_returns: HashMap<String, Vec<Type>>,
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

impl Checker<'_> {
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
            related: Vec::new(),
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
    fn type_of(&self, expression: ExprId) -> Option<Type> {
        let ast = self.ast;
        match ast.expr(expression) {
            Expression::Identifier(name) => {
                self.lookup(ast.name(*name)).cloned()
            }
            Expression::FieldAccess(base, field) => {
                let base_type = self.type_of(*base)?;
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
                ast.fields_in(*self.fields.get(Type::template_of(&name))?)
                    .iter()
                    .find(|declared| declared.name == *field)
                    .map(|declared| declared.field_type.clone())
            }
            // An element of an array or a slice, so `rows[i][j]` names the inner
            // element rather than nothing. A `str` indexes to a byte, and a
            // borrow indexes as the place it names.
            Expression::Index(base, _) => match self.type_of(*base)? {
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
                Some(Type::Ref(Box::new(self.type_of(*place)?)))
            }
            Expression::BorrowMut(place) => {
                Some(Type::RefMut(Box::new(self.type_of(*place)?)))
            }
            Expression::Dereference(inner) => match self.type_of(*inner)? {
                Type::Ptr(pointee)
                | Type::Ref(pointee)
                | Type::RefMut(pointee) => Some(*pointee),
                _ => None,
            },
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(name) = ast.expr(*callee) else {
                    return None;
                };
                // The type builtins parse as calls to these names and answer
                // constants, so their types are known here without a
                // declaration to read.
                match ast.name(*name) {
                    "sizeof" | "type_id" => return Some(Type::I64),
                    "typename" => return Some(Type::Str),
                    _ => {}
                }
                let declared = self.returns.get(ast.name(*name))?;
                let Some(parameters) = self.generics.get(ast.name(*name))
                else {
                    return Some(declared.clone());
                };
                let mut bound = HashMap::new();
                for (position, parameter) in parameters {
                    if let Some(argument) =
                        ast.exprs_in(*arguments).get(*position)
                        && let Expression::TypeValue(argument) =
                            ast.expr(*argument)
                    {
                        bound.insert(parameter.clone(), argument.clone());
                    }
                }
                Some(crate::ir_build::substitute_type(declared, &bound))
            }
            // A block's value is what its last statement answers with, and
            // `ptr_cast` is written inside one, so the type of what comes out is
            // read through it rather than lost at the boundary.
            Expression::Unsafe(body) => {
                match ast.stmt(*ast.stmts_in(*body).last()?) {
                    Statement::Expression(value) => self.produced_type(*value),
                    _ => None,
                }
            }
            Expression::If(_, consequence, alternative) => {
                [Some(*consequence), *alternative]
                    .into_iter()
                    .flatten()
                    .find_map(|block| {
                        match ast.stmt(*ast.stmts_in(block).last()?) {
                            Statement::Expression(value) => {
                                self.produced_type(*value)
                            }
                            _ => None,
                        }
                    })
            }
            Expression::Switch(_, cases) => {
                ast.cases_in(*cases).iter().find_map(|case| {
                    match ast.stmt(*ast.stmts_in(case.body).last()?) {
                        Statement::Expression(value) => {
                            self.produced_type(*value)
                        }
                        _ => None,
                    }
                })
            }
            Expression::StructInit(name, _) => {
                Some(Type::Struct(ast.name(*name).to_string()))
            }
            Expression::EnumVariantInit(name, _, _) => {
                Some(Type::Enum(ast.name(*name).to_string()))
            }
            // A written-out array. Its length is how many elements it holds, and
            // its element type is whatever the first one is, which is the whole
            // of what the index rule needs: a written array is not a pointer.
            Expression::Literal(Literal::Array(elements)) => {
                let element = ast
                    .exprs_in(*elements)
                    .first()
                    .and_then(|first| self.produced_type(*first))
                    .unwrap_or(Type::Unknown);
                Some(Type::Array(Box::new(element), elements.len()))
            }
            // `[value; N]` where `N` is a generic's value parameter. How many is
            // not known until the specialization, and the index rule does not
            // ask how many.
            Expression::ArrayRepeat(value, count) => {
                let element =
                    self.produced_type(*value).unwrap_or(Type::Unknown);
                Some(Type::ArrayGeneric(
                    Box::new(element),
                    ast.name(*count).to_string(),
                ))
            }
            Expression::Literal(Literal::String(_)) => Some(Type::Str),
            Expression::Literal(Literal::Integer(_)) => Some(Type::I64),
            Expression::Literal(Literal::Float(_)) => Some(Type::F64),
            Expression::Literal(Literal::Float32(_)) => Some(Type::F32),
            Expression::Literal(Literal::Boolean(_))
            | Expression::Boolean(_) => Some(Type::Bool),
            _ => None,
        }
    }

    fn block(&mut self, block: Range32) {
        self.scope.push(HashMap::new());
        for statement in self.ast.stmts_in(block) {
            self.statement(*statement);
        }
        self.scope.pop();
    }

    fn statement(&mut self, statement: StmtId) {
        let at = self.ast.stmt_position(statement);
        self.statement_at(statement, at);
    }

    fn statement_at(&mut self, statement: StmtId, at: Position) {
        let ast = self.ast;
        match ast.stmt(statement) {
            Statement::Let {
                name,
                type_annotation,
                value,
                ..
            } => {
                self.expression(*value, at);
                let known = type_annotation
                    .clone()
                    .or_else(|| self.produced_type(*value));
                self.bind(ast.name(*name), known);
            }
            // The multiple-return lowering runs after this pass, so a call
            // bound to several names is still written this way here. Walking
            // past it left an unchecked call with no block around it.
            Statement::LetMultiple(bindings, value) => {
                self.expression(*value, at);
                // Each binding takes the type of the value it was given. Left
                // untyped, the index rule met a base it could not name and
                // refused `view[0]` after `view, count := split()`, which is
                // ordinary code and holds nothing unchecked.
                let held = match ast.expr(*value) {
                    Expression::Call(callee, _) => match ast.expr(*callee) {
                        Expression::Identifier(name) => {
                            self.multi_returns.get(ast.name(*name)).cloned()
                        }
                        _ => None,
                    },
                    _ => None,
                };
                for (index, binding) in
                    ast.bindings_in(*bindings).iter().enumerate()
                {
                    let ty = held
                        .as_ref()
                        .and_then(|types| types.get(index).cloned());
                    self.bind(ast.name(binding.name), ty);
                }
            }
            Statement::Constant(_, value) | Statement::Return(value) => {
                self.expression(*value, at);
            }
            Statement::Expression(value) => self.expression(*value, at),
            Statement::Assignment(place, value) => {
                self.expression(*place, at);
                self.expression(*value, at);
            }
            Statement::Defer(inner) => {
                self.statement_at(*inner, at);
            }
            // Two names bind the index and then the element, so the first is an
            // integer either way. One name over a range is an integer too, and
            // one over a sequence is an element whose type this pass cannot
            // name, which is left unknown rather than assumed.
            Statement::For(name, element, range, body) => {
                self.expression(*range, at);
                self.scope.push(HashMap::new());
                let counts = element.is_some()
                    || matches!(ast.expr(*range), Expression::Range(..));
                self.bind(ast.name(*name), counts.then_some(Type::I64));
                if let Some(element) = element {
                    self.bind(ast.name(*element), None);
                }
                self.block(*body);
                self.scope.pop();
            }
            Statement::While(condition, body) => {
                self.expression(*condition, at);
                self.block(*body);
            }
            Statement::With(_, body) => self.block(*body),
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
    fn produced_type(&self, value: ExprId) -> Option<Type> {
        let ast = self.ast;
        match ast.expr(value) {
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(name) = ast.expr(*callee) else {
                    return None;
                };
                match ast.name(*name) {
                    "ptr_cast" => match ast
                        .exprs_in(*arguments)
                        .first()
                        .map(|first| ast.expr(*first))
                    {
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
                        ast.exprs_in(*arguments)
                            .first()
                            .and_then(|argument| self.type_of(*argument))
                            .unwrap_or(Type::Void),
                    ))),
                    // Anything else answers with whatever its signature says,
                    // which `type_of` reads off the declaration.
                    _ => self.type_of(value),
                }
            }
            Expression::AddressOf(inner) => {
                Some(Type::Ptr(Box::new(self.type_of(*inner)?)))
            }
            _ => self.type_of(value),
        }
    }

    fn expression(&mut self, value: ExprId, at: Position) {
        let ast = self.ast;
        match ast.expr(value) {
            Expression::Unsafe(body) => {
                if self.audit {
                    if self.depth > 0 {
                        self.diagnostics.push(Diagnostic {
                            position: at,
                            message: "this `unsafe` block is inside another one, which already vouches for what is in it".to_string(),
                            related: Vec::new(),
                        });
                    }
                    self.vouched.push(false);
                }
                self.depth += 1;
                self.block(*body);
                self.depth -= 1;
                if self.audit
                    && let Some(used) = self.vouched.pop()
                    && !used
                {
                    self.diagnostics.push(Diagnostic {
                        position: at,
                        message: "this `unsafe` block holds no unchecked operation, so it vouches for nothing".to_string(),
                        related: Vec::new(),
                    });
                }
            }
            // An `unsafe fn`'s body is an implicit unsafe block. The whole
            // function is the dangerous region, so the gated operations are
            // allowed throughout it without a nested block.
            Expression::UnsafeFn(inner) => {
                self.depth += 1;
                self.expression(*inner, at);
                self.depth -= 1;
            }
            Expression::Dereference(inner) => {
                self.refuse("reading through a raw pointer", at);
                self.expression(*inner, at);
            }
            // An array, a slice and a `str` each carry a length and are checked,
            // so indexing one is not gated. A raw pointer carries neither, so
            // indexing one is, and a borrow of one is the same pointer under
            // another name. A base this pass cannot name might be either, and a
            // gate that lets the unknown through reports what it happened to
            // recognize rather than what a program can reach. Refusing here is
            // what makes the list of blocks the whole list.
            Expression::Index(base, index) => {
                match self.type_of(*base).map(without_borrow) {
                    Some(Type::Ptr(_)) => {
                        self.refuse("indexing a raw pointer", at);
                    }
                    Some(_) => {}
                    None => self.refuse(
                        "indexing a value whose type is not known here",
                        at,
                    ),
                }
                self.expression(*base, at);
                self.expression(*index, at);
            }
            Expression::Call(callee, arguments) => {
                if let Expression::Identifier(name) = ast.expr(*callee) {
                    let name = ast.name(*name);
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
                self.expression(*callee, at);
                for argument in ast.exprs_in(*arguments) {
                    self.expression(*argument, at);
                }
            }
            Expression::Function(parameters, signature, body)
            | Expression::Proc(parameters, signature, body) => {
                self.scope.push(HashMap::new());
                for parameter in ast.params_in(*parameters) {
                    let annotation = parameter.type_annotation.clone();
                    self.bind(ast.name(parameter.name), annotation);
                }
                // An allocation capability is threaded in as a parameter by a
                // lowering that runs after this check, so the body names it
                // and nothing has declared it yet. Without its type the index
                // rule cannot tell `arena.data[0]` from a reach through a raw
                // pointer, and it refuses what it cannot name: no `uses Arena`
                // function could index its own arena outside an `unsafe` block.
                for capability in &ast.signature(*signature).uses {
                    self.bind(
                        &crate::regions::capability_binding(capability),
                        Some(capability.clone()),
                    );
                }
                self.block(*body);
                self.scope.pop();
            }
            Expression::If(condition, consequence, alternative) => {
                self.expression(*condition, at);
                self.block(*consequence);
                if let Some(alternative) = alternative {
                    self.block(*alternative);
                }
            }
            Expression::Switch(subject, cases) => {
                self.expression(*subject, at);
                for case in ast.cases_in(*cases) {
                    self.block(case.body);
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
                self.expression(*inner, at);
            }
            Expression::Infix(left, _, right)
            | Expression::Range(left, right, _) => {
                self.expression(*left, at);
                self.expression(*right, at);
            }
            Expression::Tuple(parts) => {
                for part in ast.exprs_in(*parts) {
                    self.expression(*part, at);
                }
            }
            Expression::StructInit(_, initializers)
            | Expression::EnumVariantInit(_, _, initializers) => {
                for initializer in ast.named_in(*initializers) {
                    self.expression(initializer.value, at);
                }
            }
            // Listed rather than caught by `_`, so a new expression form is a
            // compile error here instead of walking past whatever it holds.
            Expression::Identifier(_)
            | Expression::Literal(_)
            | Expression::Boolean(_)
            | Expression::TypeValue(_) => {}
        }
    }
}
