use std::collections::HashMap;

use crate::ast::{
    Ast, ExprId, Expression, Parameter, Pattern, PatternId, Range32, Statement,
    StmtId, Symbol,
};
use crate::parser::ParamMode;
use crate::types::Type;

// A parameter declares `$T: Type`, which is erased from the runtime signature.
fn is_type_parameter(ast: &Ast, parameter: &Parameter) -> bool {
    matches!(
        &parameter.type_annotation,
        Some(Type::TypeParam(name)) if name.as_str() == ast.name(parameter.name)
    )
}

// The reference-or-value type a parameter's mode desugars to. An explicit
// reference type (legacy `&`/`&mut`) is left as written. `read` borrows a
// non-copy value and copies a copy one. `mut` borrows exclusively. `move` takes
// the value.
fn effective_type(parameter: &Parameter) -> Option<Type> {
    let ty = parameter.type_annotation.as_ref()?;
    if matches!(ty, Type::Ref(_) | Type::RefMut(_)) {
        return Some(ty.clone());
    }
    Some(match parameter.mode {
        ParamMode::Move => ty.clone(),
        ParamMode::Write => Type::RefMut(Box::new(ty.clone())),
        // `value` says how the bytes cross to C, not what the caller gives up.
        // C is handed a copy, so the caller still holds its own value and the
        // parameter borrows exactly as an unmarked one does. The copy is made
        // at the call, by the backend, from this same borrow.
        ParamMode::Read | ParamMode::Value if ty.is_copy() => ty.clone(),
        ParamMode::Read | ParamMode::Value => Type::Ref(Box::new(ty.clone())),
    })
}

// The effective types of the runtime (non-type) parameters, in order.
fn runtime_param_types(ast: &Ast, params: Range32) -> Vec<Option<Type>> {
    ast.params_in(params)
        .iter()
        .filter(|parameter| !is_type_parameter(ast, parameter))
        .map(effective_type)
        .collect()
}

// Turn parameter modes into the reference types the rest of the compiler already
// handles, and insert the borrows those references need at every call site, so
// `f(x)` against a `read`/`mut` parameter borrows `x` automatically.
pub fn lower_param_modes(ast: &mut Ast, roots: &[StmtId]) {
    let mut signatures: HashMap<String, Vec<Option<Type>>> = HashMap::new();
    collect_signatures(ast, roots, &mut signatures);
    for statement in roots {
        rewrite_statement(ast, *statement, &signatures);
    }
}

fn collect_signatures(
    ast: &Ast,
    roots: &[StmtId],
    signatures: &mut HashMap<String, Vec<Option<Type>>>,
) {
    for statement in roots {
        match ast.stmt(*statement) {
            Statement::Constant(name, value) => {
                if let Expression::Function(params, _, _)
                | Expression::Proc(params, _, _) = ast.expr(*value)
                {
                    signatures.insert(
                        ast.name(*name).to_string(),
                        runtime_param_types(ast, *params),
                    );
                }
            }
            Statement::Extern { name, params, .. }
            | Statement::Declared { name, params, .. } => {
                signatures.insert(
                    ast.name(*name).to_string(),
                    runtime_param_types(ast, *params),
                );
            }
            _ => {}
        }
    }
}

fn rewrite_parameters(ast: &mut Ast, params: Range32) {
    for index in params.indices() {
        let parameter = &ast.parameters[index];
        if is_type_parameter(ast, parameter) {
            continue;
        }
        if let Some(effective) = effective_type(parameter) {
            ast.parameters[index].type_annotation = Some(effective);
        }
    }
}

fn rewrite_statement(
    ast: &mut Ast,
    statement: StmtId,
    signatures: &HashMap<String, Vec<Option<Type>>>,
) {
    match ast.stmt(statement).clone() {
        Statement::Let { value, .. } | Statement::Constant(_, value) => {
            rewrite_expression(ast, value, signatures);
        }
        Statement::Return(expression) | Statement::Expression(expression) => {
            rewrite_expression(ast, expression, signatures);
        }
        Statement::Defer(inner) | Statement::ErrDefer(inner) => {
            rewrite_statement(ast, inner, signatures)
        }
        Statement::Assignment(place, value) => {
            rewrite_expression(ast, place, signatures);
            rewrite_expression(ast, value, signatures);
        }
        Statement::For(_, _, iterable, body) => {
            rewrite_expression(ast, iterable, signatures);
            rewrite_block(ast, body, signatures);
        }
        Statement::While(condition, body) => {
            rewrite_expression(ast, condition, signatures);
            rewrite_block(ast, body, signatures);
        }
        // A declared signature has to lower its modes the same way the
        // definition did, or the call this program emits and the object that
        // defines it would disagree about what a parameter is.
        Statement::Declared { params, .. } => rewrite_parameters(ast, params),
        _ => {}
    }
}

fn rewrite_block(
    ast: &mut Ast,
    block: Range32,
    signatures: &HashMap<String, Vec<Option<Type>>>,
) {
    for index in block.indices() {
        let statement = ast.stmt_list[index];
        rewrite_statement(ast, statement, signatures);
    }
}

// Read a `mut` parameter through the reference it became, everywhere the body
// names it. A binding of the same name inside the body is a different thing and
// stops the rewrite from there on.
type Bound = Vec<Vec<Symbol>>;

fn shadowed(name: Symbol, bound: &Bound) -> bool {
    bound.iter().any(|frame| frame.contains(&name))
}

fn read_through_block(
    ast: &mut Ast,
    block: Range32,
    through: &[Symbol],
    bound: &mut Bound,
) {
    bound.push(Vec::new());
    for index in block.indices() {
        let statement = ast.stmt_list[index];
        read_through_statement(ast, statement, through, bound);
    }
    bound.pop();
}

fn read_through_statement(
    ast: &mut Ast,
    statement: StmtId,
    through: &[Symbol],
    bound: &mut Bound,
) {
    match ast.stmt(statement).clone() {
        Statement::Let {
            name,
            value,
            type_annotation: _,
            mutable: _,
        } => {
            read_through_expression(ast, value, through, bound);
            if let Some(frame) = bound.last_mut() {
                frame.push(name);
            }
        }
        Statement::LetMultiple(bindings, value) => {
            read_through_expression(ast, value, through, bound);
            if let Some(frame) = bound.last_mut() {
                for binding in ast.bindings_in(bindings) {
                    frame.push(binding.name);
                }
            }
        }
        Statement::Constant(_, value)
        | Statement::Return(value)
        | Statement::Expression(value) => {
            read_through_expression(ast, value, through, bound)
        }
        Statement::Assignment(place, value) => {
            read_through_expression(ast, place, through, bound);
            read_through_expression(ast, value, through, bound);
        }
        Statement::Defer(inner) | Statement::ErrDefer(inner) => {
            read_through_statement(ast, inner, through, bound)
        }
        Statement::For(variable, _, iterable, body) => {
            read_through_expression(ast, iterable, through, bound);
            bound.push(vec![variable]);
            read_through_block(ast, body, through, bound);
            bound.pop();
        }
        Statement::While(condition, body) => {
            read_through_expression(ast, condition, through, bound);
            read_through_block(ast, body, through, bound);
        }
        Statement::With(_, body) => {
            read_through_block(ast, body, through, bound)
        }
        // Declarations name types rather than values, so nothing to read.
        Statement::Struct(..)
        | Statement::Enum(..)
        | Statement::Flags(..)
        | Statement::TypeAlias(..)
        | Statement::Extern { .. }
        | Statement::Declared { .. }
        | Statement::Break
        | Statement::Continue
        | Statement::Import(..) => {}
    }
}

fn read_through_expression(
    ast: &mut Ast,
    expression: ExprId,
    through: &[Symbol],
    bound: &mut Bound,
) {
    if let Expression::Identifier(name) = ast.expr(expression)
        && through.iter().any(|held| held == name)
        && !shadowed(*name, bound)
    {
        let name = *name;
        let span = ast.expr_span(expression);
        let inner = ast.push_expr(Expression::Identifier(name), span);
        ast.expressions[expression.0 as usize] = Expression::Dereference(inner);
        return;
    }
    match ast.expr(expression).clone() {
        Expression::PackMap(inner, _, _)
        | Expression::Prefix(_, inner)
        | Expression::AddressOf(inner)
        | Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::Try(inner)
        | Expression::Dereference(inner)
        | Expression::ArrayRepeat(inner, _)
        | Expression::FieldAccess(inner, _) => {
            read_through_expression(ast, inner, through, bound)
        }
        Expression::Infix(left, _, right)
        | Expression::Index(left, right)
        | Expression::Range(left, right, _) => {
            read_through_expression(ast, left, through, bound);
            read_through_expression(ast, right, through, bound);
        }
        Expression::Call(callee, arguments) => {
            read_through_expression(ast, callee, through, bound);
            for index in arguments.indices() {
                let argument = ast.expr_list[index];
                read_through_expression(ast, argument, through, bound);
            }
        }
        Expression::If(condition, consequence, alternative) => {
            read_through_expression(ast, condition, through, bound);
            read_through_block(ast, consequence, through, bound);
            if let Some(block) = alternative {
                read_through_block(ast, block, through, bound);
            }
        }
        Expression::StructInit(_, fields)
        | Expression::EnumVariantInit(_, _, fields) => {
            for index in fields.indices() {
                let value = ast.named_exprs[index].value;
                read_through_expression(ast, value, through, bound);
            }
        }
        Expression::Tuple(items) => {
            for index in items.indices() {
                let item = ast.expr_list[index];
                read_through_expression(ast, item, through, bound);
            }
        }
        Expression::Switch(scrutinee, cases) => {
            read_through_expression(ast, scrutinee, through, bound);
            for index in cases.indices() {
                let case = ast.cases[index];
                // A pattern's bindings are the arm's own names, so one that
                // spells a parameter shadows it for the arm.
                bound.push(pattern_bindings(ast, case.pattern));
                read_through_block(ast, case.body, through, bound);
                bound.pop();
            }
        }
        Expression::Unsafe(body) => {
            read_through_block(ast, body, through, bound)
        }
        // A nested function has parameters of its own and does not see these.
        Expression::Function(..) | Expression::Proc(..) => {}
        // Listed rather than caught by `_`, so a new expression form is a
        // compile error here instead of silently reading through nothing.
        Expression::Identifier(_)
        | Expression::Literal(_)
        | Expression::Boolean(_)
        | Expression::TypeValue(_)
        | Expression::UnsafeFn(_) => {}
    }
}

fn rewrite_expression(
    ast: &mut Ast,
    expression: ExprId,
    signatures: &HashMap<String, Vec<Option<Type>>>,
) {
    match ast.expr(expression).clone() {
        Expression::Function(params, ret, body)
        | Expression::Proc(params, ret, body) => {
            // A `mut` parameter is a reference the body never asked for, so
            // every mention of it reads through that reference. Field access
            // would deref on its own, but a whole-value use would not, and
            // `a = b` on an unrewritten name assigns to the local reference
            // instead of through it, which silently leaves the caller's value
            // alone. A parameter already written as a reference type is not
            // this pass's doing and is left as it is.
            let mut through: Vec<Symbol> = Vec::new();
            for parameter in ast.params_in(params) {
                if !is_type_parameter(ast, parameter)
                    && parameter.mode == ParamMode::Write
                    && parameter.type_annotation.as_ref().is_some_and(|ty| {
                        !matches!(ty, Type::Ref(_) | Type::RefMut(_))
                    })
                {
                    through.push(parameter.name);
                }
            }
            rewrite_parameters(ast, params);
            let _ = ret;
            rewrite_block(ast, body, signatures);
            if !through.is_empty() {
                let mut bound: Bound = Vec::new();
                read_through_block(ast, body, &through, &mut bound);
            }
        }
        Expression::Call(callee, arguments) => {
            rewrite_expression(ast, callee, signatures);
            for index in arguments.indices() {
                let argument = ast.expr_list[index];
                rewrite_expression(ast, argument, signatures);
            }
            // Auto-borrow at call sites needs the argument's type (to avoid
            // re-borrowing a value that is already a reference), which only
            // exists during lowering. Done in ir::build, not here.
            let _ = auto_borrow_call;
        }
        Expression::PackMap(inner, _, _)
        | Expression::Prefix(_, inner)
        | Expression::AddressOf(inner)
        | Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::Dereference(inner) => {
            rewrite_expression(ast, inner, signatures);
        }
        Expression::Infix(left, _, right) | Expression::Index(left, right) => {
            rewrite_expression(ast, left, signatures);
            rewrite_expression(ast, right, signatures);
        }
        Expression::FieldAccess(inner, _)
        | Expression::ArrayRepeat(inner, _) => {
            rewrite_expression(ast, inner, signatures);
        }
        Expression::If(condition, then_block, else_block) => {
            rewrite_expression(ast, condition, signatures);
            rewrite_block(ast, then_block, signatures);
            if let Some(block) = else_block {
                rewrite_block(ast, block, signatures);
            }
        }
        Expression::StructInit(_, fields)
        | Expression::EnumVariantInit(_, _, fields) => {
            for index in fields.indices() {
                let value = ast.named_exprs[index].value;
                rewrite_expression(ast, value, signatures);
            }
        }
        Expression::Range(low, high, _) => {
            rewrite_expression(ast, low, signatures);
            rewrite_expression(ast, high, signatures);
        }
        Expression::Switch(scrutinee, cases) => {
            rewrite_expression(ast, scrutinee, signatures);
            for index in cases.indices() {
                let body = ast.cases[index].body;
                rewrite_block(ast, body, signatures);
            }
        }
        Expression::Tuple(items) => {
            for index in items.indices() {
                let item = ast.expr_list[index];
                rewrite_expression(ast, item, signatures);
            }
        }
        Expression::Unsafe(block) => rewrite_block(ast, block, signatures),
        Expression::Try(inner) => rewrite_expression(ast, inner, signatures),
        Expression::Identifier(_)
        | Expression::Literal(_)
        | Expression::Boolean(_)
        | Expression::TypeValue(_)
        | Expression::UnsafeFn(_) => {}
    }
}

fn auto_borrow_call(
    ast: &mut Ast,
    callee: ExprId,
    arguments: Range32,
    signatures: &HashMap<String, Vec<Option<Type>>>,
) {
    let Expression::Identifier(name) = ast.expr(callee) else {
        return;
    };
    let Some(param_types) = signatures.get(ast.name(*name)) else {
        return;
    };
    // Align runtime (non-type-value) arguments with runtime parameters.
    let mut param_index = 0;
    for index in arguments.indices() {
        let argument = ast.expr_list[index];
        if matches!(ast.expr(argument), Expression::TypeValue(_)) {
            continue;
        }
        let Some(effective) = param_types.get(param_index) else {
            break;
        };
        param_index += 1;
        let Some(effective) = effective else { continue };
        match effective {
            Type::Ref(_)
                if !matches!(
                    ast.expr(argument),
                    Expression::Borrow(_) | Expression::BorrowMut(_)
                ) =>
            {
                let inner = ast.expr(argument).clone();
                let span = ast.expr_span(argument);
                let inner = ast.push_expr(inner, span);
                ast.expressions[argument.0 as usize] =
                    Expression::Borrow(inner);
            }
            Type::RefMut(_)
                if !matches!(
                    ast.expr(argument),
                    Expression::Borrow(_) | Expression::BorrowMut(_)
                ) =>
            {
                let inner = ast.expr(argument).clone();
                let span = ast.expr_span(argument);
                let inner = ast.push_expr(inner, span);
                ast.expressions[argument.0 as usize] =
                    Expression::BorrowMut(inner);
            }
            _ => {}
        }
    }
}

// The names a pattern binds in the arm it belongs to.
fn pattern_bindings(ast: &Ast, pattern: PatternId) -> Vec<Symbol> {
    match ast.pattern(pattern) {
        Pattern::EnumVariant { bindings, .. } => ast
            .pattern_bindings_in(*bindings)
            .iter()
            .map(|held| held.binding)
            .collect(),
        Pattern::Tuple(patterns) | Pattern::Or(patterns) => ast
            .patterns_in(*patterns)
            .iter()
            .flat_map(|held| pattern_bindings(ast, *held))
            .collect(),
        Pattern::Wildcard | Pattern::Literal(_) | Pattern::Range { .. } => {
            Vec::new()
        }
    }
}
