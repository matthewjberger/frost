use std::collections::HashMap;

use crate::parser::{
    Block, Expression, ParamMode, Parameter, Program, Statement, SwitchCase,
};
use crate::types::Type;

// A parameter declares `$T: Type`, which is erased from the runtime signature.
fn is_type_parameter(parameter: &Parameter) -> bool {
    matches!(
        &parameter.type_annotation,
        Some(Type::TypeParam(name)) if name == &parameter.name
    )
}

// The reference-or-value type a parameter's mode desugars to. An explicit
// reference type (legacy `&`/`&mut`) is left as written. `read` borrows a
// non-copy value and copies a copy one; `mut` borrows exclusively; `move` takes
// the value.
fn effective_type(parameter: &Parameter) -> Option<Type> {
    let ty = parameter.type_annotation.as_ref()?;
    if matches!(ty, Type::Ref(_) | Type::RefMut(_)) {
        return Some(ty.clone());
    }
    Some(match parameter.mode {
        ParamMode::Move => ty.clone(),
        ParamMode::Write => Type::RefMut(Box::new(ty.clone())),
        ParamMode::Read if ty.is_copy() => ty.clone(),
        ParamMode::Read => Type::Ref(Box::new(ty.clone())),
    })
}

// The effective types of the runtime (non-type) parameters, in order.
fn runtime_param_types(params: &[Parameter]) -> Vec<Option<Type>> {
    params
        .iter()
        .filter(|parameter| !is_type_parameter(parameter))
        .map(effective_type)
        .collect()
}

// Turn parameter modes into the reference types the rest of the compiler already
// handles, and insert the borrows those references need at every call site, so
// `f(x)` against a `read`/`mut` parameter borrows `x` automatically.
pub fn lower_param_modes(program: &mut Program) {
    let mut signatures: HashMap<String, Vec<Option<Type>>> = HashMap::new();
    collect_signatures(program, &mut signatures);
    for statement in program.iter_mut() {
        rewrite_statement(&mut statement.node, &signatures);
    }
}

fn collect_signatures(
    program: &Program,
    signatures: &mut HashMap<String, Vec<Option<Type>>>,
) {
    for statement in program {
        match &statement.node {
            Statement::Constant(name, Expression::Function(params, _, _))
            | Statement::Constant(name, Expression::Proc(params, _, _)) => {
                signatures.insert(name.clone(), runtime_param_types(params));
            }
            Statement::Extern { name, params, .. } => {
                signatures.insert(name.clone(), runtime_param_types(params));
            }
            _ => {}
        }
    }
}

fn rewrite_parameters(params: &mut [Parameter]) {
    for parameter in params.iter_mut() {
        if is_type_parameter(parameter) {
            continue;
        }
        if let Some(effective) = effective_type(parameter) {
            parameter.type_annotation = Some(effective);
        }
    }
}

fn rewrite_statement(
    statement: &mut Statement,
    signatures: &HashMap<String, Vec<Option<Type>>>,
) {
    match statement {
        Statement::Let { value, .. } | Statement::Constant(_, value) => {
            rewrite_expression(value, signatures);
        }
        Statement::Return(expression) | Statement::Expression(expression) => {
            rewrite_expression(expression, signatures);
        }
        Statement::Defer(inner) => rewrite_statement(inner, signatures),
        Statement::Assignment(place, value) => {
            rewrite_expression(place, signatures);
            rewrite_expression(value, signatures);
        }
        Statement::For(_, iterable, body) => {
            rewrite_expression(iterable, signatures);
            rewrite_block(body, signatures);
        }
        Statement::While(condition, body) => {
            rewrite_expression(condition, signatures);
            rewrite_block(body, signatures);
        }
        _ => {}
    }
}

fn rewrite_block(
    block: &mut Block,
    signatures: &HashMap<String, Vec<Option<Type>>>,
) {
    for statement in block.iter_mut() {
        rewrite_statement(&mut statement.node, signatures);
    }
}

fn rewrite_expression(
    expression: &mut Expression,
    signatures: &HashMap<String, Vec<Option<Type>>>,
) {
    match expression {
        Expression::Function(params, ret, body)
        | Expression::Proc(params, ret, body) => {
            rewrite_parameters(params);
            let _ = ret;
            rewrite_block(body, signatures);
        }
        Expression::Call(callee, arguments) => {
            rewrite_expression(callee, signatures);
            for argument in arguments.iter_mut() {
                rewrite_expression(argument, signatures);
            }
            // Auto-borrow at call sites needs the argument's type (to avoid
            // re-borrowing a value that is already a reference), which only
            // exists during lowering. Done in ir_build, not here.
            let _ = auto_borrow_call;
        }
        Expression::Prefix(_, inner)
        | Expression::AddressOf(inner)
        | Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::Dereference(inner) => {
            rewrite_expression(inner, signatures);
        }
        Expression::Infix(left, _, right) | Expression::Index(left, right) => {
            rewrite_expression(left, signatures);
            rewrite_expression(right, signatures);
        }
        Expression::FieldAccess(inner, _) => {
            rewrite_expression(inner, signatures);
        }
        Expression::If(condition, then_block, else_block) => {
            rewrite_expression(condition, signatures);
            rewrite_block(then_block, signatures);
            if let Some(block) = else_block {
                rewrite_block(block, signatures);
            }
        }
        Expression::StructInit(_, fields)
        | Expression::EnumVariantInit(_, _, fields) => {
            for (_, value) in fields.iter_mut() {
                rewrite_expression(value, signatures);
            }
        }
        Expression::Range(low, high, _) => {
            rewrite_expression(low, signatures);
            rewrite_expression(high, signatures);
        }
        Expression::Switch(scrutinee, cases) => {
            rewrite_expression(scrutinee, signatures);
            for SwitchCase { body, .. } in cases.iter_mut() {
                rewrite_block(body, signatures);
            }
        }
        Expression::Tuple(items) => {
            for item in items.iter_mut() {
                rewrite_expression(item, signatures);
            }
        }
        Expression::Unsafe(block) => rewrite_block(block, signatures),
        _ => {}
    }
}

fn auto_borrow_call(
    callee: &Expression,
    arguments: &mut [Expression],
    signatures: &HashMap<String, Vec<Option<Type>>>,
) {
    let Expression::Identifier(name) = callee else {
        return;
    };
    let Some(param_types) = signatures.get(name) else {
        return;
    };
    // Align runtime (non-type-value) arguments with runtime parameters.
    let mut param_index = 0;
    for argument in arguments.iter_mut() {
        if matches!(argument, Expression::TypeValue(_)) {
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
                    argument,
                    Expression::Borrow(_) | Expression::BorrowMut(_)
                ) =>
            {
                let inner =
                    std::mem::replace(argument, Expression::Boolean(false));
                *argument = Expression::Borrow(Box::new(inner));
            }
            Type::RefMut(_)
                if !matches!(
                    argument,
                    Expression::Borrow(_) | Expression::BorrowMut(_)
                ) =>
            {
                let inner =
                    std::mem::replace(argument, Expression::Boolean(false));
                *argument = Expression::BorrowMut(Box::new(inner));
            }
            _ => {}
        }
    }
}
