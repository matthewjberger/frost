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

// Read a `mut` scalar parameter through the reference it became, everywhere the
// body names it. A binding of the same name inside the body is a different
// thing and stops the rewrite from there on.
type Bound = Vec<Vec<String>>;

fn shadowed(name: &str, bound: &Bound) -> bool {
    bound
        .iter()
        .any(|frame| frame.iter().any(|held| held == name))
}

fn read_through_block(
    block: &mut Block,
    through: &[String],
    bound: &mut Bound,
) {
    bound.push(Vec::new());
    for statement in block.iter_mut() {
        read_through_statement(&mut statement.node, through, bound);
    }
    bound.pop();
}

fn read_through_statement(
    statement: &mut Statement,
    through: &[String],
    bound: &mut Bound,
) {
    match statement {
        Statement::Let {
            name,
            value,
            type_annotation: _,
            mutable: _,
        } => {
            read_through_expression(value, through, bound);
            if let Some(frame) = bound.last_mut() {
                frame.push(name.clone());
            }
        }
        Statement::Constant(_, value)
        | Statement::Return(value)
        | Statement::Expression(value) => {
            read_through_expression(value, through, bound)
        }
        Statement::Assignment(place, value) => {
            read_through_expression(place, through, bound);
            read_through_expression(value, through, bound);
        }
        Statement::Defer(inner) => {
            read_through_statement(inner, through, bound)
        }
        Statement::For(variable, iterable, body) => {
            read_through_expression(iterable, through, bound);
            bound.push(vec![variable.clone()]);
            read_through_block(body, through, bound);
            bound.pop();
        }
        Statement::While(condition, body) => {
            read_through_expression(condition, through, bound);
            read_through_block(body, through, bound);
        }
        Statement::With(_, body) => read_through_block(body, through, bound),
        // Declarations name types rather than values, so nothing to read.
        Statement::Struct(..)
        | Statement::Enum(..)
        | Statement::TypeAlias(..)
        | Statement::Extern { .. }
        | Statement::Break
        | Statement::Continue
        | Statement::Import(_) => {}
    }
}

fn read_through_expression(
    expression: &mut Expression,
    through: &[String],
    bound: &mut Bound,
) {
    if let Expression::Identifier(name) = expression
        && through.iter().any(|held| held == name)
        && !shadowed(name, bound)
    {
        *expression = Expression::Dereference(Box::new(expression.clone()));
        return;
    }
    match expression {
        Expression::Prefix(_, inner)
        | Expression::AddressOf(inner)
        | Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::Try(inner)
        | Expression::Dereference(inner)
        | Expression::FieldAccess(inner, _) => {
            read_through_expression(inner, through, bound)
        }
        Expression::Infix(left, _, right)
        | Expression::Index(left, right)
        | Expression::Range(left, right, _) => {
            read_through_expression(left, through, bound);
            read_through_expression(right, through, bound);
        }
        Expression::Call(callee, arguments) => {
            read_through_expression(callee, through, bound);
            for argument in arguments.iter_mut() {
                read_through_expression(argument, through, bound);
            }
        }
        Expression::If(condition, consequence, alternative) => {
            read_through_expression(condition, through, bound);
            read_through_block(consequence, through, bound);
            if let Some(block) = alternative {
                read_through_block(block, through, bound);
            }
        }
        Expression::StructInit(_, fields)
        | Expression::EnumVariantInit(_, _, fields) => {
            for (_, value) in fields.iter_mut() {
                read_through_expression(value, through, bound);
            }
        }
        Expression::Tuple(items) => {
            for item in items.iter_mut() {
                read_through_expression(item, through, bound);
            }
        }
        Expression::Switch(scrutinee, cases) => {
            read_through_expression(scrutinee, through, bound);
            for case in cases.iter_mut() {
                // A pattern's bindings are the arm's own names, so one that
                // spells a parameter shadows it for the arm.
                bound.push(pattern_bindings(&case.pattern));
                read_through_block(&mut case.body, through, bound);
                bound.pop();
            }
        }
        Expression::Unsafe(body) => read_through_block(body, through, bound),
        // A nested function has parameters of its own and does not see these.
        Expression::Function(..) | Expression::Proc(..) => {}
        _ => {}
    }
}

fn rewrite_expression(
    expression: &mut Expression,
    signatures: &HashMap<String, Vec<Option<Type>>>,
) {
    match expression {
        Expression::Function(params, ret, body)
        | Expression::Proc(params, ret, body) => {
            // A `mut` parameter of a copy type is a reference the body never
            // asked for. A struct one reads through itself, since a field
            // access derefs on the way, but a scalar is used as a whole value
            // and has to be read through explicitly.
            let through: Vec<String> = params
                .iter()
                .filter(|parameter| {
                    !is_type_parameter(parameter)
                        && parameter.mode == ParamMode::Write
                        && parameter.type_annotation.as_ref().is_some_and(
                            |ty| {
                                ty.is_copy()
                                    && !matches!(
                                        ty,
                                        Type::Ref(_) | Type::RefMut(_)
                                    )
                            },
                        )
                })
                .map(|parameter| parameter.name.clone())
                .collect();
            rewrite_parameters(params);
            let _ = ret;
            rewrite_block(body, signatures);
            if !through.is_empty() {
                let mut bound: Vec<Vec<String>> = Vec::new();
                read_through_block(body, &through, &mut bound);
            }
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

// The names a pattern binds in the arm it belongs to.
fn pattern_bindings(pattern: &crate::parser::Pattern) -> Vec<String> {
    use crate::parser::Pattern;
    match pattern {
        Pattern::Identifier(name) => vec![name.clone()],
        Pattern::EnumVariant { bindings, .. } => {
            bindings.iter().map(|(_, name)| name.clone()).collect()
        }
        Pattern::Tuple(patterns) => {
            patterns.iter().flat_map(pattern_bindings).collect()
        }
        Pattern::Wildcard | Pattern::Literal(_) => Vec::new(),
    }
}
