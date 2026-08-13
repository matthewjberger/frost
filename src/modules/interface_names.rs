use crate::ast::{
    Ast, ExprId, Expression, Range32, ReturnKind, Statement, StmtId,
};
use crate::types::Type;

// Every named type or function a declaration refers to, so an interface can
// carry the things an exported name depends on. Deliberately over-approximate:
// naming something that turns out not to be a type costs an entry that nothing
// matches, while missing one costs a caller that cannot compile.
pub(crate) fn names_in_statement(
    ast: &Ast,
    statement: StmtId,
    out: &mut Vec<String>,
) {
    match ast.stmt(statement) {
        Statement::Constant(_, value) => names_in_expression(ast, *value, out),
        Statement::Declared {
            params, return_sig, ..
        } => {
            for param in ast.params_in(*params) {
                if let Some(ty) = &param.type_annotation {
                    names_in_type(ty, out);
                }
                if let Some(ty) = &param.compile_time_signature {
                    names_in_type(ty, out);
                }
            }
            let signature = ast.signature(*return_sig);
            match &signature.kind {
                ReturnKind::None => {}
                ReturnKind::Single(ty) => names_in_type(ty, out),
                ReturnKind::Multiple(values) => {
                    for held in ast.return_values_in(*values) {
                        names_in_type(&held.value_type, out);
                    }
                }
                ReturnKind::Fallible(value, failure) => {
                    names_in_type(value, out);
                    names_in_type(failure, out);
                }
            }
            for capability in &signature.uses {
                names_in_type(capability, out);
            }
        }
        Statement::Struct(_, _, fields) => {
            for field in ast.fields_in(*fields) {
                names_in_type(&field.field_type, out);
            }
        }
        Statement::Enum(_, _, variants) => {
            for variant in ast.variants_in(*variants) {
                if let Some(fields) = variant.fields {
                    for field in ast.fields_in(fields) {
                        names_in_type(&field.field_type, out);
                    }
                }
            }
        }
        Statement::TypeAlias(_, ty) | Statement::Flags(_, ty, _) => {
            names_in_type(ty, out)
        }
        Statement::Extern {
            params,
            return_type,
            ..
        } => {
            for parameter in ast.params_in(*params) {
                if let Some(ty) = &parameter.type_annotation {
                    names_in_type(ty, out);
                }
            }
            if let Some(ty) = return_type {
                names_in_type(ty, out);
            }
        }
        Statement::Let {
            type_annotation,
            value,
            ..
        } => {
            if let Some(ty) = type_annotation {
                names_in_type(ty, out);
            }
            names_in_expression(ast, *value, out);
        }
        Statement::LetMultiple(_, value)
        | Statement::Return(value)
        | Statement::Expression(value) => names_in_expression(ast, *value, out),
        Statement::Assignment(place, value) => {
            names_in_expression(ast, *place, out);
            names_in_expression(ast, *value, out);
        }
        Statement::Defer(inner) | Statement::ErrDefer(inner) => {
            names_in_statement(ast, *inner, out)
        }
        Statement::For(_, _, iterable, body) => {
            names_in_expression(ast, *iterable, out);
            names_in_block(ast, *body, out);
        }
        Statement::While(condition, body) => {
            names_in_expression(ast, *condition, out);
            names_in_block(ast, *body, out);
        }
        Statement::With(capability, body) => {
            out.push(ast.name(*capability).to_string());
            names_in_block(ast, *body, out);
        }
        Statement::Break | Statement::Continue | Statement::Import(..) => {}
    }
}

fn names_in_block(ast: &Ast, block: Range32, out: &mut Vec<String>) {
    for statement in ast.stmts_in(block) {
        names_in_statement(ast, *statement, out);
    }
}

pub(crate) fn names_in_expression(
    ast: &Ast,
    expression: ExprId,
    out: &mut Vec<String>,
) {
    match ast.expr(expression) {
        // A call names its callee, which for a generic body is how a template
        // reaches a helper the module did not export.
        Expression::Identifier(name) => out.push(ast.name(*name).to_string()),
        Expression::StructInit(name, fields) => {
            out.push(ast.name(*name).to_string());
            for field in ast.named_in(*fields) {
                names_in_expression(ast, field.value, out);
            }
        }
        Expression::EnumVariantInit(name, _, fields) => {
            out.push(ast.name(*name).to_string());
            for field in ast.named_in(*fields) {
                names_in_expression(ast, field.value, out);
            }
        }
        Expression::TypeValue(ty) => names_in_type(ty, out),
        Expression::Function(params, return_sig, body)
        | Expression::Proc(params, return_sig, body) => {
            for parameter in ast.params_in(*params) {
                if let Some(ty) = &parameter.type_annotation {
                    names_in_type(ty, out);
                }
                if let Some(ty) = &parameter.compile_time_signature {
                    names_in_type(ty, out);
                }
            }
            let signature = ast.signature(*return_sig);
            if let Some(ty) = ast.signature_to_type(signature) {
                names_in_type(&ty, out);
            }
            for ty in &signature.uses {
                names_in_type(ty, out);
            }
            names_in_block(ast, *body, out);
        }
        Expression::Call(callee, arguments) => {
            names_in_expression(ast, *callee, out);
            for argument in ast.exprs_in(*arguments) {
                names_in_expression(ast, *argument, out);
            }
        }
        Expression::PackMap(inner, _, _)
        | Expression::Prefix(_, inner)
        | Expression::AddressOf(inner)
        | Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::Try(inner)
        | Expression::Dereference(inner)
        | Expression::FieldAccess(inner, _) => {
            names_in_expression(ast, *inner, out)
        }
        Expression::Infix(left, _, right)
        | Expression::Index(left, right)
        | Expression::Range(left, right, _) => {
            names_in_expression(ast, *left, out);
            names_in_expression(ast, *right, out);
        }
        Expression::If(condition, consequence, alternative) => {
            names_in_expression(ast, *condition, out);
            names_in_block(ast, *consequence, out);
            if let Some(block) = alternative {
                names_in_block(ast, *block, out);
            }
        }
        Expression::Switch(scrutinee, cases) => {
            names_in_expression(ast, *scrutinee, out);
            for case in ast.cases_in(*cases) {
                if let crate::ast::Pattern::EnumVariant {
                    enum_name: Some(name),
                    ..
                } = ast.pattern(case.pattern)
                {
                    out.push(ast.name(*name).to_string());
                }
                names_in_block(ast, case.body, out);
            }
        }
        Expression::Tuple(elements) => {
            for element in ast.exprs_in(*elements) {
                names_in_expression(ast, *element, out);
            }
        }
        Expression::Unsafe(body) => names_in_block(ast, *body, out),
        Expression::UnsafeFn(inner) => names_in_expression(ast, *inner, out),
        Expression::ArrayRepeat(value, _) => {
            names_in_expression(ast, *value, out)
        }
        // A run written out holds expressions, so a name written inside one is
        // a name this declaration reaches.
        Expression::Literal(crate::ast::Literal::Array(elements)) => {
            for element in ast.exprs_in(*elements) {
                names_in_expression(ast, *element, out);
            }
        }
        Expression::Literal(_) | Expression::Boolean(_) => {}
    }
}

fn names_in_type(ty: &Type, out: &mut Vec<String>) {
    match ty {
        Type::Struct(name) | Type::Enum(name) => {
            // A generic instance is written `Pair<i64>`, and the base name is
            // what a caller has to be able to find.
            match name.split_once('<') {
                Some((base, _)) => out.push(base.to_string()),
                None => out.push(name.clone()),
            }
        }
        Type::ConstFn(name) => out.push(name.clone()),
        Type::Ptr(inner)
        | Type::Ref(inner)
        | Type::RefMut(inner)
        | Type::Slice(inner)
        | Type::Array(inner, _)
        | Type::ArrayGeneric(inner, _)
        | Type::Distinct(_, inner)
        | Type::Handle(inner) => names_in_type(inner, out),
        Type::Proc(params, ret) => {
            for parameter in params {
                names_in_type(parameter, out);
            }
            names_in_type(ret, out);
        }
        _ => {}
    }
}
