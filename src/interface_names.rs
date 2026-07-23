use crate::parser::{Block, Expression, Statement};
use crate::types::Type;

// Every named type or function a declaration refers to, so an interface can
// carry the things an exported name depends on. Deliberately over-approximate:
// naming something that turns out not to be a type costs an entry that nothing
// matches, while missing one costs a caller that cannot compile.
pub(crate) fn names_in_statement(statement: &Statement, out: &mut Vec<String>) {
    match statement {
        Statement::Constant(_, value) => names_in_expression(value, out),
        Statement::Declared {
            params, return_sig, ..
        } => {
            for param in params {
                if let Some(ty) = &param.type_annotation {
                    names_in_type(ty, out);
                }
                if let Some(ty) = &param.compile_time_signature {
                    names_in_type(ty, out);
                }
            }
            match &return_sig.kind {
                crate::parser::ReturnKind::None => {}
                crate::parser::ReturnKind::Single(ty) => names_in_type(ty, out),
                crate::parser::ReturnKind::Named(params) => {
                    for param in params {
                        names_in_type(&param.param_type, out);
                    }
                }
                crate::parser::ReturnKind::Fallible(value, failure) => {
                    names_in_type(value, out);
                    names_in_type(failure, out);
                }
            }
            for capability in &return_sig.uses {
                names_in_type(capability, out);
            }
        }
        Statement::Struct(_, _, fields) => {
            for field in fields {
                names_in_type(&field.field_type, out);
            }
        }
        Statement::Enum(_, _, variants) => {
            for variant in variants {
                if let Some(fields) = &variant.fields {
                    for field in fields {
                        names_in_type(&field.field_type, out);
                    }
                }
            }
        }
        Statement::TypeAlias(_, ty) => names_in_type(ty, out),
        Statement::Extern {
            params,
            return_type,
            ..
        } => {
            for parameter in params {
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
            names_in_expression(value, out);
        }
        Statement::Return(value) | Statement::Expression(value) => {
            names_in_expression(value, out)
        }
        Statement::Assignment(place, value) => {
            names_in_expression(place, out);
            names_in_expression(value, out);
        }
        Statement::Defer(inner) => names_in_statement(inner, out),
        Statement::For(_, iterable, body) => {
            names_in_expression(iterable, out);
            names_in_block(body, out);
        }
        Statement::While(condition, body) => {
            names_in_expression(condition, out);
            names_in_block(body, out);
        }
        Statement::With(capability, body) => {
            out.push(capability.clone());
            names_in_block(body, out);
        }
        Statement::Break | Statement::Continue | Statement::Import(_) => {}
    }
}

fn names_in_block(block: &Block, out: &mut Vec<String>) {
    for statement in block {
        names_in_statement(&statement.node, out);
    }
}

fn names_in_expression(expression: &Expression, out: &mut Vec<String>) {
    match expression {
        // A call names its callee, which for a generic body is how a template
        // reaches a helper the module did not export.
        Expression::Identifier(name) => out.push(name.clone()),
        Expression::StructInit(name, fields) => {
            out.push(name.clone());
            for (_, value) in fields {
                names_in_expression(value, out);
            }
        }
        Expression::EnumVariantInit(name, _, fields) => {
            out.push(name.clone());
            for (_, value) in fields {
                names_in_expression(value, out);
            }
        }
        Expression::Sizeof(ty) | Expression::TypeValue(ty) => {
            names_in_type(ty, out)
        }
        Expression::Function(params, return_sig, body)
        | Expression::Proc(params, return_sig, body) => {
            for parameter in params {
                if let Some(ty) = &parameter.type_annotation {
                    names_in_type(ty, out);
                }
                if let Some(ty) = &parameter.compile_time_signature {
                    names_in_type(ty, out);
                }
            }
            if let Some(ty) = return_sig.to_type() {
                names_in_type(&ty, out);
            }
            for ty in &return_sig.uses {
                names_in_type(ty, out);
            }
            names_in_block(body, out);
        }
        Expression::Call(callee, arguments) => {
            names_in_expression(callee, out);
            for argument in arguments {
                names_in_expression(argument, out);
            }
        }
        Expression::Prefix(_, inner)
        | Expression::AddressOf(inner)
        | Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::Try(inner)
        | Expression::Dereference(inner)
        | Expression::FieldAccess(inner, _) => names_in_expression(inner, out),
        Expression::Infix(left, _, right)
        | Expression::Index(left, right)
        | Expression::Range(left, right, _) => {
            names_in_expression(left, out);
            names_in_expression(right, out);
        }
        Expression::If(condition, consequence, alternative) => {
            names_in_expression(condition, out);
            names_in_block(consequence, out);
            if let Some(block) = alternative {
                names_in_block(block, out);
            }
        }
        Expression::Switch(scrutinee, cases) => {
            names_in_expression(scrutinee, out);
            for case in cases {
                if let crate::parser::Pattern::EnumVariant {
                    enum_name: Some(name),
                    ..
                } = &case.pattern
                {
                    out.push(name.clone());
                }
                names_in_block(&case.body, out);
            }
        }
        Expression::Tuple(elements) => {
            for element in elements {
                names_in_expression(element, out);
            }
        }
        Expression::Unsafe(body) => names_in_block(body, out),
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
        | Type::Distinct(inner)
        | Type::Handle(inner)
        | Type::Optional(inner) => names_in_type(inner, out),
        Type::Proc(params, ret) => {
            for parameter in params {
                names_in_type(parameter, out);
            }
            names_in_type(ret, out);
        }
        _ => {}
    }
}
