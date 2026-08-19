// Every type name a signature writes has to be a type something declares.
//
// A name nothing declares is the shape a wrong guess takes: `fn(v: Absent)`
// names a type that is not there, and read as an opaque name it goes through
// every pass without a word until something asks what is in it, which for an
// unused parameter is never.

use std::collections::HashSet;

use crate::ast::{Ast, Statement, StmtId};
use crate::diagnostic::Diagnostic;
use crate::types::Type;

/// The phrase a report about a name nothing declares is written with. Named
/// here so the driver, which holds back later checks on it, reads the words the
/// check writes rather than a copy that a rewording leaves behind.
pub const UNDECLARED_TYPE: &str = "is not a type this program declares";

/// Every signature and every field, checked against the types the program
/// declares.
pub fn check_declared_types(ast: &Ast, roots: &[StmtId]) -> Vec<Diagnostic> {
    let known = declared(ast, roots);
    let values = valued(ast, roots);
    let mut found = Vec::new();
    for statement in roots {
        let position = ast.stmt_position(*statement);
        match ast.stmt(*statement) {
            Statement::Constant(_, value) => {
                let (Expression::Function(params, signature, body)
                | Expression::Proc(params, signature, body)) = ast.expr(*value)
                else {
                    continue;
                };
                // A binding writes a type the same way a parameter does, so a
                // name nothing declares is the same fault wherever it is
                // written. Left out, `held: Widget = 3` read as a binding of a
                // type the program had, and what the reader was told was that
                // an `i64` is not a `Widget`.
                report_unknown_in_block(
                    ast, *body, &known, &values, &mut found,
                );
                for parameter in ast.params_in(*params) {
                    if let Some(ty) = &parameter.type_annotation {
                        // At the type, which is what the report names. The
                        // declaration's own place put the caret on the word the
                        // reader wrote first.
                        report_unknown(ty, &known, parameter.at, &mut found);
                    }
                }
                let signature = ast.signature(*signature);
                if let crate::ast::ReturnKind::Single(ty)
                | crate::ast::ReturnKind::Fallible(ty, _) = &signature.kind
                {
                    // At the type, which is what the report names, the way a
                    // parameter's already is.
                    report_unknown(ty, &known, signature.at, &mut found);
                }
            }
            Statement::Struct(_, _, fields) => {
                // A `for name in fields(T)` in the body binds its name as a
                // type for the field it writes, so the name is known inside
                // this declaration and nowhere else.
                let mut known = known.clone();
                for field in ast.fields_in(*fields) {
                    if field.walk_over.is_some() {
                        known.insert(ast.name(field.name));
                    }
                }
                for field in ast.fields_in(*fields) {
                    // At the type the field names. The declaration's own place
                    // put the caret on the struct, which is not the word the
                    // reader has to change.
                    report_unknown(
                        &field.field_type,
                        &known,
                        field.at,
                        &mut found,
                    );
                }
            }
            // A variant's payload writes a type the same way a struct's field
            // does. Left out, an enum was the one declaration a name nothing
            // declares could hide in.
            Statement::Enum(_, _, variants) => {
                for variant in ast.variants_in(*variants) {
                    let Some(fields) = variant.fields else {
                        continue;
                    };
                    for field in ast.fields_in(fields) {
                        report_unknown(
                            &field.field_type,
                            &known,
                            field.at,
                            &mut found,
                        );
                    }
                }
            }
            Statement::Extern {
                params,
                return_type,
                ..
            } => {
                for parameter in ast.params_in(*params) {
                    if let Some(ty) = &parameter.type_annotation {
                        report_unknown(ty, &known, parameter.at, &mut found);
                    }
                }
                if let Some(ty) = return_type {
                    report_unknown(ty, &known, position, &mut found);
                }
            }
            _ => {}
        }
    }
    found
}

use crate::ast::Expression;

/// The names the program declares as types, and the names a generic binds.
fn declared<'a>(ast: &'a Ast, roots: &[StmtId]) -> HashSet<&'a str> {
    let mut held = HashSet::new();
    for statement in roots {
        match ast.stmt(*statement) {
            Statement::Struct(name, generics, _)
            | Statement::Enum(name, generics, _) => {
                held.insert(ast.name(*name));
                for parameter in ast.symbols_in(*generics) {
                    held.insert(ast.name(*parameter));
                }
            }
            Statement::Flags(name, ..) | Statement::TypeAlias(name, _) => {
                held.insert(ast.name(*name));
            }
            _ => {}
        }
    }
    // A generic's type parameters name types inside its own body, bound where
    // the generic is expanded. Read from every parameter the program holds,
    // since a generic struct binds them the same way a generic function does.
    for parameter in &ast.parameters {
        if let Some(ty) = &parameter.type_annotation {
            bound(ty, &mut held);
        }
        if parameter.compile_time_signature.is_some() {
            held.insert(ast.name(parameter.name));
        }
    }
    held
}

/// The names the program binds to values rather than to types. A compile-time
/// argument may name one - `write($to_stdout, ..)` hands a function over the
/// same way `sizeof(Widget)` hands a type - and the two are one shape in the
/// tree, so what is asked about a name written as an argument is whether the
/// program binds it at all.
fn valued<'a>(ast: &'a Ast, roots: &[StmtId]) -> HashSet<&'a str> {
    let mut held = HashSet::new();
    for statement in roots {
        // A module whose object is being linked rather than rebuilt
        // contributes an ordinary function as its signature alone, and that
        // still binds the name to a value: a caller may hand it over as a
        // compile-time argument the way it hands over one it can see the body
        // of.
        match ast.stmt(*statement) {
            Statement::Constant(name, _)
            | Statement::Declared { name, .. } => {
                held.insert(ast.name(*name));
            }
            _ => {}
        }
    }
    held
}

/// Every name a `$T` annotation binds, wherever it sits inside the type.
fn bound<'a>(ty: &'a Type, held: &mut HashSet<&'a str>) {
    match ty {
        Type::TypeParam(name) => {
            held.insert(name.as_str());
        }
        Type::Ptr(inner)
        | Type::Ref(inner)
        | Type::RefMut(inner)
        | Type::Slice(inner)
        | Type::Array(inner, _)
        | Type::ArrayGeneric(inner, _)
        | Type::Handle(inner)
        | Type::Distinct(_, inner) => bound(inner, held),
        Type::Proc(params, answer) => {
            for parameter in params {
                bound(parameter, held);
            }
            bound(answer, held);
        }
        _ => {}
    }
}

/// Every binding written inside a body, and inside the bodies inside it.
///
/// A block is reached through whatever holds it, so each statement that holds
/// one names it here. A `defer` holds a single statement rather than a block,
/// which is why it is followed the same way and not with the others.
fn report_unknown_in_block(
    ast: &Ast,
    block: crate::ast::Range32,
    known: &HashSet<&str>,
    values: &HashSet<&str>,
    found: &mut Vec<Diagnostic>,
) {
    // What this block declares. A body may bind a name to a value the way the
    // top level does, and `apply($twice, 3)` names one of those, so the set of
    // names bound to values grows as the walk goes inward.
    let mut inner = values.clone();
    for statement in ast.stmts_in(block) {
        if let Statement::Constant(name, _) = ast.stmt(*statement) {
            inner.insert(ast.name(*name));
        }
    }
    for statement in ast.stmts_in(block) {
        report_unknown_in_statement(ast, *statement, known, &inner, found);
    }
}

/// One statement: the type it writes, and whatever it holds.
fn report_unknown_in_statement(
    ast: &Ast,
    statement: StmtId,
    known: &HashSet<&str>,
    values: &HashSet<&str>,
    found: &mut Vec<Diagnostic>,
) {
    match ast.stmt(statement) {
        Statement::Let {
            type_annotation,
            type_at,
            value,
            ..
        } => {
            if let Some(ty) = type_annotation {
                // At the type, which is what the report names, the way a
                // parameter's already is.
                report_unknown(ty, known, *type_at, found);
            }
            report_unknown_in_expression(ast, *value, known, values, found);
        }
        Statement::While(condition, body) => {
            report_unknown_in_expression(ast, *condition, known, values, found);
            report_unknown_in_block(ast, *body, known, values, found);
        }
        // `for field in fields(T)` binds a name that stands for a type inside
        // the loop, so `sizeof(field)` names it and nothing declares it. The
        // binding is not a type the program wrote down, and asking about it is
        // the wrong question.
        Statement::For(binding, second, over, body) => {
            report_unknown_in_expression(ast, *over, known, values, found);
            let mut inner = values.clone();
            inner.insert(ast.name(*binding));
            if let Some(held) = second {
                inner.insert(ast.name(*held));
            }
            report_unknown_in_block(ast, *body, known, &inner, found);
        }
        Statement::With(_, body) => {
            report_unknown_in_block(ast, *body, known, values, found);
        }
        // A `defer` holds one statement rather than a block.
        Statement::Defer(held) | Statement::ErrDefer(held) => {
            report_unknown_in_statement(ast, *held, known, values, found);
        }
        Statement::Expression(value) | Statement::Return(value) => {
            report_unknown_in_expression(ast, *value, known, values, found);
        }
        Statement::Assignment(place, value) => {
            report_unknown_in_expression(ast, *place, known, values, found);
            report_unknown_in_expression(ast, *value, known, values, found);
        }
        _ => {}
    }
}

/// The blocks an expression holds. Only the shapes that hold one are followed:
/// the rest have no statements under them and nothing to declare.
fn report_unknown_in_expression(
    ast: &Ast,
    value: crate::ast::ExprId,
    known: &HashSet<&str>,
    values: &HashSet<&str>,
    found: &mut Vec<Diagnostic>,
) {
    match ast.expr(value) {
        Expression::If(condition, then, otherwise) => {
            report_unknown_in_expression(ast, *condition, known, values, found);
            report_unknown_in_block(ast, *then, known, values, found);
            if let Some(held) = otherwise {
                report_unknown_in_block(ast, *held, known, values, found);
            }
        }
        Expression::Unsafe(held) => {
            report_unknown_in_block(ast, *held, known, values, found);
        }
        Expression::Switch(subject, cases) => {
            report_unknown_in_expression(ast, *subject, known, values, found);
            for case in ast.cases_in(*cases) {
                report_unknown_in_block(ast, case.body, known, values, found);
            }
        }
        // A type written as an argument. `sizeof(Widget)` says the name where
        // the reader wrote it, rather than leaving `sizeof` to say it has no
        // layout for a type that is not there to have one.
        Expression::Call(_, arguments) => {
            for argument in ast.exprs_in(*arguments) {
                report_unknown_in_expression(
                    ast, *argument, known, values, found,
                );
            }
        }
        Expression::TypeValue(ty) => {
            if let Type::Struct(name) | Type::Enum(name) = ty
                && values.contains(name.as_str())
            {
                return;
            }
            report_unknown(ty, known, ast.expr_position(value), found);
        }
        // The type a literal is written under. `Alias { v = 1 }` names one the
        // same way a declared type does, and read only where a type is written
        // out it went unasked: what the reader was told came from lowering,
        // which asks for a layout rather than for the name.
        Expression::StructInit(name, fields) => {
            let named = Type::Struct(ast.name(*name).to_string());
            report_unknown(&named, known, ast.expr_position(value), found);
            for field in ast.named_in(*fields).to_vec() {
                report_unknown_in_expression(
                    ast, field.value, known, values, found,
                );
            }
        }
        _ => {}
    }
}

/// A named type nothing declares, reported once per place it is written.
///
/// Only a plain name is asked about. A generic instance carries its arguments
/// in the name it is spelled with, and a type parameter is bound where the
/// generic is expanded, so neither is a name this can answer for.
fn report_unknown(
    ty: &Type,
    known: &HashSet<&str>,
    position: crate::lexer::Position,
    found: &mut Vec<Diagnostic>,
) {
    match ty {
        // A `$T` naming nothing. Read as a type parameter and left unasked,
        // `sizeof($Nope)` measured a name the program does not have and
        // answered zero without a word; the same call written without the
        // sigil was refused.
        Type::Struct(name) | Type::Enum(name) | Type::TypeParam(name) => {
            if name.contains('<') || name.is_empty() {
                return;
            }
            if known.contains(name.as_str()) {
                return;
            }
            found.push(Diagnostic::new(
                position,
                format!("'{name}' {UNDECLARED_TYPE}"),
            ));
        }
        Type::Ptr(inner)
        | Type::Ref(inner)
        | Type::RefMut(inner)
        | Type::Slice(inner)
        | Type::Array(inner, _)
        | Type::ArrayGeneric(inner, _)
        | Type::Handle(inner)
        | Type::Distinct(_, inner) => {
            report_unknown(inner, known, position, found)
        }
        Type::Proc(params, answer) => {
            for parameter in params {
                report_unknown(parameter, known, position, found);
            }
            report_unknown(answer, known, position, found);
        }
        _ => {}
    }
}
