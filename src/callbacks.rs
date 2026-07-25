use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::parser::{ParamMode, Parameter, Spanned, Statement};
use crate::types::Type;

// docs/callbacks.md. An `extern fn` with a `$handler` parameter bound to a
// function signature is a callback registration. The compiler will emit a
// trampoline with the C ABI the library expects, and that trampoline is the
// only code that casts the untyped userdata back to the context type.
//
// This is the declaration check, which is what makes the rest derivable. The
// handler's one `mut` parameter is the context, wherever it is written; every
// other parameter is an argument C passes through. The extern parameter of that
// same type is the one the userdata is taken from, found by type rather than by
// position because libraries put the userdata on either side of the function
// pointer.
//
// The context used to have to come first, which is the order a C library takes
// when it puts the userdata first and the wrong order for one that puts it
// last. wgpu-native does, so its callbacks could not be declared at all while
// the same function pointer in a struct field went unchecked. The position is
// not what makes the context identifiable; being the one parameter the handler
// can write is.
pub fn check_callback_declarations(
    program: &[Spanned<Statement>],
) -> Result<()> {
    for statement in program {
        let Statement::Extern { name, params, .. } = &statement.node else {
            continue;
        };
        for parameter in params {
            let Some(bound) = &parameter.compile_time_signature else {
                continue;
            };
            check_registration(name, params, parameter, bound)?;
        }
    }
    Ok(())
}

// Which argument of a registration is the callback and which is the context,
// and what the context's type is. Everything downstream is derived from this:
// the region check needs the context's position, and lowering needs all three.
#[derive(Debug, Clone, PartialEq)]
pub struct CallbackShape {
    pub handler: usize,
    pub context: usize,
    pub context_type: Type,
}

// The shape of an `extern fn`'s parameter list read as a registration, or
// `None` when it is an ordinary extern. See docs/callbacks.md.
pub fn callback_shape(params: &[Parameter]) -> Option<CallbackShape> {
    for (handler, parameter) in params.iter().enumerate() {
        let Some(Type::Proc(handler_params, _)) =
            &parameter.compile_time_signature
        else {
            continue;
        };
        let Some(context_type) = sole_context(handler_params) else {
            continue;
        };
        let context = params.iter().position(|parameter| {
            parameter.type_annotation.as_ref() == Some(context_type)
        })?;
        return Some(CallbackShape {
            handler,
            context,
            context_type: context_type.clone(),
        });
    }
    None
}

// The handler's context: its one `mut` parameter. None when it has no such
// parameter, and none when it has more than one, since then nothing says which
// of them the library is being asked to keep.
fn sole_context(handler_params: &[Type]) -> Option<&Type> {
    let mut found = None;
    for parameter in handler_params {
        if let Type::RefMut(context) = parameter {
            if found.is_some() {
                return None;
            }
            found = Some(context.as_ref());
        }
    }
    found
}

// Every callback registration in a program, by name.
pub fn callback_registrations(
    program: &[Spanned<Statement>],
) -> HashMap<String, CallbackShape> {
    let mut registrations = HashMap::new();
    for statement in program {
        let Statement::Extern { name, params, .. } = &statement.node else {
            continue;
        };
        if let Some(shape) = callback_shape(params) {
            registrations.insert(name.clone(), shape);
        }
    }
    registrations
}

fn check_registration(
    name: &str,
    params: &[Parameter],
    handler: &Parameter,
    bound: &Type,
) -> Result<()> {
    let Type::Proc(handler_params, _) = bound else {
        bail!(
            "the compile-time parameter '${}' of the extern '{name}' is bound to '{bound}', which is not a function signature, so there is no callback to build",
            handler.name
        );
    };
    let writable = handler_params
        .iter()
        .filter(|parameter| matches!(parameter, Type::RefMut(_)))
        .count();
    // A callback that cannot write its context cannot do anything, and the
    // read-only case is what a plain function pointer already covers.
    if writable == 0 {
        bail!(
            "the callback '${}' of the extern '{name}' has no 'mut' parameter, so it has no context, and a callback that cannot write its context has nothing to do that a plain function pointer does not",
            handler.name
        );
    }
    if writable > 1 {
        bail!(
            "the callback '${}' of the extern '{name}' has {writable} 'mut' parameters, so nothing says which one is the context the library is being asked to keep; a callback has one context",
            handler.name
        );
    }
    let Some(context) = sole_context(handler_params) else {
        unreachable!("exactly one mut parameter was just counted")
    };

    let carrier = params
        .iter()
        .find(|parameter| parameter.type_annotation.as_ref() == Some(context));
    let Some(carrier) = carrier else {
        bail!(
            "the callback '${}' of the extern '{name}' takes a context of type '{context}', but '{name}' has no parameter of that type to take it from",
            handler.name
        );
    };
    // The registration keeps the context past the call, so the caller must not
    // still be able to reach it. See the ownership argument in
    // docs/callbacks.md.
    if carrier.mode != ParamMode::Move {
        bail!(
            "'{}' is the context of the callback '${}' of the extern '{name}', so it has to be taken by 'move': the callback can fire at any time while it is registered, and the caller must not still hold it",
            carrier.name,
            handler.name
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn check(source: &str) -> Result<()> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let statements = parser.parse().unwrap();
        check_callback_declarations(&statements)
    }

    const CONTEXT: &str = "Ctx :: struct { hits: i64 }\n";

    #[test]
    fn a_registration_declares_its_context_by_move() {
        check(&format!(
            "{CONTEXT}register :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> i64\n"
        ))
        .unwrap();
    }

    #[test]
    fn the_context_may_come_before_the_handler() {
        check(&format!(
            "{CONTEXT}register :: extern fn(move ctx: Ctx, $handler: fn(mut Ctx, i64)) -> i64\n"
        ))
        .unwrap();
    }

    // wgpu-native's shape: the arguments C passes through come first and the
    // userdata comes last. This used to be undeclarable, so the same function
    // pointer had to go through a struct field, where nothing checks it.
    #[test]
    fn the_context_may_come_last_in_the_handler() {
        check(&format!(
            "{CONTEXT}request :: extern fn($handler: fn(i32, i64, mut Ctx), move ctx: Ctx) -> i64\n"
        ))
        .unwrap();
    }

    #[test]
    fn a_context_written_last_is_still_found() {
        let source = format!(
            "{CONTEXT}request :: extern fn($handler: fn(i32, mut Ctx), move ctx: Ctx) -> i64\n"
        );
        let mut lexer = Lexer::new(&source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let statements = parser.parse().unwrap();
        let shape = callback_registrations(&statements);
        let shape = shape.get("request").unwrap();
        assert_eq!(shape.handler, 0);
        assert_eq!(shape.context, 1);
        assert_eq!(shape.context_type, Type::Struct("Ctx".to_string()));
    }

    // Two writable parameters and nothing says which the library keeps, so the
    // position rule is replaced by a uniqueness rule rather than dropped.
    #[test]
    fn a_callback_with_two_contexts_is_rejected() {
        let message = check(
            "Ctx :: struct { hits: i64 }\n\
             register :: extern fn($handler: fn(mut Ctx, mut Ctx), move ctx: Ctx) -> i64\n",
        )
        .unwrap_err()
        .to_string();
        assert!(message.contains("which one is the context"), "{message}");
    }

    #[test]
    fn a_borrowed_context_is_rejected() {
        let message = check(&format!(
            "{CONTEXT}register :: extern fn($handler: fn(mut Ctx, i64), ctx: Ctx) -> i64\n"
        ))
        .unwrap_err()
        .to_string();
        assert!(message.contains("'move'"), "{message}");
    }

    #[test]
    fn a_context_the_callback_cannot_write_is_rejected() {
        let message = check(&format!(
            "{CONTEXT}register :: extern fn($handler: fn(Ctx, i64), move ctx: Ctx) -> i64\n"
        ))
        .unwrap_err()
        .to_string();
        assert!(message.contains("'mut'"), "{message}");
    }

    #[test]
    fn a_registration_with_nowhere_to_put_the_context_is_rejected() {
        let message = check(&format!(
            "{CONTEXT}register :: extern fn($handler: fn(mut Ctx, i64), code: i64) -> i64\n"
        ))
        .unwrap_err()
        .to_string();
        assert!(message.contains("no parameter of that type"), "{message}");
    }

    #[test]
    fn a_callback_with_no_context_is_rejected() {
        let message =
            check("register :: extern fn($handler: fn() -> i64) -> i64\n")
                .unwrap_err()
                .to_string();
        assert!(message.contains("no context"), "{message}");
    }

    #[test]
    fn registrations_are_found_with_the_context_position() {
        let source = format!(
            "{CONTEXT}register :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> i64\n"
        );
        let mut lexer = Lexer::new(&source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let statements = parser.parse().unwrap();
        let found = callback_registrations(&statements);
        let shape = found.get("register").unwrap();
        assert_eq!(shape.handler, 0);
        assert_eq!(shape.context, 1);
        assert_eq!(shape.context_type, Type::Struct("Ctx".to_string()));
    }

    // An ordinary generic is not a registration and must not be dragged into
    // this check, which is the mistake that would make every existing program
    // with a compile-time function argument stop compiling.
    #[test]
    fn an_ordinary_compile_time_function_argument_is_untouched() {
        check(
            "ascending :: fn(a: i64, b: i64) -> bool { a < b }\n\
             best :: fn($T: Type, $before: fn(T, T) -> bool, move x: $T, move y: $T) -> $T {\n\
             \x20   mut result := x\n    if (before(y, result)) { result = y }\n    result\n}\n",
        )
        .unwrap();
    }
}
