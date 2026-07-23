use anyhow::{Result, bail};

use crate::parser::{ParamMode, Parameter, Program, Statement};
use crate::types::Type;

// docs/callbacks.md. An `extern fn` with a `$handler` parameter bound to a
// function signature is a callback registration: the compiler will emit a
// trampoline with the C ABI the library expects, and that trampoline is the
// only code that casts the untyped userdata back to the context type.
//
// This is the declaration check, which is what makes the rest derivable. The
// handler's first parameter is the context, so the trampoline knows what to
// cast to; every parameter after it is an argument C passes through. The extern
// parameter of that same type is the one the userdata is taken from, found by
// type rather than by position because libraries put the userdata on either
// side of the function pointer.
pub fn check_callback_declarations(program: &Program) -> Result<()> {
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
    let Some(first) = handler_params.first() else {
        bail!(
            "the callback '${}' of the extern '{name}' takes no parameters, so it has no context, and a callback with no context is an ordinary function pointer",
            handler.name
        );
    };
    // A callback that cannot write its context cannot do anything, and the
    // read-only case is what a plain function pointer already covers.
    let Type::RefMut(context) = first else {
        bail!(
            "the first parameter of the callback '${}' of the extern '{name}' is the context and has to be written 'mut', since a callback that cannot write its context has nothing to do",
            handler.name
        );
    };

    let carrier = params.iter().find(|parameter| {
        parameter.type_annotation.as_ref() == Some(context.as_ref())
    });
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
