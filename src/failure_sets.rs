use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::lexer::Position;
use crate::parser::{
    Block, EnumVariant, Expression, Pattern, Program, ReturnKind, Spanned,
    Statement, StructField, SwitchCase,
};
use crate::types::Type;

// Desugars failure sets and the `?` operator into the enum and match machinery
// the compiler already has. A `-> T ! E` function returns a synthesized
// `Result { Ok { value: T }, Err { error: E } }` enum. A `return` wraps its value
// as Ok (or as Err when it constructs an E variant), and `expr?` becomes a match
// that yields the Ok value or returns the enclosing function's Err. After this
// pass nothing downstream knows failure sets exist.
struct Lowerer {
    // The linear types of the program, and the Result enums that turned out to
    // hold one. A linear value handed back through a failure set is still a
    // linear value: the obligation to consume it belongs to whatever the caller
    // does with the result, so the result carries it.
    linear: HashSet<String>,
    // Result enum name for each (value, error) pair, deduplicated.
    results: HashMap<String, String>,
    enums: Vec<Spanned<Statement>>,
    // Fallible function name to its Result enum name.
    fallible: HashMap<String, String>,
    // Every enum's variant names, so a `return .Denied` can be told from a
    // returned value: the leading dot names no enum, and the failure set is
    // what it belongs to when the error type has that variant.
    variants: HashMap<String, Vec<String>>,
    // Whether the function being rewritten answers with a struct or an enum.
    // An untyped `{ ... }` in its `return` is that value when it does and the
    // failure otherwise, since only one of the two can be written without a
    // name.
    value_is_aggregate: bool,
    counter: usize,
}

fn spanned(statement: Statement) -> Spanned<Statement> {
    Spanned {
        node: statement,
        position: Position::default(),
    }
}

pub fn lower_failure_sets(
    program: &mut Program,
    linear: &mut HashSet<String>,
) -> Result<()> {
    let mut lowerer = Lowerer {
        linear: linear.clone(),
        results: HashMap::new(),
        enums: Vec::new(),
        fallible: HashMap::new(),
        variants: HashMap::new(),
        value_is_aggregate: false,
        counter: 0,
    };

    for statement in program.iter() {
        if let Statement::Enum(name, _, variants) = &statement.node {
            lowerer.variants.insert(
                name.clone(),
                variants
                    .iter()
                    .map(|variant| variant.name.clone())
                    .collect(),
            );
        }
    }

    // First pass. Give every fallible function a Result enum.
    for statement in program.iter() {
        if let Statement::Constant(
            name,
            Expression::Function(_, sig, _) | Expression::Proc(_, sig, _),
        ) = &statement.node
            && let ReturnKind::Fallible(value, error) = &sig.kind
        {
            let result = lowerer.result_enum(value, error);
            lowerer.fallible.insert(name.clone(), result);
        }
    }

    // Second pass. Rewrite bodies and return signatures. A `?` in a function
    // that declares no failure set has nowhere to propagate to, so reject it.
    for statement in program.iter_mut() {
        if let Statement::Constant(
            name,
            Expression::Function(_, sig, body) | Expression::Proc(_, sig, body),
        ) = &mut statement.node
        {
            if let ReturnKind::Fallible(value, error) = sig.kind.clone() {
                let result = lowerer.fallible.get(name).unwrap().clone();
                sig.kind = ReturnKind::Single(Type::Enum(result.clone()));
                lowerer.value_is_aggregate =
                    matches!(value, Type::Struct(_) | Type::Enum(_));
                lowerer.rewrite_block(body, &result, &error);
            } else if block_has_try(body) {
                bail!(
                    "the `?` operator is only allowed in a function with a failure set; '{name}' must declare `-> T ! E`"
                );
            }
        }
    }

    // Prepend the synthesized enums so they are declared before use.
    let mut synthesized = std::mem::take(&mut lowerer.enums);
    synthesized.append(program);
    *program = synthesized;
    // The results that hold a linear value, so the check that follows knows
    // them for what they are.
    *linear = lowerer.linear;
    Ok(())
}

fn block_has_try(block: &Block) -> bool {
    block
        .iter()
        .any(|statement| statement_has_try(&statement.node))
}

fn statement_has_try(statement: &Statement) -> bool {
    match statement {
        Statement::Return(value)
        | Statement::Let { value, .. }
        | Statement::Constant(_, value)
        | Statement::Expression(value) => expression_has_try(value),
        Statement::Assignment(place, value) => {
            expression_has_try(place) || expression_has_try(value)
        }
        Statement::Defer(inner) => statement_has_try(inner),
        Statement::While(condition, body) => {
            expression_has_try(condition) || block_has_try(body)
        }
        Statement::For(_, _, iterable, body) => {
            expression_has_try(iterable) || block_has_try(body)
        }
        Statement::With(_, body) => block_has_try(body),
        _ => false,
    }
}

fn expression_has_try(expression: &Expression) -> bool {
    match expression {
        Expression::Try(_) => true,
        Expression::PackMap(inner, _, _)
        | Expression::Prefix(_, inner)
        | Expression::AddressOf(inner)
        | Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::Dereference(inner)
        | Expression::ArrayRepeat(inner, _)
        | Expression::FieldAccess(inner, _) => expression_has_try(inner),
        Expression::Infix(left, _, right) | Expression::Index(left, right) => {
            expression_has_try(left) || expression_has_try(right)
        }
        Expression::Call(callee, arguments) => {
            expression_has_try(callee)
                || arguments.iter().any(expression_has_try)
        }
        Expression::If(condition, then_block, else_block) => {
            expression_has_try(condition)
                || block_has_try(then_block)
                || else_block.as_ref().is_some_and(block_has_try)
        }
        Expression::StructInit(_, fields)
        | Expression::EnumVariantInit(_, _, fields) => {
            fields.iter().any(|(_, value)| expression_has_try(value))
        }
        Expression::Switch(scrutinee, cases) => {
            expression_has_try(scrutinee)
                || cases.iter().any(|case| block_has_try(&case.body))
        }
        Expression::Unsafe(body) => block_has_try(body),
        Expression::Tuple(items) => items.iter().any(expression_has_try),
        // Listed rather than caught by `_`, so a new expression form is a
        // compile error here instead of silently reporting no `?` inside it.
        Expression::Identifier(_)
        | Expression::Literal(_)
        | Expression::Boolean(_)
        | Expression::Sizeof(_)
        | Expression::TypeId(_)
        | Expression::TypeName(_)
        | Expression::TypeValue(_)
        | Expression::Range(..)
        | Expression::Function(..)
        | Expression::Proc(..)
        | Expression::UnsafeFn(_) => false,
    }
}

impl Lowerer {
    fn result_enum(&mut self, value: &Type, error: &Type) -> String {
        let key = format!("{value}!{error}");
        if let Some(name) = self.results.get(&key) {
            return name.clone();
        }
        let name = format!("__Result_{}", self.counter);
        self.counter += 1;
        let variants = vec![
            EnumVariant {
                name: "Ok".to_string(),
                fields: Some(vec![StructField {
                    name: "value".to_string(),
                    field_type: value.clone(),
                }]),
            },
            EnumVariant {
                name: "Err".to_string(),
                fields: Some(vec![StructField {
                    name: "error".to_string(),
                    field_type: error.clone(),
                }]),
            },
        ];
        self.enums.push(spanned(Statement::Enum(
            name.clone(),
            Vec::new(),
            variants,
        )));
        // A result holding a linear value, or a linear failure, is linear.
        // Without this a fallible call whose value must be consumed could be
        // ignored, and the resource it answered with would be leaked.
        if self.names_linear(value) || self.names_linear(error) {
            self.linear.insert(name.clone());
        }
        self.results.insert(key, name.clone());
        name
    }

    fn names_linear(&self, ty: &Type) -> bool {
        match ty {
            Type::Struct(name) | Type::Enum(name) => self.linear.contains(name),
            _ => false,
        }
    }

    // Does this expression build the failure type? A failure set may be an enum
    // or a struct, and the two are written differently: `Denied {}` names a
    // variant and `Blocked { at = 3 }` is a struct literal. Only the first used
    // to count, so a struct failure was wrapped as the Ok value instead and
    // reached the backend as a struct where the value type belonged.
    fn is_error_construction(
        &self,
        expression: &Expression,
        error: &Type,
    ) -> bool {
        let error_name = match error {
            Type::Enum(name) | Type::Struct(name) => name,
            _ => return false,
        };
        match expression {
            // `return .Denied`: the dot names no enum, so it is the failure
            // when the error type is an enum with that variant. A value type
            // that happens to share the name is not a case that can arise,
            // since the two are different enums and the reader wrote the one
            // the error declares.
            Expression::EnumVariantInit(name, variant, _)
                if name.is_empty() =>
            {
                self.variants
                    .get(error_name)
                    .is_some_and(|names| names.contains(variant))
            }
            // `return { at = 3 }`: the literal names no type either. It is the
            // failure when the value the function answers with is not itself a
            // struct or an enum, since then only the failure can be written
            // this way. A function that answers with one names it.
            Expression::StructInit(name, _) if name.is_empty() => {
                !self.value_is_aggregate
                    && matches!(error, Type::Struct(_) | Type::Enum(_))
            }
            Expression::EnumVariantInit(name, _, _)
            | Expression::StructInit(name, _) => name == error_name,
            _ => false,
        }
    }

    fn wrap_return(
        &self,
        value: Expression,
        result: &str,
        error: &Type,
    ) -> Expression {
        let (variant, field) = if self.is_error_construction(&value, error) {
            ("Err", "error")
        } else {
            ("Ok", "value")
        };
        Expression::EnumVariantInit(
            result.to_string(),
            variant.to_string(),
            vec![(field.to_string(), value)],
        )
    }

    fn rewrite_block(&mut self, block: &mut Block, result: &str, error: &Type) {
        for statement in block.iter_mut() {
            self.rewrite_statement(&mut statement.node, result, error);
        }
        // A trailing expression statement is the implicit return value.
        if let Some(last) = block.last_mut()
            && let Statement::Expression(expression) = &mut last.node
        {
            let value =
                std::mem::replace(expression, Expression::Boolean(false));
            last.node =
                Statement::Return(self.wrap_return(value, result, error));
        }
    }

    fn rewrite_statement(
        &mut self,
        statement: &mut Statement,
        result: &str,
        error: &Type,
    ) {
        match statement {
            Statement::Return(expression) => {
                self.rewrite_expression(expression, result, error);
                let value =
                    std::mem::replace(expression, Expression::Boolean(false));
                *expression = self.wrap_return(value, result, error);
            }
            Statement::Let { value, .. }
            | Statement::Constant(_, value)
            | Statement::Expression(value) => {
                self.rewrite_expression(value, result, error);
            }
            Statement::Assignment(place, value) => {
                self.rewrite_expression(place, result, error);
                self.rewrite_expression(value, result, error);
            }
            Statement::Defer(inner) => {
                self.rewrite_statement(inner, result, error)
            }
            Statement::While(condition, body) => {
                self.rewrite_expression(condition, result, error);
                self.rewrite_inner_block(body, result, error);
            }
            Statement::For(_, _, iterable, body) => {
                self.rewrite_expression(iterable, result, error);
                self.rewrite_inner_block(body, result, error);
            }
            Statement::With(_, body) => {
                self.rewrite_inner_block(body, result, error)
            }
            _ => {}
        }
    }

    // A nested block (a loop or branch body) whose trailing expression is not a
    // function-level return, so only statements are rewritten.
    fn rewrite_inner_block(
        &mut self,
        block: &mut Block,
        result: &str,
        error: &Type,
    ) {
        for statement in block.iter_mut() {
            self.rewrite_statement(&mut statement.node, result, error);
        }
    }

    fn rewrite_expression(
        &mut self,
        expression: &mut Expression,
        result: &str,
        error: &Type,
    ) {
        match expression {
            Expression::Try(inner) => {
                self.rewrite_expression(inner, result, error);
                let inner = std::mem::replace(
                    inner.as_mut(),
                    Expression::Boolean(false),
                );
                *expression = self.desugar_try(inner, result);
            }
            Expression::Call(callee, arguments) => {
                self.rewrite_expression(callee, result, error);
                for argument in arguments.iter_mut() {
                    self.rewrite_expression(argument, result, error);
                }
            }
            Expression::PackMap(inner, _, _)
            | Expression::Prefix(_, inner)
            | Expression::AddressOf(inner)
            | Expression::Borrow(inner)
            | Expression::BorrowMut(inner)
            | Expression::Dereference(inner)
            | Expression::ArrayRepeat(inner, _)
            | Expression::FieldAccess(inner, _) => {
                self.rewrite_expression(inner, result, error);
            }
            Expression::Infix(left, _, right)
            | Expression::Index(left, right) => {
                self.rewrite_expression(left, result, error);
                self.rewrite_expression(right, result, error);
            }
            Expression::If(condition, then_block, else_block) => {
                self.rewrite_expression(condition, result, error);
                self.rewrite_inner_block(then_block, result, error);
                if let Some(block) = else_block {
                    self.rewrite_inner_block(block, result, error);
                }
            }
            Expression::StructInit(_, fields)
            | Expression::EnumVariantInit(_, _, fields) => {
                for (_, value) in fields.iter_mut() {
                    self.rewrite_expression(value, result, error);
                }
            }
            Expression::Switch(scrutinee, cases) => {
                self.rewrite_expression(scrutinee, result, error);
                for case in cases.iter_mut() {
                    self.rewrite_inner_block(&mut case.body, result, error);
                }
            }
            Expression::Tuple(items) => {
                for item in items.iter_mut() {
                    self.rewrite_expression(item, result, error);
                }
            }
            Expression::Unsafe(body) => {
                self.rewrite_inner_block(body, result, error)
            }
            Expression::Identifier(_)
            | Expression::Literal(_)
            | Expression::Boolean(_)
            | Expression::Sizeof(_)
            | Expression::TypeId(_)
            | Expression::TypeName(_)
            | Expression::TypeValue(_)
            | Expression::Range(..)
            | Expression::Function(..)
            | Expression::Proc(..)
            | Expression::UnsafeFn(_) => {}
        }
    }

    // `inner?` becomes a match. The Ok value flows out, the Err returns the
    // enclosing function's Err carrying the same error.
    fn desugar_try(
        &mut self,
        inner: Expression,
        enclosing: &str,
    ) -> Expression {
        let callee_result = match &inner {
            Expression::Call(callee, _) => match callee.as_ref() {
                Expression::Identifier(name) => {
                    self.fallible.get(name).cloned()
                }
                _ => None,
            },
            _ => None,
        };
        let callee_result =
            callee_result.unwrap_or_else(|| enclosing.to_string());

        let value_binding = format!("__try_v{}", self.counter);
        let error_binding = format!("__try_e{}", self.counter);
        self.counter += 1;

        let ok_case = SwitchCase {
            pattern: Pattern::EnumVariant {
                enum_name: Some(callee_result.clone()),
                variant_name: "Ok".to_string(),
                bindings: vec![("value".to_string(), value_binding.clone())],
            },
            body: vec![spanned(Statement::Expression(Expression::Identifier(
                value_binding,
            )))],
        };
        let err_case = SwitchCase {
            pattern: Pattern::EnumVariant {
                enum_name: Some(callee_result),
                variant_name: "Err".to_string(),
                bindings: vec![("error".to_string(), error_binding.clone())],
            },
            body: vec![spanned(Statement::Return(
                Expression::EnumVariantInit(
                    enclosing.to_string(),
                    "Err".to_string(),
                    vec![(
                        "error".to_string(),
                        Expression::Identifier(error_binding),
                    )],
                ),
            ))],
        };
        Expression::Switch(Box::new(inner), vec![ok_case, err_case])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn parse(source: &str) -> Program {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let positions = lexer.positions().to_vec();
        let mut parser = Parser::with_positions(&tokens, &positions);
        parser.parse().unwrap()
    }

    #[test]
    fn rewrites_fallible_signature() {
        let source =
            "read_size :: fn(ok: i64) -> i64 ! FileError {\n    return 42\n}\n";
        let mut program = parse(source);
        let before = format!("{:?}", program);
        assert!(before.contains("Fallible"), "parsed sig: {before}");
        lower_failure_sets(&mut program, &mut HashSet::new()).unwrap();
        let after = format!("{:?}", program);
        assert!(after.contains("__Result_0"), "after: {after}");
        assert!(!after.contains("Fallible"), "after still fallible: {after}");
    }

    #[test]
    fn rejects_try_without_a_failure_set() {
        let source = "src :: fn() -> i64 ! E { return 1 }\nuse_it :: fn() -> i64 { src()? }\n";
        let mut program = parse(source);
        let error =
            lower_failure_sets(&mut program, &mut HashSet::new()).unwrap_err();
        assert!(
            error.to_string().contains("failure set"),
            "unexpected error: {error}"
        );
    }
}
