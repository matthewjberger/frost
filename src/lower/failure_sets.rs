use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::ast::{
    Ast, EnumVariant, ExprId, Expression, NamedExpr, Pattern, PatternBinding,
    Range32, ReturnKind, Statement, StmtId, StructField, SwitchCase, TokenSpan,
};
use crate::parser::Program;
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
    enums: Vec<StmtId>,
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

pub fn lower_failure_sets(
    program: &mut Program,
    linear: &mut HashSet<String>,
) -> Result<()> {
    let ast = &mut program.ast;
    let mut lowerer = Lowerer {
        linear: linear.clone(),
        results: HashMap::new(),
        enums: Vec::new(),
        fallible: HashMap::new(),
        variants: HashMap::new(),
        value_is_aggregate: false,
        counter: 0,
    };

    for statement in &program.roots {
        if let Statement::Enum(name, _, variants) = ast.stmt(*statement) {
            lowerer.variants.insert(
                ast.name(*name).to_string(),
                ast.variants_in(*variants)
                    .iter()
                    .map(|variant| ast.name(variant.name).to_string())
                    .collect(),
            );
        }
    }

    // First pass. Give every fallible function a Result enum.
    for statement in &program.roots {
        let Statement::Constant(name, value) = ast.stmt(*statement) else {
            continue;
        };
        let (name, value) = (*name, *value);
        let signature = match ast.expr(value) {
            Expression::Function(_, signature, _)
            | Expression::Proc(_, signature, _) => *signature,
            _ => continue,
        };
        let ReturnKind::Fallible(value, error) =
            ast.signature(signature).kind.clone()
        else {
            continue;
        };
        let result = lowerer.result_enum(ast, &value, &error);
        lowerer.fallible.insert(ast.name(name).to_string(), result);
    }

    // Second pass. Rewrite bodies and return signatures. A `?` in a function
    // that declares no failure set has nowhere to propagate to, so reject it.
    for statement in &program.roots {
        let Statement::Constant(name, value) = ast.stmt(*statement) else {
            continue;
        };
        let (name, value) = (*name, *value);
        let (signature, body) = match ast.expr(value) {
            Expression::Function(_, signature, body)
            | Expression::Proc(_, signature, body) => (*signature, *body),
            _ => continue,
        };
        if let ReturnKind::Fallible(value, error) =
            ast.signature(signature).kind.clone()
        {
            let result = lowerer.fallible.get(ast.name(name)).unwrap().clone();
            ast.signatures[signature.0 as usize].kind =
                ReturnKind::Single(Type::Enum(result.clone()));
            lowerer.value_is_aggregate =
                matches!(value, Type::Struct(_) | Type::Enum(_));
            lowerer.rewrite_block(ast, body, &result, &error);
        } else if block_has_try(ast, body) {
            let name = ast.name(name);
            bail!(
                "the `?` operator is only allowed in a function with a failure set; '{name}' must declare `-> T ! E`"
            );
        }
    }

    // Prepend the synthesized enums so they are declared before use.
    let mut roots = std::mem::take(&mut lowerer.enums);
    roots.append(&mut program.roots);
    program.roots = roots;
    // The results that hold a linear value, so the check that follows knows
    // them for what they are.
    *linear = lowerer.linear;
    Ok(())
}

fn block_has_try(ast: &Ast, block: Range32) -> bool {
    ast.stmts_in(block)
        .iter()
        .any(|statement| statement_has_try(ast, *statement))
}

fn statement_has_try(ast: &Ast, statement: StmtId) -> bool {
    match ast.stmt(statement) {
        Statement::Return(value)
        | Statement::Let { value, .. }
        | Statement::Constant(_, value)
        | Statement::Expression(value) => expression_has_try(ast, *value),
        Statement::Assignment(place, value) => {
            expression_has_try(ast, *place) || expression_has_try(ast, *value)
        }
        Statement::Defer(inner) | Statement::ErrDefer(inner) => {
            statement_has_try(ast, *inner)
        }
        Statement::While(condition, body) => {
            expression_has_try(ast, *condition) || block_has_try(ast, *body)
        }
        Statement::For(_, _, iterable, body) => {
            expression_has_try(ast, *iterable) || block_has_try(ast, *body)
        }
        Statement::With(_, body) => block_has_try(ast, *body),
        _ => false,
    }
}

fn expression_has_try(ast: &Ast, expression: ExprId) -> bool {
    match ast.expr(expression) {
        Expression::Try(_) => true,
        Expression::PackMap(inner, _, _)
        | Expression::Prefix(_, inner)
        | Expression::AddressOf(inner)
        | Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::Dereference(inner)
        | Expression::ArrayRepeat(inner, _)
        | Expression::FieldAccess(inner, _) => expression_has_try(ast, *inner),
        Expression::Infix(left, _, right) | Expression::Index(left, right) => {
            expression_has_try(ast, *left) || expression_has_try(ast, *right)
        }
        Expression::Call(callee, arguments) => {
            expression_has_try(ast, *callee)
                || ast
                    .exprs_in(*arguments)
                    .iter()
                    .any(|argument| expression_has_try(ast, *argument))
        }
        Expression::If(condition, then_block, else_block) => {
            expression_has_try(ast, *condition)
                || block_has_try(ast, *then_block)
                || else_block.is_some_and(|block| block_has_try(ast, block))
        }
        Expression::StructInit(_, fields)
        | Expression::EnumVariantInit(_, _, fields) => ast
            .named_in(*fields)
            .iter()
            .any(|field| expression_has_try(ast, field.value)),
        Expression::Switch(scrutinee, cases) => {
            expression_has_try(ast, *scrutinee)
                || ast
                    .cases_in(*cases)
                    .iter()
                    .any(|case| block_has_try(ast, case.body))
        }
        Expression::Unsafe(body) => block_has_try(ast, *body),
        Expression::Tuple(items) => ast
            .exprs_in(*items)
            .iter()
            .any(|item| expression_has_try(ast, *item)),
        // Listed rather than caught by `_`, so a new expression form is a
        // compile error here instead of silently reporting no `?` inside it.
        Expression::Identifier(_)
        | Expression::Literal(_)
        | Expression::Boolean(_)
        | Expression::TypeValue(_)
        | Expression::Range(..)
        | Expression::Function(..)
        | Expression::Proc(..)
        | Expression::UnsafeFn(_) => false,
    }
}

impl Lowerer {
    fn result_enum(
        &mut self,
        ast: &mut Ast,
        value: &Type,
        error: &Type,
    ) -> String {
        let key = format!("{value}!{error}");
        if let Some(name) = self.results.get(&key) {
            return name.clone();
        }
        // Named as the instance of a generic enum it is: one per `(T, E)`, the
        // same shape every time. The name is what a program writes to reach the
        // two variants, so `Result::Ok` reads here the way `Option::Some` reads
        // against an `Option<i64>`, and the values under a failure set need no
        // spelling of their own.
        let name = format!("Result<{value}, {error}>");
        let recorded = ast.intern(&name);
        ast.failure_results.push(recorded);
        let value_field = ast.intern("value");
        let error_field = ast.intern("error");
        let ok_name = ast.intern("Ok");
        let err_name = ast.intern("Err");
        let ok_fields = ast.add_struct_fields(vec![StructField {
            name: value_field,
            field_type: value.clone(),
            align: None,
        }]);
        let err_fields = ast.add_struct_fields(vec![StructField {
            name: error_field,
            field_type: error.clone(),
            align: None,
        }]);
        let variants = ast.add_enum_variants(&[
            EnumVariant {
                name: ok_name,
                fields: Some(ok_fields),
            },
            EnumVariant {
                name: err_name,
                fields: Some(err_fields),
            },
        ]);
        let name_symbol = ast.intern(&name);
        let declaration = ast.push_stmt(
            Statement::Enum(name_symbol, Range32::EMPTY, variants),
            TokenSpan::NONE,
        );
        self.enums.push(declaration);
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
        ast: &Ast,
        expression: ExprId,
        error: &Type,
    ) -> bool {
        let error_name = match error {
            Type::Enum(name) | Type::Struct(name) => name,
            _ => return false,
        };
        match ast.expr(expression) {
            // `return .Denied`: the dot names no enum, and is refused once the
            // types are read. Which side it was meant for is still worked out
            // here, because that decides the type the refusal names, and a
            // reader told to write `Fault::Denied` is told the useful half.
            Expression::EnumVariantInit(name, variant, _)
                if ast.name(*name).is_empty() =>
            {
                self.variants.get(error_name).is_some_and(|names| {
                    names.iter().any(|held| held.as_str() == ast.name(*variant))
                })
            }
            // `return { at = 3 }`: the literal names no type either. It is the
            // failure when the value the function answers with is not itself a
            // struct or an enum, since then only the failure can be written
            // this way. A function that answers with one names it.
            Expression::StructInit(name, _) if ast.name(*name).is_empty() => {
                !self.value_is_aggregate
                    && matches!(error, Type::Struct(_) | Type::Enum(_))
            }
            Expression::EnumVariantInit(name, _, _)
            | Expression::StructInit(name, _) => ast.name(*name) == error_name,
            _ => false,
        }
    }

    fn wrap_return(
        &self,
        ast: &mut Ast,
        value: ExprId,
        result: &str,
        error: &Type,
    ) {
        let (variant, field) = if self.is_error_construction(ast, value, error)
        {
            ("Err", "error")
        } else {
            ("Ok", "value")
        };
        let span = ast.expr_span(value);
        let inner = ast.expr(value).clone();
        let inner = ast.push_expr(inner, span);
        let result_symbol = ast.intern(result);
        let variant_symbol = ast.intern(variant);
        let field_symbol = ast.intern(field);
        let fields = ast.add_named_exprs(&[NamedExpr {
            name: field_symbol,
            value: inner,
        }]);
        ast.expressions[value.0 as usize] =
            Expression::EnumVariantInit(result_symbol, variant_symbol, fields);
    }

    fn rewrite_block(
        &mut self,
        ast: &mut Ast,
        block: Range32,
        result: &str,
        error: &Type,
    ) {
        for index in block.indices() {
            let statement = ast.stmt_list[index];
            self.rewrite_statement(ast, statement, result, error);
        }
        // A trailing expression statement is the implicit return value.
        if let Some(last) = ast.stmts_in(block).last().copied()
            && let Statement::Expression(expression) = ast.stmt(last)
        {
            let expression = *expression;
            self.wrap_return(ast, expression, result, error);
            ast.statements[last.0 as usize] = Statement::Return(expression);
        }
    }

    fn rewrite_statement(
        &mut self,
        ast: &mut Ast,
        statement: StmtId,
        result: &str,
        error: &Type,
    ) {
        match ast.stmt(statement).clone() {
            Statement::Return(expression) => {
                self.rewrite_expression(ast, expression, result, error);
                self.wrap_return(ast, expression, result, error);
            }
            Statement::Let { value, .. }
            | Statement::Constant(_, value)
            | Statement::Expression(value) => {
                self.rewrite_expression(ast, value, result, error);
            }
            Statement::Assignment(place, value) => {
                self.rewrite_expression(ast, place, result, error);
                self.rewrite_expression(ast, value, result, error);
            }
            Statement::Defer(inner) | Statement::ErrDefer(inner) => {
                self.rewrite_statement(ast, inner, result, error)
            }
            Statement::While(condition, body) => {
                self.rewrite_expression(ast, condition, result, error);
                self.rewrite_inner_block(ast, body, result, error);
            }
            Statement::For(_, _, iterable, body) => {
                self.rewrite_expression(ast, iterable, result, error);
                self.rewrite_inner_block(ast, body, result, error);
            }
            Statement::With(_, body) => {
                self.rewrite_inner_block(ast, body, result, error)
            }
            _ => {}
        }
    }

    // A nested block (a loop or branch body) whose trailing expression is not a
    // function-level return, so only statements are rewritten.
    fn rewrite_inner_block(
        &mut self,
        ast: &mut Ast,
        block: Range32,
        result: &str,
        error: &Type,
    ) {
        for index in block.indices() {
            let statement = ast.stmt_list[index];
            self.rewrite_statement(ast, statement, result, error);
        }
    }

    fn rewrite_expression(
        &mut self,
        ast: &mut Ast,
        expression: ExprId,
        result: &str,
        error: &Type,
    ) {
        match ast.expr(expression).clone() {
            Expression::Try(inner) => {
                self.rewrite_expression(ast, inner, result, error);
                self.desugar_try(ast, expression, inner, result);
            }
            Expression::Call(callee, arguments) => {
                self.rewrite_expression(ast, callee, result, error);
                for index in arguments.indices() {
                    let argument = ast.expr_list[index];
                    self.rewrite_expression(ast, argument, result, error);
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
                self.rewrite_expression(ast, inner, result, error);
            }
            Expression::Infix(left, _, right)
            | Expression::Index(left, right) => {
                self.rewrite_expression(ast, left, result, error);
                self.rewrite_expression(ast, right, result, error);
            }
            Expression::If(condition, then_block, else_block) => {
                self.rewrite_expression(ast, condition, result, error);
                self.rewrite_inner_block(ast, then_block, result, error);
                if let Some(block) = else_block {
                    self.rewrite_inner_block(ast, block, result, error);
                }
            }
            Expression::StructInit(_, fields)
            | Expression::EnumVariantInit(_, _, fields) => {
                for index in fields.indices() {
                    let value = ast.named_exprs[index].value;
                    self.rewrite_expression(ast, value, result, error);
                }
            }
            Expression::Switch(scrutinee, cases) => {
                self.rewrite_expression(ast, scrutinee, result, error);
                for index in cases.indices() {
                    let body = ast.cases[index].body;
                    self.rewrite_inner_block(ast, body, result, error);
                }
            }
            Expression::Tuple(items) => {
                for index in items.indices() {
                    let item = ast.expr_list[index];
                    self.rewrite_expression(ast, item, result, error);
                }
            }
            Expression::Unsafe(body) => {
                self.rewrite_inner_block(ast, body, result, error)
            }
            Expression::Identifier(_)
            | Expression::Literal(_)
            | Expression::Boolean(_)
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
        ast: &mut Ast,
        expression: ExprId,
        inner: ExprId,
        enclosing: &str,
    ) {
        let callee_result = match ast.expr(inner) {
            Expression::Call(callee, _) => match ast.expr(*callee) {
                Expression::Identifier(name) => {
                    self.fallible.get(ast.name(*name)).cloned()
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

        let span = ast.expr_span(expression);
        let callee_result_symbol = ast.intern(&callee_result);
        let enclosing_symbol = ast.intern(enclosing);
        let value_binding_symbol = ast.intern(&value_binding);
        let error_binding_symbol = ast.intern(&error_binding);
        let value_field = ast.intern("value");
        let error_field = ast.intern("error");
        let ok_variant = ast.intern("Ok");
        let err_variant = ast.intern("Err");

        let ok_bindings = ast.add_pattern_bindings(&[PatternBinding {
            field: value_field,
            binding: value_binding_symbol,
        }]);
        // The `?` wrote these two arms, so they stand where it does.
        let ok_pattern = ast.push_pattern(
            Pattern::EnumVariant {
                enum_name: Some(callee_result_symbol),
                variant_name: ok_variant,
                bindings: ok_bindings,
            },
            span,
        );
        let ok_value =
            ast.push_expr(Expression::Identifier(value_binding_symbol), span);
        let ok_statement = ast.push_stmt(Statement::Expression(ok_value), span);
        let ok_body = ast.add_stmt_list(&[ok_statement]);

        let err_bindings = ast.add_pattern_bindings(&[PatternBinding {
            field: error_field,
            binding: error_binding_symbol,
        }]);
        let err_pattern = ast.push_pattern(
            Pattern::EnumVariant {
                enum_name: Some(callee_result_symbol),
                variant_name: err_variant,
                bindings: err_bindings,
            },
            span,
        );
        let carried =
            ast.push_expr(Expression::Identifier(error_binding_symbol), span);
        let err_fields = ast.add_named_exprs(&[NamedExpr {
            name: error_field,
            value: carried,
        }]);
        let err_value = ast.push_expr(
            Expression::EnumVariantInit(
                enclosing_symbol,
                err_variant,
                err_fields,
            ),
            span,
        );
        let err_statement = ast.push_stmt(Statement::Return(err_value), span);
        let err_body = ast.add_stmt_list(&[err_statement]);

        let cases = ast.add_cases(&[
            SwitchCase {
                pattern: ok_pattern,
                body: ok_body,
            },
            SwitchCase {
                pattern: err_pattern,
                body: err_body,
            },
        ]);
        ast.expressions[expression.0 as usize] =
            Expression::Switch(inner, cases);
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
