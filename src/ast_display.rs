// The rendered form of a node, the same text the old tree types' Display
// wrote. Diagnostics quote these strings and the parity suite pins fragments
// of them, so the shapes here are load-bearing down to the parentheses.

use crate::ast::{
    Ast, ExprId, Expression, Literal, Parameter, Pattern, PatternId, Range32,
    ReturnKind, ReturnSignature, SignatureId, Statement, StmtId,
};

pub fn display_parameter(ast: &Ast, parameter: &Parameter) -> String {
    match &parameter.type_annotation {
        Some(typ) => format!("{}: {}", ast.name(parameter.name), typ),
        None => ast.name(parameter.name).to_string(),
    }
}

fn display_parameters(ast: &Ast, parameters: Range32) -> String {
    ast.params_in(parameters)
        .iter()
        .map(|parameter| display_parameter(ast, parameter))
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn display_signature(ast: &Ast, signature: SignatureId) -> String {
    display_signature_value(ast, ast.signature(signature))
}

pub fn display_signature_value(
    ast: &Ast,
    signature: &ReturnSignature,
) -> String {
    let mut out = String::new();
    match &signature.kind {
        ReturnKind::None => {}
        ReturnKind::Single(t) => out.push_str(&format!(" -> {}", t)),
        ReturnKind::Fallible(value, error) => {
            out.push_str(&format!(" -> {} ! {}", value, error))
        }
        ReturnKind::Multiple(values) => {
            let parts: Vec<String> = ast
                .return_values_in(*values)
                .iter()
                .map(|held| match held.name {
                    Some(name) => {
                        format!("{}: {}", ast.name(name), held.value_type)
                    }
                    None => held.value_type.to_string(),
                })
                .collect();
            out.push_str(&format!(" -> ({})", parts.join(", ")));
        }
    }
    for capability in &signature.uses {
        out.push_str(&format!(" uses {}", capability));
    }
    if let Some(bound) = signature.bound {
        out.push_str(&format!(" where {}", display_expr(ast, bound)));
    }
    out
}

fn display_block(ast: &Ast, block: Range32, separator: &str) -> String {
    ast.stmts_in(block)
        .iter()
        .map(|statement| display_stmt(ast, *statement))
        .collect::<Vec<_>>()
        .join(separator)
}

pub fn display_stmt(ast: &Ast, statement: StmtId) -> String {
    match ast.stmt(statement) {
        Statement::Let {
            name,
            type_annotation,
            value,
            mutable,
        } => {
            let mut_str = if *mutable { "mut " } else { "" };
            match type_annotation {
                Some(typ) => format!(
                    "{}{} : {} = {};",
                    mut_str,
                    ast.name(*name),
                    typ,
                    display_expr(ast, *value)
                ),
                None => format!(
                    "{}{} := {};",
                    mut_str,
                    ast.name(*name),
                    display_expr(ast, *value)
                ),
            }
        }
        Statement::LetMultiple(bindings, value) => {
            let names: Vec<String> = ast
                .bindings_in(*bindings)
                .iter()
                .map(|binding| {
                    let prefix = if binding.mutable { "mut " } else { "" };
                    format!("{}{}", prefix, ast.name(binding.name))
                })
                .collect();
            format!("{} := {};", names.join(", "), display_expr(ast, *value))
        }
        Statement::Constant(identifier, expression) => {
            format!(
                "{} :: {};",
                ast.name(*identifier),
                display_expr(ast, *expression)
            )
        }
        Statement::Return(expression) => {
            format!("return {};", display_expr(ast, *expression))
        }
        Statement::Print(expression, arguments) => {
            let mut written =
                format!("print {}", display_expr(ast, *expression));
            for argument in ast.exprs_in(*arguments) {
                written
                    .push_str(&format!(", {}", display_expr(ast, *argument)));
            }
            written.push(char::from(59));
            written
        }
        Statement::Expression(expression) => display_expr(ast, *expression),
        Statement::Struct(name, type_params, fields) => {
            let field_strs: Vec<String> = ast
                .fields_in(*fields)
                .iter()
                .map(|field| {
                    format!("{}: {}", ast.name(field.name), field.field_type)
                })
                .collect();
            if type_params.is_empty() {
                format!(
                    "{} :: struct {{ {} }}",
                    ast.name(*name),
                    field_strs.join(", ")
                )
            } else {
                let params_str = ast
                    .symbols_in(*type_params)
                    .iter()
                    .map(|p| format!("${}: Type", ast.name(*p)))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "{} :: struct({}) {{ {} }}",
                    ast.name(*name),
                    params_str,
                    field_strs.join(", ")
                )
            }
        }
        Statement::Enum(name, _, variants) => {
            let variant_strs: Vec<String> = ast
                .variants_in(*variants)
                .iter()
                .map(|v| match v.fields {
                    Some(fields) => {
                        let field_strs: Vec<String> = ast
                            .fields_in(fields)
                            .iter()
                            .map(|f| {
                                format!(
                                    "{}: {}",
                                    ast.name(f.name),
                                    f.field_type
                                )
                            })
                            .collect();
                        format!(
                            "{} {{ {} }}",
                            ast.name(v.name),
                            field_strs.join(", ")
                        )
                    }
                    None => ast.name(v.name).to_string(),
                })
                .collect();
            format!(
                "{} :: enum {{ {} }}",
                ast.name(*name),
                variant_strs.join(", ")
            )
        }
        Statement::Flags(name, repr, bits) => {
            let bit_strs: Vec<String> = ast
                .flag_bits_in(*bits)
                .iter()
                .map(|bit| format!("{} = {}", ast.name(bit.name), bit.value))
                .collect();
            format!(
                "{} :: flags {} {{ {} }}",
                ast.name(*name),
                repr,
                bit_strs.join(", ")
            )
        }
        Statement::TypeAlias(name, typ) => {
            format!("{} :: {};", ast.name(*name), typ)
        }
        Statement::Defer(inner) => {
            format!("defer {}", display_stmt(ast, *inner))
        }
        Statement::Assignment(lhs, rhs) => {
            format!("{} = {}", display_expr(ast, *lhs), display_expr(ast, *rhs))
        }
        Statement::For(iterator, second, range, body) => {
            let names = match second {
                Some(second) => {
                    format!("{}, {}", ast.name(*iterator), ast.name(*second))
                }
                None => ast.name(*iterator).to_string(),
            };
            format!(
                "for {} in {} {{ {} }}",
                names,
                display_expr(ast, *range),
                display_block(ast, *body, "; ")
            )
        }
        Statement::While(condition, body) => {
            format!(
                "while ({}) {{ {} }}",
                display_expr(ast, *condition),
                display_block(ast, *body, "; ")
            )
        }
        Statement::With(capability, body) => {
            format!(
                "with {} {{ {} }}",
                ast.name(*capability),
                display_block(ast, *body, "; ")
            )
        }
        Statement::Break => "break".to_string(),
        Statement::Continue => "continue".to_string(),
        Statement::Import(path, renames) => {
            if renames.is_empty() {
                format!("import \"{}\"", path)
            } else {
                let parts: Vec<String> = ast
                    .renames_in(*renames)
                    .iter()
                    .map(|held| {
                        format!(
                            "{} as {}",
                            ast.name(held.exported),
                            ast.name(held.local)
                        )
                    })
                    .collect();
                format!("import \"{}\" ({})", path, parts.join(", "))
            }
        }
        Statement::Declared {
            name,
            params,
            return_sig,
        } => {
            format!(
                "{} :: fn({}){}",
                ast.name(*name),
                display_parameters(ast, *params),
                display_signature(ast, *return_sig)
            )
        }
        Statement::Extern {
            name,
            params,
            return_type,
            safe,
        } => {
            let params_str = display_parameters(ast, *params);
            let marker = if *safe { "safe extern" } else { "extern" };
            match return_type {
                Some(typ) => format!(
                    "{} :: {} fn({}) -> {}",
                    ast.name(*name),
                    marker,
                    params_str,
                    typ
                ),
                None => format!(
                    "{} :: {} fn({})",
                    ast.name(*name),
                    marker,
                    params_str
                ),
            }
        }
    }
}

pub fn display_literal(ast: &Ast, literal: &Literal) -> String {
    match literal {
        Literal::Integer(x) => x.to_string(),
        Literal::Float(x) => x.to_string(),
        Literal::Float32(x) => format!("{}f32", x),
        Literal::Boolean(x) => x.to_string(),
        Literal::String(x) => x.to_string(),
        Literal::Array(array) => {
            let expressions = ast
                .exprs_in(*array)
                .iter()
                .map(|e| display_expr(ast, *e))
                .collect::<Vec<_>>();
            format!("[{}]", expressions.join(", "))
        }
    }
}

pub fn display_pattern(ast: &Ast, pattern: PatternId) -> String {
    match ast.pattern(pattern) {
        Pattern::Wildcard => "_".to_string(),
        Pattern::Literal(lit) => display_literal(ast, lit),
        Pattern::Identifier(id) => ast.name(*id).to_string(),
        Pattern::EnumVariant {
            enum_name,
            variant_name,
            bindings,
        } => {
            let prefix = match enum_name {
                Some(e) => format!("{}::", ast.name(*e)),
                None => ".".to_string(),
            };
            if bindings.is_empty() {
                format!("{}{}", prefix, ast.name(*variant_name))
            } else {
                let binding_strs: Vec<String> = ast
                    .pattern_bindings_in(*bindings)
                    .iter()
                    .map(|held| {
                        if held.field == held.binding {
                            ast.name(held.field).to_string()
                        } else {
                            format!(
                                "{} = {}",
                                ast.name(held.field),
                                ast.name(held.binding)
                            )
                        }
                    })
                    .collect();
                format!(
                    "{}{} {{ {} }}",
                    prefix,
                    ast.name(*variant_name),
                    binding_strs.join(", ")
                )
            }
        }
        Pattern::Tuple(patterns) => {
            let pat_strs: Vec<String> = ast
                .patterns_in(*patterns)
                .iter()
                .map(|p| display_pattern(ast, *p))
                .collect();
            format!("({})", pat_strs.join(", "))
        }
    }
}

pub fn display_expr(ast: &Ast, expression: ExprId) -> String {
    match ast.expr(expression) {
        Expression::Try(inner) => format!("{}?", display_expr(ast, *inner)),
        Expression::PackMap(body, variable, list) => {
            format!(
                "{} for {} in {}",
                display_expr(ast, *body),
                ast.name(*variable),
                ast.name(*list)
            )
        }
        Expression::ArrayRepeat(value, count) => {
            format!("[{}; {}]", display_expr(ast, *value), ast.name(*count))
        }
        Expression::Identifier(identifier) => ast.name(*identifier).to_string(),
        Expression::Literal(literal) => display_literal(ast, literal),
        Expression::Boolean(boolean) => boolean.to_string(),
        Expression::Prefix(operator, inner) => {
            format!("({}{})", operator, display_expr(ast, *inner))
        }
        Expression::Infix(left, operator, right) => {
            format!(
                "({} {} {})",
                display_expr(ast, *left),
                operator,
                display_expr(ast, *right)
            )
        }
        Expression::If(condition, consequence, alternative) => {
            let mut result = format!(
                "if ({}) {{ {} }}",
                display_expr(ast, *condition),
                display_block(ast, *consequence, "\n"),
            );
            if let Some(alternative) = alternative {
                result.push_str(&format!(
                    "else {{ {} }}",
                    display_block(ast, *alternative, "\n")
                ));
            }
            result
        }
        Expression::Function(parameters, return_sig, body) => {
            format!(
                "fn({}){}{{ {} }}",
                display_parameters(ast, *parameters),
                display_signature(ast, *return_sig),
                display_block(ast, *body, "\n")
            )
        }
        Expression::Proc(parameters, return_sig, body) => {
            format!(
                "fn({}){}{{ {} }}",
                display_parameters(ast, *parameters),
                display_signature(ast, *return_sig),
                display_block(ast, *body, "\n")
            )
        }
        Expression::Call(callee, arguments) => {
            let rendered = ast
                .exprs_in(*arguments)
                .iter()
                .map(|argument| display_expr(ast, *argument))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}({})", display_expr(ast, *callee), rendered)
        }
        Expression::Index(left_expression, index_expression) => {
            format!(
                "({}[{}])",
                display_expr(ast, *left_expression),
                display_expr(ast, *index_expression)
            )
        }
        Expression::FieldAccess(inner, field) => {
            format!("{}.{}", display_expr(ast, *inner), ast.name(*field))
        }
        Expression::AddressOf(inner) | Expression::Borrow(inner) => {
            format!("(&{})", display_expr(ast, *inner))
        }
        Expression::BorrowMut(inner) => {
            format!("(&mut {})", display_expr(ast, *inner))
        }
        Expression::Dereference(inner) => {
            format!("({}^)", display_expr(ast, *inner))
        }
        Expression::StructInit(name, fields) => {
            let field_strs: Vec<String> = ast
                .named_in(*fields)
                .iter()
                .map(|held| {
                    format!(
                        "{} = {}",
                        ast.name(held.name),
                        display_expr(ast, held.value)
                    )
                })
                .collect();
            format!("{} {{ {} }}", ast.name(*name), field_strs.join(", "))
        }
        Expression::Range(start, end, inclusive) => {
            if *inclusive {
                format!(
                    "{}..={}",
                    display_expr(ast, *start),
                    display_expr(ast, *end)
                )
            } else {
                format!(
                    "{}..{}",
                    display_expr(ast, *start),
                    display_expr(ast, *end)
                )
            }
        }
        Expression::Switch(scrutinee, cases) => {
            let case_strs: Vec<String> = ast
                .cases_in(*cases)
                .iter()
                .map(|c| {
                    format!(
                        "case {}: {{ {} }}",
                        display_pattern(ast, c.pattern),
                        display_block(ast, c.body, "; ")
                    )
                })
                .collect();
            format!(
                "match {} {{ {} }}",
                display_expr(ast, *scrutinee),
                case_strs.join(" ")
            )
        }
        Expression::Tuple(elements) => {
            let elem_strs: Vec<String> = ast
                .exprs_in(*elements)
                .iter()
                .map(|e| display_expr(ast, *e))
                .collect();
            format!("({})", elem_strs.join(", "))
        }
        Expression::EnumVariantInit(enum_name, variant_name, fields) => {
            let field_strs: Vec<String> = ast
                .named_in(*fields)
                .iter()
                .map(|held| {
                    format!(
                        "{} = {}",
                        ast.name(held.name),
                        display_expr(ast, held.value)
                    )
                })
                .collect();
            format!(
                "{}::{} {{ {} }}",
                ast.name(*enum_name),
                ast.name(*variant_name),
                field_strs.join(", ")
            )
        }
        Expression::TypeValue(typ) => format!("{}", typ),
        Expression::Unsafe(body) => {
            format!("unsafe {{ {} }}", display_block(ast, *body, "; "))
        }
        Expression::UnsafeFn(inner) => {
            format!("unsafe {}", display_expr(ast, *inner))
        }
    }
}
