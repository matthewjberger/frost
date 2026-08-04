// Multiple return values, lowered to one struct.
//
// `-> (i64, i64)` declares a return type list, `return quotient, remainder`
// produces one, and `quotient, remainder := divide(a, b)` takes it apart. There
// is no tuple type behind any of that. Each distinct list of types gets one
// struct, the signature becomes a plain return of that struct, the `return`
// becomes a struct literal, and the binding becomes a temporary and a field
// read per name. Everything after this pass sees a function that returns a
// struct, which every backend already handles, including the C ABI.
//
// Two functions that return the same list of types share the struct, since the
// name is derived from the types.

use crate::ast::{
    Ast, ExprId, Expression, NamedExpr, Range32, ReturnKind, Statement, StmtId,
    StructField, TokenSpan,
};
use crate::types::Type;
use anyhow::{Result, bail};
use std::collections::{BTreeMap, HashMap};

pub fn lower_multiple_returns(
    ast: &mut Ast,
    roots: &mut Vec<StmtId>,
) -> Result<()> {
    let mut lowering = Lowering {
        signatures: HashMap::new(),
        structs: BTreeMap::new(),
        counter: 0,
    };
    lowering.collect_signatures(ast, roots)?;
    if lowering.signatures.is_empty() {
        // Nothing declares a return type list, so there is nothing to rewrite
        // and no call to reject.
        return Ok(());
    }
    let rewritten = lowering.rewrite_statements(ast, roots, None)?;
    *roots = rewritten;

    for (name, types) in std::mem::take(&mut lowering.structs) {
        let field_types: Vec<Type> = ast
            .return_values_in(types)
            .iter()
            .map(|held| held.value_type.clone())
            .collect();
        let mut fields = Vec::with_capacity(field_types.len());
        for (index, field_type) in field_types.into_iter().enumerate() {
            let field_name = ast.multi_return_field_name(types, index);
            fields.push(StructField {
                name: ast.intern(&field_name),
                field_type,
            });
        }
        let fields = ast.add_struct_fields(fields);
        let name = ast.intern(&name);
        roots.push(ast.push_stmt(
            Statement::Struct(name, Range32::EMPTY, fields),
            TokenSpan::NONE,
        ));
    }
    Ok(())
}

struct Lowering {
    // Every function that returns a type list, by name.
    signatures: HashMap<String, Range32>,
    structs: BTreeMap<String, Range32>,
    counter: usize,
}

impl Lowering {
    fn collect_signatures(
        &mut self,
        ast: &Ast,
        statements: &[StmtId],
    ) -> Result<()> {
        for statement in statements {
            let (name, parameters, signature) = match ast.stmt(*statement) {
                Statement::Constant(name, value) => match ast.expr(*value) {
                    Expression::Function(parameters, signature, _)
                    | Expression::Proc(parameters, signature, _) => {
                        (*name, Some(*parameters), *signature)
                    }
                    _ => continue,
                },
                Statement::Declared {
                    name, return_sig, ..
                } => (*name, None, *return_sig),
                _ => continue,
            };
            let ReturnKind::Multiple(types) = &ast.signature(signature).kind
            else {
                continue;
            };
            let types = *types;
            if let Some(parameters) = parameters
                && ast
                    .params_in(parameters)
                    .iter()
                    .any(|parameter| ast.name(parameter.name).starts_with('$'))
            {
                let name = ast.name(name);
                bail!(
                    "'{name}' takes a compile-time argument and returns a type list, and a return type list is one struct rather than one per specialization; return a named struct instead"
                );
            }
            self.signatures.insert(ast.name(name).to_string(), types);
        }
        Ok(())
    }

    fn struct_for(&mut self, ast: &Ast, types: Range32) -> String {
        let name = ast.multi_return_struct_name(types);
        self.structs.entry(name.clone()).or_insert(types);
        name
    }

    fn rewrite_statements(
        &mut self,
        ast: &mut Ast,
        statements: &[StmtId],
        returns: Option<Range32>,
    ) -> Result<Vec<StmtId>> {
        let mut rewritten: Vec<StmtId> = Vec::with_capacity(statements.len());
        for statement in statements {
            match ast.stmt(*statement) {
                Statement::LetMultiple(bindings, value) => {
                    let (bindings, value) = (*bindings, *value);
                    rewritten.extend(
                        self.expand_binding(ast, *statement, bindings, value)?,
                    );
                }
                _ => {
                    // Located here rather than inside, because this is where
                    // the statement and its position are both in hand.
                    let position = ast.stmt_position(*statement);
                    crate::source_map::locate(
                        self.rewrite_statement(ast, *statement, returns),
                        position,
                    )?;
                    rewritten.push(*statement);
                }
            }
        }
        Ok(rewritten)
    }

    fn rewrite_block(
        &mut self,
        ast: &mut Ast,
        block: Range32,
        returns: Option<Range32>,
    ) -> Result<Range32> {
        let statements: Vec<StmtId> = ast.stmts_in(block).to_vec();
        let rewritten = self.rewrite_statements(ast, &statements, returns)?;
        if rewritten == statements {
            Ok(block)
        } else {
            Ok(ast.add_stmt_list(&rewritten))
        }
    }

    fn rewrite_statement(
        &mut self,
        ast: &mut Ast,
        statement: StmtId,
        returns: Option<Range32>,
    ) -> Result<()> {
        match ast.stmt(statement).clone() {
            Statement::LetMultiple(..) => {
                unreachable!("a list binding is expanded by rewrite_statements")
            }
            Statement::Constant(name, value) => {
                if let Expression::Function(_, signature, body)
                | Expression::Proc(_, signature, body) = ast.expr(value)
                {
                    let (signature, body) = (*signature, *body);
                    let types = self.signatures.get(ast.name(name)).copied();
                    if let Some(types) = types {
                        let struct_name = self.struct_for(ast, types);
                        ast.signatures[signature.0 as usize].kind =
                            ReturnKind::Single(Type::Struct(struct_name));
                    }
                    let body = self.rewrite_block(ast, body, types)?;
                    let (Expression::Function(_, _, held)
                    | Expression::Proc(_, _, held)) =
                        &mut ast.expressions[value.0 as usize]
                    else {
                        return Ok(());
                    };
                    *held = body;
                    return Ok(());
                }
                self.check_expression(ast, value, None)
            }
            Statement::Declared {
                name, return_sig, ..
            } => {
                if let Some(types) =
                    self.signatures.get(ast.name(name)).copied()
                {
                    let struct_name = self.struct_for(ast, types);
                    ast.signatures[return_sig.0 as usize].kind =
                        ReturnKind::Single(Type::Struct(struct_name));
                }
                Ok(())
            }
            Statement::Return(value) => {
                if let Expression::Tuple(values) = ast.expr(value)
                    && !values.is_empty()
                {
                    let values = *values;
                    let Some(types) = returns else {
                        bail!(
                            "this `return` lists {} values and the function returns one; write a return type list to return several",
                            values.len()
                        );
                    };
                    if values.len() != types.len() {
                        bail!(
                            "this `return` lists {} values and the function returns {}",
                            values.len(),
                            types.len()
                        );
                    }
                    let elements = ast.exprs_in(values).to_vec();
                    for held in &elements {
                        self.check_expression(ast, *held, returns)?;
                    }
                    let struct_name = self.struct_for(ast, types);
                    let mut fields = Vec::with_capacity(elements.len());
                    for (index, held) in elements.into_iter().enumerate() {
                        let field_name =
                            ast.multi_return_field_name(types, index);
                        fields.push(NamedExpr {
                            name: ast.intern(&field_name),
                            value: held,
                        });
                    }
                    let fields = ast.add_named_exprs(&fields);
                    let name = ast.intern(&struct_name);
                    ast.expressions[value.0 as usize] =
                        Expression::StructInit(name, fields);
                    return Ok(());
                }
                // `return { quotient = a / b, remainder = a % b }`: the values
                // by the names the signature gave them, which is the same
                // struct the list form builds and reads at the return site the
                // way the signature reads at the definition.
                if let Some(types) = returns
                    && let Expression::StructInit(name, fields) =
                        ast.expr(value)
                    && ast.name(*name).is_empty()
                {
                    let fields = *fields;
                    if ast
                        .return_values_in(types)
                        .iter()
                        .any(|held| held.name.is_none())
                    {
                        bail!(
                            "this `return` names its values and the signature does not; name them there too, or list the values in order"
                        );
                    }
                    for field in ast.named_in(fields).to_vec() {
                        self.check_expression(ast, field.value, returns)?;
                    }
                    let struct_name = self.struct_for(ast, types);
                    let name = ast.intern(&struct_name);
                    ast.expressions[value.0 as usize] =
                        Expression::StructInit(name, fields);
                    return Ok(());
                }
                if let Some(types) = returns {
                    bail!(
                        "this function returns {} values, so its `return` lists them or names them",
                        types.len()
                    );
                }
                self.check_expression(ast, value, returns)
            }
            Statement::Let { value, .. } | Statement::Expression(value) => {
                self.check_expression(ast, value, returns)
            }
            Statement::Print(value, arguments) => {
                self.check_expression(ast, value, returns)?;
                for argument in ast.exprs_in(arguments).to_vec() {
                    self.check_expression(ast, argument, returns)?;
                }
                Ok(())
            }
            Statement::Assignment(place, value) => {
                self.check_expression(ast, place, returns)?;
                self.check_expression(ast, value, returns)
            }
            Statement::Defer(inner) => {
                self.rewrite_statement(ast, inner, returns)
            }
            Statement::For(_, _, sequence, body) => {
                self.check_expression(ast, sequence, returns)?;
                let body = self.rewrite_block(ast, body, returns)?;
                let Statement::For(_, _, _, held) =
                    &mut ast.statements[statement.0 as usize]
                else {
                    return Ok(());
                };
                *held = body;
                Ok(())
            }
            Statement::While(condition, body) => {
                self.check_expression(ast, condition, returns)?;
                let body = self.rewrite_block(ast, body, returns)?;
                let Statement::While(_, held) =
                    &mut ast.statements[statement.0 as usize]
                else {
                    return Ok(());
                };
                *held = body;
                Ok(())
            }
            Statement::With(_, body) => {
                let body = self.rewrite_block(ast, body, returns)?;
                let Statement::With(_, held) =
                    &mut ast.statements[statement.0 as usize]
                else {
                    return Ok(());
                };
                *held = body;
                Ok(())
            }
            Statement::Struct(..)
            | Statement::Enum(..)
            | Statement::Flags(..)
            | Statement::TypeAlias(..)
            | Statement::Break
            | Statement::Continue
            | Statement::Import(..)
            | Statement::Extern { .. } => Ok(()),
        }
    }

    // `quotient, remainder := divide(a, b)` becomes the call bound to a
    // temporary and one field read per name.
    fn expand_binding(
        &mut self,
        ast: &mut Ast,
        statement: StmtId,
        bindings: Range32,
        value: ExprId,
    ) -> Result<Vec<StmtId>> {
        let Some(types) = self.called_signature(ast, value) else {
            bail!(
                "a list of names binds the values of a call to a function whose return signature is a type list"
            );
        };
        if bindings.len() != types.len() {
            bail!(
                "this binding names {} values and the call returns {}",
                bindings.len(),
                types.len()
            );
        }
        let arguments = match ast.expr(value) {
            Expression::Call(_, arguments) => Some(*arguments),
            _ => None,
        };
        if let Some(arguments) = arguments {
            for argument in ast.exprs_in(arguments).to_vec() {
                self.check_expression(ast, argument, None)?;
            }
        }

        let span = ast.stmt_span(statement);
        let temporary = format!("__multi_result{}", self.counter);
        self.counter += 1;
        let temporary = ast.intern(&temporary);
        let mut expanded = vec![ast.push_stmt(
            Statement::Let {
                name: temporary,
                type_annotation: None,
                value,
                mutable: false,
            },
            span,
        )];
        let bindings = ast.bindings_in(bindings).to_vec();
        for (index, binding) in bindings.into_iter().enumerate() {
            let field_name = ast.multi_return_field_name(types, index);
            let field = ast.intern(&field_name);
            let base = ast.push_expr(Expression::Identifier(temporary), span);
            let access =
                ast.push_expr(Expression::FieldAccess(base, field), span);
            expanded.push(ast.push_stmt(
                Statement::Let {
                    name: binding.name,
                    type_annotation: None,
                    value: access,
                    mutable: binding.mutable,
                },
                span,
            ));
        }
        Ok(expanded)
    }

    // The return type list of the function a call names, if it has one.
    fn called_signature(&self, ast: &Ast, value: ExprId) -> Option<Range32> {
        let Expression::Call(callee, _) = ast.expr(value) else {
            return None;
        };
        let Expression::Identifier(name) = ast.expr(*callee) else {
            return None;
        };
        self.signatures.get(ast.name(*name)).copied()
    }

    // A call that returns several values is bound by a list of names and used
    // nowhere else, since the struct behind the list is not a type anyone can
    // write.
    fn check_expression(
        &mut self,
        ast: &mut Ast,
        expression: ExprId,
        returns: Option<Range32>,
    ) -> Result<()> {
        match ast.expr(expression).clone() {
            Expression::Call(callee, arguments) => {
                if let Expression::Identifier(name) = ast.expr(callee)
                    && let Some(types) =
                        self.signatures.get(ast.name(*name)).copied()
                {
                    let name = ast.name(*name);
                    bail!(
                        "'{name}' returns {} values, so its call is bound by a list of names",
                        types.len()
                    );
                }
                self.check_expression(ast, callee, returns)?;
                for argument in ast.exprs_in(arguments).to_vec() {
                    self.check_expression(ast, argument, returns)?;
                }
                Ok(())
            }
            Expression::Function(_, signature, body)
            | Expression::Proc(_, signature, body) => {
                let types = match &ast.signature(signature).kind {
                    ReturnKind::Multiple(values) => Some(*values),
                    _ => None,
                };
                if let Some(types) = types {
                    let struct_name = self.struct_for(ast, types);
                    ast.signatures[signature.0 as usize].kind =
                        ReturnKind::Single(Type::Struct(struct_name));
                }
                let body = self.rewrite_block(ast, body, types)?;
                let (Expression::Function(_, _, held)
                | Expression::Proc(_, _, held)) =
                    &mut ast.expressions[expression.0 as usize]
                else {
                    return Ok(());
                };
                *held = body;
                Ok(())
            }
            Expression::PackMap(inner, _, _)
            | Expression::Prefix(_, inner)
            | Expression::AddressOf(inner)
            | Expression::Borrow(inner)
            | Expression::BorrowMut(inner)
            | Expression::Dereference(inner)
            | Expression::UnsafeFn(inner)
            | Expression::Try(inner) => {
                self.check_expression(ast, inner, returns)
            }
            Expression::Infix(left, _, right)
            | Expression::Index(left, right)
            | Expression::Range(left, right, _) => {
                self.check_expression(ast, left, returns)?;
                self.check_expression(ast, right, returns)
            }
            Expression::FieldAccess(inner, _) => {
                self.check_expression(ast, inner, returns)
            }
            Expression::If(condition, then_block, else_block) => {
                self.check_expression(ast, condition, returns)?;
                let then_block =
                    self.rewrite_block(ast, then_block, returns)?;
                let else_block = match else_block {
                    Some(block) => {
                        Some(self.rewrite_block(ast, block, returns)?)
                    }
                    None => None,
                };
                let Expression::If(_, held_then, held_else) =
                    &mut ast.expressions[expression.0 as usize]
                else {
                    return Ok(());
                };
                *held_then = then_block;
                *held_else = else_block;
                Ok(())
            }
            Expression::Unsafe(block) => {
                let block = self.rewrite_block(ast, block, returns)?;
                let Expression::Unsafe(held) =
                    &mut ast.expressions[expression.0 as usize]
                else {
                    return Ok(());
                };
                *held = block;
                Ok(())
            }
            Expression::Switch(scrutinee, cases) => {
                self.check_expression(ast, scrutinee, returns)?;
                for index in cases.indices() {
                    let body = ast.cases[index].body;
                    let body = self.rewrite_block(ast, body, returns)?;
                    ast.cases[index].body = body;
                }
                Ok(())
            }
            Expression::Tuple(values) => {
                for held in ast.exprs_in(values).to_vec() {
                    self.check_expression(ast, held, returns)?;
                }
                Ok(())
            }
            Expression::StructInit(_, fields)
            | Expression::EnumVariantInit(_, _, fields) => {
                for field in ast.named_in(fields).to_vec() {
                    self.check_expression(ast, field.value, returns)?;
                }
                Ok(())
            }
            Expression::ArrayRepeat(value, _) => {
                self.check_expression(ast, value, returns)
            }
            Expression::Identifier(_)
            | Expression::Literal(_)
            | Expression::Boolean(_)
            | Expression::TypeValue(_) => Ok(()),
        }
    }
}
