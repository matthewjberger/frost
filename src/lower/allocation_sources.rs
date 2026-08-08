use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::ast::{
    Ast, ExprId, Expression, Parameter, Range32, Statement, StmtId, TokenSpan,
};
use crate::parser::ParamMode;
use crate::types::Type;

// Lowers allocation sources. A `uses A` function draws an allocation capability
// of type A. The capability is an implicit `&mut A` parameter, in scope under the
// type's name (first letter lowercased), threaded automatically. A call from
// another `uses A` function forwards the caller's capability, and a call inside a
// `with arena { ... }` block is supplied `&mut arena`. A call from neither is
// rejected. After this pass a `uses A` function is an ordinary function with a
// trailing `&mut A` parameter, and `with` blocks are plain scopes.

// What a `uses` call may draw from at this point: the capability parameters the
// enclosing function holds, and the arenas the `with` blocks around the call
// provide, innermost last. A function may draw more than one, so this is a set
// rather than a single answer.
#[derive(Default, Clone)]
struct Provider {
    sources: Vec<Source>,
}

#[derive(Clone)]
struct Source {
    // The name the capability is reached by: the type's name with its first
    // letter lowercased for a parameter, and the written name for a `with`
    // block.
    name: String,
    // A `with` block names a variable the caller owns, so the call takes its
    // address. A forwarded capability parameter is already a reference.
    borrow: bool,
}

impl Provider {
    fn extended(&self, name: String, borrow: bool) -> Provider {
        let mut sources = self.sources.clone();
        sources.push(Source { name, borrow });
        Provider { sources }
    }

    fn named(&self, wanted: &str) -> Option<&Source> {
        self.sources
            .iter()
            .rev()
            .find(|source| source.name == wanted)
    }

    fn innermost(&self) -> Option<&Source> {
        self.sources.last()
    }
}

impl Source {
    fn expression(&self, ast: &mut Ast, span: TokenSpan) -> ExprId {
        let symbol = ast.intern(&self.name);
        let name = ast.push_expr(Expression::Identifier(symbol), span);
        if self.borrow {
            return ast.push_expr(Expression::BorrowMut(name), span);
        }
        name
    }
}

pub fn lower_allocation_sources(ast: &mut Ast, roots: &[StmtId]) -> Result<()> {
    let mut uses_functions: HashMap<String, Vec<Type>> = HashMap::new();

    // First pass. Give every `uses` function one implicit capability parameter
    // per source it draws, in the order they were declared.
    for statement in roots {
        let Statement::Constant(name, value) = ast.stmt(*statement) else {
            continue;
        };
        let (name, value) = (*name, *value);
        let (params, signature, body, function) = match ast.expr(value) {
            Expression::Function(params, signature, body) => {
                (*params, *signature, *body, true)
            }
            Expression::Proc(params, signature, body) => {
                (*params, *signature, *body, false)
            }
            _ => continue,
        };
        if ast.signature(signature).uses.is_empty() {
            continue;
        }
        let capabilities = ast.signature(signature).uses.clone();
        let mut parameters = ast.params_in(params).to_vec();
        for capability in &capabilities {
            let binding = ast.intern(&capability_binding(capability));
            parameters.push(Parameter {
                name: binding,
                type_annotation: Some(capability.clone()),
                mutable: true,
                mode: ParamMode::Write,
                compile_time_signature: None,
                pack: false,
                format: false,
            });
        }
        let widened = ast.add_parameters(parameters);
        ast.signatures[signature.0 as usize].uses.clear();
        ast.expressions[value.0 as usize] = if function {
            Expression::Function(widened, signature, body)
        } else {
            Expression::Proc(widened, signature, body)
        };
        uses_functions.insert(ast.name(name).to_string(), capabilities);
    }

    // Second pass. Thread the capability argument through calls and inline the
    // `with` blocks that provide it.
    let threader = Threader { uses_functions };
    for statement in roots {
        let Statement::Constant(name, value) = ast.stmt(*statement) else {
            continue;
        };
        let (name, value) = (*name, *value);
        let (params, signature, body, function) = match ast.expr(value) {
            Expression::Function(params, signature, body) => {
                (*params, *signature, *body, true)
            }
            Expression::Proc(params, signature, body) => {
                (*params, *signature, *body, false)
            }
            _ => continue,
        };
        let mut provider = Provider::default();
        if let Some(capabilities) = threader.uses_functions.get(ast.name(name))
        {
            for capability in capabilities {
                provider =
                    provider.extended(capability_binding(capability), false);
            }
        }
        let threaded = threader.thread_block(ast, body, &provider)?;
        if threaded != body {
            ast.expressions[value.0 as usize] = if function {
                Expression::Function(params, signature, threaded)
            } else {
                Expression::Proc(params, signature, threaded)
            };
        }
    }

    Ok(())
}

// The capability variable name for a type: the type's base name with its first
// letter lowercased, so `Arena<256>` binds `arena`.
fn capability_binding(capability: &Type) -> String {
    let name = match capability {
        Type::Struct(name) | Type::Enum(name) => name.clone(),
        other => other.to_string(),
    };
    let base = name.split('<').next().unwrap_or(&name);
    let mut characters = base.chars();
    match characters.next() {
        Some(first) => {
            first.to_lowercase().collect::<String>() + characters.as_str()
        }
        None => base.to_string(),
    }
}

struct Threader {
    uses_functions: HashMap<String, Vec<Type>>,
}

impl Threader {
    fn thread_block(
        &self,
        ast: &mut Ast,
        block: Range32,
        provider: &Provider,
    ) -> Result<Range32> {
        let statements = ast.stmts_in(block).to_vec();
        let mut threaded = Vec::with_capacity(statements.len());
        let mut inlined = false;
        for statement in statements {
            if let Statement::With(capability, body) = ast.stmt(statement) {
                // The block is a region. Inline it with the arena it names
                // added to what a call inside it may draw from.
                let (capability, body) = (*capability, *body);
                let extended =
                    provider.extended(ast.name(capability).to_string(), true);
                let inner = self.thread_block(ast, body, &extended)?;
                threaded.extend_from_slice(ast.stmts_in(inner));
                inlined = true;
            } else {
                self.thread_statement(ast, statement, provider)?;
                threaded.push(statement);
            }
        }
        if inlined {
            Ok(ast.add_stmt_list(&threaded))
        } else {
            Ok(block)
        }
    }

    fn thread_statement(
        &self,
        ast: &mut Ast,
        statement: StmtId,
        provider: &Provider,
    ) -> Result<()> {
        match ast.stmt(statement).clone() {
            Statement::Let { value, .. }
            | Statement::Constant(_, value)
            | Statement::Return(value)
            | Statement::Expression(value) => {
                self.thread_expression(ast, value, provider)?;
            }
            Statement::Assignment(place, value) => {
                self.thread_expression(ast, place, provider)?;
                self.thread_expression(ast, value, provider)?;
            }
            Statement::Defer(inner) | Statement::ErrDefer(inner) => {
                self.thread_statement(ast, inner, provider)?;
            }
            Statement::For(variable, second, iterable, body) => {
                self.thread_expression(ast, iterable, provider)?;
                let threaded = self.thread_block(ast, body, provider)?;
                if threaded != body {
                    ast.statements[statement.0 as usize] =
                        Statement::For(variable, second, iterable, threaded);
                }
            }
            Statement::While(condition, body) => {
                self.thread_expression(ast, condition, provider)?;
                let threaded = self.thread_block(ast, body, provider)?;
                if threaded != body {
                    ast.statements[statement.0 as usize] =
                        Statement::While(condition, threaded);
                }
            }
            Statement::With(..) => {
                unreachable!("`with` is inlined by thread_block")
            }
            _ => {}
        }
        Ok(())
    }

    fn thread_expression(
        &self,
        ast: &mut Ast,
        expression: ExprId,
        provider: &Provider,
    ) -> Result<()> {
        match ast.expr(expression).clone() {
            Expression::Call(callee, arguments) => {
                // A bare name in callee position is the call's own, and the
                // arm below refuses one that draws a capability. What is
                // refused there is a function taken as a value, which a callee
                // is not.
                if !matches!(ast.expr(callee), Expression::Identifier(_)) {
                    self.thread_expression(ast, callee, provider)?;
                }
                for argument in ast.exprs_in(arguments).to_vec() {
                    self.thread_expression(ast, argument, provider)?;
                }
                if let Expression::Identifier(name) = ast.expr(callee) {
                    let name = ast.name(*name).to_string();
                    if self.uses_functions.contains_key(&name) {
                        let span = ast.expr_span(expression);
                        let extra = self
                            .capability_arguments(ast, &name, provider, span)?;
                        let mut lowered = ast.exprs_in(arguments).to_vec();
                        lowered.extend(extra);
                        let widened = ast.add_expr_list(&lowered);
                        ast.expressions[expression.0 as usize] =
                            Expression::Call(callee, widened);
                    }
                }
            }
            // A function that draws a capability has one more parameter than
            // its signature was written with, and this pass is what fills it,
            // at each call. A function taken as a value has no call to fill it
            // at: the address goes somewhere that will call it through a type
            // saying nothing about the capability, so the callee reads the
            // register nobody wrote and the first use of the arena writes
            // through it. That built and faulted with no `unsafe` anywhere.
            Expression::Identifier(name) => {
                let name = ast.name(name).to_string();
                if self.uses_functions.contains_key(&name) {
                    return Err(anyhow::Error::new(
                        crate::diagnostic::LocatedError {
                            position: ast.expr_position(expression),
                            message: format!(
                                "'{name}' draws a capability, which is one more parameter, so it cannot be taken as a value: a call through a function value supplies what its type says and nothing else"
                            ),
                        },
                    ));
                }
            }
            Expression::Try(inner)
            | Expression::Prefix(_, inner)
            | Expression::AddressOf(inner)
            | Expression::Borrow(inner)
            | Expression::BorrowMut(inner)
            | Expression::Dereference(inner)
            | Expression::FieldAccess(inner, _) => {
                self.thread_expression(ast, inner, provider)?;
            }
            Expression::Infix(left, _, right)
            | Expression::Index(left, right)
            | Expression::Range(left, right, _) => {
                self.thread_expression(ast, left, provider)?;
                self.thread_expression(ast, right, provider)?;
            }
            Expression::If(condition, consequence, alternative) => {
                self.thread_expression(ast, condition, provider)?;
                let threaded_consequence =
                    self.thread_block(ast, consequence, provider)?;
                let threaded_alternative = match alternative {
                    Some(block) => {
                        Some(self.thread_block(ast, block, provider)?)
                    }
                    None => None,
                };
                if threaded_consequence != consequence
                    || threaded_alternative != alternative
                {
                    ast.expressions[expression.0 as usize] = Expression::If(
                        condition,
                        threaded_consequence,
                        threaded_alternative,
                    );
                }
            }
            Expression::StructInit(_, fields)
            | Expression::EnumVariantInit(_, _, fields) => {
                for field in ast.named_in(fields).to_vec() {
                    self.thread_expression(ast, field.value, provider)?;
                }
            }
            Expression::Switch(scrutinee, cases) => {
                self.thread_expression(ast, scrutinee, provider)?;
                for index in cases.indices() {
                    let body = ast.cases[index].body;
                    let threaded = self.thread_block(ast, body, provider)?;
                    if threaded != body {
                        ast.cases[index].body = threaded;
                    }
                }
            }
            Expression::Tuple(items) => {
                for item in ast.exprs_in(items).to_vec() {
                    self.thread_expression(ast, item, provider)?;
                }
            }
            Expression::Unsafe(body) => {
                let threaded = self.thread_block(ast, body, provider)?;
                if threaded != body {
                    ast.expressions[expression.0 as usize] =
                        Expression::Unsafe(threaded);
                }
            }
            // A nested function literal cannot see the enclosing capability, so
            // its body threads with no provider of its own.
            Expression::Function(parameters, signature, body) => {
                let threaded =
                    self.thread_block(ast, body, &Provider::default())?;
                if threaded != body {
                    ast.expressions[expression.0 as usize] =
                        Expression::Function(parameters, signature, threaded);
                }
            }
            Expression::Proc(parameters, signature, body) => {
                let threaded =
                    self.thread_block(ast, body, &Provider::default())?;
                if threaded != body {
                    ast.expressions[expression.0 as usize] =
                        Expression::Proc(parameters, signature, threaded);
                }
            }
            _ => {}
        }
        Ok(())
    }

    // One argument per source the callee draws, chosen by the name the
    // capability is reached by. A callee drawing a single source takes whatever
    // is innermost whatever it is called, which is what lets a `with scratch`
    // block supply a `uses Arena`. A callee drawing several has to tell them
    // apart, and the name is what tells them apart.
    fn capability_arguments(
        &self,
        ast: &mut Ast,
        callee: &str,
        provider: &Provider,
        span: TokenSpan,
    ) -> Result<Vec<ExprId>> {
        let capabilities = self.uses_functions.get(callee).unwrap();
        let mut arguments = Vec::with_capacity(capabilities.len());
        for capability in capabilities {
            let wanted = capability_binding(capability);
            let mut source = provider.named(&wanted);
            if source.is_none() && capabilities.len() == 1 {
                source = provider.innermost();
            }
            let Some(source) = source else {
                if provider.sources.is_empty() {
                    bail!(
                        "calling '{callee}' needs an allocation capability; declare `uses {capability}` on the caller or wrap the call in a `with` block"
                    )
                }
                let available: Vec<&str> = provider
                    .sources
                    .iter()
                    .map(|source| source.name.as_str())
                    .collect();
                bail!(
                    "calling '{callee}' needs the allocation capability '{wanted}' of type '{capability}', and what is in scope here is {}",
                    available.join(", ")
                )
            };
            arguments.push(source.expression(ast, span));
        }
        Ok(arguments)
    }
}
