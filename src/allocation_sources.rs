use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::parser::{
    Block, Expression, ParamMode, Parameter, Program, Spanned, Statement,
};
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
    fn expression(&self) -> Expression {
        let name = Expression::Identifier(self.name.clone());
        if self.borrow {
            return Expression::BorrowMut(Box::new(name));
        }
        name
    }
}

pub fn lower_allocation_sources(program: &mut Program) -> Result<()> {
    let mut uses_functions: HashMap<String, Vec<Type>> = HashMap::new();

    // First pass. Give every `uses` function one implicit capability parameter
    // per source it draws, in the order they were declared.
    for statement in program.iter_mut() {
        if let Statement::Constant(
            _,
            Expression::Function(parameters, signature, _)
            | Expression::Proc(parameters, signature, _),
        ) = &mut statement.node
            && !signature.uses.is_empty()
        {
            let capabilities = signature.uses.clone();
            for capability in &capabilities {
                parameters.push(Parameter {
                    name: capability_binding(capability),
                    type_annotation: Some(capability.clone()),
                    mutable: true,
                    mode: ParamMode::Write,
                    compile_time_signature: None,
                    pack: false,
                });
            }
            signature.uses.clear();
            if let Statement::Constant(name, _) = &statement.node {
                uses_functions.insert(name.clone(), capabilities);
            }
        }
    }

    // Second pass. Thread the capability argument through calls and inline the
    // `with` blocks that provide it.
    let threader = Threader { uses_functions };
    for statement in program.iter_mut() {
        if let Statement::Constant(
            name,
            Expression::Function(_, _, body) | Expression::Proc(_, _, body),
        ) = &mut statement.node
        {
            let mut provider = Provider::default();
            if let Some(capabilities) = threader.uses_functions.get(name) {
                for capability in capabilities {
                    provider = provider
                        .extended(capability_binding(capability), false);
                }
            }
            let taken = std::mem::take(body);
            *body = threader.thread_block(taken, &provider)?;
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
    fn thread_block(&self, block: Block, provider: &Provider) -> Result<Block> {
        let mut threaded = Vec::with_capacity(block.len());
        for statement in block {
            if let Statement::With(capability, body) = statement.node {
                // The block is a region. Inline it with the arena it names
                // added to what a call inside it may draw from.
                let inner = self
                    .thread_block(body, &provider.extended(capability, true))?;
                threaded.extend(inner);
            } else {
                let position = statement.position;
                let node = self.thread_statement(statement.node, provider)?;
                threaded.push(Spanned { node, position });
            }
        }
        Ok(threaded)
    }

    fn thread_statement(
        &self,
        statement: Statement,
        provider: &Provider,
    ) -> Result<Statement> {
        let threaded = match statement {
            Statement::Let {
                name,
                type_annotation,
                value,
                mutable,
            } => Statement::Let {
                name,
                type_annotation,
                value: self.thread_expression(value, provider)?,
                mutable,
            },
            Statement::Constant(name, value) => Statement::Constant(
                name,
                self.thread_expression(value, provider)?,
            ),
            Statement::Return(value) => {
                Statement::Return(self.thread_expression(value, provider)?)
            }
            Statement::Expression(value) => {
                Statement::Expression(self.thread_expression(value, provider)?)
            }
            Statement::Assignment(place, value) => Statement::Assignment(
                self.thread_expression(place, provider)?,
                self.thread_expression(value, provider)?,
            ),
            Statement::Defer(inner) => Statement::Defer(Box::new(
                self.thread_statement(*inner, provider)?,
            )),
            Statement::For(variable, second, iterable, body) => Statement::For(
                variable,
                second,
                self.thread_expression(iterable, provider)?,
                self.thread_block(body, provider)?,
            ),
            Statement::While(condition, body) => Statement::While(
                self.thread_expression(condition, provider)?,
                self.thread_block(body, provider)?,
            ),
            Statement::With(..) => {
                unreachable!("`with` is inlined by thread_block")
            }
            other => other,
        };
        Ok(threaded)
    }

    fn thread_expression(
        &self,
        expression: Expression,
        provider: &Provider,
    ) -> Result<Expression> {
        let threaded = match expression {
            Expression::Call(callee, arguments) => {
                let callee = self.thread_expression(*callee, provider)?;
                let mut lowered = Vec::with_capacity(arguments.len() + 1);
                for argument in arguments {
                    lowered.push(self.thread_expression(argument, provider)?);
                }
                if let Expression::Identifier(name) = &callee
                    && self.uses_functions.contains_key(name)
                {
                    lowered.extend(self.capability_arguments(name, provider)?);
                }
                Expression::Call(Box::new(callee), lowered)
            }
            Expression::Try(inner) => Expression::Try(Box::new(
                self.thread_expression(*inner, provider)?,
            )),
            Expression::Prefix(operator, inner) => Expression::Prefix(
                operator,
                Box::new(self.thread_expression(*inner, provider)?),
            ),
            Expression::AddressOf(inner) => Expression::AddressOf(Box::new(
                self.thread_expression(*inner, provider)?,
            )),
            Expression::Borrow(inner) => Expression::Borrow(Box::new(
                self.thread_expression(*inner, provider)?,
            )),
            Expression::BorrowMut(inner) => Expression::BorrowMut(Box::new(
                self.thread_expression(*inner, provider)?,
            )),
            Expression::Dereference(inner) => Expression::Dereference(
                Box::new(self.thread_expression(*inner, provider)?),
            ),
            Expression::FieldAccess(base, field) => Expression::FieldAccess(
                Box::new(self.thread_expression(*base, provider)?),
                field,
            ),
            Expression::Infix(left, operator, right) => Expression::Infix(
                Box::new(self.thread_expression(*left, provider)?),
                operator,
                Box::new(self.thread_expression(*right, provider)?),
            ),
            Expression::Index(base, index) => Expression::Index(
                Box::new(self.thread_expression(*base, provider)?),
                Box::new(self.thread_expression(*index, provider)?),
            ),
            Expression::Range(start, end, inclusive) => Expression::Range(
                Box::new(self.thread_expression(*start, provider)?),
                Box::new(self.thread_expression(*end, provider)?),
                inclusive,
            ),
            Expression::If(condition, consequence, alternative) => {
                Expression::If(
                    Box::new(self.thread_expression(*condition, provider)?),
                    self.thread_block(consequence, provider)?,
                    match alternative {
                        Some(block) => {
                            Some(self.thread_block(block, provider)?)
                        }
                        None => None,
                    },
                )
            }
            Expression::StructInit(name, fields) => Expression::StructInit(
                name,
                self.thread_fields(fields, provider)?,
            ),
            Expression::EnumVariantInit(name, variant, fields) => {
                Expression::EnumVariantInit(
                    name,
                    variant,
                    self.thread_fields(fields, provider)?,
                )
            }
            Expression::Switch(scrutinee, cases) => {
                let scrutinee =
                    Box::new(self.thread_expression(*scrutinee, provider)?);
                let mut threaded = Vec::with_capacity(cases.len());
                for case in cases {
                    threaded.push(crate::parser::SwitchCase {
                        pattern: case.pattern,
                        body: self.thread_block(case.body, provider)?,
                    });
                }
                Expression::Switch(scrutinee, threaded)
            }
            Expression::Tuple(items) => {
                let mut threaded = Vec::with_capacity(items.len());
                for item in items {
                    threaded.push(self.thread_expression(item, provider)?);
                }
                Expression::Tuple(threaded)
            }
            Expression::Unsafe(body) => {
                Expression::Unsafe(self.thread_block(body, provider)?)
            }
            // A nested function literal cannot see the enclosing capability, so
            // its body threads with no provider of its own.
            Expression::Function(parameters, signature, body) => {
                Expression::Function(
                    parameters,
                    signature,
                    self.thread_block(body, &Provider::default())?,
                )
            }
            Expression::Proc(parameters, signature, body) => Expression::Proc(
                parameters,
                signature,
                self.thread_block(body, &Provider::default())?,
            ),
            other => other,
        };
        Ok(threaded)
    }

    fn thread_fields(
        &self,
        fields: Vec<(String, Expression)>,
        provider: &Provider,
    ) -> Result<Vec<(String, Expression)>> {
        let mut threaded = Vec::with_capacity(fields.len());
        for (name, value) in fields {
            threaded.push((name, self.thread_expression(value, provider)?));
        }
        Ok(threaded)
    }

    // One argument per source the callee draws, chosen by the name the
    // capability is reached by. A callee drawing a single source takes whatever
    // is innermost whatever it is called, which is what lets a `with scratch`
    // block supply a `uses Arena`. A callee drawing several has to tell them
    // apart, and the name is what tells them apart.
    fn capability_arguments(
        &self,
        callee: &str,
        provider: &Provider,
    ) -> Result<Vec<Expression>> {
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
            arguments.push(source.expression());
        }
        Ok(arguments)
    }
}
