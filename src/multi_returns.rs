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

use crate::parser::{
    Expression, MultiBinding, ReturnKind, ReturnValue, Spanned, Statement,
    StructField, SwitchCase, multi_return_field_name, multi_return_struct_name,
};
use crate::types::Type;
use anyhow::{Result, bail};
use std::collections::{BTreeMap, HashMap};

pub fn lower_multiple_returns(
    statements: &mut Vec<Spanned<Statement>>,
) -> Result<()> {
    let mut lowering = Lowering {
        signatures: HashMap::new(),
        structs: BTreeMap::new(),
        counter: 0,
    };
    lowering.collect_signatures(statements)?;
    if lowering.signatures.is_empty() {
        // Nothing declares a return type list, so there is nothing to rewrite
        // and no call to reject.
        return Ok(());
    }
    lowering.rewrite_statements(statements, None)?;

    for (name, types) in std::mem::take(&mut lowering.structs) {
        let fields = types
            .iter()
            .enumerate()
            .map(|(index, held)| StructField {
                name: multi_return_field_name(&types, index),
                field_type: held.value_type.clone(),
            })
            .collect();
        statements.push(Spanned::new(
            Statement::Struct(name, Vec::new(), fields),
            Default::default(),
        ));
    }
    Ok(())
}

struct Lowering {
    // Every function that returns a type list, by name.
    signatures: HashMap<String, Vec<ReturnValue>>,
    structs: BTreeMap<String, Vec<ReturnValue>>,
    counter: usize,
}

impl Lowering {
    fn collect_signatures(
        &mut self,
        statements: &[Spanned<Statement>],
    ) -> Result<()> {
        for statement in statements {
            let (name, parameters, kind) = match &statement.node {
                Statement::Constant(
                    name,
                    Expression::Function(parameters, signature, _)
                    | Expression::Proc(parameters, signature, _),
                ) => (name, Some(parameters), &signature.kind),
                Statement::Declared {
                    name, return_sig, ..
                } => (name, None, &return_sig.kind),
                _ => continue,
            };
            let ReturnKind::Multiple(types) = kind else {
                continue;
            };
            if let Some(parameters) = parameters
                && parameters
                    .iter()
                    .any(|parameter| parameter.name.starts_with('$'))
            {
                bail!(
                    "'{name}' takes a compile-time argument and returns a type list, and a return type list is one struct rather than one per specialization; return a named struct instead"
                );
            }
            self.signatures.insert(name.clone(), types.clone());
        }
        Ok(())
    }

    fn struct_for(&mut self, types: &[ReturnValue]) -> String {
        let name = multi_return_struct_name(types);
        self.structs
            .entry(name.clone())
            .or_insert_with(|| types.to_vec());
        name
    }

    fn rewrite_statements(
        &mut self,
        statements: &mut Vec<Spanned<Statement>>,
        returns: Option<&[ReturnValue]>,
    ) -> Result<()> {
        let mut rewritten: Vec<Spanned<Statement>> =
            Vec::with_capacity(statements.len());
        for statement in std::mem::take(statements) {
            let position = statement.position;
            match statement.node {
                Statement::LetMultiple(bindings, value) => {
                    for expanded in self.expand_binding(bindings, value)? {
                        rewritten.push(Spanned::new(expanded, position));
                    }
                }
                other => {
                    let mut node = other;
                    self.rewrite_statement(&mut node, returns)?;
                    rewritten.push(Spanned::new(node, position));
                }
            }
        }
        *statements = rewritten;
        Ok(())
    }

    fn rewrite_statement(
        &mut self,
        statement: &mut Statement,
        returns: Option<&[ReturnValue]>,
    ) -> Result<()> {
        match statement {
            Statement::LetMultiple(..) => {
                unreachable!("a list binding is expanded by rewrite_statements")
            }
            Statement::Constant(name, value) => {
                if let Expression::Function(_, signature, body)
                | Expression::Proc(_, signature, body) = value
                {
                    let types = self.signatures.get(name).cloned();
                    if let Some(types) = &types {
                        let struct_name = self.struct_for(types);
                        signature.kind =
                            ReturnKind::Single(Type::Struct(struct_name));
                    }
                    return self.rewrite_statements(body, types.as_deref());
                }
                self.check_expression(value, None)
            }
            Statement::Declared {
                name, return_sig, ..
            } => {
                if let Some(types) = self.signatures.get(name).cloned() {
                    let struct_name = self.struct_for(&types);
                    return_sig.kind =
                        ReturnKind::Single(Type::Struct(struct_name));
                }
                Ok(())
            }
            Statement::Return(value) => {
                if let Expression::Tuple(values) = value
                    && !values.is_empty()
                {
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
                    for held in values.iter_mut() {
                        self.check_expression(held, returns)?;
                    }
                    let struct_name = self.struct_for(types);
                    let fields = std::mem::take(values)
                        .into_iter()
                        .enumerate()
                        .map(|(index, held)| {
                            (multi_return_field_name(types, index), held)
                        })
                        .collect();
                    *value = Expression::StructInit(struct_name, fields);
                    return Ok(());
                }
                // `return { quotient = a / b, remainder = a % b }`: the values
                // by the names the signature gave them, which is the same
                // struct the list form builds and reads at the return site the
                // way the signature reads at the definition.
                if let Some(types) = returns
                    && let Expression::StructInit(name, fields) = value
                    && name.is_empty()
                {
                    if types.iter().any(|held| held.name.is_none()) {
                        bail!(
                            "this `return` names its values and the signature does not; name them there too, or list the values in order"
                        );
                    }
                    for (_, held) in fields.iter_mut() {
                        self.check_expression(held, returns)?;
                    }
                    let fields = std::mem::take(fields);
                    *name = self.struct_for(types);
                    *value = Expression::StructInit(name.clone(), fields);
                    return Ok(());
                }
                if let Some(types) = returns {
                    bail!(
                        "this function returns {} values, so its `return` lists them or names them",
                        types.len()
                    );
                }
                self.check_expression(value, returns)
            }
            Statement::Let { value, .. } | Statement::Expression(value) => {
                self.check_expression(value, returns)
            }
            Statement::Print(value, arguments) => {
                self.check_expression(value, returns)?;
                for argument in arguments {
                    self.check_expression(argument, returns)?;
                }
                Ok(())
            }
            Statement::Assignment(place, value) => {
                self.check_expression(place, returns)?;
                self.check_expression(value, returns)
            }
            Statement::Defer(inner) => self.rewrite_statement(inner, returns),
            Statement::For(_, _, sequence, body) => {
                self.check_expression(sequence, returns)?;
                self.rewrite_statements(body, returns)
            }
            Statement::While(condition, body) => {
                self.check_expression(condition, returns)?;
                self.rewrite_statements(body, returns)
            }
            Statement::With(_, body) => self.rewrite_statements(body, returns),
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
        bindings: Vec<MultiBinding>,
        mut value: Expression,
    ) -> Result<Vec<Statement>> {
        let Some(types) = self.called_signature(&value) else {
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
        if let Expression::Call(_, arguments) = &mut value {
            for argument in arguments.iter_mut() {
                self.check_expression(argument, None)?;
            }
        }

        let temporary = format!("__multi_result{}", self.counter);
        self.counter += 1;
        let mut expanded = vec![Statement::Let {
            name: temporary.clone(),
            type_annotation: None,
            value,
            mutable: false,
        }];
        for (index, binding) in bindings.into_iter().enumerate() {
            expanded.push(Statement::Let {
                name: binding.name,
                type_annotation: None,
                value: Expression::FieldAccess(
                    Box::new(Expression::Identifier(temporary.clone())),
                    multi_return_field_name(&types, index),
                ),
                mutable: binding.mutable,
            });
        }
        Ok(expanded)
    }

    // The return type list of the function a call names, if it has one.
    fn called_signature(&self, value: &Expression) -> Option<Vec<ReturnValue>> {
        let Expression::Call(callee, _) = value else {
            return None;
        };
        let Expression::Identifier(name) = callee.as_ref() else {
            return None;
        };
        self.signatures.get(name).cloned()
    }

    // A call that returns several values is bound by a list of names and used
    // nowhere else, since the struct behind the list is not a type anyone can
    // write.
    fn check_expression(
        &mut self,
        expression: &mut Expression,
        returns: Option<&[ReturnValue]>,
    ) -> Result<()> {
        match expression {
            Expression::Call(callee, arguments) => {
                if let Expression::Identifier(name) = callee.as_ref()
                    && let Some(types) = self.signatures.get(name)
                {
                    bail!(
                        "'{name}' returns {} values, so its call is bound by a list of names",
                        types.len()
                    );
                }
                self.check_expression(callee, returns)?;
                for argument in arguments {
                    self.check_expression(argument, returns)?;
                }
                Ok(())
            }
            Expression::Function(_, signature, body)
            | Expression::Proc(_, signature, body) => {
                if let ReturnKind::Multiple(types) = &signature.kind {
                    let types = types.clone();
                    let struct_name = self.struct_for(&types);
                    signature.kind =
                        ReturnKind::Single(Type::Struct(struct_name));
                    return self.rewrite_statements(body, Some(&types));
                }
                self.rewrite_statements(body, None)
            }
            Expression::PackMap(inner, _, _)
            | Expression::Prefix(_, inner)
            | Expression::AddressOf(inner)
            | Expression::Borrow(inner)
            | Expression::BorrowMut(inner)
            | Expression::Dereference(inner)
            | Expression::UnsafeFn(inner)
            | Expression::Try(inner) => self.check_expression(inner, returns),
            Expression::Infix(left, _, right)
            | Expression::Index(left, right)
            | Expression::Range(left, right, _) => {
                self.check_expression(left, returns)?;
                self.check_expression(right, returns)
            }
            Expression::FieldAccess(inner, _) => {
                self.check_expression(inner, returns)
            }
            Expression::If(condition, then_block, else_block) => {
                self.check_expression(condition, returns)?;
                self.rewrite_statements(then_block, returns)?;
                if let Some(block) = else_block {
                    self.rewrite_statements(block, returns)?;
                }
                Ok(())
            }
            Expression::Unsafe(block) => {
                self.rewrite_statements(block, returns)
            }
            Expression::Switch(scrutinee, cases) => {
                self.check_expression(scrutinee, returns)?;
                for SwitchCase { body, .. } in cases {
                    self.rewrite_statements(body, returns)?;
                }
                Ok(())
            }
            Expression::Tuple(values) => {
                for held in values {
                    self.check_expression(held, returns)?;
                }
                Ok(())
            }
            Expression::StructInit(_, fields)
            | Expression::EnumVariantInit(_, _, fields) => {
                for (_, held) in fields {
                    self.check_expression(held, returns)?;
                }
                Ok(())
            }
            Expression::ArrayRepeat(value, _) => {
                self.check_expression(value, returns)
            }
            Expression::Identifier(_)
            | Expression::Literal(_)
            | Expression::Boolean(_)
            | Expression::Sizeof(_)
            | Expression::TypeValue(_) => Ok(()),
        }
    }
}
