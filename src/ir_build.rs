use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::ir::{
    BlockId, FieldLayout, IrBinOp, IrBlock, IrConstant, IrExtern, IrFunction,
    IrLocal, IrModule, IrOperand, IrRvalue, IrStatement, IrTerminator, IrUnOp,
    LocalId, StructLayout,
};
use crate::parser::{
    Block, Expression, Parameter, ReturnSignature, Statement, StructField,
};
use crate::types::Type;
use crate::{Literal, Operator};

struct FunctionSignature {
    parameters: Vec<Type>,
    return_type: Type,
}

pub struct IrBuilder {
    signatures: HashMap<String, FunctionSignature>,
    structs: HashMap<String, StructLayout>,
}

pub fn build_module(statements: &[Statement]) -> Result<IrModule> {
    let mut builder = IrBuilder {
        signatures: HashMap::new(),
        structs: compute_struct_layouts(statements),
    };
    builder.collect_signatures(statements);

    let mut functions = Vec::new();
    let mut externs = Vec::new();
    let mut top_level = Vec::new();
    let mut has_main = false;

    for statement in statements {
        match statement {
            Statement::Constant(
                name,
                Expression::Function(parameters, return_sig, body)
                | Expression::Proc(parameters, return_sig, body),
            ) => {
                if name == "main" {
                    has_main = true;
                }
                functions.push(
                    builder
                        .lower_function(name, parameters, return_sig, body)?,
                );
            }
            Statement::Extern {
                name,
                params,
                return_type,
            } => {
                externs.push(IrExtern {
                    name: name.clone(),
                    params: params
                        .iter()
                        .map(|parameter| {
                            parameter
                                .type_annotation
                                .clone()
                                .unwrap_or(Type::I64)
                        })
                        .collect(),
                    return_type: return_type.clone().unwrap_or(Type::Void),
                });
            }
            Statement::Struct(..)
            | Statement::Enum(..)
            | Statement::TypeAlias(..)
            | Statement::Import(..) => {}
            other => top_level.push(other.clone()),
        }
    }

    if !has_main && !top_level.is_empty() {
        let empty_params: Vec<Parameter> = Vec::new();
        functions.push(builder.lower_function(
            "main",
            &empty_params,
            &ReturnSignature::Single(Type::I64),
            &top_level,
        )?);
    }

    Ok(IrModule { functions, externs })
}

impl IrBuilder {
    fn collect_signatures(&mut self, statements: &[Statement]) {
        for statement in statements {
            match statement {
                Statement::Constant(
                    name,
                    Expression::Function(parameters, return_sig, _)
                    | Expression::Proc(parameters, return_sig, _),
                ) => {
                    self.signatures.insert(
                        name.clone(),
                        FunctionSignature {
                            parameters: parameters
                                .iter()
                                .map(parameter_type)
                                .collect(),
                            return_type: return_sig
                                .to_type()
                                .unwrap_or(Type::Void),
                        },
                    );
                }
                Statement::Extern {
                    name,
                    params,
                    return_type,
                } => {
                    self.signatures.insert(
                        name.clone(),
                        FunctionSignature {
                            parameters: params
                                .iter()
                                .map(parameter_type)
                                .collect(),
                            return_type: return_type
                                .clone()
                                .unwrap_or(Type::Void),
                        },
                    );
                }
                _ => {}
            }
        }
    }

    fn lower_function(
        &self,
        name: &str,
        parameters: &[Parameter],
        return_sig: &ReturnSignature,
        body: &Block,
    ) -> Result<IrFunction> {
        let return_type = return_sig.to_type().unwrap_or(Type::Void);
        let mut function = FunctionLowering::new(self, return_type.clone());

        for parameter in parameters {
            let ty = parameter_type(parameter);
            let local = function.fresh_local(ty, Some(parameter.name.clone()));
            function.define_variable(&parameter.name, local);
        }

        let (value, value_type) =
            function.lower_block(body, Some(&return_type))?;

        if !function.current_is_terminated() {
            if matches!(return_type, Type::Void) {
                function.set_terminator(IrTerminator::Return(None));
            } else {
                let operand = function.coerce(value, &value_type, &return_type);
                function.set_terminator(IrTerminator::Return(Some(operand)));
            }
        }

        let (locals, blocks) = function.finish();
        Ok(IrFunction {
            name: name.to_string(),
            param_count: parameters.len(),
            return_type,
            locals,
            blocks,
            entry: 0,
        })
    }

    fn signature(&self, name: &str) -> Option<&FunctionSignature> {
        self.signatures.get(name)
    }

    fn struct_layout(&self, name: &str) -> Option<&StructLayout> {
        self.structs.get(name)
    }

    fn byte_size(&self, ty: &Type) -> usize {
        size_and_align(ty, &self.structs)
            .map(|(size, _)| size)
            .unwrap_or(0)
    }
}

fn parameter_type(parameter: &Parameter) -> Type {
    parameter.type_annotation.clone().unwrap_or(Type::I64)
}

fn needs_memory(ty: &Type) -> bool {
    matches!(ty, Type::Struct(_) | Type::Array(_, _))
}

fn compute_struct_layouts(
    statements: &[Statement],
) -> HashMap<String, StructLayout> {
    let definitions: Vec<(&String, &Vec<StructField>)> = statements
        .iter()
        .filter_map(|statement| match statement {
            Statement::Struct(name, _, fields) => Some((name, fields)),
            _ => None,
        })
        .collect();

    let mut layouts: HashMap<String, StructLayout> = HashMap::new();
    loop {
        let mut progress = false;
        for (name, fields) in &definitions {
            if layouts.contains_key(*name) {
                continue;
            }
            if let Some(layout) = try_layout(fields, &layouts) {
                layouts.insert((*name).clone(), layout);
                progress = true;
            }
        }
        if !progress {
            break;
        }
    }
    layouts
}

fn try_layout(
    fields: &[StructField],
    layouts: &HashMap<String, StructLayout>,
) -> Option<StructLayout> {
    let mut offset = 0;
    let mut align = 1;
    let mut field_layouts = Vec::with_capacity(fields.len());
    for field in fields {
        let (field_size, field_align) =
            size_and_align(&field.field_type, layouts)?;
        offset = round_up(offset, field_align);
        field_layouts.push(FieldLayout {
            name: field.name.clone(),
            ty: field.field_type.clone(),
            offset,
        });
        offset += field_size;
        align = align.max(field_align);
    }
    Some(StructLayout {
        size: round_up(offset, align),
        align,
        fields: field_layouts,
    })
}

fn size_and_align(
    ty: &Type,
    layouts: &HashMap<String, StructLayout>,
) -> Option<(usize, usize)> {
    match ty {
        Type::Struct(name) => {
            layouts.get(name).map(|layout| (layout.size, layout.align))
        }
        Type::Array(inner, count) => {
            let (size, align) = size_and_align(inner, layouts)?;
            Some((size * count, align))
        }
        other => Some((other.size_of(), other.align_of())),
    }
}

fn round_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    value.div_ceil(align) * align
}

struct BlockUnderConstruction {
    statements: Vec<IrStatement>,
    terminator: Option<IrTerminator>,
}

struct LoopTargets {
    continue_block: BlockId,
    break_block: BlockId,
}

struct FunctionLowering<'a> {
    builder: &'a IrBuilder,
    locals: Vec<IrLocal>,
    blocks: Vec<BlockUnderConstruction>,
    scopes: Vec<HashMap<String, LocalId>>,
    loops: Vec<LoopTargets>,
    current: BlockId,
    return_type: Type,
}

impl<'a> FunctionLowering<'a> {
    fn new(builder: &'a IrBuilder, return_type: Type) -> Self {
        let entry = BlockUnderConstruction {
            statements: Vec::new(),
            terminator: None,
        };
        FunctionLowering {
            builder,
            locals: Vec::new(),
            blocks: vec![entry],
            scopes: vec![HashMap::new()],
            loops: Vec::new(),
            current: 0,
            return_type,
        }
    }

    fn fresh_local(&mut self, ty: Type, name: Option<String>) -> LocalId {
        let id = self.locals.len();
        let size = self.builder.byte_size(&ty);
        let in_memory = needs_memory(&ty);
        self.locals.push(IrLocal {
            ty,
            name,
            in_memory,
            size,
        });
        id
    }

    fn mark_in_memory(&mut self, local: LocalId) {
        self.locals[local].in_memory = true;
    }

    fn new_block(&mut self) -> BlockId {
        let id = self.blocks.len();
        self.blocks.push(BlockUnderConstruction {
            statements: Vec::new(),
            terminator: None,
        });
        id
    }

    fn switch_to(&mut self, block: BlockId) {
        self.current = block;
    }

    fn current_is_terminated(&self) -> bool {
        self.blocks[self.current].terminator.is_some()
    }

    fn emit(&mut self, statement: IrStatement) {
        if self.current_is_terminated() {
            let block = self.new_block();
            self.switch_to(block);
        }
        self.blocks[self.current].statements.push(statement);
    }

    fn set_terminator(&mut self, terminator: IrTerminator) {
        if self.blocks[self.current].terminator.is_none() {
            self.blocks[self.current].terminator = Some(terminator);
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define_variable(&mut self, name: &str, local: LocalId) {
        self.scopes
            .last_mut()
            .unwrap()
            .insert(name.to_string(), local);
    }

    fn resolve_variable(&self, name: &str) -> Option<LocalId> {
        for scope in self.scopes.iter().rev() {
            if let Some(local) = scope.get(name) {
                return Some(*local);
            }
        }
        None
    }

    fn finish(self) -> (Vec<IrLocal>, Vec<IrBlock>) {
        let blocks = self
            .blocks
            .into_iter()
            .map(|block| IrBlock {
                statements: block.statements,
                terminator: block
                    .terminator
                    .unwrap_or(IrTerminator::Unreachable),
            })
            .collect();
        (self.locals, blocks)
    }

    fn lower_block(
        &mut self,
        block: &Block,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        self.push_scope();
        let mut result = (unit_operand(), Type::Void);
        for (index, statement) in block.iter().enumerate() {
            let is_last = index + 1 == block.len();
            if is_last && let Statement::Expression(expression) = statement {
                result = self.lower_expression(expression, expected)?;
            } else {
                self.lower_statement(statement)?;
            }
        }
        self.pop_scope();
        Ok(result)
    }

    fn lower_statement(&mut self, statement: &Statement) -> Result<()> {
        match statement {
            Statement::Let {
                name,
                type_annotation,
                value,
                ..
            } => {
                if let Expression::StructInit(struct_name, field_inits) = value
                {
                    let ty = Type::Struct(struct_name.clone());
                    let local = self.fresh_local(ty, Some(name.clone()));
                    self.init_struct(local, struct_name, field_inits)?;
                    self.define_variable(name, local);
                    return Ok(());
                }
                let (operand, value_type) =
                    self.lower_expression(value, type_annotation.as_ref())?;
                let declared = type_annotation.clone().unwrap_or(value_type);
                let local =
                    self.fresh_local(declared.clone(), Some(name.clone()));
                let coerced =
                    self.coerce(operand, &self.type_of_local(local), &declared);
                self.emit(IrStatement::Assign(local, IrRvalue::Use(coerced)));
                self.define_variable(name, local);
                Ok(())
            }
            Statement::Constant(name, value) => {
                let (operand, value_type) =
                    self.lower_expression(value, None)?;
                let local = self.fresh_local(value_type, Some(name.clone()));
                self.emit(IrStatement::Assign(local, IrRvalue::Use(operand)));
                self.define_variable(name, local);
                Ok(())
            }
            Statement::Assignment(target, value) => {
                self.lower_assignment(target, value)
            }
            Statement::Return(expression) => {
                let return_type = self.return_type.clone();
                if matches!(return_type, Type::Void) {
                    self.set_terminator(IrTerminator::Return(None));
                } else {
                    let (operand, value_type) =
                        self.lower_expression(expression, Some(&return_type))?;
                    let coerced =
                        self.coerce(operand, &value_type, &return_type);
                    self.set_terminator(IrTerminator::Return(Some(coerced)));
                }
                Ok(())
            }
            Statement::Expression(expression) => {
                self.lower_expression(expression, None)?;
                Ok(())
            }
            Statement::While(condition, body) => {
                self.lower_while(condition, body)
            }
            Statement::For(variable, range, body) => {
                self.lower_for(variable, range, body)
            }
            Statement::Break => {
                let Some(targets) = self.loops.last() else {
                    bail!("native backend: break outside loop");
                };
                self.set_terminator(IrTerminator::Jump(targets.break_block));
                Ok(())
            }
            Statement::Continue => {
                let Some(targets) = self.loops.last() else {
                    bail!("native backend: continue outside loop");
                };
                self.set_terminator(IrTerminator::Jump(targets.continue_block));
                Ok(())
            }
            other => bail!("native backend: unsupported statement: {other}"),
        }
    }

    fn lower_while(
        &mut self,
        condition: &Expression,
        body: &Block,
    ) -> Result<()> {
        let header = self.new_block();
        let body_block = self.new_block();
        let exit = self.new_block();

        self.set_terminator(IrTerminator::Jump(header));
        self.switch_to(header);
        let (condition_operand, _) =
            self.lower_expression(condition, Some(&Type::Bool))?;
        self.set_terminator(IrTerminator::Branch {
            condition: condition_operand,
            then_block: body_block,
            else_block: exit,
        });

        self.switch_to(body_block);
        self.loops.push(LoopTargets {
            continue_block: header,
            break_block: exit,
        });
        self.lower_block(body, None)?;
        self.loops.pop();
        self.set_terminator(IrTerminator::Jump(header));

        self.switch_to(exit);
        Ok(())
    }

    fn lower_for(
        &mut self,
        variable: &str,
        range: &Expression,
        body: &Block,
    ) -> Result<()> {
        let Expression::Range(start, end, inclusive) = range else {
            bail!("native backend: for loop requires a range");
        };

        let (start_operand, start_type) =
            self.lower_expression(start, Some(&Type::I64))?;
        let index =
            self.fresh_local(start_type.clone(), Some(variable.to_string()));
        let start_coerced =
            self.coerce(start_operand, &start_type, &start_type);
        self.emit(IrStatement::Assign(index, IrRvalue::Use(start_coerced)));

        let (end_operand, end_type) =
            self.lower_expression(end, Some(&start_type))?;
        let end_local = self.fresh_local(end_type.clone(), None);
        let end_coerced = self.coerce(end_operand, &end_type, &start_type);
        self.emit(IrStatement::Assign(end_local, IrRvalue::Use(end_coerced)));

        let header = self.new_block();
        let body_block = self.new_block();
        let step_block = self.new_block();
        let exit = self.new_block();

        self.set_terminator(IrTerminator::Jump(header));
        self.switch_to(header);
        let condition = self.fresh_local(Type::Bool, None);
        let compare = if *inclusive {
            IrBinOp::LessThanOrEqual
        } else {
            IrBinOp::LessThan
        };
        self.emit(IrStatement::Assign(
            condition,
            IrRvalue::Binary(
                compare,
                IrOperand::Local(index),
                IrOperand::Local(end_local),
            ),
        ));
        self.set_terminator(IrTerminator::Branch {
            condition: IrOperand::Local(condition),
            then_block: body_block,
            else_block: exit,
        });

        self.switch_to(body_block);
        self.push_scope();
        self.define_variable(variable, index);
        self.loops.push(LoopTargets {
            continue_block: step_block,
            break_block: exit,
        });
        self.lower_block(body, None)?;
        self.loops.pop();
        self.pop_scope();
        self.set_terminator(IrTerminator::Jump(step_block));

        self.switch_to(step_block);
        let one =
            IrOperand::Constant(IrConstant::Integer(1, start_type.clone()));
        self.emit(IrStatement::Assign(
            index,
            IrRvalue::Binary(IrBinOp::Add, IrOperand::Local(index), one),
        ));
        self.set_terminator(IrTerminator::Jump(header));

        self.switch_to(exit);
        Ok(())
    }

    fn lower_expression(
        &mut self,
        expression: &Expression,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        match expression {
            Expression::Literal(literal) => {
                self.lower_literal(literal, expected)
            }
            Expression::Boolean(value) => {
                Ok((IrOperand::Constant(IrConstant::Bool(*value)), Type::Bool))
            }
            Expression::Identifier(name) => {
                let Some(local) = self.resolve_variable(name) else {
                    bail!("native backend: unknown variable '{name}'");
                };
                Ok((IrOperand::Local(local), self.type_of_local(local)))
            }
            Expression::Prefix(operator, operand) => {
                self.lower_prefix(*operator, operand, expected)
            }
            Expression::Infix(left, operator, right) => {
                self.lower_infix(left, *operator, right, expected)
            }
            Expression::If(condition, consequence, alternative) => self
                .lower_if(
                    condition,
                    consequence,
                    alternative.as_ref(),
                    expected,
                ),
            Expression::Call(callee, arguments) => {
                self.lower_call(callee, arguments)
            }
            Expression::Borrow(inner) => {
                self.lower_address_of(inner, RefKind::Ref)
            }
            Expression::BorrowMut(inner) => {
                self.lower_address_of(inner, RefKind::RefMut)
            }
            Expression::AddressOf(inner) => {
                self.lower_address_of(inner, RefKind::Ptr)
            }
            Expression::Dereference(inner) => self.lower_dereference(inner),
            Expression::FieldAccess(base, field) => {
                self.lower_field_read(base, field)
            }
            Expression::StructInit(..) => {
                bail!(
                    "native backend: struct literals are only supported as a variable initializer"
                )
            }
            other => {
                bail!("native backend: unsupported expression: {other}")
            }
        }
    }

    fn lower_literal(
        &mut self,
        literal: &Literal,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        match literal {
            Literal::Integer(value) => {
                let ty = match expected {
                    Some(ty) if is_integer(ty) => ty.clone(),
                    _ => Type::I64,
                };
                Ok((
                    IrOperand::Constant(IrConstant::Integer(
                        *value,
                        ty.clone(),
                    )),
                    ty,
                ))
            }
            Literal::Float(value) => {
                let ty = match expected {
                    Some(Type::F32) => Type::F32,
                    _ => Type::F64,
                };
                Ok((
                    IrOperand::Constant(IrConstant::Float(*value, ty.clone())),
                    ty,
                ))
            }
            Literal::Float32(value) => Ok((
                IrOperand::Constant(IrConstant::Float(
                    *value as f64,
                    Type::F32,
                )),
                Type::F32,
            )),
            Literal::Boolean(value) => {
                Ok((IrOperand::Constant(IrConstant::Bool(*value)), Type::Bool))
            }
            Literal::String(value) => Ok((
                IrOperand::Constant(IrConstant::CString(value.clone())),
                Type::Ptr(Box::new(Type::I8)),
            )),
            other => {
                bail!("native backend: unsupported literal: {other}")
            }
        }
    }

    fn lower_prefix(
        &mut self,
        operator: Operator,
        operand: &Expression,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        match operator {
            Operator::Negate => {
                let (value, ty) = self.lower_expression(operand, expected)?;
                let result = self.fresh_local(ty.clone(), None);
                self.emit(IrStatement::Assign(
                    result,
                    IrRvalue::Unary(IrUnOp::Negate, value),
                ));
                Ok((IrOperand::Local(result), ty))
            }
            Operator::Not => {
                let (value, _) =
                    self.lower_expression(operand, Some(&Type::Bool))?;
                let result = self.fresh_local(Type::Bool, None);
                self.emit(IrStatement::Assign(
                    result,
                    IrRvalue::Unary(IrUnOp::Not, value),
                ));
                Ok((IrOperand::Local(result), Type::Bool))
            }
            other => {
                bail!("native backend: unsupported prefix operator: {other}")
            }
        }
    }

    fn lower_infix(
        &mut self,
        left: &Expression,
        operator: Operator,
        right: &Expression,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        if matches!(operator, Operator::And | Operator::Or) {
            return self.lower_logical(left, operator, right);
        }

        let binop = binop_of(operator)?;
        if binop.is_comparison() {
            let (left_operand, left_type) =
                self.lower_expression(left, None)?;
            let (right_operand, right_type) =
                self.lower_expression(right, Some(&left_type))?;
            let operand_type = unify(&left_type, &right_type);
            let left_final =
                self.coerce(left_operand, &left_type, &operand_type);
            let right_final =
                self.coerce(right_operand, &right_type, &operand_type);
            let result = self.fresh_local(Type::Bool, None);
            self.emit(IrStatement::Assign(
                result,
                IrRvalue::Binary(binop, left_final, right_final),
            ));
            return Ok((IrOperand::Local(result), Type::Bool));
        }

        let (left_operand, left_type) =
            self.lower_expression(left, expected)?;
        let (right_operand, right_type) =
            self.lower_expression(right, Some(&left_type))?;
        let result_type = unify(&left_type, &right_type);
        let left_final = self.coerce(left_operand, &left_type, &result_type);
        let right_final = self.coerce(right_operand, &right_type, &result_type);
        let result = self.fresh_local(result_type.clone(), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Binary(binop, left_final, right_final),
        ));
        Ok((IrOperand::Local(result), result_type))
    }

    fn lower_logical(
        &mut self,
        left: &Expression,
        operator: Operator,
        right: &Expression,
    ) -> Result<(IrOperand, Type)> {
        let result = self.fresh_local(Type::Bool, None);
        let (left_operand, _) =
            self.lower_expression(left, Some(&Type::Bool))?;

        let evaluate_right = self.new_block();
        let shortcut = self.new_block();
        let merge = self.new_block();

        match operator {
            Operator::And => self.set_terminator(IrTerminator::Branch {
                condition: left_operand,
                then_block: evaluate_right,
                else_block: shortcut,
            }),
            _ => self.set_terminator(IrTerminator::Branch {
                condition: left_operand,
                then_block: shortcut,
                else_block: evaluate_right,
            }),
        }

        self.switch_to(evaluate_right);
        let (right_operand, _) =
            self.lower_expression(right, Some(&Type::Bool))?;
        self.emit(IrStatement::Assign(result, IrRvalue::Use(right_operand)));
        self.set_terminator(IrTerminator::Jump(merge));

        self.switch_to(shortcut);
        let shortcut_value = matches!(operator, Operator::Or);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Use(IrOperand::Constant(IrConstant::Bool(
                shortcut_value,
            ))),
        ));
        self.set_terminator(IrTerminator::Jump(merge));

        self.switch_to(merge);
        Ok((IrOperand::Local(result), Type::Bool))
    }

    fn lower_if(
        &mut self,
        condition: &Expression,
        consequence: &Block,
        alternative: Option<&Block>,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        let (condition_operand, _) =
            self.lower_expression(condition, Some(&Type::Bool))?;

        let then_block = self.new_block();
        let else_block = self.new_block();
        let merge = self.new_block();

        self.set_terminator(IrTerminator::Branch {
            condition: condition_operand,
            then_block,
            else_block,
        });

        self.switch_to(then_block);
        let (then_value, then_type) =
            self.lower_block(consequence, expected)?;

        let result_type = match expected {
            Some(ty) if !matches!(ty, Type::Void) => ty.clone(),
            _ => then_type.clone(),
        };
        let produces_value =
            !matches!(result_type, Type::Void) && alternative.is_some();

        let result = if produces_value {
            Some(self.fresh_local(result_type.clone(), None))
        } else {
            None
        };

        if let Some(result_local) = result {
            let coerced = self.coerce(then_value, &then_type, &result_type);
            self.emit(IrStatement::Assign(
                result_local,
                IrRvalue::Use(coerced),
            ));
        }
        self.set_terminator(IrTerminator::Jump(merge));

        self.switch_to(else_block);
        if let Some(alternative) = alternative {
            let (else_value, else_type) =
                self.lower_block(alternative, expected)?;
            if let Some(result_local) = result {
                let coerced = self.coerce(else_value, &else_type, &result_type);
                self.emit(IrStatement::Assign(
                    result_local,
                    IrRvalue::Use(coerced),
                ));
            }
        }
        self.set_terminator(IrTerminator::Jump(merge));

        self.switch_to(merge);
        match result {
            Some(result_local) => {
                Ok((IrOperand::Local(result_local), result_type))
            }
            None => Ok((unit_operand(), Type::Void)),
        }
    }

    fn lower_call(
        &mut self,
        callee: &Expression,
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        let Expression::Identifier(name) = callee else {
            bail!("native backend: only direct function calls are supported");
        };
        let Some(signature) = self.builder.signature(name) else {
            bail!("native backend: call to unknown function '{name}'");
        };
        let parameter_types = signature.parameters.clone();
        let return_type = signature.return_type.clone();

        let mut lowered = Vec::with_capacity(arguments.len());
        for (index, argument) in arguments.iter().enumerate() {
            let expected = parameter_types.get(index);
            let (operand, value_type) =
                self.lower_expression(argument, expected)?;
            let coerced = match expected {
                Some(target) => self.coerce(operand, &value_type, target),
                None => operand,
            };
            lowered.push(coerced);
        }

        let result = self.fresh_local(return_type.clone(), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Call {
                function: name.clone(),
                arguments: lowered,
            },
        ));
        Ok((IrOperand::Local(result), return_type))
    }

    fn lower_assignment(
        &mut self,
        target: &Expression,
        value: &Expression,
    ) -> Result<()> {
        if let Expression::Identifier(name) = target {
            let Some(local) = self.resolve_variable(name) else {
                bail!(
                    "native backend: assignment to unknown variable '{name}'"
                );
            };
            let target_type = self.type_of_local(local);
            let (operand, value_type) =
                self.lower_expression(value, Some(&target_type))?;
            let coerced = self.coerce(operand, &value_type, &target_type);
            self.emit(IrStatement::Assign(local, IrRvalue::Use(coerced)));
            return Ok(());
        }

        let (address, pointee) = self.place_address(target)?;
        let (operand, value_type) =
            self.lower_expression(value, Some(&pointee))?;
        let coerced = self.coerce(operand, &value_type, &pointee);
        self.emit(IrStatement::Store {
            address,
            value: coerced,
        });
        Ok(())
    }

    fn lower_address_of(
        &mut self,
        inner: &Expression,
        kind: RefKind,
    ) -> Result<(IrOperand, Type)> {
        let (address, pointee) = self.place_address(inner)?;
        let result_type = match kind {
            RefKind::Ref => Type::Ref(Box::new(pointee)),
            RefKind::RefMut => Type::RefMut(Box::new(pointee)),
            RefKind::Ptr => Type::Ptr(Box::new(pointee)),
        };
        Ok((address, result_type))
    }

    fn lower_dereference(
        &mut self,
        pointer: &Expression,
    ) -> Result<(IrOperand, Type)> {
        let (address, pointee) = self.place_address_of_deref(pointer)?;
        self.load_from(address, pointee)
    }

    fn lower_field_read(
        &mut self,
        base: &Expression,
        field: &str,
    ) -> Result<(IrOperand, Type)> {
        let (address, field_type) = self.field_address(base, field)?;
        self.load_from(address, field_type)
    }

    fn load_from(
        &mut self,
        address: IrOperand,
        ty: Type,
    ) -> Result<(IrOperand, Type)> {
        if needs_memory(&ty) {
            bail!(
                "native backend: reading an aggregate value by value is not supported yet"
            );
        }
        let result = self.fresh_local(ty.clone(), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Load {
                address,
                ty: ty.clone(),
            },
        ));
        Ok((IrOperand::Local(result), ty))
    }

    fn place_address(
        &mut self,
        place: &Expression,
    ) -> Result<(IrOperand, Type)> {
        match place {
            Expression::Identifier(name) => {
                let Some(local) = self.resolve_variable(name) else {
                    bail!(
                        "native backend: address of unknown variable '{name}'"
                    );
                };
                self.mark_in_memory(local);
                let pointee = self.type_of_local(local);
                let result = self
                    .fresh_local(Type::Ptr(Box::new(pointee.clone())), None);
                self.emit(IrStatement::Assign(
                    result,
                    IrRvalue::AddressOf { local, offset: 0 },
                ));
                Ok((IrOperand::Local(result), pointee))
            }
            Expression::FieldAccess(base, field) => {
                self.field_address(base, field)
            }
            Expression::Dereference(pointer) => {
                self.place_address_of_deref(pointer)
            }
            other => {
                bail!(
                    "native backend: expression is not an assignable place: {other}"
                )
            }
        }
    }

    fn place_address_of_deref(
        &mut self,
        pointer: &Expression,
    ) -> Result<(IrOperand, Type)> {
        let (pointer_operand, pointer_type) =
            self.lower_expression(pointer, None)?;
        let pointee = deref_target(&pointer_type)?;
        Ok((pointer_operand, pointee))
    }

    fn field_address(
        &mut self,
        base: &Expression,
        field: &str,
    ) -> Result<(IrOperand, Type)> {
        let (base_pointer, struct_name) = self.struct_place(base)?;
        let layout =
            self.builder.struct_layout(&struct_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "native backend: unknown struct '{struct_name}'"
                )
            })?;
        let field_layout = layout.field(field).ok_or_else(|| {
            anyhow::anyhow!(
                "native backend: struct '{struct_name}' has no field '{field}'"
            )
        })?;
        let field_type = field_layout.ty.clone();
        let offset = field_layout.offset;
        let result =
            self.fresh_local(Type::Ptr(Box::new(field_type.clone())), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::FieldAddress {
                base: base_pointer,
                offset,
            },
        ));
        Ok((IrOperand::Local(result), field_type))
    }

    fn struct_place(
        &mut self,
        base: &Expression,
    ) -> Result<(IrOperand, String)> {
        match base {
            Expression::Identifier(name) => {
                let Some(local) = self.resolve_variable(name) else {
                    bail!("native backend: unknown variable '{name}'");
                };
                match self.type_of_local(local) {
                    Type::Struct(struct_name) => {
                        self.mark_in_memory(local);
                        let result = self.fresh_local(
                            Type::Ptr(Box::new(Type::Struct(
                                struct_name.clone(),
                            ))),
                            None,
                        );
                        self.emit(IrStatement::Assign(
                            result,
                            IrRvalue::AddressOf { local, offset: 0 },
                        ));
                        Ok((IrOperand::Local(result), struct_name))
                    }
                    Type::Ref(inner)
                    | Type::RefMut(inner)
                    | Type::Ptr(inner)
                        if matches!(*inner, Type::Struct(_)) =>
                    {
                        let Type::Struct(struct_name) = *inner else {
                            unreachable!()
                        };
                        Ok((IrOperand::Local(local), struct_name))
                    }
                    other => bail!(
                        "native backend: '{name}' is not a struct (found {other})"
                    ),
                }
            }
            Expression::FieldAccess(inner, field) => {
                let (address, field_type) = self.field_address(inner, field)?;
                let Type::Struct(struct_name) = field_type else {
                    bail!("native backend: field '{field}' is not a struct");
                };
                Ok((address, struct_name))
            }
            Expression::Dereference(pointer) => {
                let (pointer_operand, pointer_type) =
                    self.lower_expression(pointer, None)?;
                let pointee = deref_target(&pointer_type)?;
                let Type::Struct(struct_name) = pointee else {
                    bail!("native backend: dereference is not a struct");
                };
                Ok((pointer_operand, struct_name))
            }
            other => {
                bail!("native backend: not a struct place: {other}")
            }
        }
    }

    fn init_struct(
        &mut self,
        local: LocalId,
        struct_name: &str,
        field_inits: &[(String, Expression)],
    ) -> Result<()> {
        let fields: Vec<(String, usize, Type)> = {
            let layout =
                self.builder.struct_layout(struct_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "native backend: unknown struct '{struct_name}'"
                    )
                })?;
            layout
                .fields
                .iter()
                .map(|field| {
                    (field.name.clone(), field.offset, field.ty.clone())
                })
                .collect()
        };

        for (field_name, field_value) in field_inits {
            let Some((_, offset, field_type)) =
                fields.iter().find(|(name, _, _)| name == field_name)
            else {
                bail!(
                    "native backend: struct '{struct_name}' has no field '{field_name}'"
                );
            };
            if needs_memory(field_type) {
                bail!(
                    "native backend: nested aggregate struct fields are not supported yet"
                );
            }
            let address =
                self.fresh_local(Type::Ptr(Box::new(field_type.clone())), None);
            self.emit(IrStatement::Assign(
                address,
                IrRvalue::AddressOf {
                    local,
                    offset: *offset,
                },
            ));
            let (operand, value_type) =
                self.lower_expression(field_value, Some(field_type))?;
            let coerced = self.coerce(operand, &value_type, field_type);
            self.emit(IrStatement::Store {
                address: IrOperand::Local(address),
                value: coerced,
            });
        }
        Ok(())
    }

    fn type_of_local(&self, local: LocalId) -> Type {
        self.locals[local].ty.clone()
    }

    fn coerce(
        &mut self,
        operand: IrOperand,
        from: &Type,
        to: &Type,
    ) -> IrOperand {
        if from == to || matches!(to, Type::Void | Type::Unknown) {
            return operand;
        }
        match &operand {
            IrOperand::Constant(IrConstant::Integer(value, _))
                if is_integer(to) =>
            {
                IrOperand::Constant(IrConstant::Integer(*value, to.clone()))
            }
            IrOperand::Constant(IrConstant::Float(value, _))
                if matches!(to, Type::F32 | Type::F64) =>
            {
                IrOperand::Constant(IrConstant::Float(*value, to.clone()))
            }
            _ if needs_cast(from, to) => {
                let result = self.fresh_local(to.clone(), None);
                self.emit(IrStatement::Assign(
                    result,
                    IrRvalue::Cast(operand, to.clone()),
                ));
                IrOperand::Local(result)
            }
            _ => operand,
        }
    }
}

#[derive(Clone, Copy)]
enum RefKind {
    Ref,
    RefMut,
    Ptr,
}

fn deref_target(pointer_type: &Type) -> Result<Type> {
    match pointer_type {
        Type::Ref(inner) | Type::RefMut(inner) | Type::Ptr(inner) => {
            Ok((**inner).clone())
        }
        other => {
            bail!("native backend: cannot dereference a value of type {other}")
        }
    }
}

fn unit_operand() -> IrOperand {
    IrOperand::Constant(IrConstant::Unit)
}

fn is_integer(ty: &Type) -> bool {
    matches!(
        ty,
        Type::I8
            | Type::I16
            | Type::I32
            | Type::I64
            | Type::Isize
            | Type::U8
            | Type::U16
            | Type::U32
            | Type::U64
            | Type::Usize
    )
}

fn needs_cast(from: &Type, to: &Type) -> bool {
    (is_integer(from) && is_integer(to))
        || (matches!(from, Type::F32 | Type::F64)
            && matches!(to, Type::F32 | Type::F64))
        || (is_integer(from) && matches!(to, Type::F32 | Type::F64))
        || (matches!(from, Type::F32 | Type::F64) && is_integer(to))
}

fn unify(left: &Type, right: &Type) -> Type {
    if left == right {
        return left.clone();
    }
    match (left, right) {
        (Type::I64, other) | (other, Type::I64) if is_integer(other) => {
            other.clone()
        }
        (Type::F64, Type::F32) | (Type::F32, Type::F64) => Type::F64,
        (Type::Unknown, other) | (other, Type::Unknown) => other.clone(),
        _ => left.clone(),
    }
}

fn binop_of(operator: Operator) -> Result<IrBinOp> {
    Ok(match operator {
        Operator::Add => IrBinOp::Add,
        Operator::Subtract => IrBinOp::Subtract,
        Operator::Multiply => IrBinOp::Multiply,
        Operator::Divide => IrBinOp::Divide,
        Operator::Modulo => IrBinOp::Modulo,
        Operator::BitwiseAnd => IrBinOp::BitwiseAnd,
        Operator::BitwiseOr => IrBinOp::BitwiseOr,
        Operator::ShiftLeft => IrBinOp::ShiftLeft,
        Operator::ShiftRight => IrBinOp::ShiftRight,
        Operator::Equal => IrBinOp::Equal,
        Operator::NotEqual => IrBinOp::NotEqual,
        Operator::LessThan => IrBinOp::LessThan,
        Operator::LessThanOrEqual => IrBinOp::LessThanOrEqual,
        Operator::GreaterThan => IrBinOp::GreaterThan,
        Operator::GreaterThanOrEqual => IrBinOp::GreaterThanOrEqual,
        other => bail!("native backend: unsupported binary operator: {other}"),
    })
}
