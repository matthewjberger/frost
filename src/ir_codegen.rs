use std::collections::HashMap;

use anyhow::{Result, bail};
use cranelift::codegen::ir::StackSlot;
use cranelift::prelude::*;
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};

use crate::ir::{
    IrBinOp, IrConstant, IrFunction, IrModule, IrOperand, IrRvalue,
    IrStatement, IrTerminator, IrUnOp,
};
use crate::types::Type;

pub fn compile_ir_to_object(module: &IrModule) -> Result<Vec<u8>> {
    let mut generator = Generator::new()?;
    generator.declare_strings(module)?;
    generator.declare_functions(module)?;
    // One context and one builder context for the whole module. Cranelift is
    // built to have these reused, and a program that specializes a generic
    // thousands of times is thousands of functions, each of which would
    // otherwise pay for a fresh set of arenas.
    let mut context = generator.module.make_context();
    let mut builder_context = FunctionBuilderContext::new();
    for function in &module.functions {
        generator.define_function(
            function,
            &mut context,
            &mut builder_context,
        )?;
    }
    let object = generator.module.finish();
    Ok(object.emit()?)
}

struct Generator {
    module: ObjectModule,
    functions: HashMap<String, FuncId>,
    return_types: HashMap<String, Type>,
    strings: HashMap<String, DataId>,
    pointer_type: types::Type,
}

impl Generator {
    fn new() -> Result<Self> {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed")?;
        flag_builder.set("is_pic", "true")?;
        let isa_builder = cranelift_native::builder()
            .map_err(|message| anyhow::anyhow!("ISA builder: {message}"))?;
        let isa = isa_builder.finish(settings::Flags::new(flag_builder))?;
        let pointer_type = isa.pointer_type();
        let builder = ObjectBuilder::new(
            isa,
            "frost_module",
            cranelift_module::default_libcall_names(),
        )?;
        Ok(Generator {
            module: ObjectModule::new(builder),
            functions: HashMap::new(),
            return_types: HashMap::new(),
            strings: HashMap::new(),
            pointer_type,
        })
    }

    fn declare_strings(&mut self, module: &IrModule) -> Result<()> {
        let mut counter = 0;
        collect_strings(module, &mut |text| {
            if self.strings.contains_key(text) {
                return Ok(());
            }
            let name = format!(".str.{counter}");
            counter += 1;
            let data_id = self.module.declare_data(
                &name,
                Linkage::Local,
                false,
                false,
            )?;
            let mut description = DataDescription::new();
            let mut bytes = text.as_bytes().to_vec();
            bytes.push(0);
            description.define(bytes.into_boxed_slice());
            self.module.define_data(data_id, &description)?;
            self.strings.insert(text.to_string(), data_id);
            Ok(())
        })
    }

    fn declare_functions(&mut self, module: &IrModule) -> Result<()> {
        let pointer_type = self.pointer_type;
        for external in &module.externs {
            let mut signature = self.module.make_signature();
            for parameter in &external.params {
                signature.params.push(AbiParam::new(param_abi_type(
                    pointer_type,
                    parameter,
                )?));
            }
            if !matches!(external.return_type, Type::Void) {
                signature.returns.push(AbiParam::new(clif_type(
                    pointer_type,
                    &external.return_type,
                )?));
            }
            let func_id = self.module.declare_function(
                &external.name,
                Linkage::Import,
                &signature,
            )?;
            self.functions.insert(external.name.clone(), func_id);
            self.return_types
                .insert(external.name.clone(), external.return_type.clone());
        }

        for function in &module.functions {
            let signature = self.build_signature(function)?;
            let func_id = self.module.declare_function(
                &function.name,
                Linkage::Export,
                &signature,
            )?;
            self.functions.insert(function.name.clone(), func_id);
            self.return_types
                .insert(function.name.clone(), function.return_type.clone());
        }

        if !self.functions.contains_key("memcpy") {
            let mut signature = self.module.make_signature();
            signature.params.push(AbiParam::new(pointer_type));
            signature.params.push(AbiParam::new(pointer_type));
            signature.params.push(AbiParam::new(pointer_type));
            signature.returns.push(AbiParam::new(pointer_type));
            let func_id = self.module.declare_function(
                "memcpy",
                Linkage::Import,
                &signature,
            )?;
            self.functions.insert("memcpy".to_string(), func_id);
        }

        for name in ["frost_bounds_check", "frost_generation_check"] {
            if self.functions.contains_key(name) {
                continue;
            }
            let mut signature = self.module.make_signature();
            signature.params.push(AbiParam::new(types::I64));
            signature.params.push(AbiParam::new(types::I64));
            let func_id = self.module.declare_function(
                name,
                Linkage::Import,
                &signature,
            )?;
            self.functions.insert(name.to_string(), func_id);
        }

        Ok(())
    }

    fn function_return_type(&self, function: &IrFunction) -> Type {
        if function.name == "main" {
            Type::I32
        } else {
            function.return_type.clone()
        }
    }

    fn returns_aggregate(&self, function: &IrFunction) -> bool {
        function.name != "main" && is_aggregate(&function.return_type)
    }

    fn build_signature(
        &self,
        function: &IrFunction,
    ) -> Result<cranelift::codegen::ir::Signature> {
        let pointer_type = self.pointer_type;
        let mut signature = self.module.make_signature();
        for index in 0..function.param_count {
            signature.params.push(AbiParam::new(param_abi_type(
                pointer_type,
                function.local_type(index),
            )?));
        }
        if self.returns_aggregate(function) {
            signature.params.push(AbiParam::new(pointer_type));
        } else {
            let return_type = self.function_return_type(function);
            if !matches!(return_type, Type::Void) {
                signature.returns.push(AbiParam::new(clif_type(
                    pointer_type,
                    &return_type,
                )?));
            }
        }
        Ok(signature)
    }

    fn define_function(
        &mut self,
        function: &IrFunction,
        context: &mut cranelift::codegen::Context,
        builder_context: &mut FunctionBuilderContext,
    ) -> Result<()> {
        let func_id = self.functions[&function.name];
        let pointer_type = self.pointer_type;
        let returns_aggregate = self.returns_aggregate(function);
        let return_type = self.function_return_type(function);

        context.func.signature = self.build_signature(function)?;

        let mut builder =
            FunctionBuilder::new(&mut context.func, builder_context);

        let clif_blocks: Vec<Block> = function
            .blocks
            .iter()
            .map(|_| builder.create_block())
            .collect();
        let entry = clif_blocks[function.entry];
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);

        let mut slots: HashMap<usize, StackSlot> = HashMap::new();
        for (index, local) in function.locals.iter().enumerate() {
            if matches!(local.ty, Type::Void | Type::Unknown) {
                continue;
            }
            if local.in_memory {
                let size = local.size.max(1) as u32;
                let slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    size,
                    0,
                ));
                slots.insert(index, slot);
            } else {
                builder.declare_var(
                    Variable::new(index),
                    clif_type(pointer_type, &local.ty)?,
                );
            }
        }

        let memcpy = self.functions["memcpy"];
        let params = builder.block_params(entry).to_vec();
        for (index, value) in
            params.iter().take(function.param_count).enumerate()
        {
            let local = &function.locals[index];
            if is_aggregate(&local.ty) {
                let slot = slots[&index];
                let destination =
                    builder.ins().stack_addr(pointer_type, slot, 0);
                let size =
                    builder.ins().iconst(pointer_type, local.size as i64);
                let memcpy_ref =
                    self.module.declare_func_in_func(memcpy, builder.func);
                builder.ins().call(memcpy_ref, &[destination, *value, size]);
            } else if let Some(slot) = slots.get(&index) {
                builder.ins().stack_store(*value, *slot, 0);
            } else {
                builder.def_var(Variable::new(index), *value);
            }
        }
        let out_pointer = if returns_aggregate {
            Some(params[function.param_count])
        } else {
            None
        };

        {
            let mut translator = Translator {
                module: &mut self.module,
                functions: &self.functions,
                strings: &self.strings,
                slots: &slots,
                pointer_type,
                out_pointer,
                builder: &mut builder,
                function,
                return_type: return_type.clone(),
            };

            for (block_index, ir_block) in function.blocks.iter().enumerate() {
                if block_index != function.entry {
                    translator
                        .builder
                        .switch_to_block(clif_blocks[block_index]);
                }
                for statement in &ir_block.statements {
                    translator.statement(statement)?;
                }
                translator.terminator(&ir_block.terminator, &clif_blocks)?;
            }
        }

        builder.seal_all_blocks();
        builder.finalize();

        self.module.define_function(func_id, context)?;
        self.module.clear_context(context);
        Ok(())
    }
}

fn is_aggregate(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Struct(_)
            | Type::Enum(_)
            | Type::Array(_, _)
            | Type::Str
            | Type::Slice(_)
    )
}

fn param_abi_type(pointer_type: types::Type, ty: &Type) -> Result<types::Type> {
    if is_aggregate(ty) {
        Ok(pointer_type)
    } else {
        clif_type(pointer_type, ty)
    }
}

fn clif_type(pointer_type: types::Type, ty: &Type) -> Result<types::Type> {
    Ok(match ty {
        Type::I8 | Type::U8 | Type::Bool => types::I8,
        Type::I16 | Type::U16 => types::I16,
        Type::I32 | Type::U32 => types::I32,
        Type::I64 | Type::U64 | Type::Isize | Type::Usize => types::I64,
        Type::Handle(_) => types::I64,
        Type::F32 => types::F32,
        Type::F64 => types::F64,
        Type::Ptr(_) | Type::Ref(_) | Type::RefMut(_) | Type::Proc(_, _) => {
            pointer_type
        }
        Type::Distinct(inner) => clif_type(pointer_type, inner)?,
        other => {
            bail!("native backend: type not supported in codegen: {other}")
        }
    })
}

struct Translator<'a, 'b> {
    module: &'a mut ObjectModule,
    functions: &'a HashMap<String, FuncId>,
    strings: &'a HashMap<String, DataId>,
    slots: &'a HashMap<usize, StackSlot>,
    pointer_type: types::Type,
    out_pointer: Option<Value>,
    builder: &'a mut FunctionBuilder<'b>,
    function: &'a IrFunction,
    return_type: Type,
}

impl Translator<'_, '_> {
    fn slot_address(&mut self, local: usize) -> Result<Value> {
        let slot = self.slots.get(&local).ok_or_else(|| {
            anyhow::anyhow!("native backend: aggregate local is not in memory")
        })?;
        Ok(self.builder.ins().stack_addr(self.pointer_type, *slot, 0))
    }

    fn emit_memcpy(&mut self, destination: Value, source: Value, size: usize) {
        let size_value =
            self.builder.ins().iconst(self.pointer_type, size as i64);
        let memcpy = self.functions["memcpy"];
        let memcpy_ref =
            self.module.declare_func_in_func(memcpy, self.builder.func);
        self.builder
            .ins()
            .call(memcpy_ref, &[destination, source, size_value]);
    }

    fn statement(&mut self, statement: &IrStatement) -> Result<()> {
        match statement {
            IrStatement::Assign(local, rvalue) => {
                let local_type = self.function.local_type(*local).clone();
                if matches!(local_type, Type::Void | Type::Unknown) {
                    match rvalue {
                        IrRvalue::Call {
                            function,
                            arguments,
                        } => {
                            self.emit_call(function, arguments)?;
                        }
                        IrRvalue::CallIndirect {
                            callee,
                            arguments,
                            parameter_types,
                            return_type,
                        } => {
                            self.emit_call_indirect(
                                callee,
                                arguments,
                                parameter_types,
                                return_type,
                            )?;
                        }
                        _ => {}
                    }
                    return Ok(());
                }
                if is_aggregate(&local_type) {
                    match rvalue {
                        IrRvalue::Use(IrOperand::Local(source)) => {
                            let destination = self.slot_address(*local)?;
                            let source_address = self.slot_address(*source)?;
                            let size = self.function.locals[*local].size;
                            self.emit_memcpy(destination, source_address, size);
                        }
                        IrRvalue::Call {
                            function,
                            arguments,
                        } => {
                            let out = self.slot_address(*local)?;
                            self.emit_call_with_out(function, arguments, out)?;
                        }
                        _ => bail!(
                            "native backend: unsupported aggregate assignment"
                        ),
                    }
                    return Ok(());
                }
                let value = self.rvalue(rvalue, &local_type)?;
                if let Some(slot) = self.slots.get(local) {
                    self.builder.ins().stack_store(value, *slot, 0);
                } else {
                    self.builder.def_var(Variable::new(*local), value);
                }
                Ok(())
            }
            IrStatement::Store { address, value } => {
                let address_value = self.operand(address)?;
                let value_value = self.operand(value)?;
                self.builder.ins().store(
                    MemFlags::new(),
                    value_value,
                    address_value,
                    0,
                );
                Ok(())
            }
            IrStatement::Copy {
                destination,
                source,
                size,
            } => {
                let destination_value = self.operand(destination)?;
                let source_value = self.operand(source)?;
                self.emit_memcpy(destination_value, source_value, *size);
                Ok(())
            }
            IrStatement::Own(_) | IrStatement::Consume(_) => Ok(()),
        }
    }

    fn rvalue(
        &mut self,
        rvalue: &IrRvalue,
        result_type: &Type,
    ) -> Result<Value> {
        match rvalue {
            IrRvalue::Use(operand) => self.operand(operand),
            IrRvalue::Unary(op, operand) => {
                let value = self.operand(operand)?;
                let operand_type = self.operand_type(operand);
                Ok(match op {
                    IrUnOp::Negate => {
                        if is_float(&operand_type) {
                            self.builder.ins().fneg(value)
                        } else {
                            self.builder.ins().ineg(value)
                        }
                    }
                    IrUnOp::Not => self.builder.ins().bxor_imm(value, 1),
                })
            }
            IrRvalue::Binary(op, left, right) => {
                let operand_type = self.operand_type(left);
                let left_value = self.operand(left)?;
                let right_value = self.operand(right)?;
                self.binary(*op, left_value, right_value, &operand_type)
            }
            IrRvalue::Cast(operand, target) => {
                let value = self.operand(operand)?;
                let source = self.operand_type(operand);
                self.cast(value, &source, target)
            }
            IrRvalue::AddressOf { local, offset } => {
                let Some(slot) = self.slots.get(local) else {
                    bail!(
                        "native backend: address taken of a non-memory local"
                    );
                };
                Ok(self.builder.ins().stack_addr(
                    self.pointer_type,
                    *slot,
                    *offset as i32,
                ))
            }
            IrRvalue::FieldAddress { base, offset } => {
                let base_value = self.operand(base)?;
                if *offset == 0 {
                    Ok(base_value)
                } else {
                    Ok(self.builder.ins().iadd_imm(base_value, *offset as i64))
                }
            }
            IrRvalue::ElementAddress {
                base,
                index,
                element_size,
            } => {
                let base_value = self.operand(base)?;
                let index_value = self.operand(index)?;
                let scaled = if *element_size == 1 {
                    index_value
                } else {
                    self.builder
                        .ins()
                        .imul_imm(index_value, *element_size as i64)
                };
                Ok(self.builder.ins().iadd(base_value, scaled))
            }
            IrRvalue::Load { address, ty } => {
                let address_value = self.operand(address)?;
                let clif = clif_type(self.pointer_type, ty)?;
                Ok(self.builder.ins().load(
                    clif,
                    MemFlags::new(),
                    address_value,
                    0,
                ))
            }
            IrRvalue::Call {
                function,
                arguments,
            } => {
                let results = self.emit_call(function, arguments)?;
                match results.first() {
                    Some(value) => Ok(*value),
                    None => Ok(self.zero_value(result_type)?),
                }
            }
            IrRvalue::FunctionAddress(name) => {
                let Some(func_id) = self.functions.get(name) else {
                    bail!(
                        "native backend: address of undeclared function '{name}'"
                    );
                };
                let func_ref = self
                    .module
                    .declare_func_in_func(*func_id, self.builder.func);
                Ok(self.builder.ins().func_addr(self.pointer_type, func_ref))
            }
            IrRvalue::CallIndirect {
                callee,
                arguments,
                parameter_types,
                return_type,
            } => {
                let results = self.emit_call_indirect(
                    callee,
                    arguments,
                    parameter_types,
                    return_type,
                )?;
                match results.first() {
                    Some(value) => Ok(*value),
                    None => Ok(self.zero_value(result_type)?),
                }
            }
        }
    }

    fn binary(
        &mut self,
        op: IrBinOp,
        left: Value,
        right: Value,
        operand_type: &Type,
    ) -> Result<Value> {
        let float = is_float(operand_type);
        let signed = is_signed(operand_type);
        let instructions = self.builder.ins();
        Ok(match op {
            IrBinOp::Add if float => instructions.fadd(left, right),
            IrBinOp::Add => instructions.iadd(left, right),
            IrBinOp::Subtract if float => instructions.fsub(left, right),
            IrBinOp::Subtract => instructions.isub(left, right),
            IrBinOp::Multiply if float => instructions.fmul(left, right),
            IrBinOp::Multiply => instructions.imul(left, right),
            IrBinOp::Divide if float => instructions.fdiv(left, right),
            IrBinOp::Divide if signed => instructions.sdiv(left, right),
            IrBinOp::Divide => instructions.udiv(left, right),
            IrBinOp::Modulo if signed => instructions.srem(left, right),
            IrBinOp::Modulo => instructions.urem(left, right),
            IrBinOp::BitwiseAnd => instructions.band(left, right),
            IrBinOp::BitwiseOr => instructions.bor(left, right),
            IrBinOp::ShiftLeft => instructions.ishl(left, right),
            IrBinOp::ShiftRight if signed => instructions.sshr(left, right),
            IrBinOp::ShiftRight => instructions.ushr(left, right),
            comparison => {
                return self.comparison(comparison, left, right, operand_type);
            }
        })
    }

    fn comparison(
        &mut self,
        op: IrBinOp,
        left: Value,
        right: Value,
        operand_type: &Type,
    ) -> Result<Value> {
        let float = is_float(operand_type);
        let signed = is_signed(operand_type);
        if float {
            let condition = match op {
                IrBinOp::Equal => FloatCC::Equal,
                IrBinOp::NotEqual => FloatCC::NotEqual,
                IrBinOp::LessThan => FloatCC::LessThan,
                IrBinOp::LessThanOrEqual => FloatCC::LessThanOrEqual,
                IrBinOp::GreaterThan => FloatCC::GreaterThan,
                IrBinOp::GreaterThanOrEqual => FloatCC::GreaterThanOrEqual,
                _ => bail!("native backend: invalid float comparison"),
            };
            return Ok(self.builder.ins().fcmp(condition, left, right));
        }
        let condition = match (op, signed) {
            (IrBinOp::Equal, _) => IntCC::Equal,
            (IrBinOp::NotEqual, _) => IntCC::NotEqual,
            (IrBinOp::LessThan, true) => IntCC::SignedLessThan,
            (IrBinOp::LessThan, false) => IntCC::UnsignedLessThan,
            (IrBinOp::LessThanOrEqual, true) => IntCC::SignedLessThanOrEqual,
            (IrBinOp::LessThanOrEqual, false) => IntCC::UnsignedLessThanOrEqual,
            (IrBinOp::GreaterThan, true) => IntCC::SignedGreaterThan,
            (IrBinOp::GreaterThan, false) => IntCC::UnsignedGreaterThan,
            (IrBinOp::GreaterThanOrEqual, true) => {
                IntCC::SignedGreaterThanOrEqual
            }
            (IrBinOp::GreaterThanOrEqual, false) => {
                IntCC::UnsignedGreaterThanOrEqual
            }
            _ => bail!("native backend: invalid integer comparison"),
        };
        Ok(self.builder.ins().icmp(condition, left, right))
    }

    fn cast(
        &mut self,
        value: Value,
        source: &Type,
        target: &Type,
    ) -> Result<Value> {
        let source_clif = clif_type(self.pointer_type, source)?;
        let target_clif = clif_type(self.pointer_type, target)?;
        if source_clif == target_clif {
            return Ok(value);
        }
        let source_float = is_float(source);
        let target_float = is_float(target);
        Ok(match (source_float, target_float) {
            (false, false) => {
                if target_clif.bits() > source_clif.bits() {
                    if is_signed(source) {
                        self.builder.ins().sextend(target_clif, value)
                    } else {
                        self.builder.ins().uextend(target_clif, value)
                    }
                } else {
                    self.builder.ins().ireduce(target_clif, value)
                }
            }
            (false, true) => {
                if is_signed(source) {
                    self.builder.ins().fcvt_from_sint(target_clif, value)
                } else {
                    self.builder.ins().fcvt_from_uint(target_clif, value)
                }
            }
            (true, false) => {
                if is_signed(target) {
                    self.builder.ins().fcvt_to_sint(target_clif, value)
                } else {
                    self.builder.ins().fcvt_to_uint(target_clif, value)
                }
            }
            (true, true) => {
                if target_clif.bits() > source_clif.bits() {
                    self.builder.ins().fpromote(target_clif, value)
                } else {
                    self.builder.ins().fdemote(target_clif, value)
                }
            }
        })
    }

    fn emit_call(
        &mut self,
        function: &str,
        arguments: &[IrOperand],
    ) -> Result<Vec<Value>> {
        let Some(func_id) = self.functions.get(function) else {
            bail!("native backend: call to undeclared function '{function}'");
        };
        let func_ref = self
            .module
            .declare_func_in_func(*func_id, self.builder.func);
        let mut argument_values = Vec::with_capacity(arguments.len());
        for argument in arguments {
            argument_values.push(self.operand(argument)?);
        }
        let call = self.builder.ins().call(func_ref, &argument_values);
        Ok(self.builder.inst_results(call).to_vec())
    }

    fn emit_call_indirect(
        &mut self,
        callee: &IrOperand,
        arguments: &[IrOperand],
        parameter_types: &[Type],
        return_type: &Type,
    ) -> Result<Vec<Value>> {
        let mut signature = self.module.make_signature();
        for parameter in parameter_types {
            signature.params.push(AbiParam::new(param_abi_type(
                self.pointer_type,
                parameter,
            )?));
        }
        if !matches!(return_type, Type::Void) {
            signature.returns.push(AbiParam::new(clif_type(
                self.pointer_type,
                return_type,
            )?));
        }
        let signature_ref = self.builder.import_signature(signature);
        let callee_value = self.operand(callee)?;
        let mut argument_values = Vec::with_capacity(arguments.len());
        for argument in arguments {
            argument_values.push(self.operand(argument)?);
        }
        let call = self.builder.ins().call_indirect(
            signature_ref,
            callee_value,
            &argument_values,
        );
        Ok(self.builder.inst_results(call).to_vec())
    }

    fn emit_call_with_out(
        &mut self,
        function: &str,
        arguments: &[IrOperand],
        out: Value,
    ) -> Result<()> {
        let Some(func_id) = self.functions.get(function) else {
            bail!("native backend: call to undeclared function '{function}'");
        };
        let func_ref = self
            .module
            .declare_func_in_func(*func_id, self.builder.func);
        let mut argument_values = Vec::with_capacity(arguments.len() + 1);
        for argument in arguments {
            argument_values.push(self.operand(argument)?);
        }
        argument_values.push(out);
        self.builder.ins().call(func_ref, &argument_values);
        Ok(())
    }

    fn terminator(
        &mut self,
        terminator: &IrTerminator,
        blocks: &[Block],
    ) -> Result<()> {
        match terminator {
            IrTerminator::Return(None) => {
                self.emit_return(None)?;
            }
            IrTerminator::Return(Some(operand)) => {
                self.emit_return(Some(operand))?;
            }
            IrTerminator::Jump(block) => {
                self.builder.ins().jump(blocks[*block], &[]);
            }
            IrTerminator::Branch {
                condition,
                then_block,
                else_block,
            } => {
                let condition_value = self.operand(condition)?;
                self.builder.ins().brif(
                    condition_value,
                    blocks[*then_block],
                    &[],
                    blocks[*else_block],
                    &[],
                );
            }
            IrTerminator::Unreachable => {
                self.emit_return(None)?;
            }
        }
        Ok(())
    }

    fn emit_return(&mut self, operand: Option<&IrOperand>) -> Result<()> {
        if let Some(out_pointer) = self.out_pointer {
            if let Some(IrOperand::Local(source)) = operand {
                let source_address = self.slot_address(*source)?;
                let size = self.function.locals[*source].size;
                self.emit_memcpy(out_pointer, source_address, size);
            }
            self.builder.ins().return_(&[]);
            return Ok(());
        }
        if matches!(self.return_type, Type::Void) {
            self.builder.ins().return_(&[]);
            return Ok(());
        }
        let value = match operand {
            Some(operand) => {
                let source = self.operand_type(operand);
                let value = self.operand(operand)?;
                self.cast(value, &source, &self.return_type.clone())?
            }
            None => self.zero_value(&self.return_type.clone())?,
        };
        self.builder.ins().return_(&[value]);
        Ok(())
    }

    fn operand(&mut self, operand: &IrOperand) -> Result<Value> {
        match operand {
            IrOperand::Local(local) => {
                if let Some(slot) = self.slots.get(local) {
                    let clif = clif_type(
                        self.pointer_type,
                        self.function.local_type(*local),
                    )?;
                    Ok(self.builder.ins().stack_load(clif, *slot, 0))
                } else {
                    Ok(self.builder.use_var(Variable::new(*local)))
                }
            }
            IrOperand::Constant(constant) => self.constant(constant),
        }
    }

    fn constant(&mut self, constant: &IrConstant) -> Result<Value> {
        match constant {
            IrConstant::Integer(value, ty) => {
                let clif = clif_type(self.pointer_type, ty)?;
                Ok(self.builder.ins().iconst(clif, *value))
            }
            IrConstant::Float(value, Type::F32) => {
                Ok(self.builder.ins().f32const(*value as f32))
            }
            IrConstant::Float(value, _) => {
                Ok(self.builder.ins().f64const(*value))
            }
            IrConstant::Bool(value) => {
                Ok(self.builder.ins().iconst(types::I8, i64::from(*value)))
            }
            IrConstant::CString(text) => {
                let data_id = self.strings[text];
                let local = self
                    .module
                    .declare_data_in_func(data_id, self.builder.func);
                Ok(self.builder.ins().symbol_value(self.pointer_type, local))
            }
            IrConstant::Unit => {
                bail!("native backend: unit value used as a real value")
            }
        }
    }

    fn zero_value(&mut self, ty: &Type) -> Result<Value> {
        let clif = clif_type(self.pointer_type, ty)?;
        Ok(match ty {
            Type::F32 => self.builder.ins().f32const(0.0),
            Type::F64 => self.builder.ins().f64const(0.0),
            _ => self.builder.ins().iconst(clif, 0),
        })
    }

    fn operand_type(&self, operand: &IrOperand) -> Type {
        match operand {
            IrOperand::Local(local) => self.function.local_type(*local).clone(),
            IrOperand::Constant(constant) => constant.constant_type(),
        }
    }
}

fn is_float(ty: &Type) -> bool {
    matches!(ty, Type::F32 | Type::F64)
}

fn is_signed(ty: &Type) -> bool {
    matches!(
        ty,
        Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::Isize
    )
}

fn collect_strings(
    module: &IrModule,
    handle: &mut impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    for function in &module.functions {
        for block in &function.blocks {
            for statement in &block.statements {
                match statement {
                    IrStatement::Assign(_, rvalue) => {
                        collect_rvalue_strings(rvalue, handle)?;
                    }
                    IrStatement::Store { address, value } => {
                        collect_operand_strings(address, handle)?;
                        collect_operand_strings(value, handle)?;
                    }
                    IrStatement::Copy {
                        destination,
                        source,
                        ..
                    } => {
                        collect_operand_strings(destination, handle)?;
                        collect_operand_strings(source, handle)?;
                    }
                    IrStatement::Own(_) | IrStatement::Consume(_) => {}
                }
            }
            if let IrTerminator::Return(Some(operand)) = &block.terminator {
                collect_operand_strings(operand, handle)?;
            }
        }
    }
    Ok(())
}

fn collect_rvalue_strings(
    rvalue: &IrRvalue,
    handle: &mut impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    match rvalue {
        IrRvalue::Use(operand) | IrRvalue::Unary(_, operand) => {
            collect_operand_strings(operand, handle)
        }
        IrRvalue::Cast(operand, _) => collect_operand_strings(operand, handle),
        IrRvalue::Load { address, .. } => {
            collect_operand_strings(address, handle)
        }
        IrRvalue::FieldAddress { base, .. } => {
            collect_operand_strings(base, handle)
        }
        IrRvalue::ElementAddress { base, index, .. } => {
            collect_operand_strings(base, handle)?;
            collect_operand_strings(index, handle)
        }
        IrRvalue::AddressOf { .. } => Ok(()),
        IrRvalue::Binary(_, left, right) => {
            collect_operand_strings(left, handle)?;
            collect_operand_strings(right, handle)
        }
        IrRvalue::Call { arguments, .. }
        | IrRvalue::CallIndirect { arguments, .. } => {
            for argument in arguments {
                collect_operand_strings(argument, handle)?;
            }
            Ok(())
        }
        IrRvalue::FunctionAddress(_) => Ok(()),
    }
}

fn collect_operand_strings(
    operand: &IrOperand,
    handle: &mut impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    if let IrOperand::Constant(IrConstant::CString(text)) = operand {
        handle(text)?;
    }
    Ok(())
}
