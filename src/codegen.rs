use crate::{
    types::Type, Expression, Literal, Operator, Parameter, Statement, StructField,
};
use anyhow::{bail, Result};
use cranelift::prelude::*;
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use std::collections::HashMap;
use target_lexicon::Triple;

pub struct CodegenContext {
    module: ObjectModule,
    functions: HashMap<String, FuncId>,
    structs: HashMap<String, CompiledStructInfo>,
    string_data: HashMap<String, DataId>,
    string_counter: usize,
}

#[derive(Debug, Clone)]
struct CompiledStructInfo {
    #[allow(dead_code)]
    fields: Vec<StructField>,
    field_offsets: HashMap<String, usize>,
    size: usize,
}

struct FunctionTranslator<'a> {
    builder: FunctionBuilder<'a>,
    variables: HashMap<String, Variable>,
    variable_counter: usize,
    module: &'a mut ObjectModule,
    functions: &'a HashMap<String, FuncId>,
    structs: &'a HashMap<String, CompiledStructInfo>,
    string_data: &'a mut HashMap<String, DataId>,
    string_counter: &'a mut usize,
    pointer_type: types::Type,
    loop_exit_blocks: Vec<Block>,
    loop_continue_blocks: Vec<Block>,
}

impl CodegenContext {
    pub fn new() -> Result<Self> {
        let _triple = Triple::host();
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed")?;
        flag_builder.set("is_pic", "true")?;

        let isa_builder = cranelift_native::builder()
            .map_err(|msg| anyhow::anyhow!("Failed to create ISA builder: {}", msg))?;
        let isa = isa_builder.finish(settings::Flags::new(flag_builder))?;

        let builder = ObjectBuilder::new(
            isa,
            "frost_module",
            cranelift_module::default_libcall_names(),
        )?;
        let module = ObjectModule::new(builder);

        Ok(Self {
            module,
            functions: HashMap::new(),
            structs: HashMap::new(),
            string_data: HashMap::new(),
            string_counter: 0,
        })
    }

    pub fn compile_program(&mut self, statements: &[Statement]) -> Result<()> {
        for statement in statements {
            match statement {
                Statement::Struct(name, _type_params, fields) => {
                    self.compile_struct_definition(name, fields)?;
                }
                Statement::Extern { name, params, return_type } => {
                    self.declare_extern_function(name, params, return_type.as_ref())?;
                }
                Statement::Constant(name, expr) => {
                    if let Expression::Function(params, ret_sig, body)
                    | Expression::Proc(params, ret_sig, body) = expr
                    {
                        let ret_type = ret_sig.to_type();
                        self.compile_function(name, params, ret_type.as_ref(), body)?;
                    }
                }
                _ => {}
            }
        }

        if !self.functions.contains_key("main") {
            let mut global_body = Vec::new();
            for statement in statements {
                match statement {
                    Statement::Struct(..) => {}
                    Statement::Extern { .. } => {}
                    Statement::Constant(_name, expr) => {
                        if matches!(expr, Expression::Function(_, _, _) | Expression::Proc(_, _, _)) {
                            continue;
                        }
                        global_body.push(statement.clone());
                    }
                    _ => {
                        global_body.push(statement.clone());
                    }
                }
            }

            if !global_body.is_empty() {
                self.compile_function("main", &[], None, &global_body)?;
            }
        }

        Ok(())
    }

    fn compile_struct_definition(&mut self, name: &str, fields: &[StructField]) -> Result<()> {
        let mut field_offsets = HashMap::new();
        let mut offset = 0;

        for field in fields {
            field_offsets.insert(field.name.clone(), offset);
            offset += self.type_size(&field.field_type);
        }

        self.structs.insert(name.to_string(), CompiledStructInfo {
            fields: fields.to_vec(),
            field_offsets,
            size: offset,
        });

        Ok(())
    }

    fn type_size(&self, typ: &Type) -> usize {
        match typ {
            Type::I8 | Type::U8 | Type::Bool => 1,
            Type::I16 | Type::U16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64 | Type::U64 | Type::F64 => 8,
            Type::Ptr(_) | Type::Ref(_) | Type::RefMut(_) => 8,
            Type::Array(inner, count) => self.type_size(inner) * count,
            Type::Struct(name) => {
                self.structs.get(name).map(|s| s.size).unwrap_or(8)
            }
            _ => 8,
        }
    }

    fn frost_type_to_cranelift(&self, typ: &Type) -> types::Type {
        match typ {
            Type::I8 => types::I8,
            Type::I16 => types::I16,
            Type::I32 => types::I32,
            Type::I64 => types::I64,
            Type::U8 => types::I8,
            Type::U16 => types::I16,
            Type::U32 => types::I32,
            Type::U64 => types::I64,
            Type::F32 => types::F32,
            Type::F64 => types::F64,
            Type::Bool => types::I8,
            Type::Ptr(_) | Type::Ref(_) | Type::RefMut(_) => self.module.isa().pointer_type(),
            _ => types::I64,
        }
    }

    fn declare_extern_function(
        &mut self,
        name: &str,
        params: &[Parameter],
        return_type: Option<&Type>,
    ) -> Result<()> {
        let mut sig = self.module.make_signature();

        for param in params {
            let param_type = param
                .type_annotation
                .as_ref()
                .map(|t| self.frost_type_to_cranelift(t))
                .unwrap_or(types::I64);
            sig.params.push(AbiParam::new(param_type));
        }

        if let Some(ret) = return_type {
            sig.returns.push(AbiParam::new(self.frost_type_to_cranelift(ret)));
        }

        let func_id = self.module.declare_function(name, Linkage::Import, &sig)?;
        self.functions.insert(name.to_string(), func_id);

        Ok(())
    }

    fn compile_function(
        &mut self,
        name: &str,
        params: &[Parameter],
        return_type: Option<&Type>,
        body: &[Statement],
    ) -> Result<()> {
        let mut sig = self.module.make_signature();

        for param in params {
            let param_type = param
                .type_annotation
                .as_ref()
                .map(|t| self.frost_type_to_cranelift(t))
                .unwrap_or(types::I64);
            sig.params.push(AbiParam::new(param_type));
        }

        if let Some(ret) = return_type {
            sig.returns.push(AbiParam::new(self.frost_type_to_cranelift(ret)));
        }

        let func_id = self.module.declare_function(name, Linkage::Export, &sig)?;
        self.functions.insert(name.to_string(), func_id);

        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;

        let mut func_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let pointer_type = self.module.isa().pointer_type();
            let mut translator = FunctionTranslator {
                builder,
                variables: HashMap::new(),
                variable_counter: 0,
                module: &mut self.module,
                functions: &self.functions,
                structs: &self.structs,
                string_data: &mut self.string_data,
                string_counter: &mut self.string_counter,
                pointer_type,
                loop_exit_blocks: Vec::new(),
                loop_continue_blocks: Vec::new(),
            };

            for (index, param) in params.iter().enumerate() {
                let param_value = translator.builder.block_params(entry_block)[index];
                let param_type = param
                    .type_annotation
                    .as_ref()
                    .map(|t| translator.frost_type_to_cranelift(t))
                    .unwrap_or(types::I64);
                let var = translator.declare_variable(&param.name, param_type);
                translator.builder.def_var(var, param_value);
            }

            let mut last_value = None;
            for statement in body {
                last_value = translator.translate_statement(statement)?;
            }

            if return_type.is_some() {
                if let Some(val) = last_value {
                    translator.builder.ins().return_(&[val]);
                } else {
                    let zero = translator.builder.ins().iconst(types::I64, 0);
                    translator.builder.ins().return_(&[zero]);
                }
            } else {
                translator.builder.ins().return_(&[]);
            }

            translator.builder.finalize();
        }

        self.module.define_function(func_id, &mut ctx)?;
        self.module.clear_context(&mut ctx);

        Ok(())
    }

    pub fn finish(self) -> Result<Vec<u8>> {
        let product = self.module.finish();
        Ok(product.emit()?)
    }
}

impl<'a> FunctionTranslator<'a> {
    fn declare_variable(&mut self, name: &str, ty: types::Type) -> Variable {
        let var = Variable::new(self.variable_counter);
        self.variable_counter += 1;
        self.builder.declare_var(var, ty);
        self.variables.insert(name.to_string(), var);
        var
    }

    fn frost_type_to_cranelift(&self, typ: &Type) -> types::Type {
        match typ {
            Type::I8 => types::I8,
            Type::I16 => types::I16,
            Type::I32 => types::I32,
            Type::I64 => types::I64,
            Type::U8 => types::I8,
            Type::U16 => types::I16,
            Type::U32 => types::I32,
            Type::U64 => types::I64,
            Type::F32 => types::F32,
            Type::F64 => types::F64,
            Type::Bool => types::I8,
            Type::Ptr(_) | Type::Ref(_) | Type::RefMut(_) => self.pointer_type,
            _ => types::I64,
        }
    }

    fn translate_statement(&mut self, statement: &Statement) -> Result<Option<Value>> {
        match statement {
            Statement::Let { name, type_annotation, value, mutable: _ } => {
                let val = self.translate_expression(value)?;
                let ty = type_annotation
                    .as_ref()
                    .map(|t| self.frost_type_to_cranelift(t))
                    .unwrap_or(types::I64);
                let var = self.declare_variable(name, ty);
                self.builder.def_var(var, val);
                Ok(None)
            }

            Statement::Constant(name, expr) => {
                let val = self.translate_expression(expr)?;
                let var = self.declare_variable(name, types::I64);
                self.builder.def_var(var, val);
                Ok(None)
            }

            Statement::Return(expr) => {
                let val = self.translate_expression(expr)?;
                self.builder.ins().return_(&[val]);
                Ok(Some(val))
            }

            Statement::Expression(expr) => {
                let val = self.translate_expression(expr)?;
                Ok(Some(val))
            }

            Statement::Assignment(lhs, rhs) => {
                let val = self.translate_expression(rhs)?;
                match lhs {
                    Expression::Identifier(name) => {
                        if let Some(&var) = self.variables.get(name) {
                            self.builder.def_var(var, val);
                        }
                    }
                    Expression::Dereference(inner) => {
                        let ptr = self.translate_expression(inner)?;
                        self.builder.ins().store(MemFlags::new(), val, ptr, 0);
                    }
                    Expression::FieldAccess(obj, field) => {
                        self.translate_field_store(obj, field, val)?;
                    }
                    Expression::Index(arr, idx) => {
                        let base = self.translate_expression(arr)?;
                        let index = self.translate_expression(idx)?;
                        let offset = self.builder.ins().imul_imm(index, 8);
                        let ptr = self.builder.ins().iadd(base, offset);
                        self.builder.ins().store(MemFlags::new(), val, ptr, 0);
                    }
                    _ => {}
                }
                Ok(None)
            }

            Statement::While(condition, body) => {
                let header_block = self.builder.create_block();
                let body_block = self.builder.create_block();
                let exit_block = self.builder.create_block();

                self.builder.ins().jump(header_block, &[]);

                self.builder.switch_to_block(header_block);
                let cond_val = self.translate_expression(condition)?;
                self.builder.ins().brif(cond_val, body_block, &[], exit_block, &[]);

                self.builder.switch_to_block(body_block);
                self.loop_exit_blocks.push(exit_block);
                self.loop_continue_blocks.push(header_block);

                for stmt in body {
                    self.translate_statement(stmt)?;
                }

                self.loop_exit_blocks.pop();
                self.loop_continue_blocks.pop();

                self.builder.ins().jump(header_block, &[]);

                self.builder.seal_block(header_block);
                self.builder.seal_block(body_block);
                self.builder.seal_block(exit_block);

                self.builder.switch_to_block(exit_block);

                Ok(None)
            }

            Statement::For(iter_name, range, body) => {
                if let Expression::Range(start, end, inclusive) = range {
                    let start_val = self.translate_expression(start)?;
                    let end_val = self.translate_expression(end)?;

                    let iter_var = self.declare_variable(iter_name, types::I64);
                    self.builder.def_var(iter_var, start_val);

                    let header_block = self.builder.create_block();
                    let body_block = self.builder.create_block();
                    let exit_block = self.builder.create_block();

                    self.builder.ins().jump(header_block, &[]);

                    self.builder.switch_to_block(header_block);
                    let current = self.builder.use_var(iter_var);
                    let compare = if *inclusive {
                        IntCC::SignedLessThanOrEqual
                    } else {
                        IntCC::SignedLessThan
                    };
                    let cond = self.builder.ins().icmp(compare, current, end_val);
                    self.builder.ins().brif(cond, body_block, &[], exit_block, &[]);

                    self.builder.switch_to_block(body_block);
                    self.loop_exit_blocks.push(exit_block);
                    self.loop_continue_blocks.push(header_block);

                    for stmt in body {
                        self.translate_statement(stmt)?;
                    }

                    let current = self.builder.use_var(iter_var);
                    let incremented = self.builder.ins().iadd_imm(current, 1);
                    self.builder.def_var(iter_var, incremented);

                    self.loop_exit_blocks.pop();
                    self.loop_continue_blocks.pop();

                    self.builder.ins().jump(header_block, &[]);

                    self.builder.seal_block(header_block);
                    self.builder.seal_block(body_block);
                    self.builder.seal_block(exit_block);

                    self.builder.switch_to_block(exit_block);
                }
                Ok(None)
            }

            Statement::Break => {
                if let Some(&exit_block) = self.loop_exit_blocks.last() {
                    self.builder.ins().jump(exit_block, &[]);
                    let unreachable = self.builder.create_block();
                    self.builder.switch_to_block(unreachable);
                    self.builder.seal_block(unreachable);
                }
                Ok(None)
            }

            Statement::Continue => {
                if let Some(&continue_block) = self.loop_continue_blocks.last() {
                    self.builder.ins().jump(continue_block, &[]);
                    let unreachable = self.builder.create_block();
                    self.builder.switch_to_block(unreachable);
                    self.builder.seal_block(unreachable);
                }
                Ok(None)
            }

            Statement::Struct(..) => Ok(None),
            Statement::Enum(_, _) => Ok(None),
            Statement::TypeAlias(_, _) => Ok(None),
            Statement::Import(_) => Ok(None),
            Statement::Defer(_) => Ok(None),
            Statement::InterpolatedConstant(_, _) => Ok(None),
            Statement::Extern { .. } => Ok(None),
            Statement::PushContext { body, .. } => {
                for statement in body {
                    self.translate_statement(statement)?;
                }
                Ok(None)
            }
            Statement::PushAllocator { body, .. } => {
                for statement in body {
                    self.translate_statement(statement)?;
                }
                Ok(None)
            }
        }
    }

    fn translate_expression(&mut self, expr: &Expression) -> Result<Value> {
        match expr {
            Expression::Literal(lit) => self.translate_literal(lit),

            Expression::Boolean(b) => {
                let val = if *b { 1i64 } else { 0i64 };
                Ok(self.builder.ins().iconst(types::I8, val))
            }

            Expression::Identifier(name) => {
                if let Some(&var) = self.variables.get(name) {
                    Ok(self.builder.use_var(var))
                } else {
                    bail!("Undefined variable: {}", name)
                }
            }

            Expression::Prefix(op, inner) => {
                let val = self.translate_expression(inner)?;
                match op {
                    Operator::Negate => Ok(self.builder.ins().ineg(val)),
                    Operator::Not => {
                        let one = self.builder.ins().iconst(types::I8, 1);
                        Ok(self.builder.ins().bxor(val, one))
                    }
                    _ => bail!("Unsupported prefix operator: {:?}", op),
                }
            }

            Expression::Infix(left, op, right) => {
                let lhs = self.translate_expression(left)?;
                let rhs = self.translate_expression(right)?;
                self.translate_binop(*op, lhs, rhs)
            }

            Expression::If(condition, then_block, else_block) => {
                let cond_val = self.translate_expression(condition)?;

                let then_bb = self.builder.create_block();
                let else_bb = self.builder.create_block();
                let merge_bb = self.builder.create_block();

                self.builder.append_block_param(merge_bb, types::I64);

                self.builder.ins().brif(cond_val, then_bb, &[], else_bb, &[]);

                self.builder.switch_to_block(then_bb);
                self.builder.seal_block(then_bb);
                let mut then_val = self.builder.ins().iconst(types::I64, 0);
                for stmt in then_block {
                    if let Some(val) = self.translate_statement(stmt)? {
                        then_val = val;
                    }
                }
                self.builder.ins().jump(merge_bb, &[then_val]);

                self.builder.switch_to_block(else_bb);
                self.builder.seal_block(else_bb);
                let else_val = if let Some(else_stmts) = else_block {
                    let mut val = self.builder.ins().iconst(types::I64, 0);
                    for stmt in else_stmts {
                        if let Some(v) = self.translate_statement(stmt)? {
                            val = v;
                        }
                    }
                    val
                } else {
                    self.builder.ins().iconst(types::I64, 0)
                };
                self.builder.ins().jump(merge_bb, &[else_val]);

                self.builder.switch_to_block(merge_bb);
                self.builder.seal_block(merge_bb);

                Ok(self.builder.block_params(merge_bb)[0])
            }

            Expression::Call(func, args) => {
                let mut arg_vals = Vec::new();
                for arg in args {
                    arg_vals.push(self.translate_expression(arg)?);
                }

                if let Expression::Identifier(name) = func.as_ref() {
                    if let Some(&func_id) = self.functions.get(name) {
                        let local_callee = self.module.declare_func_in_func(func_id, self.builder.func);
                        let call = self.builder.ins().call(local_callee, &arg_vals);
                        let results = self.builder.inst_results(call);
                        if results.is_empty() {
                            return Ok(self.builder.ins().iconst(types::I64, 0));
                        }
                        return Ok(results[0]);
                    }
                }

                Ok(self.builder.ins().iconst(types::I64, 0))
            }

            Expression::Index(arr, idx) => {
                let base = self.translate_expression(arr)?;
                let index = self.translate_expression(idx)?;
                let offset = self.builder.ins().imul_imm(index, 8);
                let ptr = self.builder.ins().iadd(base, offset);
                Ok(self.builder.ins().load(types::I64, MemFlags::new(), ptr, 0))
            }

            Expression::FieldAccess(obj, field) => {
                self.translate_field_access(obj, field)
            }

            Expression::AddressOf(inner) | Expression::Borrow(inner) | Expression::BorrowMut(inner) => {
                if let Expression::Identifier(name) = inner.as_ref() {
                    if let Some(&var) = self.variables.get(name) {
                        let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
                            StackSlotKind::ExplicitSlot,
                            8,
                            0,
                        ));
                        let val = self.builder.use_var(var);
                        self.builder.ins().stack_store(val, slot, 0);
                        return Ok(self.builder.ins().stack_addr(self.pointer_type, slot, 0));
                    }
                }
                bail!("Cannot take address of expression")
            }

            Expression::Dereference(inner) => {
                let ptr = self.translate_expression(inner)?;
                Ok(self.builder.ins().load(types::I64, MemFlags::new(), ptr, 0))
            }

            Expression::StructInit(name, fields) => {
                let struct_info = self.structs.get(name).cloned();
                if let Some(info) = struct_info {
                    let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        info.size as u32,
                        0,
                    ));

                    for (field_name, field_expr) in fields {
                        let val = self.translate_expression(field_expr)?;
                        if let Some(&offset) = info.field_offsets.get(field_name) {
                            self.builder.ins().stack_store(val, slot, offset as i32);
                        }
                    }

                    Ok(self.builder.ins().stack_addr(self.pointer_type, slot, 0))
                } else {
                    bail!("Unknown struct: {}", name)
                }
            }

            Expression::Sizeof(typ) => {
                let size = self.type_size(typ);
                Ok(self.builder.ins().iconst(types::I64, size as i64))
            }

            Expression::Range(_, _, _) => {
                bail!("Range expressions should be handled in for loops")
            }

            Expression::Tuple(elements) => {
                if elements.is_empty() {
                    return Ok(self.builder.ins().iconst(types::I64, 0));
                }
                self.translate_expression(&elements[0])
            }

            Expression::Function(_, _, _) | Expression::Proc(_, _, _) => {
                Ok(self.builder.ins().iconst(types::I64, 0))
            }

            Expression::Switch(_, _) => {
                Ok(self.builder.ins().iconst(types::I64, 0))
            }

            Expression::EnumVariantInit(_, _, _) => {
                Ok(self.builder.ins().iconst(types::I64, 0))
            }

            Expression::ComptimeBlock(_) => {
                Ok(self.builder.ins().iconst(types::I64, 0))
            }

            Expression::ComptimeFor { .. } => {
                Ok(self.builder.ins().iconst(types::I64, 0))
            }

            Expression::TypeValue(_) => {
                Ok(self.builder.ins().iconst(types::I64, 0))
            }

            Expression::Typename(_) => {
                Ok(self.builder.ins().iconst(types::I64, 0))
            }

            Expression::InterpolatedIdent(_) => {
                Ok(self.builder.ins().iconst(types::I64, 0))
            }

            Expression::Unsafe(body) => {
                let mut result = self.builder.ins().iconst(types::I64, 0);
                for statement in body {
                    if let Some(value) = self.translate_statement(statement)? {
                        result = value;
                    }
                }
                Ok(result)
            }
            Expression::ContextAccess => {
                Ok(self.builder.ins().iconst(types::I64, 0))
            }
            Expression::IfLet(_, _, _, _) => {
                Ok(self.builder.ins().iconst(types::I64, 0))
            }
        }
    }

    fn translate_literal(&mut self, lit: &Literal) -> Result<Value> {
        match lit {
            Literal::Integer(n) => Ok(self.builder.ins().iconst(types::I64, *n)),
            Literal::Float(f) => Ok(self.builder.ins().f64const(*f)),
            Literal::Float32(f) => Ok(self.builder.ins().f32const(*f)),
            Literal::String(s) => {
                let data_id = if let Some(&existing) = self.string_data.get(s) {
                    existing
                } else {
                    let name = format!(".str.{}", *self.string_counter);
                    *self.string_counter += 1;

                    let data_id = self.module.declare_data(&name, Linkage::Local, false, false)?;
                    let mut data_desc = DataDescription::new();
                    let mut bytes = s.as_bytes().to_vec();
                    bytes.push(0);
                    data_desc.define(bytes.into_boxed_slice());
                    self.module.define_data(data_id, &data_desc)?;
                    self.string_data.insert(s.clone(), data_id);
                    data_id
                };

                let local_data = self.module.declare_data_in_func(data_id, self.builder.func);
                let ptr = self.builder.ins().symbol_value(self.pointer_type, local_data);
                Ok(ptr)
            }
            Literal::Array(elements) => {
                if elements.is_empty() {
                    return Ok(self.builder.ins().iconst(self.pointer_type, 0));
                }

                let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    (elements.len() * 8) as u32,
                    0,
                ));

                for (index, elem) in elements.iter().enumerate() {
                    let val = self.translate_expression(elem)?;
                    self.builder.ins().stack_store(val, slot, (index * 8) as i32);
                }

                Ok(self.builder.ins().stack_addr(self.pointer_type, slot, 0))
            }
            Literal::Boolean(b) => Ok(self.builder.ins().iconst(types::I64, if *b { 1 } else { 0 })),
            Literal::HashMap(pairs) => {
                let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    (pairs.len() * 16) as u32,
                    0,
                ));
                Ok(self.builder.ins().stack_addr(self.pointer_type, slot, 0))
            }
        }
    }

    fn translate_binop(&mut self, op: Operator, lhs: Value, rhs: Value) -> Result<Value> {
        match op {
            Operator::Add => Ok(self.builder.ins().iadd(lhs, rhs)),
            Operator::Subtract => Ok(self.builder.ins().isub(lhs, rhs)),
            Operator::Multiply => Ok(self.builder.ins().imul(lhs, rhs)),
            Operator::Divide => Ok(self.builder.ins().sdiv(lhs, rhs)),
            Operator::Modulo => Ok(self.builder.ins().srem(lhs, rhs)),

            Operator::Equal => {
                let cmp = self.builder.ins().icmp(IntCC::Equal, lhs, rhs);
                Ok(self.builder.ins().uextend(types::I64, cmp))
            }
            Operator::NotEqual => {
                let cmp = self.builder.ins().icmp(IntCC::NotEqual, lhs, rhs);
                Ok(self.builder.ins().uextend(types::I64, cmp))
            }
            Operator::LessThan => {
                let cmp = self.builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs);
                Ok(self.builder.ins().uextend(types::I64, cmp))
            }
            Operator::LessThanOrEqual => {
                let cmp = self.builder.ins().icmp(IntCC::SignedLessThanOrEqual, lhs, rhs);
                Ok(self.builder.ins().uextend(types::I64, cmp))
            }
            Operator::GreaterThan => {
                let cmp = self.builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs);
                Ok(self.builder.ins().uextend(types::I64, cmp))
            }
            Operator::GreaterThanOrEqual => {
                let cmp = self.builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs);
                Ok(self.builder.ins().uextend(types::I64, cmp))
            }

            Operator::And => Ok(self.builder.ins().band(lhs, rhs)),
            Operator::Or => Ok(self.builder.ins().bor(lhs, rhs)),
            Operator::BitwiseAnd => Ok(self.builder.ins().band(lhs, rhs)),
            Operator::BitwiseOr => Ok(self.builder.ins().bor(lhs, rhs)),
            Operator::ShiftLeft => Ok(self.builder.ins().ishl(lhs, rhs)),
            Operator::ShiftRight => Ok(self.builder.ins().sshr(lhs, rhs)),

            _ => bail!("Unsupported operator: {:?}", op),
        }
    }

    fn translate_field_access(&mut self, obj: &Expression, field: &str) -> Result<Value> {
        let base = self.translate_expression(obj)?;

        if let Expression::Identifier(name) = obj {
            if let Some(var) = self.variables.get(name) {
                let _ = var;
            }
        }

        for (struct_name, info) in self.structs.iter() {
            if let Some(&offset) = info.field_offsets.get(field) {
                let _ = struct_name;
                let ptr = if offset > 0 {
                    self.builder.ins().iadd_imm(base, offset as i64)
                } else {
                    base
                };
                return Ok(self.builder.ins().load(types::I64, MemFlags::new(), ptr, 0));
            }
        }

        Ok(self.builder.ins().load(types::I64, MemFlags::new(), base, 0))
    }

    fn translate_field_store(&mut self, obj: &Expression, field: &str, val: Value) -> Result<()> {
        let base = self.translate_expression(obj)?;

        for (_, info) in self.structs.iter() {
            if let Some(&offset) = info.field_offsets.get(field) {
                let ptr = if offset > 0 {
                    self.builder.ins().iadd_imm(base, offset as i64)
                } else {
                    base
                };
                self.builder.ins().store(MemFlags::new(), val, ptr, 0);
                return Ok(());
            }
        }

        self.builder.ins().store(MemFlags::new(), val, base, 0);
        Ok(())
    }

    fn type_size(&self, typ: &Type) -> usize {
        match typ {
            Type::I8 | Type::U8 | Type::Bool => 1,
            Type::I16 | Type::U16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64 | Type::U64 | Type::F64 => 8,
            Type::Ptr(_) | Type::Ref(_) | Type::RefMut(_) => 8,
            Type::Array(inner, count) => self.type_size(inner) * count,
            Type::Struct(name) => {
                self.structs.get(name).map(|s| s.size).unwrap_or(8)
            }
            _ => 8,
        }
    }
}

pub fn compile_to_object(statements: &[Statement]) -> Result<Vec<u8>> {
    let mut ctx = CodegenContext::new()?;
    ctx.compile_program(statements)?;
    ctx.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Lexer, Parser};

    fn parse_and_compile(source: &str) -> Result<Vec<u8>> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let statements = parser.parse()?;
        compile_to_object(&statements)
    }

    #[test]
    fn test_compile_simple_function() {
        let source = r#"
            add :: fn(a: i64, b: i64) -> i64 {
                a + b
            }
        "#;
        let result = parse_and_compile(source);
        assert!(result.is_ok());
        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_compile_arithmetic() {
        let source = r#"
            calc :: fn(x: i64) -> i64 {
                y := x * 2;
                z := y + 10;
                z - 5
            }
        "#;
        let result = parse_and_compile(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_if_expression() {
        let source = r#"
            max :: fn(a: i64, b: i64) -> i64 {
                if (a > b) { a } else { b }
            }
        "#;
        let result = parse_and_compile(source);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_while_loop() {
        let source = r#"
            sum_to :: fn(n: i64) -> i64 {
                mut result := 0;
                mut i := 0;
                while (i < n) {
                    result = result + i;
                    i = i + 1;
                }
                result
            }
        "#;
        let result = parse_and_compile(source);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_for_loop() {
        let source = r#"
            sum_range :: fn() -> i64 {
                mut result := 0;
                for i in 0..10 {
                    result = result + i;
                }
                result
            }
        "#;
        let result = parse_and_compile(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_struct() {
        let source = r#"
            Point :: struct {
                x: i64,
                y: i64,
            }

            make_point :: fn() -> i64 {
                p := Point { x = 10, y = 20 };
                0
            }
        "#;
        let result = parse_and_compile(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_function_call() {
        let source = r#"
            double :: fn(x: i64) -> i64 {
                x * 2
            }

            quad :: fn(x: i64) -> i64 {
                double(double(x))
            }
        "#;
        let result = parse_and_compile(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_comparison_operators() {
        let source = r#"
            compare :: fn(a: i64, b: i64) -> i64 {
                if (a == b) { 0 }
                else if (a < b) { 1 }
                else if (a > b) { 2 }
                else if (a <= b) { 3 }
                else if (a >= b) { 4 }
                else if (a != b) { 5 }
                else { 6 }
            }
        "#;
        let result = parse_and_compile(source);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_logical_operators() {
        let source = r#"
            logic :: fn(a: bool, b: bool) -> i64 {
                if (a && b) { 1 }
                else if (a || b) { 2 }
                else { 0 }
            }
        "#;
        let result = parse_and_compile(source);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_bitwise_operators() {
        let source = r#"
            bits :: fn(a: i64, b: i64) -> i64 {
                c := a & b;
                d := a | b;
                e := a << 2;
                f := a >> 1;
                c + d + e + f
            }
        "#;
        let result = parse_and_compile(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_break_continue() {
        let source = r#"
            with_break :: fn() -> i64 {
                mut i := 0;
                while (true) {
                    if (i > 10) { break; }
                    i = i + 1;
                }
                i
            }
        "#;
        let result = parse_and_compile(source);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_object_file_has_valid_header() {
        let source = r#"
            main :: fn() -> i64 {
                42
            }
        "#;
        let bytes = parse_and_compile(source).unwrap();

        assert!(bytes.len() > 4);
    }

    #[test]
    fn test_compile_extern_function() {
        let source = r#"
            puts :: extern fn(s: ^i8) -> i32

            main :: fn() -> i64 {
                0
            }
        "#;
        let result = parse_and_compile(source);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_extern_call() {
        let source = r#"
            add_extern :: extern fn(a: i64, b: i64) -> i64

            main :: fn() -> i64 {
                add_extern(10, 20)
            }
        "#;
        let result = parse_and_compile(source);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_string_literal() {
        let source = r#"
            puts :: extern fn(s: ^i8) -> i32

            main :: fn() -> i64 {
                puts("Hello, World!");
                0
            }
        "#;
        let result = parse_and_compile(source);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_multiple_strings() {
        let source = r#"
            puts :: extern fn(s: ^i8) -> i32

            main :: fn() -> i64 {
                puts("First string");
                puts("Second string");
                puts("First string");
                0
            }
        "#;
        let result = parse_and_compile(source);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_with_libc_functions() {
        let source = r#"
            puts :: extern fn(s: ^i8) -> i32

            main :: fn() -> i64 {
                puts("Hello from Frost!");
                puts("Another line");
                0
            }
        "#;
        let result = parse_and_compile(source);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_array_literal() {
        let source = r#"
            main :: fn() -> i64 {
                arr := [1, 2, 3, 4, 5];
                arr[2]
            }
        "#;
        let result = parse_and_compile(source);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_array_index_assignment() {
        let source = r#"
            main :: fn() -> i64 {
                mut arr := [10, 20, 30];
                arr[1] = 42;
                arr[1]
            }
        "#;
        let result = parse_and_compile(source);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_array_in_loop() {
        let source = r#"
            main :: fn() -> i64 {
                mut arr := [0, 0, 0, 0, 0];
                for i in 0..5 {
                    arr[i] = i * 2;
                }
                arr[3]
            }
        "#;
        let result = parse_and_compile(source);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
    }
}
