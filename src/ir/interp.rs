use crate::ir::{
    IrBinOp, IrConstant, IrFunction, IrModule, IrOperand, IrRvalue,
    IrStatement, IrTerminator, IrUnOp,
};
use crate::types::Type;
use std::collections::HashMap;

pub enum RunOutcome {
    Output(String),
    Unsupported(String),
}

#[derive(Clone, Copy)]
enum Value {
    Int(i64),
    Float(f64),
}

impl Value {
    fn as_i64(self) -> i64 {
        match self {
            Value::Int(value) => value,
            Value::Float(value) => value as i64,
        }
    }

    fn as_f64(self) -> f64 {
        match self {
            Value::Int(value) => value as f64,
            Value::Float(value) => value,
        }
    }
}

enum Signal {
    Unsupported(String),
}

type Eval<T> = std::result::Result<T, Signal>;

fn unsupported<T>(reason: impl Into<String>) -> Eval<T> {
    Err(Signal::Unsupported(reason.into()))
}

const MAX_DEPTH: usize = 2_000;

pub fn run_module(module: &IrModule) -> RunOutcome {
    let Some(entry) = module.functions.iter().find(|f| f.name == "main") else {
        return RunOutcome::Unsupported("no main function".to_string());
    };
    let mut interpreter = Interpreter {
        module,
        output: String::new(),
        memory: vec![0; FIRST_ADDRESS],
        literals: HashMap::new(),
    };
    // Every literal in the program is laid down before anything runs, so that
    // reading one is a read of storage like any other and mentioning the same
    // literal twice names one address.
    if let Err(Signal::Unsupported(reason)) = interpreter.place_literals() {
        return RunOutcome::Unsupported(reason);
    }
    match interpreter.call(entry, &[], 0) {
        Ok(_) => RunOutcome::Output(interpreter.output),
        Err(Signal::Unsupported(reason)) => RunOutcome::Unsupported(reason),
    }
}

// Nothing is placed at zero, so a null pointer stays a value nothing answers
// for and reaching through one is a fault here rather than a read of whatever
// happened to be laid down first.
const FIRST_ADDRESS: usize = 8;
const MEMORY_LIMIT: usize = 1 << 26;

struct Interpreter<'a> {
    module: &'a IrModule,
    output: String,
    // One flat run of bytes standing for every frame and every literal. A local
    // the build put in memory is given a place in it and the local's value is
    // that address, which is the same shape the native backend gives a stack
    // slot. Frames are never reclaimed: a run bounded by MEMORY_LIMIT is enough
    // for the programs this answers for, and reclaiming would make an address
    // handed back from a call read as storage that had been reused.
    memory: Vec<u8>,
    literals: HashMap<String, i64>,
}

enum Flow {
    Jump(usize),
    Return(Value),
}

// What a body is being read against: the function it belongs to, the values its
// locals hold, and how deep the calls are. They travel together because every
// question about an operand needs all three.
struct Frame<'f> {
    function: &'f IrFunction,
    locals: &'f [Value],
    depth: usize,
}

impl<'a> Interpreter<'a> {
    fn call(
        &mut self,
        function: &IrFunction,
        arguments: &[Value],
        depth: usize,
    ) -> Eval<Value> {
        if depth > MAX_DEPTH {
            return unsupported("recursion limit reached");
        }
        // A local the build put in memory is given its storage here, and its
        // value is the address of it. Every read and write of such a local
        // goes through that address, which is what makes `ptr_to` on one
        // answer with something the rest of the walk can follow.
        let mut locals: Vec<Value> = Vec::with_capacity(function.locals.len());
        for local in &function.locals {
            if local.in_memory {
                let address = self.reserve(local.size.max(1))?;
                locals.push(Value::Int(address));
                continue;
            }
            if !is_scalar(&local.ty) {
                return unsupported(format!(
                    "a {} local that the build left out of memory",
                    local.ty
                ));
            }
            locals.push(default_value(&local.ty));
        }
        // How a parameter arrives, matching what the native backend collects:
        // an aggregate as the address of the caller's copy, which the callee
        // copies into its own storage, and everything else as the value, stored
        // into the callee's storage where it has some.
        let mut copies: Vec<(i64, i64, usize)> = Vec::new();
        let mut writes: Vec<(i64, Value, Type)> = Vec::new();
        for (index, slot) in
            locals.iter_mut().enumerate().take(function.param_count)
        {
            let Some(incoming) = arguments.get(index) else {
                break;
            };
            let local = &function.locals[index];
            if is_aggregate(&local.ty) {
                copies.push((slot.as_i64(), incoming.as_i64(), local.size));
                continue;
            }
            if local.in_memory {
                writes.push((slot.as_i64(), *incoming, local.ty.clone()));
                continue;
            }
            *slot = *incoming;
        }
        for (to, from, size) in copies {
            self.copy_bytes(to, from, size)?;
        }
        for (address, value, ty) in writes {
            self.write_at(address, value, &ty)?;
        }
        // A function answering with an aggregate is handed the storage to put
        // it in as one more argument past the ones it declares.
        let out_pointer = if returns_aggregate(function) {
            arguments
                .get(function.param_count)
                .map(|held| held.as_i64())
        } else {
            None
        };

        let mut block_index = function.entry;
        loop {
            let block = &function.blocks[block_index];
            for statement in &block.statements {
                self.execute(function, statement, &mut locals, depth)?;
            }
            match self.terminate(function, block_index, &locals, out_pointer)? {
                Flow::Return(value) => return Ok(value),
                Flow::Jump(target) => block_index = target,
            }
        }
    }

    fn execute(
        &mut self,
        function: &IrFunction,
        statement: &IrStatement,
        locals: &mut [Value],
        depth: usize,
    ) -> Eval<()> {
        match statement {
            IrStatement::Assign(local, rvalue) => {
                let ty = function.local_type(*local).clone();
                if matches!(ty, Type::Void | Type::Unknown) {
                    // A call made for what it does rather than what it answers
                    // with still has to happen.
                    if matches!(
                        rvalue,
                        IrRvalue::Call { .. } | IrRvalue::CallIndirect { .. }
                    ) {
                        self.evaluate(function, rvalue, locals, depth)?;
                    }
                    return Ok(());
                }
                // An aggregate lands in the storage the local already names,
                // rather than replacing what the local is: a copy from another
                // one is bytes moved, and a call is handed the storage to
                // answer into.
                if is_aggregate(&ty) {
                    let destination = locals[*local].as_i64();
                    let size = function.locals[*local].size;
                    match rvalue {
                        IrRvalue::Use(IrOperand::Local(source)) => {
                            let from = self
                                .aggregate_address(function, *source, locals);
                            self.copy_bytes(destination, from, size)?;
                        }
                        IrRvalue::Call { .. }
                        | IrRvalue::CallIndirect { .. } => {
                            let frame = Frame {
                                function,
                                locals,
                                depth,
                            };
                            self.evaluate_into(
                                &frame,
                                rvalue,
                                Some(destination),
                            )?;
                        }
                        _ => {
                            return unsupported(
                                "an aggregate assigned from something that is neither a local nor a call",
                            );
                        }
                    }
                    return Ok(());
                }
                let value = self.evaluate(function, rvalue, locals, depth)?;
                if function.locals[*local].in_memory {
                    let address = locals[*local].as_i64();
                    self.write_at(address, value, &ty)?;
                    return Ok(());
                }
                locals[*local] = value;
                Ok(())
            }
            IrStatement::Own(_) | IrStatement::Consume(_) => Ok(()),
            IrStatement::Store { address, value } => {
                let ty = operand_type(function, value);
                let target =
                    self.operand_value(function, address, locals).as_i64();
                let held = self.operand_value(function, value, locals);
                self.write_at(target, held, &ty)
            }
            IrStatement::Copy {
                destination,
                source,
                size,
            } => {
                let to =
                    self.operand_value(function, destination, locals).as_i64();
                let from =
                    self.operand_value(function, source, locals).as_i64();
                self.copy_bytes(to, from, *size)
            }
        }
    }

    // Where the aggregate a local names lives. A local of aggregate type is its
    // own storage; a local of reference type holds someone else's address
    // already, so its value is the answer rather than where that value sits.
    fn aggregate_address(
        &self,
        function: &IrFunction,
        local: usize,
        locals: &[Value],
    ) -> i64 {
        if matches!(function.local_type(local), Type::Ref(_) | Type::RefMut(_))
        {
            return self
                .operand_value(function, &IrOperand::Local(local), locals)
                .as_i64();
        }
        locals[local].as_i64()
    }

    fn reserve(&mut self, size: usize) -> Eval<i64> {
        let rounded = size.div_ceil(8) * 8;
        let at = self.memory.len();
        if at + rounded > MEMORY_LIMIT {
            return unsupported(
                "the program asked for more storage than the interpreter holds",
            );
        }
        self.memory.resize(at + rounded, 0);
        Ok(at as i64)
    }

    fn span(&self, address: i64, width: usize) -> Eval<usize> {
        let Ok(at) = usize::try_from(address) else {
            return unsupported("a read through an address below zero");
        };
        if at < FIRST_ADDRESS || at + width > self.memory.len() {
            return unsupported("a read or write outside any storage");
        }
        Ok(at)
    }

    fn write_at(&mut self, address: i64, value: Value, ty: &Type) -> Eval<()> {
        let width = stored_width(ty);
        let at = self.span(address, width)?;
        let bits = match ty {
            Type::F32 => (value.as_f64() as f32).to_bits() as u64,
            Type::F64 => value.as_f64().to_bits(),
            _ => value.as_i64() as u64,
        };
        self.memory[at..at + width]
            .copy_from_slice(&bits.to_le_bytes()[..width]);
        Ok(())
    }

    fn read_at(&self, address: i64, ty: &Type) -> Eval<Value> {
        let width = stored_width(ty);
        let at = self.span(address, width)?;
        let mut bytes = [0u8; 8];
        bytes[..width].copy_from_slice(&self.memory[at..at + width]);
        let raw = u64::from_le_bytes(bytes);
        Ok(match ty {
            Type::F32 => Value::Float(f32::from_bits(raw as u32) as f64),
            Type::F64 => Value::Float(f64::from_bits(raw)),
            _ => {
                let (bits, signed) = integer_info(ty);
                Value::Int(normalize(
                    raw as i64,
                    bits.min(width as u32 * 8),
                    signed,
                ))
            }
        })
    }

    fn copy_bytes(&mut self, to: i64, from: i64, size: usize) -> Eval<()> {
        if size == 0 {
            return Ok(());
        }
        let source = self.span(from, size)?;
        let destination = self.span(to, size)?;
        self.memory.copy_within(source..source + size, destination);
        Ok(())
    }

    // Where a literal's bytes sit, laid down once and shared by every mention.
    // A trailing NUL is written after them, since a `^i8` handed to C is read
    // until one.
    fn literal_address(&mut self, text: &str) -> Eval<i64> {
        if let Some(address) = self.literals.get(text) {
            return Ok(*address);
        }
        let bytes = text.as_bytes();
        let address = self.reserve(bytes.len() + 1)?;
        let at = address as usize;
        self.memory[at..at + bytes.len()].copy_from_slice(bytes);
        self.literals.insert(text.to_string(), address);
        Ok(address)
    }

    fn place_literals(&mut self) -> Eval<()> {
        let mut found: Vec<String> = Vec::new();
        for function in &self.module.functions {
            for block in &function.blocks {
                for statement in &block.statements {
                    match statement {
                        IrStatement::Assign(_, rvalue) => {
                            collect_rvalue_literals(rvalue, &mut found);
                        }
                        IrStatement::Store { address, value } => {
                            collect_literal(address, &mut found);
                            collect_literal(value, &mut found);
                        }
                        IrStatement::Copy {
                            destination,
                            source,
                            ..
                        } => {
                            collect_literal(destination, &mut found);
                            collect_literal(source, &mut found);
                        }
                        IrStatement::Own(_) | IrStatement::Consume(_) => {}
                    }
                }
                match &block.terminator {
                    IrTerminator::Return(Some(operand)) => {
                        collect_literal(operand, &mut found);
                    }
                    IrTerminator::Branch { condition, .. } => {
                        collect_literal(condition, &mut found);
                    }
                    _ => {}
                }
            }
        }
        for text in found {
            self.literal_address(&text)?;
        }
        Ok(())
    }

    fn terminate(
        &mut self,
        function: &IrFunction,
        block_index: usize,
        locals: &[Value],
        out_pointer: Option<i64>,
    ) -> Eval<Flow> {
        match &function.blocks[block_index].terminator {
            IrTerminator::Return(None) => Ok(Flow::Return(Value::Int(0))),
            IrTerminator::Return(Some(operand)) => {
                // A function answering with an aggregate copies it into the
                // storage the caller handed in, and answers with nothing.
                if let Some(out) = out_pointer {
                    if let IrOperand::Local(source) = operand {
                        let from =
                            self.aggregate_address(function, *source, locals);
                        let size = function.locals[*source].size;
                        self.copy_bytes(out, from, size)?;
                    }
                    return Ok(Flow::Return(Value::Int(0)));
                }
                Ok(Flow::Return(self.operand_value(function, operand, locals)))
            }
            IrTerminator::Jump(target) => Ok(Flow::Jump(*target)),
            IrTerminator::Branch {
                condition,
                then_block,
                else_block,
            } => {
                let taken =
                    self.operand_value(function, condition, locals).as_i64()
                        != 0;
                Ok(Flow::Jump(if taken { *then_block } else { *else_block }))
            }
            IrTerminator::Unreachable => unsupported("reached unreachable"),
        }
    }

    fn evaluate(
        &mut self,
        function: &IrFunction,
        rvalue: &IrRvalue,
        locals: &[Value],
        depth: usize,
    ) -> Eval<Value> {
        let frame = Frame {
            function,
            locals,
            depth,
        };
        self.evaluate_into(&frame, rvalue, None)
    }

    fn evaluate_into(
        &mut self,
        frame: &Frame,
        rvalue: &IrRvalue,
        out: Option<i64>,
    ) -> Eval<Value> {
        let function = frame.function;
        let locals = frame.locals;
        let depth = frame.depth;
        match rvalue {
            IrRvalue::Use(operand) => {
                Ok(self.operand_value(function, operand, locals))
            }
            IrRvalue::Binary(op, left, right) => {
                let ty = operand_type(function, left);
                binary(
                    *op,
                    self.operand_value(function, left, locals),
                    self.operand_value(function, right, locals),
                    &ty,
                )
            }
            IrRvalue::Unary(op, operand) => {
                let ty = operand_type(function, operand);
                let value = self.operand_value(function, operand, locals);
                Ok(unary(*op, value, &ty))
            }
            IrRvalue::Cast(operand, target) => {
                let source = operand_type(function, operand);
                let value = self.operand_value(function, operand, locals);
                Ok(cast(value, &source, target))
            }
            IrRvalue::Call {
                function: callee,
                arguments,
            } => {
                self.call_named(frame, callee, arguments, out)
            }
            IrRvalue::CallIndirect {
                callee, arguments, ..
            } => {
                let index =
                    self.operand_value(function, callee, locals).as_i64();
                let Some(target) = usize::try_from(index)
                    .ok()
                    .and_then(|index| self.module.functions.get(index))
                else {
                    return unsupported("indirect call to unknown target");
                };
                let mut values =
                    self.argument_values(function, arguments, locals);
                if let Some(address) = out {
                    values.push(Value::Int(address));
                }
                self.call(target, &values, depth + 1)
            }
            IrRvalue::FunctionAddress(name) => {
                match self.module.functions.iter().position(|f| &f.name == name)
                {
                    Some(index) => Ok(Value::Int(index as i64)),
                    None => unsupported("address of unknown function"),
                }
            }
            IrRvalue::AddressOf { local, offset } => {
                if !function.locals[*local].in_memory {
                    return unsupported(
                        "the address of a local the build left out of memory",
                    );
                }
                Ok(Value::Int(locals[*local].as_i64() + *offset as i64))
            }
            IrRvalue::FieldAddress { base, offset } => {
                let address =
                    self.operand_value(function, base, locals).as_i64();
                Ok(Value::Int(address + *offset as i64))
            }
            IrRvalue::ElementAddress {
                base,
                index,
                element_size,
            } => {
                let address =
                    self.operand_value(function, base, locals).as_i64();
                let step = self.operand_value(function, index, locals).as_i64();
                Ok(Value::Int(address + step * *element_size as i64))
            }
            IrRvalue::Load { address, ty } => {
                let at = self.operand_value(function, address, locals).as_i64();
                self.read_at(at, ty)
            }
        }
    }

    fn argument_values(
        &mut self,
        function: &IrFunction,
        arguments: &[IrOperand],
        locals: &[Value],
    ) -> Vec<Value> {
        arguments
            .iter()
            .map(|argument| self.operand_value(function, argument, locals))
            .collect()
    }

    fn call_named(
        &mut self,
        frame: &Frame,
        callee: &str,
        arguments: &[IrOperand],
        out: Option<i64>,
    ) -> Eval<Value> {
        let function = frame.function;
        let locals = frame.locals;
        let depth = frame.depth;
        if callee == "printf" {
            return self.printf(function, arguments, locals);
        }
        // What std/io.frost writes through. The interpreter is one of the
        // backends a program is checked against, so it has to write the same
        // bytes the ones that link the runtime do.
        if callee == "frost_rt_write_bytes" {
            return self.write_bytes(function, arguments, locals);
        }
        if callee == "frost_rt_write_i64" {
            let value = self.operand_value(function, &arguments[0], locals);
            let Value::Int(number) = value else {
                return unsupported("writing a value that is not an integer");
            };
            self.output.push_str(&number.to_string());
            return Ok(Value::Int(0));
        }
        // A float is written the way C writes `%g`, which this does not
        // reproduce, so a program printing one is left to the backends that
        // link the runtime rather than answered differently here.
        if callee == "frost_rt_write_f64" {
            return unsupported("writing a float");
        }
        if callee == "frost_rt_write_char" {
            let value = self.operand_value(function, &arguments[0], locals);
            let Value::Int(byte) = value else {
                return unsupported("writing a byte that is not an integer");
            };
            self.output.push(byte as u8 as char);
            return Ok(Value::Int(0));
        }
        // An index against the run it reaches into. Both forms are answered
        // here, since every indexed read and write in a compiled program goes
        // through one of them and a walk that stopped at them would reach no
        // program that indexes anything.
        if callee == "frost_rt_bounds_check" || callee == "frost_rt_check_index"
        {
            let index = self.operand_value(function, &arguments[0], locals);
            let length = self.operand_value(function, &arguments[1], locals);
            let (index, length) = (index.as_i64(), length.as_i64());
            if index < 0 || index >= length {
                return unsupported(format!(
                    "index {index} out of bounds for length {length}"
                ));
            }
            return Ok(Value::Int(index));
        }
        // A view against the run it is taken from, and the bytes a count of a
        // width comes to.
        if callee == "frost_rt_check_span" {
            let from = self.operand_value(function, &arguments[0], locals);
            let count = self.operand_value(function, &arguments[1], locals);
            let room = self.operand_value(function, &arguments[2], locals);
            let (from, count, room) =
                (from.as_i64(), count.as_i64(), room.as_i64());
            if from < 0 || from > room {
                return unsupported(format!(
                    "a view cannot start {from} elements into a run of {room}"
                ));
            }
            if count < 0 || count > room - from {
                return unsupported(format!(
                    "a view of {count} elements starting at {from} reaches past a run of {room}"
                ));
            }
            return Ok(Value::Int(count));
        }
        if callee == "frost_rt_check_size" {
            let count = self.operand_value(function, &arguments[0], locals);
            let width = self.operand_value(function, &arguments[1], locals);
            let (count, width) = (count.as_i64(), width.as_i64());
            if count < 0 || width <= 0 {
                return unsupported(format!(
                    "cannot allocate {count} elements of {width} bytes"
                ));
            }
            if count > i64::MAX / width {
                return unsupported(format!(
                    "{count} elements of {width} bytes is more memory than can be addressed"
                ));
            }
            return Ok(Value::Int(count * width));
        }
        // Every slice is built through this, so the interpreter has to answer it
        // the way the linked runtime does or a program that builds one stops
        // here rather than being compared against the other two backends.
        if callee == "frost_rt_check_length" {
            let value = self.operand_value(function, &arguments[0], locals);
            let Value::Int(length) = value else {
                return unsupported("a slice length that is not an integer");
            };
            if length < 0 {
                return unsupported(format!(
                    "a slice cannot be {length} elements long"
                ));
            }
            return Ok(Value::Int(length));
        }
        if let Some(target) =
            self.module.functions.iter().find(|f| f.name == callee)
        {
            let mut values = self.argument_values(function, arguments, locals);
            if let Some(address) = out {
                values.push(Value::Int(address));
            }
            return self.call(target, &values, depth + 1);
        }
        unsupported(format!("call to external '{callee}'"))
    }

    fn printf(
        &mut self,
        function: &IrFunction,
        arguments: &[IrOperand],
        locals: &[Value],
    ) -> Eval<Value> {
        let Some(IrOperand::Constant(IrConstant::CString(format))) =
            arguments.first()
        else {
            return unsupported("printf with a non-literal format");
        };
        let format = format.clone();
        let rendered =
            self.render_format(function, &format, &arguments[1..], locals)?;
        let length = rendered.len();
        self.output.push_str(&rendered);
        Ok(Value::Int(length as i64))
    }

    // A run of bytes given as a pointer and a length, read out of the storage
    // the pointer names. A literal is laid into that storage the first time it
    // is mentioned, so a literal and a `str` the program built arrive the same
    // way and neither is a special case.
    fn write_bytes(
        &mut self,
        function: &IrFunction,
        arguments: &[IrOperand],
        locals: &[Value],
    ) -> Eval<Value> {
        let (Some(address), Some(length)) =
            (arguments.first(), arguments.get(1))
        else {
            return unsupported("writing bytes without a pointer and a length");
        };
        let at = self.operand_value(function, address, locals).as_i64();
        let count = self.operand_value(function, length, locals).as_i64();
        let Ok(count) = usize::try_from(count) else {
            return unsupported("writing a run of fewer than no bytes");
        };
        if count == 0 {
            return Ok(Value::Int(0));
        }
        let start = self.span(at, count)?;
        let bytes = &self.memory[start..start + count];
        self.output.push_str(&String::from_utf8_lossy(bytes));
        Ok(Value::Int(0))
    }

    fn operand_value(
        &self,
        function: &IrFunction,
        operand: &IrOperand,
        locals: &[Value],
    ) -> Value {
        match operand {
            // A local the build put in memory holds its storage rather than its
            // value, so reading one is a read of that storage. An aggregate is
            // the exception: what names it is the storage.
            IrOperand::Local(local) => {
                let held = function.local_type(*local);
                if function.locals[*local].in_memory && !is_aggregate(held) {
                    return self
                        .read_at(locals[*local].as_i64(), held)
                        .unwrap_or(Value::Int(0));
                }
                locals[*local]
            }
            IrOperand::Constant(constant) => match constant {
                IrConstant::Integer(value, _) => Value::Int(*value),
                IrConstant::Float(value, _) => Value::Float(*value),
                IrConstant::Bool(value) => Value::Int(*value as i64),
                IrConstant::Unit => Value::Int(0),
                IrConstant::CString(text) => {
                    Value::Int(self.literals.get(text).copied().unwrap_or(0))
                }
            },
        }
    }
}

impl Interpreter<'_> {
    fn render_format(
        &self,
        function: &IrFunction,
        format: &str,
        arguments: &[IrOperand],
        locals: &[Value],
    ) -> Eval<String> {
        let mut result = String::new();
        let mut argument_index = 0;
        let mut characters = format.chars().peekable();
        while let Some(character) = characters.next() {
            if character != '%' {
                result.push(character);
                continue;
            }
            let mut specifier = String::new();
            loop {
                let Some(next) = characters.next() else {
                    return unsupported("truncated format specifier");
                };
                if next == '%' && specifier.is_empty() {
                    result.push('%');
                    break;
                }
                if is_conversion(next) {
                    let value = arguments
                        .get(argument_index)
                        .map(|held| self.operand_value(function, held, locals));
                    argument_index += 1;
                    let text =
                        self.format_argument(&specifier, next, value)?;
                    result.push_str(&text);
                    break;
                }
                specifier.push(next);
            }
        }
        Ok(result)
    }

    fn format_argument(
        &self,
        flags: &str,
        conversion: char,
        argument: Option<Value>,
    ) -> Eval<String> {
        let trimmed = flags.trim_end_matches('l');
        let Some(held) = argument else {
            return unsupported("printf missing argument");
        };
        match conversion {
            'd' | 'i' => Ok(held.as_i64().to_string()),
            'u' => Ok((held.as_i64() as u64).to_string()),
            'x' => Ok(format!("{:x}", held.as_i64() as u64)),
            'c' => {
                let value = held.as_i64();
                match u32::try_from(value).ok().and_then(char::from_u32) {
                    Some(character) => Ok(character.to_string()),
                    None => unsupported("invalid character in printf"),
                }
            }
            'f' | 'F' => {
                let value = held.as_f64();
                let precision = parse_precision(trimmed).unwrap_or(6);
                Ok(format!("{value:.precision$}"))
            }
            's' => {
                let at = held.as_i64();
                let start = self.span(at, 1)?;
                let end = self.memory[start..]
                    .iter()
                    .position(|byte| *byte == 0)
                    .map_or(self.memory.len(), |offset| start + offset);
                Ok(String::from_utf8_lossy(&self.memory[start..end])
                    .into_owned())
            }
            _ => unsupported(format!(
                "unsupported printf conversion '%{conversion}'"
            )),
        }
    }
}

fn is_conversion(character: char) -> bool {
    matches!(
        character,
        'd' | 'i'
            | 'u'
            | 'o'
            | 'x'
            | 'X'
            | 'e'
            | 'E'
            | 'f'
            | 'F'
            | 'g'
            | 'G'
            | 'c'
            | 's'
            | 'p'
    )
}

fn parse_precision(flags: &str) -> Option<usize> {
    let dot = flags.find('.')?;
    flags[dot + 1..].parse().ok()
}

fn collect_literal(operand: &IrOperand, found: &mut Vec<String>) {
    if let IrOperand::Constant(IrConstant::CString(text)) = operand {
        found.push(text.clone());
    }
}

fn collect_rvalue_literals(rvalue: &IrRvalue, found: &mut Vec<String>) {
    match rvalue {
        IrRvalue::Use(operand)
        | IrRvalue::Unary(_, operand)
        | IrRvalue::Cast(operand, _)
        | IrRvalue::FieldAddress { base: operand, .. }
        | IrRvalue::Load {
            address: operand, ..
        } => collect_literal(operand, found),
        IrRvalue::Binary(_, left, right) => {
            collect_literal(left, found);
            collect_literal(right, found);
        }
        IrRvalue::ElementAddress { base, index, .. } => {
            collect_literal(base, found);
            collect_literal(index, found);
        }
        IrRvalue::Call { arguments, .. } => {
            for argument in arguments {
                collect_literal(argument, found);
            }
        }
        IrRvalue::CallIndirect {
            callee, arguments, ..
        } => {
            collect_literal(callee, found);
            for argument in arguments {
                collect_literal(argument, found);
            }
        }
        IrRvalue::AddressOf { .. } | IrRvalue::FunctionAddress(_) => {}
    }
}

// How many bytes a value of this type occupies where it is stored. A `bool`
// takes one, the way the native backend stores it, so a struct laid out with
// one beside a number reads back what was written into it.
fn stored_width(ty: &Type) -> usize {
    match ty {
        Type::Distinct(_, inner) => stored_width(inner),
        Type::F32 => 4,
        Type::F64 => 8,
        Type::Bool => 1,
        Type::Enum(_) => 4,
        Type::Ptr(_) | Type::Ref(_) | Type::RefMut(_) | Type::Proc(_, _) => 8,
        other => {
            let (bits, _) = integer_info(other);
            (bits as usize).div_ceil(8).clamp(1, 8)
        }
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

fn returns_aggregate(function: &IrFunction) -> bool {
    function.name != "main" && is_aggregate(&function.return_type)
}

fn operand_type(function: &IrFunction, operand: &IrOperand) -> Type {
    match operand {
        IrOperand::Local(local) => function.local_type(*local).clone(),
        IrOperand::Constant(constant) => constant.constant_type(),
    }
}

fn binary(op: IrBinOp, left: Value, right: Value, ty: &Type) -> Eval<Value> {
    if ty.is_float() {
        return binary_float(op, left.as_f64(), right.as_f64(), ty);
    }
    let (bits, signed) = integer_info(ty);
    let left = normalize(left.as_i64(), bits, signed);
    let right = normalize(right.as_i64(), bits, signed);
    // Arithmetic is done where it cannot overflow and the answer is then held to
    // the range its type means, which is one rule for every width and matches
    // what the two compiled backends check for.
    let wide = |value: i128| -> Eval<i64> {
        let (low, high) = if signed {
            (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
        } else {
            (0, (1i128 << bits) - 1)
        };
        if value < low || value > high {
            return unsupported("arithmetic left the range of its type");
        }
        Ok(value as i64)
    };
    let as_wide = |value: i64| -> i128 {
        if signed {
            value as i128
        } else {
            value as u64 as i128
        }
    };
    let result = match op {
        IrBinOp::Add => wide(as_wide(left) + as_wide(right))?,
        IrBinOp::Subtract => wide(as_wide(left) - as_wide(right))?,
        IrBinOp::Multiply => wide(as_wide(left) * as_wide(right))?,
        IrBinOp::WrappingAdd => left.wrapping_add(right),
        IrBinOp::WrappingSubtract => left.wrapping_sub(right),
        IrBinOp::WrappingMultiply => left.wrapping_mul(right),
        IrBinOp::Divide => {
            if right == 0 {
                return unsupported("division by zero");
            }
            if signed {
                wide(as_wide(left) / as_wide(right))?
            } else {
                ((left as u64) / (right as u64)) as i64
            }
        }
        IrBinOp::Modulo => {
            if right == 0 {
                return unsupported("remainder by zero");
            }
            if signed {
                (as_wide(left) % as_wide(right)) as i64
            } else {
                ((left as u64) % (right as u64)) as i64
            }
        }
        IrBinOp::BitwiseAnd => left & right,
        IrBinOp::BitwiseOr => left | right,
        IrBinOp::ShiftLeft => {
            if !(0..bits as i64).contains(&right) {
                return unsupported("shift out of range");
            }
            wide(as_wide(left) << right)?
        }
        IrBinOp::ShiftRight => {
            if !(0..bits as i64).contains(&right) {
                return unsupported("shift out of range");
            }
            if signed {
                left.wrapping_shr(right as u32)
            } else {
                ((left as u64) >> (right as u32)) as i64
            }
        }
        IrBinOp::Equal => return Ok(boolean(left == right)),
        IrBinOp::NotEqual => return Ok(boolean(left != right)),
        IrBinOp::LessThan => return Ok(boolean(compare(left, right, signed))),
        IrBinOp::LessThanOrEqual => {
            return Ok(boolean(left == right || compare(left, right, signed)));
        }
        IrBinOp::GreaterThan => {
            return Ok(boolean(left != right && !compare(left, right, signed)));
        }
        IrBinOp::GreaterThanOrEqual => {
            return Ok(boolean(!compare(left, right, signed)));
        }
    };
    Ok(Value::Int(normalize(result, bits, signed)))
}

fn compare(left: i64, right: i64, signed: bool) -> bool {
    if signed {
        left < right
    } else {
        (left as u64) < (right as u64)
    }
}

fn binary_float(op: IrBinOp, left: f64, right: f64, ty: &Type) -> Eval<Value> {
    let value = match op {
        IrBinOp::Add => left + right,
        IrBinOp::Subtract => left - right,
        IrBinOp::Multiply => left * right,
        IrBinOp::Divide => left / right,
        IrBinOp::Equal => return Ok(boolean(left == right)),
        IrBinOp::NotEqual => return Ok(boolean(left != right)),
        IrBinOp::LessThan => return Ok(boolean(left < right)),
        IrBinOp::LessThanOrEqual => return Ok(boolean(left <= right)),
        IrBinOp::GreaterThan => return Ok(boolean(left > right)),
        IrBinOp::GreaterThanOrEqual => return Ok(boolean(left >= right)),
        _ => return unsupported("unsupported floating-point operation"),
    };
    Ok(Value::Float(round_to_type(value, ty)))
}

fn unary(op: IrUnOp, value: Value, ty: &Type) -> Value {
    match op {
        IrUnOp::Negate => {
            if ty.is_float() {
                Value::Float(round_to_type(-value.as_f64(), ty))
            } else {
                let (bits, signed) = integer_info(ty);
                Value::Int(normalize(
                    value.as_i64().wrapping_neg(),
                    bits,
                    signed,
                ))
            }
        }
        IrUnOp::Not => boolean(value.as_i64() == 0),
        IrUnOp::TrailingZeros => {
            Value::Int(value.as_i64().trailing_zeros() as i64)
        }
    }
}

fn cast(value: Value, source: &Type, target: &Type) -> Value {
    if target.is_float() {
        return Value::Float(round_to_type(value.as_f64(), target));
    }
    let (bits, signed) = integer_info(target);
    if source.is_float() {
        return Value::Int(normalize(value.as_f64() as i64, bits, signed));
    }
    Value::Int(normalize(value.as_i64(), bits, signed))
}

fn round_to_type(value: f64, ty: &Type) -> f64 {
    if matches!(ty, Type::F32) {
        value as f32 as f64
    } else {
        value
    }
}

fn boolean(value: bool) -> Value {
    Value::Int(value as i64)
}

fn normalize(value: i64, bits: u32, signed: bool) -> i64 {
    if bits >= 64 {
        return value;
    }
    let mask: i128 = (1i128 << bits) - 1;
    let masked = (value as i128) & mask;
    if signed && masked & (1i128 << (bits - 1)) != 0 {
        (masked - (1i128 << bits)) as i64
    } else {
        masked as i64
    }
}

fn integer_info(ty: &Type) -> (u32, bool) {
    match ty {
        Type::I8 => (8, true),
        Type::I16 => (16, true),
        Type::I32 => (32, true),
        Type::I64 | Type::Isize => (64, true),
        Type::U8 => (8, false),
        Type::U16 => (16, false),
        Type::U32 => (32, false),
        Type::U64 | Type::Usize => (64, false),
        Type::Bool => (8, false),
        Type::Enum(_) => (32, false),
        Type::Distinct(_, inner) => integer_info(inner),
        _ => (64, false),
    }
}

fn is_scalar(ty: &Type) -> bool {
    match ty {
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
        | Type::F32
        | Type::F64
        | Type::Bool
        | Type::Void
        | Type::Ptr(_)
        // A borrow is the address of someone else's storage, which is a value
        // this holds like any other pointer.
        | Type::Ref(_)
        | Type::RefMut(_)
        | Type::Handle(_)
        | Type::Proc(_, _)
        | Type::Enum(_) => true,
        Type::Distinct(_, inner) => is_scalar(inner),
        _ => false,
    }
}

fn default_value(ty: &Type) -> Value {
    if ty.is_float() {
        Value::Float(0.0)
    } else {
        Value::Int(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{IrBlock, IrLocal};

    fn ok(result: Eval<Value>) -> Value {
        match result {
            Ok(value) => value,
            Err(Signal::Unsupported(reason)) => {
                panic!("unexpected decline: {reason}")
            }
        }
    }

    fn int(value: i64) -> Value {
        Value::Int(value)
    }

    #[test]
    fn normalize_wraps_and_sign_extends() {
        assert_eq!(normalize(300, 8, false), 44);
        assert_eq!(normalize(200, 8, true), -56);
        assert_eq!(normalize(-1, 64, false), -1);
        assert_eq!(normalize(5, 32, true), 5);
        assert_eq!(normalize(256, 8, false), 0);
    }

    #[test]
    fn binary_add_stops_where_the_width_ends() {
        assert!(binary(IrBinOp::Add, int(200), int(100), &Type::U8).is_err());
        assert!(
            binary(
                IrBinOp::Add,
                int(4_000_000_000),
                int(1_000_000_000),
                &Type::U32
            )
            .is_err()
        );
        assert!(binary(IrBinOp::Add, int(100), int(100), &Type::I8).is_err());
    }

    #[test]
    fn binary_wrapping_add_keeps_the_low_bits() {
        assert_eq!(
            ok(binary(IrBinOp::WrappingAdd, int(200), int(100), &Type::U8))
                .as_i64(),
            44
        );
        assert_eq!(
            ok(binary(
                IrBinOp::WrappingAdd,
                int(4_000_000_000),
                int(1_000_000_000),
                &Type::U32
            ))
            .as_i64(),
            705_032_704
        );
        assert_eq!(
            ok(binary(IrBinOp::WrappingAdd, int(100), int(100), &Type::I8))
                .as_i64(),
            -56
        );
    }

    #[test]
    fn binary_division_respects_signedness() {
        assert_eq!(
            ok(binary(IrBinOp::Divide, int(-7), int(2), &Type::I64)).as_i64(),
            -3
        );
        assert_eq!(
            ok(binary(IrBinOp::Modulo, int(-7), int(2), &Type::I64)).as_i64(),
            -1
        );
        assert_eq!(
            ok(binary(IrBinOp::Divide, int(200), int(3), &Type::U8)).as_i64(),
            66
        );
    }

    #[test]
    fn binary_comparison_respects_signedness() {
        assert_eq!(
            ok(binary(IrBinOp::LessThan, int(-1), int(1), &Type::I64)).as_i64(),
            1
        );
        assert_eq!(
            ok(binary(IrBinOp::LessThan, int(-1), int(1), &Type::U64)).as_i64(),
            0
        );
        assert_eq!(
            ok(binary(
                IrBinOp::GreaterThanOrEqual,
                int(5),
                int(5),
                &Type::I64
            ))
            .as_i64(),
            1
        );
    }

    #[test]
    fn binary_shift_in_and_out_of_range() {
        assert_eq!(
            ok(binary(IrBinOp::ShiftLeft, int(1), int(10), &Type::I64))
                .as_i64(),
            1024
        );
        assert!(
            binary(IrBinOp::ShiftLeft, int(1), int(64), &Type::I64).is_err()
        );
        assert!(
            binary(IrBinOp::ShiftRight, int(1), int(-1), &Type::I64).is_err()
        );
    }

    #[test]
    fn binary_division_by_zero_declines() {
        assert!(binary(IrBinOp::Divide, int(1), int(0), &Type::I64).is_err());
        assert!(binary(IrBinOp::Modulo, int(1), int(0), &Type::I64).is_err());
    }

    #[test]
    fn binary_float_arithmetic_and_rounding() {
        assert_eq!(
            ok(binary(
                IrBinOp::Add,
                Value::Float(1.5),
                Value::Float(1.5),
                &Type::F64
            ))
            .as_f64(),
            3.0
        );
        let rounded = ok(binary(
            IrBinOp::Add,
            Value::Float(0.1),
            Value::Float(0.0),
            &Type::F32,
        ))
        .as_f64();
        assert_eq!(rounded, 0.1_f32 as f64);
    }

    #[test]
    fn cast_between_int_and_float() {
        assert_eq!(cast(int(3), &Type::I64, &Type::F64).as_f64(), 3.0);
        assert_eq!(cast(Value::Float(3.9), &Type::F64, &Type::I64).as_i64(), 3);
        assert_eq!(cast(int(300), &Type::I64, &Type::U8).as_i64(), 44);
        assert_eq!(
            cast(Value::Float(0.1), &Type::F64, &Type::F32).as_f64(),
            0.1_f32 as f64
        );
    }

    #[test]
    fn unary_negate_and_not() {
        assert_eq!(unary(IrUnOp::Negate, int(5), &Type::I64).as_i64(), -5);
        assert_eq!(
            unary(IrUnOp::Negate, Value::Float(2.0), &Type::F64).as_f64(),
            -2.0
        );
        assert_eq!(unary(IrUnOp::Not, int(0), &Type::Bool).as_i64(), 1);
        assert_eq!(unary(IrUnOp::Not, int(1), &Type::Bool).as_i64(), 0);
    }

    fn scalar_local(ty: Type) -> IrLocal {
        IrLocal {
            size: ty.size_of(),
            ty,
            name: None,
            in_memory: false,
            linear: false,
            position: Default::default(),
        }
    }

    #[test]
    fn run_module_interprets_printf() {
        let module = IrModule {
            externs: Vec::new(),
            imported: Vec::new(),
            functions: vec![IrFunction {
                name: "main".to_string(),
                param_count: 0,
                param_layouts: vec![None; 0],
                return_type: Type::I64,
                locals: vec![scalar_local(Type::I32)],
                blocks: vec![IrBlock {
                    statements: vec![IrStatement::Assign(
                        0,
                        IrRvalue::Call {
                            function: "printf".to_string(),
                            arguments: vec![
                                IrOperand::Constant(IrConstant::CString(
                                    "value=%lld\n".to_string(),
                                )),
                                IrOperand::Constant(IrConstant::Integer(
                                    42,
                                    Type::I64,
                                )),
                            ],
                        },
                    )],
                    terminator: IrTerminator::Return(Some(
                        IrOperand::Constant(IrConstant::Integer(0, Type::I64)),
                    )),
                }],
                entry: 0,
                module: 0,
                local: false,
                keeps_name: false,
                instantiated: None,
            }],
        };
        match run_module(&module) {
            RunOutcome::Output(output) => assert_eq!(output, "value=42\n"),
            RunOutcome::Unsupported(reason) => panic!("declined: {reason}"),
        }
    }

    #[test]
    fn run_module_declines_on_memory_ops() {
        let module = IrModule {
            externs: Vec::new(),
            imported: Vec::new(),
            functions: vec![IrFunction {
                name: "main".to_string(),
                param_count: 0,
                param_layouts: vec![None; 0],
                return_type: Type::I64,
                locals: vec![scalar_local(Type::I64)],
                blocks: vec![IrBlock {
                    statements: vec![IrStatement::Store {
                        address: IrOperand::Local(0),
                        value: IrOperand::Constant(IrConstant::Integer(
                            1,
                            Type::I64,
                        )),
                    }],
                    terminator: IrTerminator::Return(Some(
                        IrOperand::Constant(IrConstant::Integer(0, Type::I64)),
                    )),
                }],
                entry: 0,
                module: 0,
                local: false,
                keeps_name: false,
                instantiated: None,
            }],
        };
        assert!(matches!(run_module(&module), RunOutcome::Unsupported(_)));
    }

    #[test]
    fn run_module_declines_without_main() {
        let module = IrModule {
            externs: Vec::new(),
            imported: Vec::new(),
            functions: Vec::new(),
        };
        assert!(matches!(run_module(&module), RunOutcome::Unsupported(_)));
    }
}
