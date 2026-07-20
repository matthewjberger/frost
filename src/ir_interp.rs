use crate::ir::{
    IrBinOp, IrConstant, IrFunction, IrModule, IrOperand, IrRvalue,
    IrStatement, IrTerminator, IrUnOp,
};
use crate::types::Type;

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

const MAX_DEPTH: usize = 20_000;

pub fn run_module(module: &IrModule) -> RunOutcome {
    let Some(entry) = module.functions.iter().find(|f| f.name == "main") else {
        return RunOutcome::Unsupported("no main function".to_string());
    };
    let mut interpreter = Interpreter {
        module,
        output: String::new(),
    };
    match interpreter.call(entry, &[], 0) {
        Ok(_) => RunOutcome::Output(interpreter.output),
        Err(Signal::Unsupported(reason)) => RunOutcome::Unsupported(reason),
    }
}

struct Interpreter<'a> {
    module: &'a IrModule,
    output: String,
}

enum Flow {
    Jump(usize),
    Return(Value),
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
        for local in &function.locals {
            if local.in_memory || !is_scalar(&local.ty) {
                return unsupported(format!(
                    "non-scalar local of type {}",
                    local.ty
                ));
            }
        }
        let mut locals: Vec<Value> = function
            .locals
            .iter()
            .map(|local| default_value(&local.ty))
            .collect();
        for (slot, value) in arguments.iter().enumerate() {
            locals[slot] = *value;
        }

        let mut block_index = function.entry;
        loop {
            let block = &function.blocks[block_index];
            for statement in &block.statements {
                self.execute(function, statement, &mut locals, depth)?;
            }
            match self.terminate(function, block_index, &locals)? {
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
                let value = self.evaluate(function, rvalue, locals, depth)?;
                locals[*local] = value;
                Ok(())
            }
            IrStatement::Store { .. } | IrStatement::Copy { .. } => {
                unsupported("memory store or copy")
            }
        }
    }

    fn terminate(
        &mut self,
        function: &IrFunction,
        block_index: usize,
        locals: &[Value],
    ) -> Eval<Flow> {
        match &function.blocks[block_index].terminator {
            IrTerminator::Return(None) => Ok(Flow::Return(Value::Int(0))),
            IrTerminator::Return(Some(operand)) => {
                Ok(Flow::Return(operand_value(operand, locals)))
            }
            IrTerminator::Jump(target) => Ok(Flow::Jump(*target)),
            IrTerminator::Branch {
                condition,
                then_block,
                else_block,
            } => {
                let taken = operand_value(condition, locals).as_i64() != 0;
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
        match rvalue {
            IrRvalue::Use(operand) => Ok(operand_value(operand, locals)),
            IrRvalue::Binary(op, left, right) => {
                let ty = operand_type(function, left);
                binary(
                    *op,
                    operand_value(left, locals),
                    operand_value(right, locals),
                    &ty,
                )
            }
            IrRvalue::Unary(op, operand) => {
                let ty = operand_type(function, operand);
                Ok(unary(*op, operand_value(operand, locals), &ty))
            }
            IrRvalue::Cast(operand, target) => {
                let source = operand_type(function, operand);
                Ok(cast(operand_value(operand, locals), &source, target))
            }
            IrRvalue::Call {
                function: callee,
                arguments,
            } => self.call_named(callee, arguments, locals, depth),
            IrRvalue::CallIndirect {
                callee, arguments, ..
            } => {
                let index = operand_value(callee, locals).as_i64();
                let Some(target) = usize::try_from(index)
                    .ok()
                    .and_then(|index| self.module.functions.get(index))
                else {
                    return unsupported("indirect call to unknown target");
                };
                let values = self.argument_values(arguments, locals);
                self.call(target, &values, depth + 1)
            }
            IrRvalue::FunctionAddress(name) => {
                match self.module.functions.iter().position(|f| &f.name == name)
                {
                    Some(index) => Ok(Value::Int(index as i64)),
                    None => unsupported("address of unknown function"),
                }
            }
            IrRvalue::AddressOf { .. }
            | IrRvalue::FieldAddress { .. }
            | IrRvalue::ElementAddress { .. }
            | IrRvalue::Load { .. } => unsupported("memory addressing"),
        }
    }

    fn argument_values(
        &self,
        arguments: &[IrOperand],
        locals: &[Value],
    ) -> Vec<Value> {
        arguments
            .iter()
            .map(|argument| operand_value(argument, locals))
            .collect()
    }

    fn call_named(
        &mut self,
        callee: &str,
        arguments: &[IrOperand],
        locals: &[Value],
        depth: usize,
    ) -> Eval<Value> {
        if callee == "printf" {
            return self.printf(arguments, locals);
        }
        if let Some(target) =
            self.module.functions.iter().find(|f| f.name == callee)
        {
            let values = self.argument_values(arguments, locals);
            return self.call(target, &values, depth + 1);
        }
        unsupported(format!("call to external '{callee}'"))
    }

    fn printf(
        &mut self,
        arguments: &[IrOperand],
        locals: &[Value],
    ) -> Eval<Value> {
        let Some(IrOperand::Constant(IrConstant::CString(format))) =
            arguments.first()
        else {
            return unsupported("printf with a non-literal format");
        };
        let rendered = render_format(format, &arguments[1..], locals)?;
        let length = rendered.len();
        self.output.push_str(&rendered);
        Ok(Value::Int(length as i64))
    }
}

fn render_format(
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
                let value = arguments.get(argument_index);
                argument_index += 1;
                let text = format_argument(&specifier, next, value, locals)?;
                result.push_str(&text);
                break;
            }
            specifier.push(next);
        }
    }
    Ok(result)
}

fn format_argument(
    flags: &str,
    conversion: char,
    argument: Option<&IrOperand>,
    locals: &[Value],
) -> Eval<String> {
    let trimmed = flags.trim_end_matches('l');
    match conversion {
        'd' | 'i' => {
            let value = require_scalar(argument, locals)?.as_i64();
            Ok(value.to_string())
        }
        'u' => {
            let value = require_scalar(argument, locals)?.as_i64();
            Ok((value as u64).to_string())
        }
        'x' => {
            let value = require_scalar(argument, locals)?.as_i64();
            Ok(format!("{:x}", value as u64))
        }
        'c' => {
            let value = require_scalar(argument, locals)?.as_i64();
            match u32::try_from(value).ok().and_then(char::from_u32) {
                Some(character) => Ok(character.to_string()),
                None => unsupported("invalid character in printf"),
            }
        }
        'f' | 'F' => {
            let value = require_scalar(argument, locals)?.as_f64();
            let precision = parse_precision(trimmed).unwrap_or(6);
            Ok(format!("{value:.precision$}"))
        }
        _ => unsupported(format!(
            "unsupported printf conversion '%{conversion}'"
        )),
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

fn require_scalar(
    argument: Option<&IrOperand>,
    locals: &[Value],
) -> Eval<Value> {
    match argument {
        Some(operand) => Ok(operand_value(operand, locals)),
        None => unsupported("printf missing argument"),
    }
}

fn operand_value(operand: &IrOperand, locals: &[Value]) -> Value {
    match operand {
        IrOperand::Local(local) => locals[*local],
        IrOperand::Constant(constant) => match constant {
            IrConstant::Integer(value, _) => Value::Int(*value),
            IrConstant::Float(value, _) => Value::Float(*value),
            IrConstant::Bool(value) => Value::Int(*value as i64),
            IrConstant::Unit => Value::Int(0),
            IrConstant::CString(_) => Value::Int(0),
        },
    }
}

fn operand_type(function: &IrFunction, operand: &IrOperand) -> Type {
    match operand {
        IrOperand::Local(local) => function.local_type(*local).clone(),
        IrOperand::Constant(constant) => constant.constant_type(),
    }
}

fn binary(op: IrBinOp, left: Value, right: Value, ty: &Type) -> Eval<Value> {
    if is_float(ty) {
        return binary_float(op, left.as_f64(), right.as_f64(), ty);
    }
    let (bits, signed) = integer_info(ty);
    let left = normalize(left.as_i64(), bits, signed);
    let right = normalize(right.as_i64(), bits, signed);
    let result = match op {
        IrBinOp::Add => left.wrapping_add(right),
        IrBinOp::Subtract => left.wrapping_sub(right),
        IrBinOp::Multiply => left.wrapping_mul(right),
        IrBinOp::Divide => {
            if right == 0 {
                return unsupported("division by zero");
            }
            if signed {
                left.wrapping_div(right)
            } else {
                ((left as u64) / (right as u64)) as i64
            }
        }
        IrBinOp::Modulo => {
            if right == 0 {
                return unsupported("remainder by zero");
            }
            if signed {
                left.wrapping_rem(right)
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
            left.wrapping_shl(right as u32)
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
            if is_float(ty) {
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
    }
}

fn cast(value: Value, source: &Type, target: &Type) -> Value {
    if is_float(target) {
        return Value::Float(round_to_type(value.as_f64(), target));
    }
    let (bits, signed) = integer_info(target);
    if is_float(source) {
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
        Type::Distinct(inner) => integer_info(inner),
        _ => (64, false),
    }
}

fn is_float(ty: &Type) -> bool {
    match ty {
        Type::F32 | Type::F64 => true,
        Type::Distinct(inner) => is_float(inner),
        _ => false,
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
        | Type::Handle(_)
        | Type::Proc(_, _)
        | Type::Enum(_) => true,
        Type::Distinct(inner) => is_scalar(inner),
        _ => false,
    }
}

fn default_value(ty: &Type) -> Value {
    if is_float(ty) {
        Value::Float(0.0)
    } else {
        Value::Int(0)
    }
}
