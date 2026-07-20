use std::fmt::Write;

use anyhow::{Result, bail};

use crate::ir::{
    IrBinOp, IrConstant, IrFunction, IrModule, IrOperand, IrRvalue,
    IrStatement, IrTerminator, IrUnOp,
};
use crate::types::Type;

pub fn emit_c(module: &IrModule) -> Result<String> {
    let mut output = String::new();
    output.push_str("#include <stdint.h>\n\n");

    for external in &module.externs {
        let mut params = Vec::new();
        for (index, param) in external.params.iter().enumerate() {
            params.push(format!("{} a{index}", c_type(param)?));
        }
        let params = if params.is_empty() {
            "void".to_string()
        } else {
            params.join(", ")
        };
        writeln!(
            output,
            "{} {}({});",
            c_type(&external.return_type)?,
            external.name,
            params
        )?;
    }
    output.push('\n');

    for function in &module.functions {
        emit_function(&mut output, function)?;
    }

    Ok(output)
}

fn emit_function(output: &mut String, function: &IrFunction) -> Result<()> {
    let is_main = function.name == "main";
    let return_type = if is_main {
        Type::I32
    } else {
        function.return_type.clone()
    };

    let mut params = Vec::new();
    for index in 0..function.param_count {
        let ty = function.local_type(index);
        let c_ty = if is_aggregate(ty) {
            "char*".to_string()
        } else {
            c_type(ty)?
        };
        params.push(format!("{c_ty} a{index}"));
    }
    let param_list = if params.is_empty() {
        "void".to_string()
    } else {
        params.join(", ")
    };

    writeln!(
        output,
        "{} {}({}) {{",
        c_type(&return_type)?,
        function.name,
        param_list
    )?;

    for (index, local) in function.locals.iter().enumerate() {
        if matches!(local.ty, Type::Void | Type::Unknown) {
            continue;
        }
        if local.in_memory {
            writeln!(
                output,
                "  _Alignas(16) unsigned char _{index}[{}];",
                local.size.max(1)
            )?;
        } else {
            writeln!(output, "  {} _{index};", c_type(&local.ty)?)?;
        }
    }

    for index in 0..function.param_count {
        let local = &function.locals[index];
        if is_aggregate(&local.ty) {
            writeln!(
                output,
                "  __builtin_memcpy(_{index}, a{index}, {});",
                local.size.max(1)
            )?;
        } else if local.in_memory {
            writeln!(
                output,
                "  *({}*)_{index} = a{index};",
                c_type(&local.ty)?
            )?;
        } else {
            writeln!(output, "  _{index} = a{index};")?;
        }
    }

    for (block_index, block) in function.blocks.iter().enumerate() {
        writeln!(output, " block{block_index}:;")?;
        for statement in &block.statements {
            emit_statement(output, function, statement)?;
        }
        emit_terminator(output, function, &return_type, &block.terminator)?;
    }

    writeln!(output, "}}\n")?;
    Ok(())
}

fn emit_statement(
    output: &mut String,
    function: &IrFunction,
    statement: &IrStatement,
) -> Result<()> {
    match statement {
        IrStatement::Assign(local, rvalue) => {
            let local_type = function.local_type(*local).clone();
            if matches!(local_type, Type::Void | Type::Unknown) {
                if let IrRvalue::Call { .. } = rvalue {
                    writeln!(output, "  {};", rvalue_expr(function, rvalue)?)?;
                }
                return Ok(());
            }
            if is_aggregate(&local_type) {
                let IrRvalue::Use(IrOperand::Local(source)) = rvalue else {
                    bail!("C backend: unsupported aggregate assignment");
                };
                writeln!(
                    output,
                    "  __builtin_memcpy(_{local}, _{source}, {});",
                    function.locals[*local].size.max(1)
                )?;
                return Ok(());
            }
            let value = rvalue_expr(function, rvalue)?;
            if function.locals[*local].in_memory {
                writeln!(
                    output,
                    "  *({}*)_{local} = {value};",
                    c_type(&local_type)?
                )?;
            } else {
                writeln!(output, "  _{local} = {value};")?;
            }
            Ok(())
        }
        IrStatement::Store { address, value } => {
            let value_type = operand_type(function, value);
            writeln!(
                output,
                "  *({}*)({}) = {};",
                c_type(&value_type)?,
                operand_expr(function, address)?,
                operand_expr(function, value)?
            )?;
            Ok(())
        }
    }
}

fn emit_terminator(
    output: &mut String,
    function: &IrFunction,
    return_type: &Type,
    terminator: &IrTerminator,
) -> Result<()> {
    match terminator {
        IrTerminator::Return(None) | IrTerminator::Unreachable => {
            if matches!(return_type, Type::Void) {
                writeln!(output, "  return;")?;
            } else {
                writeln!(output, "  return 0;")?;
            }
        }
        IrTerminator::Return(Some(operand)) => {
            if matches!(return_type, Type::Void) {
                writeln!(output, "  return;")?;
            } else {
                writeln!(
                    output,
                    "  return ({})({});",
                    c_type(return_type)?,
                    operand_expr(function, operand)?
                )?;
            }
        }
        IrTerminator::Jump(block) => {
            writeln!(output, "  goto block{block};")?;
        }
        IrTerminator::Branch {
            condition,
            then_block,
            else_block,
        } => {
            writeln!(
                output,
                "  if ({}) goto block{then_block}; else goto block{else_block};",
                operand_expr(function, condition)?
            )?;
        }
    }
    Ok(())
}

fn rvalue_expr(function: &IrFunction, rvalue: &IrRvalue) -> Result<String> {
    Ok(match rvalue {
        IrRvalue::Use(operand) => operand_expr(function, operand)?,
        IrRvalue::Binary(op, left, right) => {
            let left_expr = operand_expr(function, left)?;
            let right_expr = operand_expr(function, right)?;
            format!("({left_expr} {} {right_expr})", binary_operator(*op))
        }
        IrRvalue::Unary(op, operand) => {
            let expr = operand_expr(function, operand)?;
            match op {
                IrUnOp::Negate => format!("(-{expr})"),
                IrUnOp::Not => format!("(!{expr})"),
            }
        }
        IrRvalue::Cast(operand, target) => {
            format!(
                "({})({})",
                c_type(target)?,
                operand_expr(function, operand)?
            )
        }
        IrRvalue::AddressOf { local, offset } => {
            format!("((char*)_{local} + {offset})")
        }
        IrRvalue::FieldAddress { base, offset } => {
            format!("((char*)({}) + {offset})", operand_expr(function, base)?)
        }
        IrRvalue::ElementAddress {
            base,
            index,
            element_size,
        } => {
            format!(
                "((char*)({}) + ({}) * {element_size})",
                operand_expr(function, base)?,
                operand_expr(function, index)?
            )
        }
        IrRvalue::Load { address, ty } => {
            format!(
                "(*({}*)({}))",
                c_type(ty)?,
                operand_expr(function, address)?
            )
        }
        IrRvalue::Call {
            function: name,
            arguments,
        } => {
            let mut args = Vec::new();
            for argument in arguments {
                args.push(operand_expr(function, argument)?);
            }
            format!("{name}({})", args.join(", "))
        }
    })
}

fn operand_expr(function: &IrFunction, operand: &IrOperand) -> Result<String> {
    Ok(match operand {
        IrOperand::Local(local) => {
            if function.locals[*local].in_memory {
                let ty = function.local_type(*local);
                format!("(*({}*)_{local})", c_type(ty)?)
            } else {
                format!("_{local}")
            }
        }
        IrOperand::Constant(constant) => constant_expr(constant)?,
    })
}

fn constant_expr(constant: &IrConstant) -> Result<String> {
    Ok(match constant {
        IrConstant::Integer(value, _) => format!("{value}LL"),
        IrConstant::Float(value, Type::F32) => format!("(({value:?})f)"),
        IrConstant::Float(value, _) => format!("((double){value:?})"),
        IrConstant::Bool(value) => {
            if *value {
                "1".to_string()
            } else {
                "0".to_string()
            }
        }
        IrConstant::CString(text) => format!("(char*){}", c_string(text)),
        IrConstant::Unit => bail!("C backend: unit value used as a value"),
    })
}

fn operand_type(function: &IrFunction, operand: &IrOperand) -> Type {
    match operand {
        IrOperand::Local(local) => function.local_type(*local).clone(),
        IrOperand::Constant(constant) => constant.constant_type(),
    }
}

fn binary_operator(op: IrBinOp) -> &'static str {
    match op {
        IrBinOp::Add => "+",
        IrBinOp::Subtract => "-",
        IrBinOp::Multiply => "*",
        IrBinOp::Divide => "/",
        IrBinOp::Modulo => "%",
        IrBinOp::BitwiseAnd => "&",
        IrBinOp::BitwiseOr => "|",
        IrBinOp::ShiftLeft => "<<",
        IrBinOp::ShiftRight => ">>",
        IrBinOp::Equal => "==",
        IrBinOp::NotEqual => "!=",
        IrBinOp::LessThan => "<",
        IrBinOp::LessThanOrEqual => "<=",
        IrBinOp::GreaterThan => ">",
        IrBinOp::GreaterThanOrEqual => ">=",
    }
}

fn is_aggregate(ty: &Type) -> bool {
    matches!(ty, Type::Struct(_) | Type::Enum(_) | Type::Array(_, _))
}

fn c_type(ty: &Type) -> Result<String> {
    Ok(match ty {
        Type::I8 => "int8_t".to_string(),
        Type::I16 => "int16_t".to_string(),
        Type::I32 => "int32_t".to_string(),
        Type::I64 | Type::Isize => "int64_t".to_string(),
        Type::U8 => "uint8_t".to_string(),
        Type::U16 => "uint16_t".to_string(),
        Type::U32 => "uint32_t".to_string(),
        Type::U64 | Type::Usize => "uint64_t".to_string(),
        Type::F32 => "float".to_string(),
        Type::F64 => "double".to_string(),
        Type::Bool => "int8_t".to_string(),
        Type::Void => "void".to_string(),
        Type::Ptr(_) | Type::Ref(_) | Type::RefMut(_) => "char*".to_string(),
        Type::Distinct(inner) => c_type(inner)?,
        other => bail!("C backend: type not supported: {other}"),
    })
}

fn c_string(text: &str) -> String {
    let mut result = String::from("\"");
    for byte in text.bytes() {
        match byte {
            b'"' => result.push_str("\\\""),
            b'\\' => result.push_str("\\\\"),
            b'\n' => result.push_str("\\n"),
            b'\t' => result.push_str("\\t"),
            b'\r' => result.push_str("\\r"),
            0 => result.push_str("\\0"),
            0x20..=0x7e => result.push(byte as char),
            other => {
                result.push_str(&format!("\\x{other:02x}"));
            }
        }
    }
    result.push('"');
    result
}
