use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use anyhow::{Result, bail};

use crate::ir::{
    IrBinOp, IrConstant, IrFunction, IrModule, IrOperand, IrRvalue,
    IrStatement, IrTerminator, IrUnOp,
};
use crate::types::Type;

// The runtime's index check. Every checked read names it, and it is emitted as
// a comparison guarding the call rather than as the call, so it is written once
// here.
const BOUNDS_CHECK: &str = "frost_rt_bounds_check";

// What the emitter needs to know about the C side: which names are external,
// and which of those return an aggregate. C returns a struct by a rule of its
// own, and the way to get that rule right in emitted C is to declare a real
// struct type and let the C compiler apply it, rather than to reimplement the
// classification here. See src/c_abi.rs, which is the same problem where there
// is no C compiler to defer to.
struct Externs {
    names: HashSet<String>,
    // Extern name to the name and size of the struct type it returns.
    aggregate_returns: HashMap<String, (String, usize)>,
    // Extern name to the parameters it takes by value, each mapped to the name
    // of the struct type declared for it. The argument is a pointer at the IR
    // level whatever the mode, so a by-value parameter is the one place the
    // pointer is read through at the call.
    value_parameters: HashMap<String, HashMap<usize, String>>,
}

impl Externs {
    fn insert(&mut self, name: &str) {
        self.names.insert(name.to_string());
    }
}

// A Frost function's C name. The prefix is `frost_u_` rather than `frost_`
// because the runtime owns `frost_rt_`: two fixed prefixes that differ at the
// same position cannot collide whatever a function is called, where a single
// shared prefix made every runtime symbol a name users could not use. A Frost
// function named `byte_at` used to become `frost_byte_at` and fail to link
// against the runtime's own.
fn c_function_name(name: &str, externs: &Externs) -> String {
    if name == "main" || externs.names.contains(name) {
        name.to_string()
    } else {
        format!("frost_u_{name}")
    }
}

pub fn emit_c(module: &IrModule) -> Result<String> {
    let mut externs = Externs {
        names: module
            .externs
            .iter()
            .map(|external| external.name.clone())
            .collect(),
        aggregate_returns: HashMap::new(),
        value_parameters: HashMap::new(),
    };
    externs.insert(BOUNDS_CHECK);

    // The runtime names this file declares itself, from the runtime's own
    // header rather than from whatever a program said about them. A program may
    // declare one of these too, and its declaration is the one that is dropped:
    // two declarations of the same C function have to agree exactly, and a
    // Frost `^u8` reaching C as `char*` does not match the `void*` the runtime
    // actually takes.
    let mut declared: HashSet<&str> = HashSet::new();
    let runtime_declarations = [
        (
            "frost_rt_bounds_check",
            "void frost_rt_bounds_check(int64_t index, int64_t length);",
        ),
        (
            "frost_rt_check_length",
            "int64_t frost_rt_check_length(int64_t length);",
        ),
        (
            "frost_rt_generation_check",
            "void frost_rt_generation_check(int64_t stored, int64_t expected);",
        ),
        (
            "frost_rt_mem_set",
            "void frost_rt_mem_set(void *destination, int64_t value, int64_t size);",
        ),
        (
            "frost_rt_write_bytes",
            "void frost_rt_write_bytes(const char* data, int64_t length);",
        ),
        (
            "frost_rt_write_i64",
            "void frost_rt_write_i64(int64_t value);",
        ),
        (
            "frost_rt_write_f64",
            "void frost_rt_write_f64(double value);",
        ),
        (
            "frost_rt_write_char",
            "void frost_rt_write_char(int64_t byte);",
        ),
    ];

    let mut output = String::new();
    output.push_str("#include <stdint.h>\n\n");
    output.push_str(crate::lower::arith_prelude::ARITH_PRELUDE);
    output.push('\n');
    externs.insert("frost_rt_arith_trap");
    declared.insert("frost_rt_arith_trap");
    for (name, declaration) in runtime_declarations {
        writeln!(output, "{declaration}")?;
        externs.insert(name);
        declared.insert(name);
    }
    output.push('\n');

    // A struct type per aggregate-returning extern, laid out field for field so
    // that the C compiler classifies it exactly as the library's own header
    // would.
    for external in &module.externs {
        let Some(layout) = &external.return_layout else {
            continue;
        };
        let name = format!("frost_ret_{}", external.name);
        writeln!(output, "typedef struct {{")?;
        for field in c_return_fields(layout)? {
            writeln!(output, "  {field};")?;
        }
        writeln!(output, "}} {name};")?;
        externs
            .aggregate_returns
            .insert(external.name.clone(), (name, layout.size));
    }
    // A struct type per by-value parameter, for the same reason: the C compiler
    // is the thing that knows how that target passes a struct, so it is given a
    // real struct rather than told the answer.
    for external in &module.externs {
        for (index, layout) in external.param_layouts.iter().enumerate() {
            let Some(layout) = layout else {
                continue;
            };
            let name = format!("frost_arg_{}_{index}", external.name);
            writeln!(output, "typedef struct {{")?;
            for field in c_return_fields(layout)? {
                writeln!(output, "  {field};")?;
            }
            writeln!(output, "}} {name};")?;
            externs
                .value_parameters
                .entry(external.name.clone())
                .or_default()
                .insert(index, name);
        }
    }
    output.push('\n');

    for external in &module.externs {
        if declared.contains(external.name.as_str()) {
            continue;
        }
        let by_value = externs.value_parameters.get(&external.name).cloned();
        let mut params = Vec::new();
        for (index, param) in external.params.iter().enumerate() {
            let c_ty = match by_value.as_ref().and_then(|by| by.get(&index)) {
                Some(name) => name.clone(),
                None if is_aggregate(param) => "char*".to_string(),
                None => c_type(param)?,
            };
            params.push(format!("{c_ty} a{index}"));
        }
        let params = if params.is_empty() {
            "void".to_string()
        } else {
            params.join(", ")
        };
        let returns = match externs.aggregate_returns.get(&external.name) {
            Some((name, _)) => name.clone(),
            None => c_type(&external.return_type)?,
        };
        writeln!(output, "{returns} {}({params});", external.name)?;
    }
    output.push('\n');

    // The same struct type per by-value parameter, for a Frost function that C
    // calls back. Declaring it is what makes the C compiler collect the bytes
    // the way it passed them.
    for function in &module.functions {
        for (index, layout) in function.param_layouts.iter().enumerate() {
            let Some(layout) = layout else {
                continue;
            };
            let name = format!("frost_val_{}_{index}", function.name);
            writeln!(output, "typedef struct {{")?;
            for field in c_return_fields(layout)? {
                writeln!(output, "  {field};")?;
            }
            writeln!(output, "}} {name};")?;
            externs
                .value_parameters
                .entry(function.name.clone())
                .or_default()
                .insert(index, name);
        }
    }
    output.push('\n');

    for function in &module.functions {
        if function.name == "main" {
            continue;
        }
        writeln!(output, "{};", function_signature(function, &externs)?)?;
    }
    output.push('\n');

    for function in &module.functions {
        emit_function(&mut output, function, &externs)?;
    }

    Ok(output)
}

fn function_signature(
    function: &IrFunction,
    externs: &Externs,
) -> Result<String> {
    let is_main = function.name == "main";
    let returns_aggregate = !is_main && is_aggregate(&function.return_type);
    let return_type = if is_main {
        Type::I32
    } else {
        function.return_type.clone()
    };

    let by_value = externs.value_parameters.get(&function.name);
    let mut params = Vec::new();
    for index in 0..function.param_count {
        let ty = function.local_type(index);
        let c_ty = match by_value.and_then(|by| by.get(&index)) {
            Some(name) => name.clone(),
            None if is_aggregate(ty) => "char*".to_string(),
            None => c_type(ty)?,
        };
        params.push(format!("{c_ty} a{index}"));
    }
    if returns_aggregate {
        params.push("char* __ret".to_string());
    }
    let param_list = if params.is_empty() {
        "void".to_string()
    } else {
        params.join(", ")
    };

    let return_type_str = if returns_aggregate {
        "void".to_string()
    } else {
        c_type(&return_type)?
    };
    Ok(format!(
        "{return_type_str} {}({param_list})",
        c_function_name(&function.name, externs)
    ))
}

fn emit_function(
    output: &mut String,
    function: &IrFunction,
    externs: &Externs,
) -> Result<()> {
    let return_type = if function.name == "main" {
        Type::I32
    } else {
        function.return_type.clone()
    };

    writeln!(output, "{} {{", function_signature(function, externs)?)?;

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
            // Naming the function and the slot, since a type C has no name for
            // is a compiler bug and the next question is always which one.
            let c_ty = c_type(&local.ty).map_err(|error| {
                anyhow::anyhow!(
                    "{error}, for local {index} of '{}'",
                    function.name
                )
            })?;
            writeln!(output, "  {c_ty} _{index};")?;
        }
    }

    let by_value = externs.value_parameters.get(&function.name);
    if let Some(by_value) = by_value {
        let mut named: Vec<_> = by_value.iter().collect();
        named.sort_by_key(|(index, _)| **index);
        for (index, ty) in named {
            writeln!(output, "  {ty} __v{index};")?;
        }
    }
    for index in 0..function.param_count {
        let local = &function.locals[index];
        // A parameter C passed as the struct arrives as a value rather than an
        // address, and the body reads it through one, so it is given somewhere
        // to point at. The copy is the callee's own, which is what by value
        // means.
        if by_value.is_some_and(|by| by.contains_key(&index)) {
            writeln!(output, "  __v{index} = a{index};")?;
            writeln!(output, "  _{index} = (char*)&__v{index};")?;
        } else if is_aggregate(&local.ty) {
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
            emit_statement(output, function, statement, externs)?;
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
    externs: &Externs,
) -> Result<()> {
    match statement {
        IrStatement::Assign(local, rvalue) => {
            let local_type = function.local_type(*local).clone();
            if matches!(local_type, Type::Void | Type::Unknown) {
                // An index check is written as the comparison it is, so the
                // call is reached only by an index that is out of range. The C
                // compiler can then drop the ones it can prove, which it cannot
                // do with a call into another translation unit.
                if let IrRvalue::Call {
                    function: name,
                    arguments,
                } = rvalue
                    && name == BOUNDS_CHECK
                    && arguments.len() == 2
                {
                    let index = operand_expr(function, &arguments[0])?;
                    let length = operand_expr(function, &arguments[1])?;
                    writeln!(
                        output,
                        "  if ((uint64_t)({index}) >= (uint64_t)({length})) {BOUNDS_CHECK}({index}, {length});"
                    )?;
                    return Ok(());
                }
                if matches!(
                    rvalue,
                    IrRvalue::Call { .. } | IrRvalue::CallIndirect { .. }
                ) {
                    writeln!(
                        output,
                        "  {};",
                        rvalue_expr(function, rvalue, externs)?
                    )?;
                }
                return Ok(());
            }
            if is_aggregate(&local_type) {
                match rvalue {
                    IrRvalue::Use(IrOperand::Local(source)) => {
                        writeln!(
                            output,
                            "  __builtin_memcpy(_{local}, _{source}, {});",
                            function.locals[*local].size.max(1)
                        )?;
                    }
                    IrRvalue::Call {
                        function: name,
                        arguments,
                    } => {
                        let mut args =
                            call_arguments(function, name, arguments, externs)?;
                        // A C function returning a struct hands back a value
                        // rather than filling in an out-pointer, so the value
                        // is taken and copied into the local's storage.
                        if let Some((returns, size)) =
                            externs.aggregate_returns.get(name)
                        {
                            writeln!(
                                output,
                                "  {{ {returns} __r = {}({});  __builtin_memcpy(_{local}, &__r, {size}); }}",
                                name,
                                args.join(", ")
                            )?;
                            return Ok(());
                        }
                        args.push(format!("_{local}"));
                        writeln!(
                            output,
                            "  {}({});",
                            c_function_name(name, externs),
                            args.join(", ")
                        )?;
                    }
                    // A function pointer hands an aggregate back the way a
                    // named function does, through the trailing out-pointer,
                    // since the pointer's type is a Frost signature.
                    IrRvalue::CallIndirect {
                        callee,
                        arguments,
                        parameter_types,
                        ..
                    } => {
                        let signature =
                            indirect_signature(parameter_types, "void", true)?;
                        let mut args = Vec::new();
                        for argument in arguments {
                            args.push(operand_expr(function, argument)?);
                        }
                        args.push(format!("_{local}"));
                        writeln!(
                            output,
                            "  (({signature})({}))({});",
                            operand_expr(function, callee)?,
                            args.join(", ")
                        )?;
                    }
                    _ => bail!("unsupported aggregate assignment"),
                }
                return Ok(());
            }
            let mut value = rvalue_expr(function, rvalue, externs)?;
            // `ptr_cast` is a copy in the IR, so a pointer can be assigned the
            // integer it reinterprets. C reads that as building a pointer out
            // of an integer and refuses, so the reinterpretation is written
            // where the IR only implied it.
            if let IrRvalue::Use(operand) = rvalue {
                let wanted = c_type(&local_type)?;
                let held = c_type(&operand_type(function, operand))?;
                if wanted != held
                    && (wanted.ends_with('*') || held.ends_with('*'))
                {
                    value = format!("({wanted})({value})");
                }
            }
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
        IrStatement::Copy {
            destination,
            source,
            size,
        } => {
            writeln!(
                output,
                "  __builtin_memcpy({}, {}, {size});",
                operand_expr(function, destination)?,
                operand_expr(function, source)?
            )?;
            Ok(())
        }
        IrStatement::Own(_) | IrStatement::Consume(_) => Ok(()),
    }
}

fn emit_terminator(
    output: &mut String,
    function: &IrFunction,
    return_type: &Type,
    terminator: &IrTerminator,
) -> Result<()> {
    let returns_aggregate =
        function.name != "main" && is_aggregate(&function.return_type);
    match terminator {
        IrTerminator::Return(None) | IrTerminator::Unreachable => {
            if returns_aggregate || matches!(return_type, Type::Void) {
                writeln!(output, "  return;")?;
            } else {
                writeln!(output, "  return 0;")?;
            }
        }
        IrTerminator::Return(Some(operand)) => {
            if returns_aggregate {
                if let IrOperand::Local(source) = operand {
                    writeln!(
                        output,
                        "  __builtin_memcpy(__ret, _{source}, {});",
                        function.locals[*source].size.max(1)
                    )?;
                }
                writeln!(output, "  return;")?;
            } else if matches!(return_type, Type::Void) {
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

// The arguments of a call, written out. An argument is the same expression
// whatever the callee, except where the callee takes that parameter by value:
// the IR hands every aggregate over as a pointer, so there the pointer is read
// through and the C compiler passes the struct itself.
fn call_arguments(
    function: &IrFunction,
    name: &str,
    arguments: &[IrOperand],
    externs: &Externs,
) -> Result<Vec<String>> {
    let by_value = externs.value_parameters.get(name);
    let mut args = Vec::new();
    for (index, argument) in arguments.iter().enumerate() {
        let expr = operand_expr(function, argument)?;
        match by_value.and_then(|by| by.get(&index)) {
            Some(ty) => args.push(format!("(*({ty}*)({expr}))")),
            None => args.push(expr),
        }
    }
    Ok(args)
}

fn rvalue_expr(
    function: &IrFunction,
    rvalue: &IrRvalue,
    externs: &Externs,
) -> Result<String> {
    Ok(match rvalue {
        IrRvalue::Use(operand) => operand_expr(function, operand)?,
        IrRvalue::Binary(op, left, right) => {
            let left_expr = operand_expr(function, left)?;
            let right_expr = operand_expr(function, right)?;
            let ty = operand_type(function, left);
            checked_binary(*op, &left_expr, &right_expr, &ty)
        }
        IrRvalue::Unary(op, operand) => {
            let expr = operand_expr(function, operand)?;
            let ty = operand_type(function, operand);
            match op {
                // The operand is parenthesised, not just the whole thing. A
                // negated negative constant came out as `--9`, which C reads as
                // a decrement and refuses because a literal is not a place.
                IrUnOp::Negate if integer_range(&ty).is_some() => {
                    format!("frost_neg_i64(({expr}))")
                }
                IrUnOp::Negate => format!("(-({expr}))"),
                IrUnOp::Not => format!("(!({expr}))"),
                IrUnOp::TrailingZeros => {
                    format!("((int64_t)__builtin_ctzll((uint64_t)({expr})))")
                }
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
            let args = call_arguments(function, name, arguments, externs)?;
            format!("{}({})", c_function_name(name, externs), args.join(", "))
        }
        IrRvalue::FunctionAddress(name) => {
            format!("(void*){}", c_function_name(name, externs))
        }
        IrRvalue::CallIndirect {
            callee,
            arguments,
            parameter_types,
            return_type,
        } => {
            let signature = indirect_signature(
                parameter_types,
                &c_type(return_type)?,
                false,
            )?;
            let mut args = Vec::new();
            for argument in arguments {
                args.push(operand_expr(function, argument)?);
            }
            format!(
                "(({signature})({}))({})",
                operand_expr(function, callee)?,
                args.join(", ")
            )
        }
    })
}

// The C type of a Frost function pointer. An aggregate parameter crosses as an
// address, and an aggregate return is the trailing out-pointer the callee
// writes through, which is why the return type is passed in rather than read
// off the signature.
fn indirect_signature(
    parameter_types: &[Type],
    return_c: &str,
    out: bool,
) -> Result<String> {
    let mut param_c = Vec::new();
    for parameter in parameter_types {
        if is_aggregate(parameter) {
            param_c.push("char*".to_string());
        } else {
            param_c.push(c_type(parameter)?);
        }
    }
    if out {
        param_c.push("char*".to_string());
    }
    Ok(format!("{return_c}(*)({})", param_c.join(", ")))
}

fn operand_expr(function: &IrFunction, operand: &IrOperand) -> Result<String> {
    Ok(match operand {
        IrOperand::Local(local) => {
            if function.locals[*local].in_memory {
                let ty = function.local_type(*local);
                {
                    let c_ty = c_type(ty).map_err(|error| {
                        anyhow::anyhow!(
                            "{error}, reading local {local} of '{}'",
                            function.name
                        )
                    })?;
                    format!("(*({c_ty}*)_{local})")
                }
            } else {
                format!("_{local}")
            }
        }
        IrOperand::Constant(constant) => constant_expr(constant)?,
    })
}

fn constant_expr(constant: &IrConstant) -> Result<String> {
    Ok(match constant {
        // The smallest i64 has no C literal of its own: `-9223372036854775808`
        // reads as a negation of a number one past the largest signed one, so
        // C makes it unsigned and warns. Written as one more than it plus one
        // less, both halves fit.
        IrConstant::Integer(i64::MIN, _) => {
            "(-9223372036854775807LL - 1)".to_string()
        }
        IrConstant::Integer(value, _) => format!("{value}LL"),
        IrConstant::Float(value, Type::F32) => format!("((float){value:?})"),
        IrConstant::Float(value, _) => format!("((double){value:?})"),
        IrConstant::Bool(value) => {
            if *value {
                "1".to_string()
            } else {
                "0".to_string()
            }
        }
        IrConstant::CString(text) => format!("(char*){}", c_string(text)),
        IrConstant::Unit => bail!("unit value used as a value"),
    })
}

fn operand_type(function: &IrFunction, operand: &IrOperand) -> Type {
    match operand {
        IrOperand::Local(local) => function.local_type(*local).clone(),
        IrOperand::Constant(constant) => constant.constant_type(),
    }
}

/// The range an integer type can hold, and whether it is signed. `None` for
/// anything that is not an integer, where arithmetic is either floating point
/// or a pointer difference and neither has this question.
fn integer_range(ty: &Type) -> Option<(bool, u32)> {
    let held = match ty {
        Type::Distinct(_, inner) => inner.as_ref(),
        other => other,
    };
    match held {
        Type::I8 => Some((true, 8)),
        Type::I16 => Some((true, 16)),
        Type::I32 => Some((true, 32)),
        Type::I64 | Type::Isize => Some((true, 64)),
        Type::U8 => Some((false, 8)),
        Type::U16 => Some((false, 16)),
        Type::U32 => Some((false, 32)),
        Type::U64 | Type::Usize => Some((false, 64)),
        _ => None,
    }
}

/// One arithmetic operation, with the check its type needs.
///
/// A 64-bit operation is checked where it is done, since there is no wider place
/// to do it. A narrower one is computed at 64 bits, where neither operand can
/// overflow, and the answer is held to the range its own type means.
fn checked_binary(op: IrBinOp, left: &str, right: &str, ty: &Type) -> String {
    let Some((signed, bits)) = integer_range(ty) else {
        return format!("({left} {} {right})", binary_operator(op));
    };
    let wide = bits == 64;
    let suffix = if signed { "i64" } else { "u64" };
    let call =
        |name: &str| format!("frost_{name}_{suffix}(({left}), ({right}))");
    let sign = i32::from(signed);
    // The narrow form names the operation too, so the sentence a program gets
    // for an overflow does not depend on which backend compiled it.
    let narrow = |expr: String, fault: i64| {
        format!("frost_narrow((int64_t){expr}, {bits}, {sign}, {fault})")
    };
    match op {
        IrBinOp::Add | IrBinOp::Subtract | IrBinOp::Multiply => {
            let (name, fault) = match op {
                IrBinOp::Add => ("add", 0),
                IrBinOp::Subtract => ("sub", 1),
                _ => ("mul", 2),
            };
            if wide {
                call(name)
            } else {
                narrow(
                    format!("(({left}) {} ({right}))", binary_operator(op)),
                    fault,
                )
            }
        }
        IrBinOp::Divide => call("div"),
        IrBinOp::Modulo => call("rem"),
        // Shifted as a bit pattern, which is what a shift is here. Moving a one
        // into the sign bit of a signed C type is undefined, and slot
        // sixty-three of a liveness word is exactly that shift.
        IrBinOp::ShiftLeft => narrow(
            format!(
                "(int64_t)((uint64_t)({left}) << frost_shift((int64_t)({right}), {bits}))"
            ),
            8,
        ),
        IrBinOp::ShiftRight => narrow(
            format!("(({left}) >> frost_shift((int64_t)({right}), {bits}))"),
            8,
        ),
        _ => format!("({left} {} {right})", binary_operator(op)),
    }
}

fn binary_operator(op: IrBinOp) -> &'static str {
    match op {
        IrBinOp::Add | IrBinOp::WrappingAdd => "+",
        IrBinOp::Subtract | IrBinOp::WrappingSubtract => "-",
        IrBinOp::Multiply | IrBinOp::WrappingMultiply => "*",
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
    matches!(
        ty,
        Type::Struct(_)
            | Type::Enum(_)
            | Type::Array(_, _)
            | Type::Str
            | Type::Slice(_)
    )
}

// The fields of the struct type an aggregate-returning extern is declared with.
// Built from the flattened scalars in offset order with explicit padding, so
// the declaration has the same size, the same offsets and the same floating
// point content as the library's own type, which is everything any C ABI
// classifies on.
//
// A scalar that overlaps one already taken is a variant of an enum, and only
// one of them can be a field. That is safe while both are non-floating, since
// then the bytes classify the same either way, and it is refused when it is not,
// rather than guessed at.
fn c_return_fields(layout: &crate::c_abi::CLayout) -> Result<Vec<String>> {
    let mut scalars = layout.scalars.clone();
    scalars.sort_by_key(|scalar| scalar.offset);
    let mut fields = Vec::new();
    let mut at = 0usize;
    for scalar in &scalars {
        let size = scalar.ty.size_of().max(1);
        if scalar.offset < at {
            if scalar.ty.is_float() {
                bail!(
                    "'{}' overlaps a floating point field with another, which C has no way to declare and whose calling convention Frost will not guess at",
                    layout.name
                );
            }
            continue;
        }
        if scalar.offset > at {
            fields.push(format!("uint8_t pad{at}[{}]", scalar.offset - at));
        }
        fields.push(format!("{} f{}", c_type(&scalar.ty)?, scalar.offset));
        at = scalar.offset + size;
    }
    if at < layout.size {
        fields.push(format!("uint8_t pad{at}[{}]", layout.size - at));
    }
    Ok(fields)
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
        Type::Proc(_, _) => "void*".to_string(),
        Type::Handle(_) => "int64_t".to_string(),
        Type::Distinct(_, inner) => c_type(inner)?,
        other => bail!("type not supported: {other}"),
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
