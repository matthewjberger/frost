use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::ir::{
    IrFunction, IrModule, IrOperand, IrRvalue, IrStatement, IrTerminator,
    LocalId,
};
use crate::types::Type;

struct Signature {
    param_count: usize,
}

const RUNTIME_INTRINSICS: &[&str] =
    &["frost_bounds_check", "frost_generation_check"];

fn is_runtime_intrinsic(name: &str) -> bool {
    RUNTIME_INTRINSICS.contains(&name)
}

pub fn check_module(module: &IrModule) -> Result<()> {
    let mut signatures: HashMap<&str, Signature> = HashMap::new();
    for function in &module.functions {
        signatures.insert(
            function.name.as_str(),
            Signature {
                param_count: function.param_count,
            },
        );
    }
    for external in &module.externs {
        signatures.insert(
            external.name.as_str(),
            Signature {
                param_count: external.params.len(),
            },
        );
    }
    for function in &module.functions {
        check_function(function, &signatures)?;
    }
    Ok(())
}

fn check_function(
    function: &IrFunction,
    signatures: &HashMap<&str, Signature>,
) -> Result<()> {
    if function.param_count > function.locals.len() {
        bail!(
            "function '{}' declares {} parameters but only {} locals",
            function.name,
            function.param_count,
            function.locals.len()
        );
    }
    let block_count = function.blocks.len();
    if function.entry >= block_count {
        bail!(
            "function '{}' entry block {} is out of range",
            function.name,
            function.entry
        );
    }
    for block in &function.blocks {
        for statement in &block.statements {
            check_statement(function, statement, signatures)?;
        }
        check_terminator(function, &block.terminator, block_count)?;
    }
    Ok(())
}

fn check_statement(
    function: &IrFunction,
    statement: &IrStatement,
    signatures: &HashMap<&str, Signature>,
) -> Result<()> {
    match statement {
        IrStatement::Assign(local, rvalue) => {
            check_local(function, *local)?;
            check_rvalue(function, rvalue, signatures)?;
        }
        IrStatement::Store { address, value } => {
            check_operand(function, address)?;
            check_operand(function, value)?;
            require_pointer(function, address, "store address")?;
        }
        IrStatement::Copy {
            destination,
            source,
            ..
        } => {
            check_operand(function, destination)?;
            check_operand(function, source)?;
        }
        IrStatement::Consume(local) => {
            check_local(function, *local)?;
        }
    }
    Ok(())
}

fn check_rvalue(
    function: &IrFunction,
    rvalue: &IrRvalue,
    signatures: &HashMap<&str, Signature>,
) -> Result<()> {
    match rvalue {
        IrRvalue::Use(operand) => check_operand(function, operand)?,
        IrRvalue::Binary(op, left, right) => {
            check_operand(function, left)?;
            check_operand(function, right)?;
            if !op.is_comparison() {
                require_numeric(function, left)?;
                require_numeric(function, right)?;
            }
        }
        IrRvalue::Unary(_, operand) => {
            check_operand(function, operand)?;
        }
        IrRvalue::Cast(operand, target) => {
            check_operand(function, operand)?;
            require_numeric(function, operand)?;
            if !is_numeric(target) {
                bail!(
                    "cast in '{}' targets non-numeric type {target}",
                    function.name
                );
            }
        }
        IrRvalue::AddressOf { local, .. } => check_local(function, *local)?,
        IrRvalue::FieldAddress { base, .. } => {
            check_operand(function, base)?;
            require_pointer(function, base, "field access base")?;
        }
        IrRvalue::ElementAddress { base, index, .. } => {
            check_operand(function, base)?;
            check_operand(function, index)?;
            require_pointer(function, base, "element access base")?;
            require_numeric(function, index)?;
        }
        IrRvalue::Load { address, .. } => {
            check_operand(function, address)?;
            require_pointer(function, address, "load address")?;
        }
        IrRvalue::Call {
            function: callee,
            arguments,
        } => {
            for argument in arguments {
                check_operand(function, argument)?;
            }
            match signatures.get(callee.as_str()) {
                Some(signature) => {
                    if arguments.len() != signature.param_count {
                        bail!(
                            "call to '{}' passes {} arguments but it takes {}",
                            callee,
                            arguments.len(),
                            signature.param_count
                        );
                    }
                }
                None if is_runtime_intrinsic(callee) => {}
                None => {
                    bail!(
                        "call to unknown function '{}' in '{}'",
                        callee,
                        function.name
                    );
                }
            }
        }
        IrRvalue::FunctionAddress(name) => {
            if !signatures.contains_key(name.as_str()) {
                bail!(
                    "address taken of unknown function '{}' in '{}'",
                    name,
                    function.name
                );
            }
        }
        IrRvalue::CallIndirect {
            callee,
            arguments,
            parameter_types,
            ..
        } => {
            check_operand(function, callee)?;
            if !matches!(operand_type(function, callee), Type::Proc(_, _)) {
                bail!(
                    "indirect call in '{}' calls a value of non-function type {}",
                    function.name,
                    operand_type(function, callee)
                );
            }
            for argument in arguments {
                check_operand(function, argument)?;
            }
            if arguments.len() != parameter_types.len() {
                bail!(
                    "indirect call in '{}' passes {} arguments but the callee \
                     type takes {}",
                    function.name,
                    arguments.len(),
                    parameter_types.len()
                );
            }
        }
    }
    Ok(())
}

fn check_terminator(
    function: &IrFunction,
    terminator: &IrTerminator,
    block_count: usize,
) -> Result<()> {
    match terminator {
        IrTerminator::Return(value) => {
            if let Some(operand) = value {
                check_operand(function, operand)?;
            } else if function.return_type != Type::Void {
                bail!(
                    "function '{}' returns {} but a block returns no value",
                    function.name,
                    function.return_type
                );
            }
        }
        IrTerminator::Jump(target) => {
            require_block(function, *target, block_count)?;
        }
        IrTerminator::Branch {
            condition,
            then_block,
            else_block,
        } => {
            check_operand(function, condition)?;
            require_block(function, *then_block, block_count)?;
            require_block(function, *else_block, block_count)?;
        }
        IrTerminator::Unreachable => {}
    }
    Ok(())
}

fn operand_type(function: &IrFunction, operand: &IrOperand) -> Type {
    match operand {
        IrOperand::Constant(constant) => constant.constant_type(),
        IrOperand::Local(local) => function.local_type(*local).clone(),
    }
}

fn check_operand(function: &IrFunction, operand: &IrOperand) -> Result<()> {
    if let IrOperand::Local(local) = operand {
        check_local(function, *local)?;
    }
    Ok(())
}

fn check_local(function: &IrFunction, local: LocalId) -> Result<()> {
    if local >= function.locals.len() {
        bail!(
            "local _{} referenced in '{}' is out of range",
            local,
            function.name
        );
    }
    Ok(())
}

fn require_block(
    function: &IrFunction,
    block: usize,
    block_count: usize,
) -> Result<()> {
    if block >= block_count {
        bail!(
            "branch to block{} in '{}' is out of range",
            block,
            function.name
        );
    }
    Ok(())
}

fn require_numeric(function: &IrFunction, operand: &IrOperand) -> Result<()> {
    if !is_numeric(&operand_type(function, operand)) {
        bail!(
            "arithmetic operand in '{}' has non-numeric type {}",
            function.name,
            operand_type(function, operand)
        );
    }
    Ok(())
}

fn require_pointer(
    function: &IrFunction,
    operand: &IrOperand,
    role: &str,
) -> Result<()> {
    let ty = operand_type(function, operand);
    if !is_pointer(&ty) {
        bail!("{role} in '{}' has non-pointer type {ty}", function.name);
    }
    Ok(())
}

fn is_pointer(ty: &Type) -> bool {
    matches!(ty, Type::Ptr(_) | Type::Ref(_) | Type::RefMut(_))
}

fn is_numeric(ty: &Type) -> bool {
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
        | Type::Bool => true,
        Type::Distinct(inner) => is_numeric(inner),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{IrBinOp, IrBlock, IrConstant, IrLocal};

    fn local(ty: Type) -> IrLocal {
        IrLocal {
            size: ty.size_of(),
            ty,
            name: None,
            in_memory: false,
            linear: false,
        }
    }

    fn single_block(
        return_type: Type,
        locals: Vec<IrLocal>,
        statements: Vec<IrStatement>,
        terminator: IrTerminator,
    ) -> IrModule {
        IrModule {
            externs: Vec::new(),
            functions: vec![IrFunction {
                name: "main".to_string(),
                param_count: 0,
                return_type,
                locals,
                blocks: vec![IrBlock {
                    statements,
                    terminator,
                }],
                entry: 0,
            }],
        }
    }

    fn integer(value: i64) -> IrOperand {
        IrOperand::Constant(IrConstant::Integer(value, Type::I64))
    }

    #[test]
    fn accepts_well_formed_function() {
        let module = single_block(
            Type::I64,
            vec![local(Type::I64)],
            vec![IrStatement::Assign(0, IrRvalue::Use(integer(7)))],
            IrTerminator::Return(Some(IrOperand::Local(0))),
        );
        assert!(check_module(&module).is_ok());
    }

    #[test]
    fn rejects_out_of_range_local() {
        let module = single_block(
            Type::I64,
            vec![local(Type::I64)],
            vec![IrStatement::Assign(0, IrRvalue::Use(IrOperand::Local(9)))],
            IrTerminator::Return(Some(IrOperand::Local(0))),
        );
        assert!(check_module(&module).is_err());
    }

    #[test]
    fn rejects_wrong_argument_count() {
        let module = single_block(
            Type::I64,
            vec![local(Type::I64)],
            vec![IrStatement::Assign(
                0,
                IrRvalue::Call {
                    function: "main".to_string(),
                    arguments: vec![integer(1)],
                },
            )],
            IrTerminator::Return(Some(IrOperand::Local(0))),
        );
        assert!(check_module(&module).is_err());
    }

    #[test]
    fn rejects_out_of_range_branch() {
        let module = single_block(
            Type::I64,
            vec![local(Type::Bool)],
            vec![],
            IrTerminator::Branch {
                condition: IrOperand::Local(0),
                then_block: 5,
                else_block: 0,
            },
        );
        assert!(check_module(&module).is_err());
    }

    #[test]
    fn rejects_arithmetic_on_non_numeric() {
        let module = single_block(
            Type::I64,
            vec![local(Type::I64), local(Type::Struct("Point".to_string()))],
            vec![IrStatement::Assign(
                0,
                IrRvalue::Binary(IrBinOp::Add, IrOperand::Local(1), integer(1)),
            )],
            IrTerminator::Return(Some(IrOperand::Local(0))),
        );
        assert!(check_module(&module).is_err());
    }

    #[test]
    fn rejects_missing_return_value() {
        let module =
            single_block(Type::I64, vec![], vec![], IrTerminator::Return(None));
        assert!(check_module(&module).is_err());
    }

    #[test]
    fn rejects_load_from_a_non_pointer() {
        let module = single_block(
            Type::I64,
            vec![local(Type::I64), local(Type::I64)],
            vec![IrStatement::Assign(
                0,
                IrRvalue::Load {
                    address: IrOperand::Local(1),
                    ty: Type::I64,
                },
            )],
            IrTerminator::Return(Some(IrOperand::Local(0))),
        );
        assert!(check_module(&module).is_err());
    }

    #[test]
    fn accepts_load_through_a_pointer() {
        let module = single_block(
            Type::I64,
            vec![local(Type::I64), local(Type::Ptr(Box::new(Type::I64)))],
            vec![IrStatement::Assign(
                0,
                IrRvalue::Load {
                    address: IrOperand::Local(1),
                    ty: Type::I64,
                },
            )],
            IrTerminator::Return(Some(IrOperand::Local(0))),
        );
        assert!(check_module(&module).is_ok());
    }

    #[test]
    fn rejects_store_to_a_non_pointer() {
        let module = single_block(
            Type::I64,
            vec![local(Type::I64), local(Type::I64)],
            vec![IrStatement::Store {
                address: IrOperand::Local(1),
                value: integer(5),
            }],
            IrTerminator::Return(Some(IrOperand::Local(0))),
        );
        assert!(check_module(&module).is_err());
    }

    #[test]
    fn rejects_cast_to_a_non_numeric_type() {
        let module = single_block(
            Type::I64,
            vec![local(Type::I64), local(Type::I64)],
            vec![IrStatement::Assign(
                0,
                IrRvalue::Cast(
                    IrOperand::Local(1),
                    Type::Struct("Point".to_string()),
                ),
            )],
            IrTerminator::Return(Some(IrOperand::Local(0))),
        );
        assert!(check_module(&module).is_err());
    }
}
