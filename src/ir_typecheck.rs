use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::ir::{
    IrFunction, IrModule, IrOperand, IrRvalue, IrStatement, IrTerminator,
    LocalId,
};
use crate::types::Type;

struct Signature {
    param_types: Vec<Type>,
    return_type: Type,
}

impl Signature {
    fn param_count(&self) -> usize {
        self.param_types.len()
    }
}

const RUNTIME_INTRINSICS: &[&str] = &[
    "frost_rt_bounds_check",
    "frost_rt_check_length",
    "frost_rt_generation_check",
    "frost_rt_mem_set",
    "frost_rt_print_i64",
    "frost_rt_print_f64",
    "frost_rt_write_bytes",
    "frost_rt_write_i64",
    "frost_rt_write_f64",
    "frost_rt_write_newline",
    "frost_rt_write_cstr",
];

fn is_runtime_intrinsic(name: &str) -> bool {
    RUNTIME_INTRINSICS.contains(&name)
}

pub fn check_module(module: &IrModule) -> Result<()> {
    let reports = check_module_recovering(module);
    if reports.is_empty() {
        return Ok(());
    }
    let rendered: Vec<String> =
        reports.iter().map(|held| held.rendered()).collect();
    Err(anyhow::anyhow!(rendered.join("\n")))
}

/// Check each function, reporting one failure per function rather than stopping
/// at the first.
///
/// The granularity is the function, not the statement. This pass derives a
/// type for each operand as it goes, so past the first mismatch it has nothing
/// sound to check the rest of the body against, and continuing would report
/// consequences rather than causes. Functions share no such state.
///
/// Each diagnostic's message is already located by `locate_instantiation`,
/// which rewrites a specialization's mangled name into what the reader wrote;
/// the position field anchors the function for a caller that wants structure,
/// at the call that instantiated it or at its first local.
pub fn check_module_recovering(
    module: &IrModule,
) -> Vec<crate::diagnostic::Diagnostic> {
    let mut signatures: HashMap<&str, Signature> = HashMap::new();
    // A function another object defines is callable here and has a signature to
    // check the call against. It has no body to check.
    for function in module.functions.iter().chain(module.imported.iter()) {
        signatures.insert(
            function.name.as_str(),
            Signature {
                // A function's parameters are its first locals, which is where
                // their types are written down.
                param_types: function.locals[..function.param_count]
                    .iter()
                    .map(|local| local.ty.clone())
                    .collect(),
                return_type: function.return_type.clone(),
            },
        );
    }
    for external in &module.externs {
        signatures.insert(
            external.name.as_str(),
            Signature {
                param_types: external.params.clone(),
                return_type: external.return_type.clone(),
            },
        );
    }
    let mut reports = Vec::new();
    for function in &module.functions {
        if let Err(error) = locate_instantiation(
            check_function(function, &signatures),
            function,
        ) {
            let anchor = function
                .instantiated
                .as_ref()
                .map(|held| held.at)
                .or_else(|| function.locals.first().map(|local| local.position))
                .unwrap_or_default();
            reports.push(crate::diagnostic::Diagnostic::new(
                anchor,
                error.to_string(),
            ));
        }
    }
    reports
}

// A type error inside a specialization names a line in the template, which is
// code the reader never wrote. The call that asked for the specialization is
// what they did write, so it goes first and the template position stays behind
// it for whoever maintains the generic.
fn locate_instantiation<T>(
    result: Result<T>,
    function: &IrFunction,
) -> Result<T> {
    let Some(instantiated) = &function.instantiated else {
        return result;
    };
    result.map_err(|error| {
        let text = crate::imports::demangle_private_names(&error.to_string());
        let name = crate::imports::demangle_private_names(&instantiated.name);
        // The mangled symbol is a compiler artifact, so where the inner message
        // names it, say what the reader wrote instead. The text has already had
        // the module tag taken off it, so the symbol looked for has to have
        // lost the same thing or it is not in there to find.
        let symbol = crate::imports::demangle_private_names(&function.name);
        let text = text.replace(&symbol, &name);
        if instantiated.at == crate::lexer::Position::default() {
            anyhow::anyhow!("instantiating '{name}': {text}")
        } else {
            anyhow::anyhow!(
                "at {}: instantiating '{name}': {text}",
                instantiated.at.describe()
            )
        }
    })
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
            if let Some(produced) = rvalue_type(function, rvalue, signatures) {
                let wanted = function.local_type(*local);
                if !fits(&produced, wanted) {
                    bail!(
                        "{}_{local} in '{}' is a {wanted} and is assigned a {produced}",
                        at(function, &IrOperand::Local(*local)),
                        function.name
                    );
                }
            }
        }
        IrStatement::Store { address, value } => {
            check_operand(function, address)?;
            check_operand(function, value)?;
            require_pointer(function, address, "store address")?;
            {
                let held = operand_type(function, address);
                let given = operand_type(function, value);
                if let Some(pointee) = pointee_of(&held)
                    && !fits(&given, pointee)
                {
                    bail!(
                        "{}a store through a {held} writes a {given}",
                        at(function, value)
                    );
                }
            }
        }
        IrStatement::Copy {
            destination,
            source,
            ..
        } => {
            check_operand(function, destination)?;
            check_operand(function, source)?;
        }
        IrStatement::Own(local) | IrStatement::Consume(local) => {
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
            {
                let held = operand_type(function, left);
                let other = operand_type(function, right);
                if !fits(&held, &other) && !fits(&other, &held) {
                    bail!(
                        "{}an operator has a {held} on one side and a {other} on the other",
                        at(function, left)
                    );
                }
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
                    if arguments.len() != signature.param_count() {
                        bail!(
                            "call to '{}' passes {} arguments but it takes {}",
                            callee,
                            arguments.len(),
                            signature.param_count()
                        );
                    }
                    {
                        for (index, (argument, wanted)) in arguments
                            .iter()
                            .zip(signature.param_types.iter())
                            .enumerate()
                        {
                            let given = operand_type(function, argument);
                            if !fits(&given, wanted) {
                                bail!(
                                    "{}argument {} of the call to '{}' is a {given}, and it takes a {wanted}",
                                    at(function, argument),
                                    index + 1,
                                    callee
                                );
                            }
                        }
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
                {
                    let given = operand_type(function, operand);
                    if !fits(&given, &function.return_type) {
                        bail!(
                            "{}'{}' answers with a {}, and this returns a {given}",
                            at(function, operand),
                            function.name,
                            function.return_type
                        );
                    }
                }
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

/// The type an rvalue produces, or `None` where this pass does not model it.
///
/// `None` rather than a guess. The three address-forming rvalues produce a
/// pointer whose pointee is the lowering's business, and answering with a
/// plausible one here would turn this from a check into a second opinion that
/// is sometimes wrong.
fn rvalue_type(
    function: &IrFunction,
    rvalue: &IrRvalue,
    signatures: &HashMap<&str, Signature>,
) -> Option<Type> {
    match rvalue {
        IrRvalue::Use(operand) => Some(operand_type(function, operand)),
        IrRvalue::Binary(op, left, _) => {
            if op.is_comparison() {
                Some(Type::Bool)
            } else {
                Some(operand_type(function, left))
            }
        }
        IrRvalue::Unary(_, operand) => Some(operand_type(function, operand)),
        IrRvalue::Cast(_, target) => Some(target.clone()),
        IrRvalue::Load { ty, .. } => Some(ty.clone()),
        IrRvalue::Call {
            function: callee, ..
        } => signatures
            .get(callee.as_str())
            .map(|signature| signature.return_type.clone()),
        IrRvalue::CallIndirect { return_type, .. } => Some(return_type.clone()),
        IrRvalue::AddressOf { .. }
        | IrRvalue::FieldAddress { .. }
        | IrRvalue::ElementAddress { .. }
        | IrRvalue::FunctionAddress(_) => None,
    }
}

fn pointee_of(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Ptr(inner) | Type::Ref(inner) | Type::RefMut(inner) => {
            Some(inner)
        }
        _ => None,
    }
}

/// A distinct type is its own type to a program and its base type to a machine,
/// and this pass is about the machine.
fn strip(ty: &Type) -> &Type {
    match ty {
        Type::Distinct(_, inner) => strip(inner),
        other => other,
    }
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

fn is_integer_like(ty: &Type) -> bool {
    is_integer(ty) || matches!(ty, Type::Handle(_))
}

fn is_float(ty: &Type) -> bool {
    matches!(ty, Type::F32 | Type::F64)
}

// The name a struct or an enum carries, so the two spellings of one declared
// type can be recognised as the same thing.
fn named(ty: &Type) -> Option<&str> {
    match ty {
        Type::Struct(name) | Type::Enum(name) => Some(name.as_str()),
        _ => None,
    }
}

fn is_text(ty: &Type) -> bool {
    match ty {
        Type::Str => true,
        Type::Slice(inner) => matches!(**inner, Type::U8 | Type::I8),
        _ => false,
    }
}

fn is_aggregate(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Struct(_)
            | Type::Enum(_)
            | Type::Array(_, _)
            | Type::Slice(_)
            | Type::Str
    )
}

/// What a pointer points at, or `None` for anything that is not one.
fn pointee(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Ptr(inner) | Type::Ref(inner) | Type::RefMut(inner) => {
            Some(inner)
        }
        _ => None,
    }
}

/// Whether a value of one type may stand where the other is wanted.
///
/// Not equality. Two integer widths fit each other here, because the language
/// converts between them wherever one is written for the other and whether it
/// should is a question about the language rather than about this pass. What
/// does not fit is a truth value where a number is wanted, or text where either
/// is, which is where the holes this was written for actually are: an i64 was
/// accepted for a `bool` parameter and answered 111, and a `str` reached a
/// backend that had nothing to say about it beyond its own name.
fn fits(given: &Type, wanted: &Type) -> bool {
    let given = strip(given);
    let wanted = strip(wanted);
    if given == wanted {
        return true;
    }
    // Text is a run of bytes. The two spellings are one type and the lowering
    // uses whichever the source wrote.
    if is_text(given) && is_text(wanted) {
        return true;
    }
    // Two function types fit when their parts do. The same aggregate-travels-by
    // -address rule applies inside a signature: a bundle field declared
    // `fn(Text) -> i64` is filled by a function whose parameter became a
    // `&Text` when the mode pass ran, and the two are one signature.
    if let (
        Type::Proc(given_params, given_ret),
        Type::Proc(wanted_params, wanted_ret),
    ) = (given, wanted)
    {
        return given_params.len() == wanted_params.len()
            && given_params
                .iter()
                .zip(wanted_params.iter())
                .all(|(a, b)| fits(a, b))
            && fits(given_ret, wanted_ret);
    }
    // An enum is laid out as a struct with a tag beside the variant's fields,
    // so the two spell one type here and which one a value carries depends on
    // whether it came from the declaration or from the layout.
    if named(given) == named(wanted) && named(given).is_some() {
        return true;
    }
    // Void is a result nobody asked for. A statement's value is assigned to one
    // and goes no further, which is the lowering saying it is discarded rather
    // than saying it is nothing.
    if matches!(wanted, Type::Void) {
        return true;
    }
    // A run fits a run when its elements do, and its length is part of what it
    // is. This is where an enum spelled as its declaration meets one spelled as
    // the layout it lowers to, a rule the top of this function has for a bare
    // name and needs inside a run as well.
    if let (
        Type::Array(given_element, given_count),
        Type::Array(wanted_element, wanted_count),
    ) = (given, wanted)
    {
        return given_count == wanted_count
            && fits(given_element, wanted_element);
    }
    if let (Type::Slice(given_element), Type::Slice(wanted_element)) =
        (given, wanted)
    {
        return fits(given_element, wanted_element);
    }
    // An aggregate travels by address: the mode pass rewrites a parameter that
    // holds one into a pointer to it, so a call passes the address where the
    // signature says the value. What travels has to be the address of the thing
    // wanted, since a pointer to something else is a different value. Where the
    // wanted type carries a length beside its address, taking one for the other
    // reads whatever sat beside the pointer as the length: `takes(p)` with
    // `p : ^i8` and a `str` parameter answered a length of two trillion and
    // indexed past the end of a two-byte string with the bounds check agreeing.
    if is_aggregate(wanted)
        && let Some(inner) = pointee(given)
        && fits(inner, wanted)
    {
        return true;
    }
    if is_aggregate(given)
        && let Some(inner) = pointee(wanted)
        && fits(given, inner)
    {
        return true;
    }
    // A pointer is a machine word and the lowering retypes them freely: an
    // address is checked where it is loaded from rather than where it is
    // carried.
    if is_pointer(given) && is_pointer(wanted) {
        return true;
    }
    // A handle is a generation and an index packed into one word, so it is an
    // integer to everything below the language.
    if is_integer_like(given) && is_integer_like(wanted) {
        return true;
    }
    if is_float(given) && is_float(wanted) {
        return true;
    }
    // A number becoming a pointer and back is how the language spells an
    // untyped address, and `no_pointer()` is written that way everywhere.
    if (is_integer(given) && is_pointer(wanted))
        || (is_pointer(given) && is_integer(wanted))
    {
        return true;
    }
    false
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

// Where an operand came from, so a type error points at source rather than only
// naming the function it happened in. A local carries the position it was bound
// at. A constant carries none, and the message stands on its own.
fn at(function: &IrFunction, operand: &IrOperand) -> String {
    let IrOperand::Local(id) = operand else {
        return String::new();
    };
    let Some(local) = function.locals.get(*id) else {
        return String::new();
    };
    if local.position == crate::lexer::Position::default() {
        return String::new();
    }
    format!("at {}: ", local.position.describe())
}

fn require_numeric(function: &IrFunction, operand: &IrOperand) -> Result<()> {
    if !is_numeric(&operand_type(function, operand)) {
        bail!(
            "{}arithmetic operand in '{}' has non-numeric type {}",
            at(function, operand),
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
        bail!(
            "{}{role} in '{}' has non-pointer type {ty}",
            at(function, operand),
            function.name
        );
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
        Type::Distinct(_, inner) => is_numeric(inner),
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
            position: Default::default(),
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
            imported: Vec::new(),
            functions: vec![IrFunction {
                name: "main".to_string(),
                param_count: 0,
                param_layouts: vec![None; 0],
                return_type,
                locals,
                blocks: vec![IrBlock {
                    statements,
                    terminator,
                }],
                entry: 0,
                module: 0,
                local: false,
                instantiated: None,
            }],
        }
    }

    fn integer(value: i64) -> IrOperand {
        IrOperand::Constant(IrConstant::Integer(value, Type::I64))
    }

    // What the strict checks are for. Each of these is a shape the pass used to
    // accept: it counted a call's arguments without looking at them, and it
    // never compared an assignment, a store or a return against the type it was
    // going into.
    #[test]
    fn refuses_an_assignment_of_the_wrong_type() {
        let module = single_block(
            Type::I64,
            vec![local(Type::I64), local(Type::Str)],
            vec![IrStatement::Assign(0, IrRvalue::Use(IrOperand::Local(1)))],
            IrTerminator::Return(Some(IrOperand::Local(0))),
        );
        let reports = check_module_recovering(&module);
        assert_eq!(reports.len(), 1, "{reports:?}");
        assert!(
            reports[0]
                .message
                .contains("is a i64 and is assigned a str")
        );
    }

    #[test]
    fn refuses_a_return_of_the_wrong_type() {
        let module = single_block(
            Type::Bool,
            vec![local(Type::Str)],
            vec![IrStatement::Assign(0, IrRvalue::Use(IrOperand::Local(0)))],
            IrTerminator::Return(Some(IrOperand::Local(0))),
        );
        let reports = check_module_recovering(&module);
        assert_eq!(reports.len(), 1, "{reports:?}");
        assert!(reports[0].message.contains("answers with a bool"));
    }

    #[test]
    fn refuses_a_store_of_the_wrong_type() {
        let module = single_block(
            Type::I64,
            vec![
                local(Type::Ptr(Box::new(Type::Bool))),
                local(Type::Str),
                local(Type::I64),
            ],
            vec![IrStatement::Store {
                address: IrOperand::Local(0),
                value: IrOperand::Local(1),
            }],
            IrTerminator::Return(Some(IrOperand::Local(2))),
        );
        let reports = check_module_recovering(&module);
        assert_eq!(reports.len(), 1, "{reports:?}");
        assert!(reports[0].message.contains("writes a str"));
    }

    #[test]
    fn refuses_an_argument_of_the_wrong_type() {
        let mut module = single_block(
            Type::I64,
            vec![local(Type::Str), local(Type::I64)],
            vec![IrStatement::Assign(
                1,
                IrRvalue::Call {
                    function: "takes_a_truth".to_string(),
                    arguments: vec![IrOperand::Local(0)],
                },
            )],
            IrTerminator::Return(Some(IrOperand::Local(1))),
        );
        module.functions.push(IrFunction {
            name: "takes_a_truth".to_string(),
            param_count: 1,
            param_layouts: vec![None],
            return_type: Type::I64,
            locals: vec![local(Type::Bool)],
            blocks: vec![IrBlock {
                statements: Vec::new(),
                terminator: IrTerminator::Return(Some(integer(0))),
            }],
            entry: 0,
            module: 0,
            local: false,
            instantiated: None,
        });
        let reports = check_module_recovering(&module);
        assert_eq!(reports.len(), 1, "{reports:?}");
        assert!(reports[0].message.contains("is a str, and it takes a bool"));
    }

    // Two integer widths do fit each other, because whether they should is a
    // question about the language rather than about this pass, and answering it
    // here would refuse programs the language accepts everywhere else.
    #[test]
    fn accepts_one_integer_width_where_another_is_wanted() {
        let module = single_block(
            Type::I64,
            vec![local(Type::I32), local(Type::I64)],
            vec![
                IrStatement::Assign(0, IrRvalue::Use(integer(7))),
                IrStatement::Assign(1, IrRvalue::Use(IrOperand::Local(0))),
            ],
            IrTerminator::Return(Some(IrOperand::Local(1))),
        );
        assert!(check_module(&module).is_ok());
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

    // Each function is checked on its own, so a module with two bad ones names
    // both rather than only whichever came first.
    #[test]
    fn reports_every_bad_function_not_only_the_first() {
        let mut module = single_block(
            Type::I64,
            vec![local(Type::I64)],
            vec![IrStatement::Assign(0, IrRvalue::Use(IrOperand::Local(9)))],
            IrTerminator::Return(Some(IrOperand::Local(0))),
        );
        let mut second = module.functions[0].clone();
        second.name = "other".to_string();
        module.functions.push(second);

        let reports = check_module_recovering(&module);
        assert_eq!(
            reports.len(),
            2,
            "expected one report per bad function, got: {reports:?}"
        );
        assert!(check_module(&module).is_err());
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
