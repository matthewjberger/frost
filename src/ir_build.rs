use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::ir::{
    BlockId, EnumLayout, EnumVariantLayout, FieldLayout, IrBinOp, IrBlock,
    IrConstant, IrExtern, IrFunction, IrLocal, IrModule, IrOperand, IrRvalue,
    IrStatement, IrTerminator, IrUnOp, LocalId, StructLayout,
};
use crate::lexer::Position;
use crate::parser::{
    Block, EnumVariant, Expression, Parameter, Pattern, ReturnSignature,
    Spanned, Statement, StructField, SwitchCase,
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
    enums: HashMap<String, EnumLayout>,
    constants: HashMap<String, Expression>,
    generic_functions: HashMap<String, GenericFunction>,
    generic_struct_defs: HashMap<String, (Vec<String>, Vec<StructField>)>,
    anon_counter: std::cell::Cell<usize>,
}

struct AnonRequest {
    name: String,
    parameters: Vec<Parameter>,
    return_sig: ReturnSignature,
    body: Block,
}

fn locate<T>(result: Result<T>, position: Position) -> Result<T> {
    result.map_err(|error| {
        let text = error.to_string();
        if position == Position::default() || text.starts_with("at line ") {
            error
        } else {
            anyhow::anyhow!(
                "at line {}, column {}: {text}",
                position.line,
                position.column
            )
        }
    })
}

pub fn build_module(statements: &[Spanned<Statement>]) -> Result<IrModule> {
    let synthetic_structs = expand_generic_structs(statements)?;
    let mut layout_statements: Vec<Statement> =
        statements.iter().map(|s| s.node.clone()).collect();
    layout_statements.extend(synthetic_structs);
    let (structs, enums) = compute_layouts(&layout_statements);
    let mut constants = HashMap::new();
    for statement in statements {
        if let Statement::Constant(name, value) = &statement.node
            && !matches!(value, Expression::Function(..) | Expression::Proc(..))
        {
            constants.insert(name.clone(), value.clone());
        }
    }
    let mut generic_functions = HashMap::new();
    for statement in statements {
        if let Statement::Constant(
            name,
            Expression::Function(parameters, return_sig, body)
            | Expression::Proc(parameters, return_sig, body),
        ) = &statement.node
            && function_is_generic(parameters)
        {
            let type_params = function_type_params(parameters);
            generic_functions.insert(
                name.clone(),
                GenericFunction {
                    type_params,
                    parameters: parameters.clone(),
                    return_sig: return_sig.clone(),
                    body: body.clone(),
                },
            );
        }
    }

    let mut generic_struct_defs = HashMap::new();
    for statement in statements {
        if let Statement::Struct(name, type_params, fields) = &statement.node
            && !type_params.is_empty()
        {
            generic_struct_defs
                .insert(name.clone(), (type_params.clone(), fields.clone()));
        }
    }

    let mut builder = IrBuilder {
        signatures: HashMap::new(),
        structs,
        enums,
        constants,
        generic_functions,
        generic_struct_defs,
        anon_counter: std::cell::Cell::new(0),
    };
    builder.collect_signatures(statements);

    let mut functions = Vec::new();
    let mut externs = Vec::new();
    let mut top_level = Vec::new();
    let mut has_main = false;
    let mut pending: Vec<Specialization> = Vec::new();
    let mut pending_anon: Vec<AnonRequest> = Vec::new();

    for statement in statements {
        let position = statement.position;
        match &statement.node {
            Statement::Constant(
                name,
                Expression::Function(parameters, return_sig, body)
                | Expression::Proc(parameters, return_sig, body),
            ) => {
                if function_is_generic(parameters) {
                    continue;
                }
                if name == "main" {
                    has_main = true;
                }
                let (function, requests, anon) = locate(
                    builder.lower_function(name, parameters, return_sig, body),
                    position,
                )?;
                functions.push(function);
                pending.extend(requests);
                pending_anon.extend(anon);
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
            _ => top_level.push(statement.clone()),
        }
    }

    if !has_main && !top_level.is_empty() {
        let empty_params: Vec<Parameter> = Vec::new();
        let (function, requests, anon) = builder.lower_function(
            "main",
            &empty_params,
            &ReturnSignature::Single(Type::I64),
            &top_level,
        )?;
        functions.push(function);
        pending.extend(requests);
        pending_anon.extend(anon);
    }

    let mut emitted: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    loop {
        if let Some(specialization) = pending.pop() {
            if !emitted.insert(specialization.mangled_name.clone()) {
                continue;
            }
            let generic = builder
                .generic_functions
                .get(&specialization.generic_name)
                .expect("specialization references a known generic function")
                .clone();
            let parameters: Vec<Parameter> = generic
                .parameters
                .iter()
                .filter(|parameter| !is_type_parameter(parameter))
                .map(|parameter| Parameter {
                    name: parameter.name.clone(),
                    type_annotation: parameter
                        .type_annotation
                        .as_ref()
                        .map(|ty| substitute_type(ty, &specialization.subst)),
                    mutable: parameter.mutable,
                })
                .collect();
            let return_sig = match generic.return_sig.to_type() {
                Some(ty) => ReturnSignature::Single(substitute_type(
                    &ty,
                    &specialization.subst,
                )),
                None => ReturnSignature::None,
            };
            let body = substitute_block(&generic.body, &specialization.subst);
            let (function, requests, anon) = builder.lower_function(
                &specialization.mangled_name,
                &parameters,
                &return_sig,
                &body,
            )?;
            functions.push(function);
            pending.extend(requests);
            pending_anon.extend(anon);
        } else if let Some(request) = pending_anon.pop() {
            let (function, requests, anon) = builder.lower_function(
                &request.name,
                &request.parameters,
                &request.return_sig,
                &request.body,
            )?;
            functions.push(function);
            pending.extend(requests);
            pending_anon.extend(anon);
        } else {
            break;
        }
    }

    Ok(IrModule { functions, externs })
}

impl IrBuilder {
    fn collect_signatures(&mut self, statements: &[Spanned<Statement>]) {
        for statement in statements {
            match &statement.node {
                Statement::Constant(
                    name,
                    Expression::Function(parameters, return_sig, _)
                    | Expression::Proc(parameters, return_sig, _),
                ) => {
                    if function_is_generic(parameters) {
                        continue;
                    }
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
    ) -> Result<(IrFunction, Vec<Specialization>, Vec<AnonRequest>)> {
        let return_type = return_sig.to_type().unwrap_or(Type::Void);
        let mut function = FunctionLowering::new(self, return_type.clone());

        for parameter in parameters {
            let ty = parameter_type(parameter);
            let local = function.fresh_local(ty, Some(parameter.name.clone()));
            function.define_variable(&parameter.name, local);
        }

        let has_defers =
            body.iter().any(|s| matches!(s.node, Statement::Defer(_)));
        if has_defers {
            function.lower_body_with_defers(body, &return_type)?;
        } else {
            let (value, value_type) =
                function.lower_block(body, Some(&return_type))?;
            if !function.current_is_terminated() {
                if matches!(return_type, Type::Void) {
                    function.set_terminator(IrTerminator::Return(None));
                } else {
                    let operand =
                        function.coerce(value, &value_type, &return_type);
                    function
                        .set_terminator(IrTerminator::Return(Some(operand)));
                }
            }
        }

        let specializations = std::mem::take(&mut function.specializations);
        let anonymous = std::mem::take(&mut function.anonymous);
        let (locals, blocks) = function.finish();
        Ok((
            IrFunction {
                name: name.to_string(),
                param_count: parameters.len(),
                return_type,
                locals,
                blocks,
                entry: 0,
            },
            specializations,
            anonymous,
        ))
    }

    fn signature(&self, name: &str) -> Option<&FunctionSignature> {
        self.signatures.get(name)
    }

    fn struct_layout(&self, name: &str) -> Option<&StructLayout> {
        self.structs.get(name)
    }

    fn enum_layout(&self, name: &str) -> Option<&EnumLayout> {
        self.enums.get(name)
    }

    fn byte_size(&self, ty: &Type) -> usize {
        size_and_align(ty, &self.structs, &self.enums)
            .map(|(size, _)| size)
            .unwrap_or(0)
    }
}

fn parameter_type(parameter: &Parameter) -> Type {
    parameter.type_annotation.clone().unwrap_or(Type::I64)
}

#[derive(Clone)]
struct GenericFunction {
    type_params: Vec<String>,
    parameters: Vec<Parameter>,
    return_sig: ReturnSignature,
    body: Block,
}

struct Specialization {
    generic_name: String,
    mangled_name: String,
    subst: HashMap<String, Type>,
}

fn function_type_params(parameters: &[Parameter]) -> Vec<String> {
    let mut names = Vec::new();
    for parameter in parameters {
        collect_type_params(&parameter_type(parameter), &mut names);
    }
    names
}

fn function_is_generic(parameters: &[Parameter]) -> bool {
    !function_type_params(parameters).is_empty()
}

fn is_type_parameter(parameter: &Parameter) -> bool {
    matches!(
        &parameter.type_annotation,
        Some(Type::TypeParam(name)) if name == &parameter.name
    )
}

fn collect_type_params(ty: &Type, out: &mut Vec<String>) {
    match ty {
        Type::TypeParam(name) => {
            if !out.contains(name) {
                out.push(name.clone());
            }
        }
        Type::Struct(name) if is_generic_instance(name) => {
            if let Some((_, arguments)) = split_instance(name) {
                for argument in arguments {
                    if let Ok(argument_type) =
                        crate::parser::type_from_string(&argument)
                    {
                        collect_type_params(&argument_type, out);
                    }
                }
            }
        }
        Type::Proc(params, ret) => {
            for param in params {
                collect_type_params(param, out);
            }
            collect_type_params(ret, out);
        }
        _ => {
            if let Some(inner) = single_inner(ty) {
                collect_type_params(inner, out);
            }
        }
    }
}

fn single_inner(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Ptr(inner)
        | Type::Ref(inner)
        | Type::RefMut(inner)
        | Type::Array(inner, _)
        | Type::Slice(inner)
        | Type::Optional(inner)
        | Type::Handle(inner)
        | Type::Distinct(inner) => Some(inner),
        _ => None,
    }
}

fn substitute_type(ty: &Type, subst: &HashMap<String, Type>) -> Type {
    match ty {
        Type::Struct(name) if is_generic_instance(name) => {
            if let Some((base, arguments)) = split_instance(name) {
                let substituted: Vec<String> = arguments
                    .iter()
                    .map(|argument| {
                        match crate::parser::type_from_string(argument) {
                            Ok(argument_type) => {
                                substitute_type(&argument_type, subst)
                                    .to_string()
                            }
                            Err(_) => argument.clone(),
                        }
                    })
                    .collect();
                return Type::Struct(format!(
                    "{}<{}>",
                    base,
                    substituted.join(", ")
                ));
            }
            ty.clone()
        }
        Type::TypeParam(name) | Type::Struct(name) => {
            if let Some(concrete) = subst.get(name) {
                return concrete.clone();
            }
            ty.clone()
        }
        Type::Ptr(inner) => Type::Ptr(Box::new(substitute_type(inner, subst))),
        Type::Ref(inner) => Type::Ref(Box::new(substitute_type(inner, subst))),
        Type::RefMut(inner) => {
            Type::RefMut(Box::new(substitute_type(inner, subst)))
        }
        Type::Array(inner, size) => {
            Type::Array(Box::new(substitute_type(inner, subst)), *size)
        }
        Type::Slice(inner) => {
            Type::Slice(Box::new(substitute_type(inner, subst)))
        }
        Type::Optional(inner) => {
            Type::Optional(Box::new(substitute_type(inner, subst)))
        }
        Type::Handle(inner) => {
            Type::Handle(Box::new(substitute_type(inner, subst)))
        }
        Type::Pool(inner) => {
            Type::Pool(Box::new(substitute_type(inner, subst)))
        }
        Type::Distinct(inner) => {
            Type::Distinct(Box::new(substitute_type(inner, subst)))
        }
        Type::Proc(params, ret) => Type::Proc(
            params.iter().map(|p| substitute_type(p, subst)).collect(),
            Box::new(substitute_type(ret, subst)),
        ),
        other => other.clone(),
    }
}

fn infer_subst_into(
    pattern: &Type,
    concrete: &Type,
    type_params: &[String],
    subst: &mut HashMap<String, Type>,
) {
    match pattern {
        Type::TypeParam(name) => {
            subst
                .entry(name.clone())
                .or_insert_with(|| concrete.clone());
            return;
        }
        Type::Struct(name) if type_params.contains(name) => {
            subst
                .entry(name.clone())
                .or_insert_with(|| concrete.clone());
            return;
        }
        _ => {}
    }
    if let (Some(pattern_inner), Some(concrete_inner)) =
        (single_inner(pattern), single_inner(concrete))
    {
        infer_subst_into(pattern_inner, concrete_inner, type_params, subst);
    } else if let (Type::Proc(pp, pr), Type::Proc(cp, cr)) = (pattern, concrete)
    {
        for (pattern_param, concrete_param) in pp.iter().zip(cp) {
            infer_subst_into(pattern_param, concrete_param, type_params, subst);
        }
        infer_subst_into(pr, cr, type_params, subst);
    } else if let (Type::Struct(pattern_name), Type::Struct(concrete_name)) =
        (pattern, concrete)
        && let (
            Some((pattern_base, pattern_args)),
            Some((concrete_base, concrete_args)),
        ) = (split_instance(pattern_name), split_instance(concrete_name))
        && pattern_base == concrete_base
        && pattern_args.len() == concrete_args.len()
    {
        for (pattern_arg, concrete_arg) in
            pattern_args.iter().zip(&concrete_args)
        {
            if let (Ok(pattern_type), Ok(concrete_type)) = (
                crate::parser::type_from_string(pattern_arg),
                crate::parser::type_from_string(concrete_arg),
            ) {
                infer_subst_into(
                    &pattern_type,
                    &concrete_type,
                    type_params,
                    subst,
                );
            }
        }
    }
}

fn sanitize_identifier(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn mangle_type(ty: &Type) -> String {
    match ty {
        Type::I8 => "i8".to_string(),
        Type::I16 => "i16".to_string(),
        Type::I32 => "i32".to_string(),
        Type::I64 => "i64".to_string(),
        Type::Isize => "isize".to_string(),
        Type::U8 => "u8".to_string(),
        Type::U16 => "u16".to_string(),
        Type::U32 => "u32".to_string(),
        Type::U64 => "u64".to_string(),
        Type::Usize => "usize".to_string(),
        Type::F32 => "f32".to_string(),
        Type::F64 => "f64".to_string(),
        Type::Bool => "bool".to_string(),
        Type::Struct(name) | Type::Enum(name) => sanitize_identifier(name),
        Type::Ptr(inner) => format!("p_{}", mangle_type(inner)),
        Type::Ref(inner) => format!("r_{}", mangle_type(inner)),
        Type::RefMut(inner) => format!("rm_{}", mangle_type(inner)),
        Type::Array(inner, size) => format!("a{}_{}", size, mangle_type(inner)),
        Type::Handle(inner) => format!("h_{}", mangle_type(inner)),
        Type::Pool(inner) => format!("pool_{}", mangle_type(inner)),
        Type::Proc(_, _) => "proc".to_string(),
        other => format!("{other}"),
    }
}

fn mangle_specialization(
    name: &str,
    type_params: &[String],
    subst: &HashMap<String, Type>,
) -> String {
    let mut mangled = name.to_string();
    for type_param in type_params {
        mangled.push_str("__");
        match subst.get(type_param) {
            Some(concrete) => mangled.push_str(&mangle_type(concrete)),
            None => mangled.push_str("unknown"),
        }
    }
    mangled
}

fn is_generic_instance(name: &str) -> bool {
    name.contains('<')
}

fn split_instance(name: &str) -> Option<(String, Vec<String>)> {
    let open = name.find('<')?;
    if !name.ends_with('>') {
        return None;
    }
    let base = name[..open].to_string();
    let inner = &name[open + 1..name.len() - 1];
    let mut arguments = Vec::new();
    let mut depth = 0usize;
    let mut current = String::new();
    for character in inner.chars() {
        match character {
            '<' => {
                depth += 1;
                current.push(character);
            }
            '>' => {
                depth -= 1;
                current.push(character);
            }
            ',' if depth == 0 => {
                arguments.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(character),
        }
    }
    if !current.trim().is_empty() {
        arguments.push(current.trim().to_string());
    }
    Some((base, arguments))
}

fn collect_instances_in_type(ty: &Type, out: &mut Vec<String>) {
    if let Type::Struct(name) = ty
        && is_generic_instance(name)
        && !out.contains(name)
    {
        out.push(name.clone());
    }
    if let Some(inner) = single_inner(ty) {
        collect_instances_in_type(inner, out);
    } else if let Type::Proc(params, ret) = ty {
        for param in params {
            collect_instances_in_type(param, out);
        }
        collect_instances_in_type(ret, out);
    }
}

fn collect_instances_in_block(block: &Block, out: &mut Vec<String>) {
    for statement in block {
        collect_instances_in_statement(statement, out);
    }
}

fn collect_instances_in_statement(
    statement: &Statement,
    out: &mut Vec<String>,
) {
    match statement {
        Statement::Let {
            type_annotation,
            value,
            ..
        } => {
            if let Some(ty) = type_annotation {
                collect_instances_in_type(ty, out);
            }
            collect_instances_in_expression(value, out);
        }
        Statement::Return(expression) | Statement::Expression(expression) => {
            collect_instances_in_expression(expression, out);
        }
        Statement::Assignment(target, value) => {
            collect_instances_in_expression(target, out);
            collect_instances_in_expression(value, out);
        }
        Statement::For(_, range, body) => {
            collect_instances_in_expression(range, out);
            collect_instances_in_block(body, out);
        }
        Statement::While(condition, body) => {
            collect_instances_in_expression(condition, out);
            collect_instances_in_block(body, out);
        }
        Statement::Defer(inner) => {
            collect_instances_in_statement(inner, out);
        }
        _ => {}
    }
}

fn collect_instances_in_expression(
    expression: &Expression,
    out: &mut Vec<String>,
) {
    match expression {
        Expression::Sizeof(ty) => collect_instances_in_type(ty, out),
        Expression::Prefix(_, operand)
        | Expression::AddressOf(operand)
        | Expression::Borrow(operand)
        | Expression::BorrowMut(operand)
        | Expression::Dereference(operand) => {
            collect_instances_in_expression(operand, out);
        }
        Expression::Infix(left, _, right) => {
            collect_instances_in_expression(left, out);
            collect_instances_in_expression(right, out);
        }
        Expression::If(condition, consequence, alternative) => {
            collect_instances_in_expression(condition, out);
            collect_instances_in_block(consequence, out);
            if let Some(block) = alternative {
                collect_instances_in_block(block, out);
            }
        }
        Expression::Call(callee, arguments) => {
            collect_instances_in_expression(callee, out);
            for argument in arguments {
                collect_instances_in_expression(argument, out);
            }
        }
        Expression::Index(base, index) => {
            collect_instances_in_expression(base, out);
            collect_instances_in_expression(index, out);
        }
        Expression::FieldAccess(base, _) => {
            collect_instances_in_expression(base, out);
        }
        Expression::StructInit(_, fields)
        | Expression::EnumVariantInit(_, _, fields) => {
            for (_, value) in fields {
                collect_instances_in_expression(value, out);
            }
        }
        Expression::Range(start, end, _) => {
            collect_instances_in_expression(start, out);
            collect_instances_in_expression(end, out);
        }
        Expression::Tuple(elements) => {
            for element in elements {
                collect_instances_in_expression(element, out);
            }
        }
        Expression::Switch(scrutinee, cases) => {
            collect_instances_in_expression(scrutinee, out);
            for case in cases {
                collect_instances_in_block(&case.body, out);
            }
        }
        _ => {}
    }
}

struct Discovery<'a> {
    functions: &'a HashMap<String, GenericFunction>,
    structs: &'a HashMap<String, (Vec<String>, Vec<StructField>)>,
}

fn infer_struct_instance_shallow(
    struct_name: &str,
    field_inits: &[(String, Expression)],
    env: &HashMap<String, Type>,
    discovery: &Discovery,
) -> Option<String> {
    let (type_params, fields) = discovery.structs.get(struct_name)?;
    let mut subst: HashMap<String, Type> = HashMap::new();
    for (field_name, value) in field_inits {
        if let Some(field) =
            fields.iter().find(|field| &field.name == field_name)
            && let Some(value_type) =
                infer_expr_type_shallow(value, env, discovery)
        {
            infer_subst_into(
                &field.field_type,
                &value_type,
                type_params,
                &mut subst,
            );
        }
    }
    let rendered: Vec<String> = type_params
        .iter()
        .map(|type_param| {
            subst
                .get(type_param)
                .map(|ty| ty.to_string())
                .unwrap_or_else(|| type_param.clone())
        })
        .collect();
    Some(format!("{struct_name}<{}>", rendered.join(", ")))
}

fn infer_expr_type_shallow(
    expression: &Expression,
    env: &HashMap<String, Type>,
    discovery: &Discovery,
) -> Option<Type> {
    match expression {
        Expression::Literal(Literal::Integer(_)) => Some(Type::I64),
        Expression::Literal(Literal::Float(_)) => Some(Type::F64),
        Expression::Literal(Literal::Float32(_)) => Some(Type::F32),
        Expression::Boolean(_) | Expression::Literal(Literal::Boolean(_)) => {
            Some(Type::Bool)
        }
        Expression::Identifier(name) => env.get(name).cloned(),
        Expression::StructInit(name, fields) => {
            if discovery.structs.contains_key(name) {
                infer_struct_instance_shallow(name, fields, env, discovery)
                    .map(Type::Struct)
            } else {
                Some(Type::Struct(name.clone()))
            }
        }
        Expression::EnumVariantInit(name, _, _) => {
            Some(Type::Enum(name.clone()))
        }
        Expression::Borrow(inner) => {
            infer_expr_type_shallow(inner, env, discovery)
                .map(|inner| Type::Ref(Box::new(inner)))
        }
        Expression::BorrowMut(inner) => {
            infer_expr_type_shallow(inner, env, discovery)
                .map(|inner| Type::RefMut(Box::new(inner)))
        }
        Expression::Call(callee, arguments) => {
            let Expression::Identifier(name) = callee.as_ref() else {
                return None;
            };
            let generic = discovery.functions.get(name)?;
            let subst = infer_call_subst(generic, arguments, env, discovery);
            generic
                .return_sig
                .to_type()
                .map(|ty| substitute_type(&ty, &subst))
        }
        _ => None,
    }
}

fn infer_call_subst(
    generic: &GenericFunction,
    arguments: &[Expression],
    env: &HashMap<String, Type>,
    discovery: &Discovery,
) -> HashMap<String, Type> {
    let mut subst = HashMap::new();
    for (parameter, argument) in generic.parameters.iter().zip(arguments) {
        if is_type_parameter(parameter)
            && let Expression::TypeValue(ty) = argument
        {
            subst.insert(parameter.name.clone(), ty.clone());
            continue;
        }
        if let Some(argument_type) =
            infer_expr_type_shallow(argument, env, discovery)
        {
            infer_subst_into(
                &parameter_type(parameter),
                &argument_type,
                &generic.type_params,
                &mut subst,
            );
        }
    }
    subst
}

fn collect_call_instances_in_block(
    block: &Block,
    env: &mut HashMap<String, Type>,
    discovery: &Discovery,
    out: &mut Vec<String>,
) {
    for statement in block {
        collect_call_instances_in_statement(statement, env, discovery, out);
    }
}

fn collect_call_instances_in_statement(
    statement: &Statement,
    env: &mut HashMap<String, Type>,
    discovery: &Discovery,
    out: &mut Vec<String>,
) {
    match statement {
        Statement::Let {
            name,
            type_annotation,
            value,
            ..
        } => {
            collect_call_instances_in_expression(value, env, discovery, out);
            let inferred = type_annotation
                .clone()
                .or_else(|| infer_expr_type_shallow(value, env, discovery));
            if let Some(ty) = inferred {
                env.insert(name.clone(), ty);
            }
        }
        Statement::Return(expression) | Statement::Expression(expression) => {
            collect_call_instances_in_expression(
                expression, env, discovery, out,
            );
        }
        Statement::Assignment(target, value) => {
            collect_call_instances_in_expression(target, env, discovery, out);
            collect_call_instances_in_expression(value, env, discovery, out);
        }
        Statement::For(variable, range, body) => {
            collect_call_instances_in_expression(range, env, discovery, out);
            env.insert(variable.clone(), Type::I64);
            collect_call_instances_in_block(body, env, discovery, out);
        }
        Statement::While(condition, body) => {
            collect_call_instances_in_expression(
                condition, env, discovery, out,
            );
            collect_call_instances_in_block(body, env, discovery, out);
        }
        Statement::Defer(inner) => {
            collect_call_instances_in_statement(inner, env, discovery, out);
        }
        _ => {}
    }
}

fn collect_call_instances_in_expression(
    expression: &Expression,
    env: &mut HashMap<String, Type>,
    discovery: &Discovery,
    out: &mut Vec<String>,
) {
    match expression {
        Expression::Call(callee, arguments) => {
            if let Expression::Identifier(name) = callee.as_ref()
                && let Some(generic) = discovery.functions.get(name)
            {
                let subst =
                    infer_call_subst(generic, arguments, env, discovery);
                if let Some(return_type) = generic.return_sig.to_type() {
                    collect_instances_in_type(
                        &substitute_type(&return_type, &subst),
                        out,
                    );
                }
                for parameter in &generic.parameters {
                    collect_instances_in_type(
                        &substitute_type(&parameter_type(parameter), &subst),
                        out,
                    );
                }
            }
            collect_call_instances_in_expression(callee, env, discovery, out);
            for argument in arguments {
                collect_call_instances_in_expression(
                    argument, env, discovery, out,
                );
            }
        }
        Expression::StructInit(name, fields) => {
            if discovery.structs.contains_key(name)
                && let Some(instance) =
                    infer_struct_instance_shallow(name, fields, env, discovery)
            {
                out.push(instance);
            }
            for (_, value) in fields {
                collect_call_instances_in_expression(
                    value, env, discovery, out,
                );
            }
        }
        Expression::Prefix(_, operand)
        | Expression::AddressOf(operand)
        | Expression::Borrow(operand)
        | Expression::BorrowMut(operand)
        | Expression::Dereference(operand) => {
            collect_call_instances_in_expression(operand, env, discovery, out);
        }
        Expression::Infix(left, _, right) => {
            collect_call_instances_in_expression(left, env, discovery, out);
            collect_call_instances_in_expression(right, env, discovery, out);
        }
        Expression::If(condition, consequence, alternative) => {
            collect_call_instances_in_expression(
                condition, env, discovery, out,
            );
            let mut branch_env = env.clone();
            collect_call_instances_in_block(
                consequence,
                &mut branch_env,
                discovery,
                out,
            );
            if let Some(block) = alternative {
                let mut branch_env = env.clone();
                collect_call_instances_in_block(
                    block,
                    &mut branch_env,
                    discovery,
                    out,
                );
            }
        }
        Expression::Index(base, index) => {
            collect_call_instances_in_expression(base, env, discovery, out);
            collect_call_instances_in_expression(index, env, discovery, out);
        }
        Expression::FieldAccess(base, _) => {
            collect_call_instances_in_expression(base, env, discovery, out);
        }
        Expression::EnumVariantInit(_, _, fields) => {
            for (_, value) in fields {
                collect_call_instances_in_expression(
                    value, env, discovery, out,
                );
            }
        }
        Expression::Switch(scrutinee, cases) => {
            collect_call_instances_in_expression(
                scrutinee, env, discovery, out,
            );
            for case in cases {
                let mut branch_env = env.clone();
                collect_call_instances_in_block(
                    &case.body,
                    &mut branch_env,
                    discovery,
                    out,
                );
            }
        }
        _ => {}
    }
}

fn expand_generic_structs(
    statements: &[Spanned<Statement>],
) -> Result<Vec<Statement>> {
    let mut generic_structs: HashMap<String, (Vec<String>, Vec<StructField>)> =
        HashMap::new();
    for statement in statements {
        let statement = &statement.node;
        if let Statement::Struct(name, type_params, fields) = statement
            && !type_params.is_empty()
        {
            generic_structs
                .insert(name.clone(), (type_params.clone(), fields.clone()));
        }
    }
    if generic_structs.is_empty() {
        return Ok(Vec::new());
    }

    let mut generic_functions: HashMap<String, GenericFunction> =
        HashMap::new();
    for statement in statements {
        let statement = &statement.node;
        if let Statement::Constant(
            name,
            Expression::Function(parameters, return_sig, body)
            | Expression::Proc(parameters, return_sig, body),
        ) = statement
            && function_is_generic(parameters)
        {
            generic_functions.insert(
                name.clone(),
                GenericFunction {
                    type_params: function_type_params(parameters),
                    parameters: parameters.clone(),
                    return_sig: return_sig.clone(),
                    body: body.clone(),
                },
            );
        }
    }

    let discovery = Discovery {
        functions: &generic_functions,
        structs: &generic_structs,
    };
    let mut queue: Vec<String> = Vec::new();
    for statement in statements {
        let statement = &statement.node;
        if let Statement::Constant(
            _,
            Expression::Function(parameters, _, body)
            | Expression::Proc(parameters, _, body),
        ) = statement
        {
            let mut env: HashMap<String, Type> = HashMap::new();
            for parameter in parameters {
                if let Some(ty) = &parameter.type_annotation {
                    env.insert(parameter.name.clone(), ty.clone());
                }
            }
            collect_call_instances_in_block(
                body, &mut env, &discovery, &mut queue,
            );
        }
        collect_instances_in_statement(statement, &mut queue);
        if let Statement::Struct(_, _, fields) = statement {
            for field in fields {
                collect_instances_in_type(&field.field_type, &mut queue);
            }
        }
        if let Statement::Enum(_, variants) = statement {
            for variant in variants {
                if let Some(fields) = &variant.fields {
                    for field in fields {
                        collect_instances_in_type(
                            &field.field_type,
                            &mut queue,
                        );
                    }
                }
            }
        }
        if let Statement::Extern {
            params,
            return_type,
            ..
        } = statement
        {
            for parameter in params {
                if let Some(ty) = &parameter.type_annotation {
                    collect_instances_in_type(ty, &mut queue);
                }
            }
            if let Some(ty) = return_type {
                collect_instances_in_type(ty, &mut queue);
            }
        }
        if let Statement::Constant(
            _,
            Expression::Function(parameters, return_sig, body)
            | Expression::Proc(parameters, return_sig, body),
        ) = statement
        {
            for parameter in parameters {
                if let Some(ty) = &parameter.type_annotation {
                    collect_instances_in_type(ty, &mut queue);
                }
            }
            if let Some(ty) = return_sig.to_type() {
                collect_instances_in_type(&ty, &mut queue);
            }
            collect_instances_in_block(body, &mut queue);
        }
    }

    let mut done: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let mut synthetic = Vec::new();
    while let Some(instance) = queue.pop() {
        if !done.insert(instance.clone()) {
            continue;
        }
        let Some((base, argument_strings)) = split_instance(&instance) else {
            continue;
        };
        let Some((type_params, fields)) = generic_structs.get(&base) else {
            continue;
        };
        if type_params.len() != argument_strings.len() {
            bail!(
                "native backend: generic struct '{base}' expects {} type argument(s) but {} were given",
                type_params.len(),
                argument_strings.len()
            );
        }
        let mut subst = HashMap::new();
        for (type_param, argument) in type_params.iter().zip(&argument_strings)
        {
            let argument_type = crate::parser::type_from_string(argument)?;
            subst.insert(type_param.clone(), argument_type);
        }
        let concrete_fields: Vec<StructField> = fields
            .iter()
            .map(|field| StructField {
                name: field.name.clone(),
                field_type: substitute_type(&field.field_type, &subst),
            })
            .collect();
        for field in &concrete_fields {
            collect_instances_in_type(&field.field_type, &mut queue);
        }
        synthetic.push(Statement::Struct(
            instance.clone(),
            Vec::new(),
            concrete_fields,
        ));
    }
    Ok(synthetic)
}

fn substitute_block(block: &Block, subst: &HashMap<String, Type>) -> Block {
    block
        .iter()
        .map(|statement| {
            Spanned::new(
                substitute_statement(&statement.node, subst),
                statement.position,
            )
        })
        .collect()
}

fn substitute_statement(
    statement: &Statement,
    subst: &HashMap<String, Type>,
) -> Statement {
    match statement {
        Statement::Let {
            name,
            type_annotation,
            value,
            mutable,
        } => Statement::Let {
            name: name.clone(),
            type_annotation: type_annotation
                .as_ref()
                .map(|ty| substitute_type(ty, subst)),
            value: substitute_expression(value, subst),
            mutable: *mutable,
        },
        Statement::Return(expression) => {
            Statement::Return(substitute_expression(expression, subst))
        }
        Statement::Expression(expression) => {
            Statement::Expression(substitute_expression(expression, subst))
        }
        Statement::Assignment(target, value) => Statement::Assignment(
            substitute_expression(target, subst),
            substitute_expression(value, subst),
        ),
        Statement::For(variable, range, body) => Statement::For(
            variable.clone(),
            substitute_expression(range, subst),
            substitute_block(body, subst),
        ),
        Statement::While(condition, body) => Statement::While(
            substitute_expression(condition, subst),
            substitute_block(body, subst),
        ),
        Statement::Defer(inner) => {
            Statement::Defer(Box::new(substitute_statement(inner, subst)))
        }
        other => other.clone(),
    }
}

fn substitute_expression(
    expression: &Expression,
    subst: &HashMap<String, Type>,
) -> Expression {
    match expression {
        Expression::Prefix(operator, operand) => Expression::Prefix(
            *operator,
            Box::new(substitute_expression(operand, subst)),
        ),
        Expression::Infix(left, operator, right) => Expression::Infix(
            Box::new(substitute_expression(left, subst)),
            *operator,
            Box::new(substitute_expression(right, subst)),
        ),
        Expression::If(condition, consequence, alternative) => Expression::If(
            Box::new(substitute_expression(condition, subst)),
            substitute_block(consequence, subst),
            alternative
                .as_ref()
                .map(|block| substitute_block(block, subst)),
        ),
        Expression::Call(callee, arguments) => Expression::Call(
            Box::new(substitute_expression(callee, subst)),
            arguments
                .iter()
                .map(|argument| substitute_expression(argument, subst))
                .collect(),
        ),
        Expression::Index(base, index) => Expression::Index(
            Box::new(substitute_expression(base, subst)),
            Box::new(substitute_expression(index, subst)),
        ),
        Expression::FieldAccess(base, field) => Expression::FieldAccess(
            Box::new(substitute_expression(base, subst)),
            field.clone(),
        ),
        Expression::AddressOf(inner) => {
            Expression::AddressOf(Box::new(substitute_expression(inner, subst)))
        }
        Expression::Borrow(inner) => {
            Expression::Borrow(Box::new(substitute_expression(inner, subst)))
        }
        Expression::BorrowMut(inner) => {
            Expression::BorrowMut(Box::new(substitute_expression(inner, subst)))
        }
        Expression::Dereference(inner) => Expression::Dereference(Box::new(
            substitute_expression(inner, subst),
        )),
        Expression::StructInit(name, fields) => Expression::StructInit(
            name.clone(),
            fields
                .iter()
                .map(|(field, value)| {
                    (field.clone(), substitute_expression(value, subst))
                })
                .collect(),
        ),
        Expression::EnumVariantInit(name, variant, fields) => {
            Expression::EnumVariantInit(
                name.clone(),
                variant.clone(),
                fields
                    .iter()
                    .map(|(field, value)| {
                        (field.clone(), substitute_expression(value, subst))
                    })
                    .collect(),
            )
        }
        Expression::Sizeof(ty) => {
            Expression::Sizeof(substitute_type(ty, subst))
        }
        Expression::Range(start, end, inclusive) => Expression::Range(
            Box::new(substitute_expression(start, subst)),
            Box::new(substitute_expression(end, subst)),
            *inclusive,
        ),
        Expression::Tuple(elements) => Expression::Tuple(
            elements
                .iter()
                .map(|element| substitute_expression(element, subst))
                .collect(),
        ),
        Expression::Switch(scrutinee, cases) => Expression::Switch(
            Box::new(substitute_expression(scrutinee, subst)),
            cases
                .iter()
                .map(|case| SwitchCase {
                    pattern: case.pattern.clone(),
                    body: substitute_block(&case.body, subst),
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

fn needs_memory(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Struct(_) | Type::Array(_, _) | Type::Enum(_) | Type::Str
    )
}

const STR_PTR_OFFSET: usize = 0;
const STR_LEN_OFFSET: usize = 8;

fn str_byte_ptr_type() -> Type {
    Type::Ptr(Box::new(Type::U8))
}

fn body_has_nested_return(body: &Block) -> bool {
    body.iter().any(|statement| match &statement.node {
        Statement::Return(_) => false,
        other => statement_contains_return(other),
    })
}

fn block_contains_return(block: &Block) -> bool {
    block
        .iter()
        .any(|statement| statement_contains_return(&statement.node))
}

fn statement_contains_return(statement: &Statement) -> bool {
    match statement {
        Statement::Return(_) => true,
        Statement::While(_, body) | Statement::For(_, _, body) => {
            block_contains_return(body)
        }
        Statement::Defer(inner) => statement_contains_return(inner),
        Statement::Expression(expression) => {
            expression_contains_return(expression)
        }
        _ => false,
    }
}

fn expression_contains_return(expression: &Expression) -> bool {
    match expression {
        Expression::If(_, consequence, alternative) => {
            block_contains_return(consequence)
                || alternative.as_ref().is_some_and(block_contains_return)
        }
        Expression::Switch(_, cases) => {
            cases.iter().any(|case| block_contains_return(&case.body))
        }
        _ => false,
    }
}

fn array_element_type(
    annotation: Option<&Type>,
    elements: &[Expression],
    signatures: &HashMap<String, FunctionSignature>,
) -> Type {
    match annotation {
        Some(Type::Array(inner, _)) | Some(Type::Slice(inner)) => {
            return (**inner).clone();
        }
        _ => {}
    }
    match elements.first() {
        Some(Expression::Literal(Literal::Integer(_))) => Type::I64,
        Some(Expression::Literal(Literal::Float(_))) => Type::F64,
        Some(Expression::Literal(Literal::Float32(_))) => Type::F32,
        Some(Expression::Literal(Literal::Boolean(_)))
        | Some(Expression::Boolean(_)) => Type::Bool,
        Some(Expression::StructInit(name, _)) => Type::Struct(name.clone()),
        Some(Expression::EnumVariantInit(name, _, _)) => {
            Type::Enum(name.clone())
        }
        Some(Expression::Identifier(name))
            if let Some(signature) = signatures.get(name) =>
        {
            Type::Proc(
                signature.parameters.clone(),
                Box::new(signature.return_type.clone()),
            )
        }
        Some(
            Expression::Function(parameters, return_sig, _)
            | Expression::Proc(parameters, return_sig, _),
        ) => Type::Proc(
            parameters.iter().map(parameter_type).collect(),
            Box::new(return_sig.to_type().unwrap_or(Type::Void)),
        ),
        Some(Expression::Literal(Literal::Array(inner))) => Type::Array(
            Box::new(array_element_type(None, inner, signatures)),
            inner.len(),
        ),
        _ => Type::I64,
    }
}

type LayoutMaps = (HashMap<String, StructLayout>, HashMap<String, EnumLayout>);

fn compute_layouts(statements: &[Statement]) -> LayoutMaps {
    let struct_defs: Vec<(&String, &Vec<StructField>)> = statements
        .iter()
        .filter_map(|statement| match statement {
            Statement::Struct(name, _, fields) => Some((name, fields)),
            _ => None,
        })
        .collect();
    let enum_defs: Vec<(&String, &Vec<EnumVariant>)> = statements
        .iter()
        .filter_map(|statement| match statement {
            Statement::Enum(name, variants) => Some((name, variants)),
            _ => None,
        })
        .collect();

    let mut structs: HashMap<String, StructLayout> = HashMap::new();
    let mut enums: HashMap<String, EnumLayout> = HashMap::new();
    loop {
        let mut progress = false;
        for (name, fields) in &struct_defs {
            if structs.contains_key(*name) {
                continue;
            }
            if let Some(layout) = try_struct_layout(fields, &structs, &enums) {
                structs.insert((*name).clone(), layout);
                progress = true;
            }
        }
        for (name, variants) in &enum_defs {
            if enums.contains_key(*name) {
                continue;
            }
            if let Some(layout) = try_enum_layout(variants, &structs, &enums) {
                enums.insert((*name).clone(), layout);
                progress = true;
            }
        }
        if !progress {
            break;
        }
    }
    (structs, enums)
}

fn try_struct_layout(
    fields: &[StructField],
    structs: &HashMap<String, StructLayout>,
    enums: &HashMap<String, EnumLayout>,
) -> Option<StructLayout> {
    let mut offset = 0;
    let mut align = 1;
    let mut field_layouts = Vec::with_capacity(fields.len());
    for field in fields {
        let (field_size, field_align) =
            size_and_align(&field.field_type, structs, enums)?;
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

fn try_enum_layout(
    variants: &[EnumVariant],
    structs: &HashMap<String, StructLayout>,
    enums: &HashMap<String, EnumLayout>,
) -> Option<EnumLayout> {
    let tag_size = 4;
    let mut payload_align = 1;
    for variant in variants {
        if let Some(fields) = &variant.fields {
            for field in fields {
                let (_, field_align) =
                    size_and_align(&field.field_type, structs, enums)?;
                payload_align = payload_align.max(field_align);
            }
        }
    }
    let payload_offset = round_up(tag_size, payload_align);

    let mut variant_layouts = Vec::with_capacity(variants.len());
    let mut max_end = payload_offset;
    for (index, variant) in variants.iter().enumerate() {
        let mut offset = payload_offset;
        let mut field_layouts = Vec::new();
        if let Some(fields) = &variant.fields {
            for field in fields {
                let (field_size, field_align) =
                    size_and_align(&field.field_type, structs, enums)?;
                offset = round_up(offset, field_align);
                field_layouts.push(FieldLayout {
                    name: field.name.clone(),
                    ty: field.field_type.clone(),
                    offset,
                });
                offset += field_size;
            }
        }
        max_end = max_end.max(offset);
        variant_layouts.push(EnumVariantLayout {
            name: variant.name.clone(),
            tag: index as u32,
            fields: field_layouts,
        });
    }

    let align = payload_align.max(tag_size);
    Some(EnumLayout {
        size: round_up(max_end, align),
        align,
        variants: variant_layouts,
    })
}

fn size_and_align(
    ty: &Type,
    structs: &HashMap<String, StructLayout>,
    enums: &HashMap<String, EnumLayout>,
) -> Option<(usize, usize)> {
    match ty {
        // A named type from an annotation parses as `Struct`, even when it
        // names an enum, so fall back to the enum registry.
        Type::Struct(name) => structs
            .get(name)
            .map(|layout| (layout.size, layout.align))
            .or_else(|| {
                enums.get(name).map(|layout| (layout.size, layout.align))
            }),
        Type::Enum(name) => {
            enums.get(name).map(|layout| (layout.size, layout.align))
        }
        Type::Array(inner, count) => {
            let (size, align) = size_and_align(inner, structs, enums)?;
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
    specializations: Vec<Specialization>,
    anonymous: Vec<AnonRequest>,
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
            specializations: Vec::new(),
            anonymous: Vec::new(),
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
            let position = statement.position;
            if is_last
                && let Statement::Expression(expression) = &statement.node
            {
                result = locate(
                    self.lower_expression(expression, expected),
                    position,
                )?;
            } else {
                locate(self.lower_statement(&statement.node), position)?;
            }
        }
        self.pop_scope();
        Ok(result)
    }

    fn lower_body_with_defers(
        &mut self,
        body: &Block,
        return_type: &Type,
    ) -> Result<()> {
        if body_has_nested_return(body) {
            bail!(
                "native backend: a return inside a branch is not supported together with defer yet"
            );
        }
        let defers: Vec<&Statement> = body
            .iter()
            .filter_map(|statement| match &statement.node {
                Statement::Defer(inner) => Some(inner.as_ref()),
                _ => None,
            })
            .collect();

        self.push_scope();
        for (index, statement) in body.iter().enumerate() {
            let is_last = index + 1 == body.len();
            let position = statement.position;
            match &statement.node {
                Statement::Defer(_) => {}
                Statement::Return(expression) => {
                    let operand = locate(
                        self.lower_return_value(expression, return_type),
                        position,
                    )?;
                    self.run_defers(&defers)?;
                    self.set_terminator(IrTerminator::Return(operand));
                }
                Statement::Expression(expression) if is_last => {
                    let (value, value_type) = locate(
                        self.lower_expression(expression, Some(return_type)),
                        position,
                    )?;
                    self.run_defers(&defers)?;
                    if !self.current_is_terminated() {
                        if matches!(return_type, Type::Void) {
                            self.set_terminator(IrTerminator::Return(None));
                        } else {
                            let operand =
                                self.coerce(value, &value_type, return_type);
                            self.set_terminator(IrTerminator::Return(Some(
                                operand,
                            )));
                        }
                    }
                }
                other => locate(self.lower_statement(other), position)?,
            }
        }
        self.pop_scope();

        if !self.current_is_terminated() {
            self.run_defers(&defers)?;
            self.set_terminator(IrTerminator::Return(None));
        }
        Ok(())
    }

    fn lower_return_value(
        &mut self,
        expression: &Expression,
        return_type: &Type,
    ) -> Result<Option<IrOperand>> {
        if matches!(return_type, Type::Void) {
            return Ok(None);
        }
        let (operand, value_type) =
            self.lower_expression(expression, Some(return_type))?;
        Ok(Some(self.coerce(operand, &value_type, return_type)))
    }

    fn run_defers(&mut self, defers: &[&Statement]) -> Result<()> {
        for deferred in defers.iter().rev() {
            self.lower_statement(deferred)?;
        }
        Ok(())
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
                    let layout_name = match type_annotation {
                        Some(Type::Struct(annotated))
                            if is_generic_instance(annotated) =>
                        {
                            annotated.clone()
                        }
                        _ if self
                            .builder
                            .generic_struct_defs
                            .contains_key(struct_name) =>
                        {
                            self.generic_instance_of(struct_name, field_inits)
                                .unwrap_or_else(|| struct_name.clone())
                        }
                        _ => struct_name.clone(),
                    };
                    let ty = Type::Struct(layout_name.clone());
                    let local = self.fresh_local(ty, Some(name.clone()));
                    self.init_struct(local, &layout_name, field_inits)?;
                    self.define_variable(name, local);
                    return Ok(());
                }
                if let Expression::Literal(Literal::Array(elements)) = value {
                    let element_type = array_element_type(
                        type_annotation.as_ref(),
                        elements,
                        &self.builder.signatures,
                    );
                    let ty = Type::Array(
                        Box::new(element_type.clone()),
                        elements.len(),
                    );
                    let local = self.fresh_local(ty, Some(name.clone()));
                    self.init_array(local, &element_type, elements)?;
                    self.define_variable(name, local);
                    return Ok(());
                }
                if let Expression::EnumVariantInit(
                    enum_name,
                    variant_name,
                    field_inits,
                ) = value
                {
                    let ty = Type::Enum(enum_name.clone());
                    let local = self.fresh_local(ty, Some(name.clone()));
                    self.init_enum(
                        local,
                        enum_name,
                        variant_name,
                        field_inits,
                    )?;
                    self.define_variable(name, local);
                    return Ok(());
                }
                let (operand, value_type) =
                    self.lower_expression(value, type_annotation.as_ref())?;
                if matches!(value_type, Type::Void) {
                    bail!(
                        "native backend: cannot bind '{name}' to a void value; this expression produces no value"
                    );
                }
                let declared = type_annotation
                    .clone()
                    .unwrap_or_else(|| value_type.clone());
                let coerced = self.coerce(operand, &value_type, &declared);
                let local =
                    self.fresh_local(declared.clone(), Some(name.clone()));
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
                if let Some(local) = self.resolve_variable(name) {
                    return Ok((
                        IrOperand::Local(local),
                        self.type_of_local(local),
                    ));
                }
                if let Some(signature) = self.builder.signature(name) {
                    let proc_type = Type::Proc(
                        signature.parameters.clone(),
                        Box::new(signature.return_type.clone()),
                    );
                    let result = self.fresh_local(proc_type.clone(), None);
                    self.emit(IrStatement::Assign(
                        result,
                        IrRvalue::FunctionAddress(name.clone()),
                    ));
                    return Ok((IrOperand::Local(result), proc_type));
                }
                if let Some(value) = self.builder.constants.get(name).cloned() {
                    return self.lower_expression(&value, expected);
                }
                bail!("native backend: unknown variable '{name}'");
            }
            Expression::Function(parameters, return_sig, body)
            | Expression::Proc(parameters, return_sig, body) => {
                let id = self.builder.anon_counter.get();
                self.builder.anon_counter.set(id + 1);
                let name = format!("__anon_{id}");
                let param_types: Vec<Type> =
                    parameters.iter().map(parameter_type).collect();
                let return_type = return_sig.to_type().unwrap_or(Type::Void);
                let proc_type = Type::Proc(param_types, Box::new(return_type));
                self.anonymous.push(AnonRequest {
                    name: name.clone(),
                    parameters: parameters.clone(),
                    return_sig: return_sig.clone(),
                    body: body.clone(),
                });
                let result = self.fresh_local(proc_type.clone(), None);
                self.emit(IrStatement::Assign(
                    result,
                    IrRvalue::FunctionAddress(name),
                ));
                Ok((IrOperand::Local(result), proc_type))
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
            Expression::Sizeof(ty) => {
                let size = self.builder.byte_size(ty) as i64;
                Ok((
                    IrOperand::Constant(IrConstant::Integer(size, Type::I64)),
                    Type::I64,
                ))
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
            Expression::Index(base, index) => {
                let (address, element_type) =
                    self.element_address(base, index)?;
                self.load_from(address, element_type)
            }
            Expression::Switch(scrutinee, cases) => {
                self.lower_match(scrutinee, cases, expected)
            }
            Expression::StructInit(struct_name, _) => {
                let ty = match expected {
                    Some(Type::Struct(instance))
                        if is_generic_instance(instance)
                            && instance
                                .starts_with(&format!("{struct_name}<")) =>
                    {
                        Type::Struct(instance.clone())
                    }
                    _ => Type::Struct(struct_name.clone()),
                };
                let temp = self.fresh_local(ty.clone(), None);
                self.materialize_aggregate(temp, expression)?;
                Ok((IrOperand::Local(temp), ty))
            }
            Expression::EnumVariantInit(struct_name, _, _) => {
                let ty = Type::Enum(struct_name.clone());
                let temp = self.fresh_local(ty.clone(), None);
                self.materialize_aggregate(temp, expression)?;
                Ok((IrOperand::Local(temp), ty))
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
            Literal::String(value) => {
                if matches!(expected, Some(Type::Ptr(_))) {
                    return Ok((
                        IrOperand::Constant(IrConstant::CString(value.clone())),
                        Type::Ptr(Box::new(Type::I8)),
                    ));
                }
                let local = self.fresh_local(Type::Str, None);
                self.build_str_value(local, value);
                Ok((IrOperand::Local(local), Type::Str))
            }
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
        if let Expression::Identifier(name) = callee
            && name == "assert"
            && self.resolve_variable(name).is_none()
            && self.builder.signature("frost_assert").is_some()
        {
            return self.lower_direct_call("frost_assert", arguments);
        }
        if let Expression::Identifier(name) = callee
            && name == "str_len"
            && self.resolve_variable(name).is_none()
            && self.builder.signature(name).is_none()
            && !self.builder.generic_functions.contains_key(name)
        {
            return self.lower_str_len(arguments);
        }
        if let Expression::Identifier(name) = callee
            && self.resolve_variable(name).is_none()
            && self.builder.signature(name).is_none()
            && !self.builder.generic_functions.contains_key(name)
        {
            match name.as_str() {
                "pool_new" => return self.lower_pool_new(arguments),
                "pool_alloc" => return self.lower_pool_alloc(arguments),
                "pool_contains" => {
                    return self.lower_pool_contains(arguments);
                }
                "pool_free" => return self.lower_pool_free(arguments),
                "pool_destroy" => {
                    return self.lower_pool_destroy(arguments);
                }
                _ => {}
            }
        }
        if let Expression::Identifier(name) = callee
            && self.resolve_variable(name).is_none()
        {
            if self.builder.generic_functions.contains_key(name) {
                return self.lower_generic_call(name, arguments);
            }
            if self.builder.signature(name).is_some() {
                return self.lower_direct_call(name, arguments);
            }
        }
        self.lower_indirect_call(callee, arguments)
    }

    fn lower_generic_call(
        &mut self,
        name: &str,
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        let generic = self
            .builder
            .generic_functions
            .get(name)
            .expect("generic function exists")
            .clone();

        if arguments.len() != generic.parameters.len() {
            bail!(
                "native backend: generic function '{name}' expects {} argument(s) but {} were given",
                generic.parameters.len(),
                arguments.len()
            );
        }

        let mut subst: HashMap<String, Type> = HashMap::new();
        let mut value_operands: Vec<(IrOperand, Type)> = Vec::new();
        for (parameter, argument) in generic.parameters.iter().zip(arguments) {
            if is_type_parameter(parameter) {
                let Expression::TypeValue(ty) = argument else {
                    bail!(
                        "native backend: type parameter '{}' of '{name}' requires a type argument like '${}'",
                        parameter.name,
                        parameter.name
                    );
                };
                subst.insert(parameter.name.clone(), ty.clone());
            } else {
                let (operand, value_type) =
                    self.lower_expression(argument, None)?;
                infer_subst_into(
                    &parameter_type(parameter),
                    &value_type,
                    &generic.type_params,
                    &mut subst,
                );
                value_operands.push((operand, value_type));
            }
        }

        let value_parameter_types: Vec<Type> = generic
            .parameters
            .iter()
            .filter(|parameter| !is_type_parameter(parameter))
            .map(|parameter| {
                substitute_type(&parameter_type(parameter), &subst)
            })
            .collect();
        let return_type = generic
            .return_sig
            .to_type()
            .map(|ty| substitute_type(&ty, &subst))
            .unwrap_or(Type::Void);
        let mangled_name =
            mangle_specialization(name, &generic.type_params, &subst);

        self.specializations.push(Specialization {
            generic_name: name.to_string(),
            mangled_name: mangled_name.clone(),
            subst,
        });

        let mut lowered = Vec::with_capacity(value_operands.len());
        for ((operand, value_type), target) in
            value_operands.into_iter().zip(&value_parameter_types)
        {
            if needs_memory(target) {
                let IrOperand::Local(local) = operand else {
                    bail!(
                        "native backend: aggregate argument to generic call is not a place"
                    );
                };
                lowered.push(self.address_of_local(local, target));
            } else {
                lowered.push(self.coerce(operand, &value_type, target));
            }
        }

        let result = self.fresh_local(return_type.clone(), None);
        if needs_memory(&return_type) {
            self.mark_in_memory(result);
        }
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Call {
                function: mangled_name,
                arguments: lowered,
            },
        ));
        Ok((IrOperand::Local(result), return_type))
    }

    fn lower_direct_call(
        &mut self,
        name: &str,
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        let signature = self.builder.signature(name).unwrap();
        let parameter_types = signature.parameters.clone();
        let return_type = signature.return_type.clone();

        if arguments.len() != parameter_types.len() {
            bail!(
                "native backend: function '{name}' expects {} argument(s) but {} were given",
                parameter_types.len(),
                arguments.len()
            );
        }

        let mut lowered = Vec::with_capacity(arguments.len());
        for (index, argument) in arguments.iter().enumerate() {
            let expected = parameter_types.get(index);
            if let Some(target) = expected
                && needs_memory(target)
            {
                let address =
                    self.aggregate_argument_address(argument, target)?;
                lowered.push(address);
                continue;
            }
            let (operand, value_type) =
                self.lower_expression(argument, expected)?;
            if let Some(Type::Ref(inner) | Type::RefMut(inner)) = expected
                && needs_memory(&value_type)
                && value_type == **inner
            {
                bail!(
                    "native backend: cannot pass a '{value_type}' by value to a reference parameter '&{value_type}'; take a reference with '&' or '&mut'"
                );
            }
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
                function: name.to_string(),
                arguments: lowered,
            },
        ));
        Ok((IrOperand::Local(result), return_type))
    }

    fn lower_indirect_call(
        &mut self,
        callee: &Expression,
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        let (callee_operand, callee_type) =
            self.lower_expression(callee, None)?;
        let Type::Proc(parameter_types, return_type) = callee_type else {
            bail!(
                "native backend: cannot call a value that is not a function pointer"
            );
        };
        let return_type = *return_type;
        if needs_memory(&return_type) {
            bail!(
                "native backend: indirect call returning an aggregate is not supported yet"
            );
        }
        if arguments.len() != parameter_types.len() {
            bail!(
                "native backend: function pointer expects {} argument(s) but {} were given",
                parameter_types.len(),
                arguments.len()
            );
        }

        let mut lowered = Vec::with_capacity(arguments.len());
        for (index, argument) in arguments.iter().enumerate() {
            let expected = parameter_types.get(index);
            if let Some(target) = expected
                && needs_memory(target)
            {
                let address =
                    self.aggregate_argument_address(argument, target)?;
                lowered.push(address);
                continue;
            }
            let (operand, value_type) =
                self.lower_expression(argument, expected)?;
            if let Some(Type::Ref(inner) | Type::RefMut(inner)) = expected
                && needs_memory(&value_type)
                && value_type == **inner
            {
                bail!(
                    "native backend: cannot pass a '{value_type}' by value to a reference parameter '&{value_type}'; take a reference with '&' or '&mut'"
                );
            }
            let coerced = match expected {
                Some(target) => self.coerce(operand, &value_type, target),
                None => operand,
            };
            lowered.push(coerced);
        }

        let result = self.fresh_local(return_type.clone(), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::CallIndirect {
                callee: callee_operand,
                arguments: lowered,
                parameter_types,
                return_type: return_type.clone(),
            },
        ));
        Ok((IrOperand::Local(result), return_type))
    }

    fn aggregate_argument_address(
        &mut self,
        argument: &Expression,
        target: &Type,
    ) -> Result<IrOperand> {
        match argument {
            Expression::Identifier(_)
            | Expression::FieldAccess(..)
            | Expression::Index(..)
            | Expression::Dereference(_) => {
                let (address, _) = self.place_address(argument)?;
                Ok(address)
            }
            Expression::StructInit(..)
            | Expression::EnumVariantInit(..)
            | Expression::Literal(Literal::Array(_)) => {
                let temp = self.fresh_local(target.clone(), None);
                self.materialize_aggregate(temp, argument)?;
                Ok(self.address_of_local(temp, target))
            }
            _ => {
                let (operand, _) =
                    self.lower_expression(argument, Some(target))?;
                let IrOperand::Local(local) = operand else {
                    bail!(
                        "native backend: cannot pass this value as an aggregate argument"
                    );
                };
                Ok(self.address_of_local(local, target))
            }
        }
    }

    fn address_of_local(&mut self, local: LocalId, ty: &Type) -> IrOperand {
        let result = self.fresh_local(Type::Ptr(Box::new(ty.clone())), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::AddressOf { local, offset: 0 },
        ));
        IrOperand::Local(result)
    }

    fn materialize_aggregate(
        &mut self,
        local: LocalId,
        expression: &Expression,
    ) -> Result<()> {
        match expression {
            Expression::StructInit(name, fields) => {
                let layout_name = match self.type_of_local(local) {
                    Type::Struct(instance)
                        if is_generic_instance(&instance) =>
                    {
                        instance
                    }
                    _ => name.clone(),
                };
                self.init_struct(local, &layout_name, fields)
            }
            Expression::EnumVariantInit(name, variant, fields) => {
                self.init_enum(local, name, variant, fields)
            }
            Expression::Literal(Literal::Array(elements)) => {
                let Type::Array(element, _) = self.type_of_local(local) else {
                    bail!("native backend: array literal has non-array type");
                };
                self.init_array(local, &element, elements)
            }
            _ => {
                bail!("native backend: cannot materialize this aggregate")
            }
        }
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
        if needs_memory(&pointee) {
            let IrOperand::Local(source_local) = operand else {
                bail!(
                    "native backend: aggregate assignment from a non-place value"
                );
            };
            let source = self.address_of_local(source_local, &pointee);
            let size = self.builder.byte_size(&pointee);
            self.emit(IrStatement::Copy {
                destination: address,
                source,
                size,
            });
            return Ok(());
        }
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
            let temp = self.fresh_local(ty.clone(), None);
            self.mark_in_memory(temp);
            let destination = self.address_of_local(temp, &ty);
            let size = self.builder.byte_size(&ty);
            self.emit(IrStatement::Copy {
                destination,
                source: address,
                size,
            });
            return Ok((IrOperand::Local(temp), ty));
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

    fn build_str_value(&mut self, local: LocalId, text: &str) {
        self.mark_in_memory(local);
        let ptr_slot =
            self.fresh_local(Type::Ptr(Box::new(str_byte_ptr_type())), None);
        self.emit(IrStatement::Assign(
            ptr_slot,
            IrRvalue::AddressOf {
                local,
                offset: STR_PTR_OFFSET,
            },
        ));
        self.emit(IrStatement::Store {
            address: IrOperand::Local(ptr_slot),
            value: IrOperand::Constant(IrConstant::CString(text.to_string())),
        });
        let len_slot = self.fresh_local(Type::Ptr(Box::new(Type::Usize)), None);
        self.emit(IrStatement::Assign(
            len_slot,
            IrRvalue::AddressOf {
                local,
                offset: STR_LEN_OFFSET,
            },
        ));
        self.emit(IrStatement::Store {
            address: IrOperand::Local(len_slot),
            value: IrOperand::Constant(IrConstant::Integer(
                text.len() as i64,
                Type::Usize,
            )),
        });
    }

    fn place_type(&self, place: &Expression) -> Option<Type> {
        match place {
            Expression::Identifier(name) => self
                .resolve_variable(name)
                .map(|local| self.type_of_local(local)),
            _ => None,
        }
    }

    fn str_value_address(
        &mut self,
        expression: &Expression,
    ) -> Result<IrOperand> {
        if matches!(self.place_type(expression), Some(Type::Str)) {
            let (address, _) = self.place_address(expression)?;
            return Ok(address);
        }
        let (operand, value_type) =
            self.lower_expression(expression, Some(&Type::Str))?;
        if value_type != Type::Str {
            bail!("native backend: expected a str value, found {value_type}");
        }
        let IrOperand::Local(local) = operand else {
            bail!("native backend: str value is not addressable");
        };
        self.mark_in_memory(local);
        Ok(self.address_of_local(local, &Type::Str))
    }

    fn str_field(
        &mut self,
        base: IrOperand,
        offset: usize,
        field_type: Type,
    ) -> IrOperand {
        let slot =
            self.fresh_local(Type::Ptr(Box::new(field_type.clone())), None);
        self.emit(IrStatement::Assign(
            slot,
            IrRvalue::FieldAddress { base, offset },
        ));
        let result = self.fresh_local(field_type.clone(), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Load {
                address: IrOperand::Local(slot),
                ty: field_type,
            },
        ));
        IrOperand::Local(result)
    }

    fn lower_pool_new(
        &mut self,
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 2 {
            bail!(
                "native backend: pool_new expects a type and a capacity, as in pool_new($Entity, 8)"
            );
        }
        let Expression::TypeValue(element) = &arguments[0] else {
            bail!(
                "native backend: pool_new's first argument must be a type, as in $Entity"
            );
        };
        let element = element.clone();
        let (capacity, capacity_type) =
            self.lower_expression(&arguments[1], Some(&Type::I64))?;
        let capacity = self.coerce(capacity, &capacity_type, &Type::I64);
        let element_size = self.builder.byte_size(&element) as i64;
        let pool_type = Type::Pool(Box::new(element));
        let result = self.fresh_local(pool_type.clone(), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Call {
                function: "pool_new".to_string(),
                arguments: vec![
                    capacity,
                    IrOperand::Constant(IrConstant::Integer(
                        element_size,
                        Type::I64,
                    )),
                ],
            },
        ));
        Ok((IrOperand::Local(result), pool_type))
    }

    fn pool_element_of(
        &mut self,
        pool: &Expression,
        operation: &str,
    ) -> Result<(IrOperand, Type)> {
        let (pool_operand, pool_type) = self.lower_expression(pool, None)?;
        let Type::Pool(element) = pool_type else {
            bail!(
                "native backend: {operation} expects a Pool<T>, but its pool argument has type '{pool_type}'. \
                 The typed pool surface (pool_new($T, capacity), pool_alloc, pool_contains, pool_free, pool_destroy) \
                 and the raw '^u8' extern surface cannot be mixed: either create the pool with pool_new so it is a Pool<T>, \
                 or declare every pool function as 'extern fn' and use the raw surface throughout"
            );
        };
        Ok((pool_operand, *element))
    }

    fn address_of_value(
        &mut self,
        expression: &Expression,
        ty: &Type,
    ) -> Result<IrOperand> {
        if needs_memory(ty) {
            return self.aggregate_argument_address(expression, ty);
        }
        let (value, value_type) =
            self.lower_expression(expression, Some(ty))?;
        let value = self.coerce(value, &value_type, ty);
        let temp = self.fresh_local(ty.clone(), None);
        self.mark_in_memory(temp);
        let address = self.address_of_local(temp, ty);
        self.emit(IrStatement::Store {
            address: address.clone(),
            value,
        });
        Ok(address)
    }

    fn lower_pool_alloc(
        &mut self,
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 2 {
            bail!(
                "native backend: pool_alloc expects a pool and a value, as in pool_alloc(world, entity)"
            );
        }
        let (pool_operand, element) =
            self.pool_element_of(&arguments[0], "pool_alloc")?;
        let value_address = self.address_of_value(&arguments[1], &element)?;
        let handle_type = Type::Handle(Box::new(element));
        let result = self.fresh_local(handle_type.clone(), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Call {
                function: "pool_alloc".to_string(),
                arguments: vec![pool_operand, value_address],
            },
        ));
        Ok((IrOperand::Local(result), handle_type))
    }

    fn lower_pool_contains(
        &mut self,
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 2 {
            bail!("native backend: pool_contains expects a pool and a handle");
        }
        let (pool_operand, _) =
            self.pool_element_of(&arguments[0], "pool_contains")?;
        let (handle, _) = self.lower_expression(&arguments[1], None)?;
        let raw = self.fresh_local(Type::I64, None);
        self.emit(IrStatement::Assign(
            raw,
            IrRvalue::Call {
                function: "pool_contains".to_string(),
                arguments: vec![pool_operand, handle],
            },
        ));
        let result = self.fresh_local(Type::Bool, None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Binary(
                IrBinOp::NotEqual,
                IrOperand::Local(raw),
                IrOperand::Constant(IrConstant::Integer(0, Type::I64)),
            ),
        ));
        Ok((IrOperand::Local(result), Type::Bool))
    }

    fn lower_pool_free(
        &mut self,
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 2 {
            bail!("native backend: pool_free expects a pool and a handle");
        }
        let (pool_operand, _) =
            self.pool_element_of(&arguments[0], "pool_free")?;
        let (handle, _) = self.lower_expression(&arguments[1], None)?;
        let result = self.fresh_local(Type::Void, None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Call {
                function: "pool_free".to_string(),
                arguments: vec![pool_operand, handle],
            },
        ));
        Ok((IrOperand::Local(result), Type::Void))
    }

    fn lower_pool_destroy(
        &mut self,
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 1 {
            bail!("native backend: pool_destroy expects a pool");
        }
        let (pool_operand, _) =
            self.pool_element_of(&arguments[0], "pool_destroy")?;
        let result = self.fresh_local(Type::Void, None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Call {
                function: "pool_destroy".to_string(),
                arguments: vec![pool_operand],
            },
        ));
        Ok((IrOperand::Local(result), Type::Void))
    }

    fn lower_str_len(
        &mut self,
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 1 {
            bail!("native backend: str_len expects one argument");
        }
        let base = self.str_value_address(&arguments[0])?;
        let length = self.str_field(base, STR_LEN_OFFSET, Type::Usize);
        Ok((length, Type::Usize))
    }

    fn str_byte_address(
        &mut self,
        base: &Expression,
        index_operand: IrOperand,
        index_type: Type,
    ) -> Result<(IrOperand, Type)> {
        let str_address = self.str_value_address(base)?;
        let data = self.str_field(
            str_address.clone(),
            STR_PTR_OFFSET,
            str_byte_ptr_type(),
        );
        let length = self.str_field(str_address, STR_LEN_OFFSET, Type::Usize);
        let index = self.coerce(index_operand, &index_type, &Type::I64);
        let length = self.coerce(length, &Type::Usize, &Type::I64);
        let check = self.fresh_local(Type::Void, None);
        self.emit(IrStatement::Assign(
            check,
            IrRvalue::Call {
                function: "frost_bounds_check".to_string(),
                arguments: vec![index.clone(), length],
            },
        ));
        let element = self.fresh_local(str_byte_ptr_type(), None);
        self.emit(IrStatement::Assign(
            element,
            IrRvalue::ElementAddress {
                base: data,
                index,
                element_size: 1,
            },
        ));
        Ok((IrOperand::Local(element), Type::U8))
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
            Expression::Index(base, index) => self.element_address(base, index),
            Expression::Dereference(pointer) => {
                self.place_address_of_deref(pointer)
            }
            Expression::StructInit(..)
            | Expression::EnumVariantInit(..)
            | Expression::Literal(Literal::Array(_)) => {
                let (operand, ty) = self.lower_expression(place, None)?;
                let IrOperand::Local(local) = operand else {
                    bail!(
                        "native backend: cannot take the address of this value"
                    );
                };
                Ok((self.address_of_local(local, &ty), ty))
            }
            other => {
                bail!(
                    "native backend: expression is not an assignable place: {other}"
                )
            }
        }
    }

    fn element_address(
        &mut self,
        base: &Expression,
        index: &Expression,
    ) -> Result<(IrOperand, Type)> {
        let (index_operand, index_type) = self.lower_expression(index, None)?;
        if let Type::Handle(element) = index_type {
            return self.pool_element_address(base, index_operand, *element);
        }
        if matches!(self.place_type(base), Some(Type::Str)) {
            return self.str_byte_address(base, index_operand, index_type);
        }
        let (base_pointer, element_type, length) =
            self.array_base_pointer(base)?;
        let element_size = self.builder.byte_size(&element_type);
        let index_operand = self.coerce(index_operand, &index_type, &Type::I64);
        if let Some(length) = length {
            let check_result = self.fresh_local(Type::Void, None);
            self.emit(IrStatement::Assign(
                check_result,
                IrRvalue::Call {
                    function: "frost_bounds_check".to_string(),
                    arguments: vec![
                        index_operand.clone(),
                        IrOperand::Constant(IrConstant::Integer(
                            length as i64,
                            Type::I64,
                        )),
                    ],
                },
            ));
        }
        let result =
            self.fresh_local(Type::Ptr(Box::new(element_type.clone())), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::ElementAddress {
                base: base_pointer,
                index: index_operand,
                element_size,
            },
        ));
        Ok((IrOperand::Local(result), element_type))
    }

    fn pool_element_address(
        &mut self,
        base: &Expression,
        handle_operand: IrOperand,
        element_type: Type,
    ) -> Result<(IrOperand, Type)> {
        let (pool_operand, _) = self.lower_expression(base, None)?;
        let result =
            self.fresh_local(Type::Ptr(Box::new(element_type.clone())), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Call {
                function: "pool_get".to_string(),
                arguments: vec![pool_operand, handle_operand],
            },
        ));
        Ok((IrOperand::Local(result), element_type))
    }

    fn array_base_pointer(
        &mut self,
        base: &Expression,
    ) -> Result<(IrOperand, Type, Option<usize>)> {
        match base {
            Expression::Identifier(name) => {
                let Some(local) = self.resolve_variable(name) else {
                    bail!("native backend: unknown variable '{name}'");
                };
                match self.type_of_local(local) {
                    Type::Array(element, count) => {
                        self.mark_in_memory(local);
                        let result = self.fresh_local(
                            Type::Ptr(Box::new((*element).clone())),
                            None,
                        );
                        self.emit(IrStatement::Assign(
                            result,
                            IrRvalue::AddressOf { local, offset: 0 },
                        ));
                        Ok((IrOperand::Local(result), *element, Some(count)))
                    }
                    Type::Ref(inner)
                    | Type::RefMut(inner)
                    | Type::Ptr(inner)
                        if matches!(*inner, Type::Array(_, _)) =>
                    {
                        let Type::Array(element, count) = *inner else {
                            unreachable!()
                        };
                        Ok((IrOperand::Local(local), *element, Some(count)))
                    }
                    other => bail!(
                        "native backend: '{name}' is not an array (found {other})"
                    ),
                }
            }
            Expression::FieldAccess(inner, field) => {
                let (address, field_type) = self.field_address(inner, field)?;
                let Type::Array(element, count) = field_type else {
                    bail!("native backend: field '{field}' is not an array");
                };
                Ok((address, *element, Some(count)))
            }
            Expression::Index(inner, index) => {
                let (address, element_type) =
                    self.element_address(inner, index)?;
                let Type::Array(element, count) = element_type else {
                    bail!("native backend: indexed value is not an array");
                };
                Ok((address, *element, Some(count)))
            }
            other => {
                bail!("native backend: cannot index into: {other}")
            }
        }
    }

    fn init_array(
        &mut self,
        local: LocalId,
        element_type: &Type,
        elements: &[Expression],
    ) -> Result<()> {
        let element_size = self.builder.byte_size(element_type);
        for (index, element) in elements.iter().enumerate() {
            let address = self
                .fresh_local(Type::Ptr(Box::new(element_type.clone())), None);
            self.emit(IrStatement::Assign(
                address,
                IrRvalue::AddressOf {
                    local,
                    offset: index * element_size,
                },
            ));
            if needs_memory(element_type) {
                let source_local =
                    self.materialize_field_value(element, element_type)?;
                let source = self.address_of_local(source_local, element_type);
                self.emit(IrStatement::Copy {
                    destination: IrOperand::Local(address),
                    source,
                    size: element_size,
                });
            } else {
                let (operand, value_type) =
                    self.lower_expression(element, Some(element_type))?;
                let coerced = self.coerce(operand, &value_type, element_type);
                self.emit(IrStatement::Store {
                    address: IrOperand::Local(address),
                    value: coerced,
                });
            }
        }
        Ok(())
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
            Expression::Index(inner, index) => {
                let (address, element_type) =
                    self.element_address(inner, index)?;
                let Type::Struct(struct_name) = element_type else {
                    bail!("native backend: indexed element is not a struct");
                };
                Ok((address, struct_name))
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
            let address =
                self.fresh_local(Type::Ptr(Box::new(field_type.clone())), None);
            self.emit(IrStatement::Assign(
                address,
                IrRvalue::AddressOf {
                    local,
                    offset: *offset,
                },
            ));
            if needs_memory(field_type) {
                let source_local =
                    self.materialize_field_value(field_value, field_type)?;
                let source = self.address_of_local(source_local, field_type);
                self.emit(IrStatement::Copy {
                    destination: IrOperand::Local(address),
                    source,
                    size: self.builder.byte_size(field_type),
                });
            } else {
                let (operand, value_type) =
                    self.lower_expression(field_value, Some(field_type))?;
                let coerced = self.coerce(operand, &value_type, field_type);
                self.emit(IrStatement::Store {
                    address: IrOperand::Local(address),
                    value: coerced,
                });
            }
        }
        Ok(())
    }

    fn materialize_field_value(
        &mut self,
        expression: &Expression,
        field_type: &Type,
    ) -> Result<LocalId> {
        match expression {
            Expression::StructInit(..)
            | Expression::EnumVariantInit(..)
            | Expression::Literal(Literal::Array(_)) => {
                let temp = self.fresh_local(field_type.clone(), None);
                self.materialize_aggregate(temp, expression)?;
                Ok(temp)
            }
            _ => {
                let (operand, _) =
                    self.lower_expression(expression, Some(field_type))?;
                let IrOperand::Local(local) = operand else {
                    bail!(
                        "native backend: cannot initialize an aggregate field from this value"
                    );
                };
                Ok(local)
            }
        }
    }

    fn init_enum(
        &mut self,
        local: LocalId,
        enum_name: &str,
        variant_name: &str,
        field_inits: &[(String, Expression)],
    ) -> Result<()> {
        let (tag, fields): (u32, Vec<(String, usize, Type)>) = {
            let layout =
                self.builder.enum_layout(enum_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "native backend: unknown enum '{enum_name}'"
                    )
                })?;
            let variant = layout.variant(variant_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "native backend: enum '{enum_name}' has no variant '{variant_name}'"
                )
            })?;
            (
                variant.tag,
                variant
                    .fields
                    .iter()
                    .map(|field| {
                        (field.name.clone(), field.offset, field.ty.clone())
                    })
                    .collect(),
            )
        };

        let tag_address =
            self.fresh_local(Type::Ptr(Box::new(Type::I32)), None);
        self.emit(IrStatement::Assign(
            tag_address,
            IrRvalue::AddressOf { local, offset: 0 },
        ));
        self.emit(IrStatement::Store {
            address: IrOperand::Local(tag_address),
            value: IrOperand::Constant(IrConstant::Integer(
                tag as i64,
                Type::I32,
            )),
        });

        for (field_name, field_value) in field_inits {
            let Some((_, offset, field_type)) =
                fields.iter().find(|(name, _, _)| name == field_name)
            else {
                bail!(
                    "native backend: enum variant '{variant_name}' has no field '{field_name}'"
                );
            };
            let address =
                self.fresh_local(Type::Ptr(Box::new(field_type.clone())), None);
            self.emit(IrStatement::Assign(
                address,
                IrRvalue::AddressOf {
                    local,
                    offset: *offset,
                },
            ));
            if needs_memory(field_type) {
                let source_local =
                    self.materialize_field_value(field_value, field_type)?;
                let source = self.address_of_local(source_local, field_type);
                self.emit(IrStatement::Copy {
                    destination: IrOperand::Local(address),
                    source,
                    size: self.builder.byte_size(field_type),
                });
            } else {
                let (operand, value_type) =
                    self.lower_expression(field_value, Some(field_type))?;
                let coerced = self.coerce(operand, &value_type, field_type);
                self.emit(IrStatement::Store {
                    address: IrOperand::Local(address),
                    value: coerced,
                });
            }
        }
        Ok(())
    }

    fn lower_match(
        &mut self,
        scrutinee: &Expression,
        cases: &[SwitchCase],
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        if cases.is_empty() {
            bail!("native backend: match with no cases");
        }

        if let Expression::Tuple(elements) = scrutinee {
            return self.lower_tuple_match(elements, cases, expected);
        }

        let enum_info = self.enum_scrutinee_address(scrutinee)?;

        let (enum_address, enum_name, tag_operand, scalar) =
            if let Some((name, address)) = enum_info {
                let tag = self.fresh_local(Type::I32, None);
                self.emit(IrStatement::Assign(
                    tag,
                    IrRvalue::Load {
                        address: address.clone(),
                        ty: Type::I32,
                    },
                ));
                (Some(address), Some(name), Some(IrOperand::Local(tag)), None)
            } else {
                let (value, value_type) =
                    self.lower_expression(scrutinee, None)?;
                (None, None, None, Some((value, value_type)))
            };

        let merge = self.new_block();
        let mut result_local: Option<LocalId> = None;
        let mut result_type = Type::Void;

        for case in cases {
            let case_block = self.new_block();
            let next_block = self.new_block();

            match &case.pattern {
                Pattern::Wildcard | Pattern::Identifier(_) => {
                    self.set_terminator(IrTerminator::Jump(case_block));
                }
                Pattern::Literal(literal) => {
                    let Some((value, value_type)) = &scalar else {
                        bail!(
                            "native backend: literal pattern requires a scalar match value"
                        );
                    };
                    let (literal_operand, _) =
                        self.lower_literal(literal, Some(value_type))?;
                    let condition = self.fresh_local(Type::Bool, None);
                    self.emit(IrStatement::Assign(
                        condition,
                        IrRvalue::Binary(
                            IrBinOp::Equal,
                            value.clone(),
                            literal_operand,
                        ),
                    ));
                    self.set_terminator(IrTerminator::Branch {
                        condition: IrOperand::Local(condition),
                        then_block: case_block,
                        else_block: next_block,
                    });
                }
                Pattern::EnumVariant { variant_name, .. } => {
                    let Some(tag) = &tag_operand else {
                        bail!(
                            "native backend: enum variant pattern requires an enum match value"
                        );
                    };
                    let enum_name = enum_name.as_ref().unwrap();
                    let variant_tag = self
                        .builder
                        .enum_layout(enum_name)
                        .and_then(|layout| layout.variant(variant_name))
                        .map(|variant| variant.tag)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "native backend: enum '{enum_name}' has no variant '{variant_name}'"
                            )
                        })?;
                    let condition = self.fresh_local(Type::Bool, None);
                    self.emit(IrStatement::Assign(
                        condition,
                        IrRvalue::Binary(
                            IrBinOp::Equal,
                            tag.clone(),
                            IrOperand::Constant(IrConstant::Integer(
                                variant_tag as i64,
                                Type::I32,
                            )),
                        ),
                    ));
                    self.set_terminator(IrTerminator::Branch {
                        condition: IrOperand::Local(condition),
                        then_block: case_block,
                        else_block: next_block,
                    });
                }
                Pattern::Tuple(_) => {
                    bail!(
                        "native backend: tuple patterns are not supported yet"
                    );
                }
            }

            self.switch_to(case_block);
            self.push_scope();
            self.bind_pattern(
                &case.pattern,
                enum_address.as_ref(),
                enum_name.as_deref(),
                scalar.as_ref(),
            )?;
            let (value, value_type) = self.lower_block(&case.body, expected)?;
            if result_local.is_none() {
                result_type = match expected {
                    Some(ty) if !matches!(ty, Type::Void) => ty.clone(),
                    _ => value_type.clone(),
                };
                result_local =
                    Some(self.fresh_local(result_type.clone(), None));
            }
            let target = result_local.unwrap();
            let coerced = self.coerce(value, &value_type, &result_type);
            self.emit(IrStatement::Assign(target, IrRvalue::Use(coerced)));
            self.pop_scope();
            self.set_terminator(IrTerminator::Jump(merge));

            self.switch_to(next_block);
        }

        let target = result_local
            .expect("match has at least one case, so a result exists");
        if !needs_memory(&result_type) {
            let zero = zero_operand(&result_type);
            self.emit(IrStatement::Assign(target, IrRvalue::Use(zero)));
        }
        self.set_terminator(IrTerminator::Jump(merge));

        self.switch_to(merge);
        Ok((IrOperand::Local(target), result_type))
    }

    fn enum_name_of(&self, ty: &Type) -> Option<String> {
        match ty {
            Type::Enum(name) => Some(name.clone()),
            Type::Struct(name) if self.builder.enum_layout(name).is_some() => {
                Some(name.clone())
            }
            _ => None,
        }
    }

    fn enum_scrutinee_address(
        &mut self,
        scrutinee: &Expression,
    ) -> Result<Option<(String, IrOperand)>> {
        if matches!(
            scrutinee,
            Expression::FieldAccess(..)
                | Expression::Index(..)
                | Expression::Dereference(..)
        ) {
            let (address, ty) = self.place_address(scrutinee)?;
            if let Some(enum_name) = self.enum_name_of(&ty) {
                return Ok(Some((enum_name, address)));
            }
            return Ok(None);
        }
        let Expression::Identifier(name) = scrutinee else {
            return Ok(None);
        };
        let Some(local) = self.resolve_variable(name) else {
            return Ok(None);
        };
        let ty = self.type_of_local(local);

        if let Some(enum_name) = self.enum_name_of(&ty) {
            self.mark_in_memory(local);
            let address = self.fresh_local(
                Type::Ptr(Box::new(Type::Enum(enum_name.clone()))),
                None,
            );
            self.emit(IrStatement::Assign(
                address,
                IrRvalue::AddressOf { local, offset: 0 },
            ));
            return Ok(Some((enum_name, IrOperand::Local(address))));
        }

        if let Type::Ref(inner) | Type::RefMut(inner) | Type::Ptr(inner) = &ty
            && let Some(enum_name) = self.enum_name_of(inner)
        {
            return Ok(Some((enum_name, IrOperand::Local(local))));
        }

        Ok(None)
    }

    fn lower_tuple_match(
        &mut self,
        elements: &[Expression],
        cases: &[SwitchCase],
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        let mut values = Vec::with_capacity(elements.len());
        for element in elements {
            values.push(self.lower_expression(element, None)?);
        }

        let merge = self.new_block();
        let mut result_local: Option<LocalId> = None;
        let mut result_type = Type::Void;

        for case in cases {
            let case_block = self.new_block();
            let next_block = self.new_block();

            let patterns: Vec<&Pattern> = match &case.pattern {
                Pattern::Tuple(patterns) => patterns.iter().collect(),
                Pattern::Wildcard | Pattern::Identifier(_) => Vec::new(),
                other => bail!(
                    "native backend: unsupported tuple match pattern: {other:?}"
                ),
            };

            let mut condition: Option<LocalId> = None;
            for (pattern, (value, value_type)) in
                patterns.iter().zip(values.iter())
            {
                if let Pattern::Literal(literal) = pattern {
                    let (literal_operand, _) =
                        self.lower_literal(literal, Some(value_type))?;
                    let test = self.fresh_local(Type::Bool, None);
                    self.emit(IrStatement::Assign(
                        test,
                        IrRvalue::Binary(
                            IrBinOp::Equal,
                            value.clone(),
                            literal_operand,
                        ),
                    ));
                    condition = Some(match condition {
                        None => test,
                        Some(previous) => {
                            let combined = self.fresh_local(Type::Bool, None);
                            self.emit(IrStatement::Assign(
                                combined,
                                IrRvalue::Binary(
                                    IrBinOp::BitwiseAnd,
                                    IrOperand::Local(previous),
                                    IrOperand::Local(test),
                                ),
                            ));
                            combined
                        }
                    });
                }
            }

            match condition {
                Some(local) => self.set_terminator(IrTerminator::Branch {
                    condition: IrOperand::Local(local),
                    then_block: case_block,
                    else_block: next_block,
                }),
                None => self.set_terminator(IrTerminator::Jump(case_block)),
            }

            self.switch_to(case_block);
            self.push_scope();
            for (pattern, (value, value_type)) in
                patterns.iter().zip(values.iter())
            {
                if let Pattern::Identifier(name) = pattern {
                    let bound = self
                        .fresh_local(value_type.clone(), Some(name.clone()));
                    self.emit(IrStatement::Assign(
                        bound,
                        IrRvalue::Use(value.clone()),
                    ));
                    self.define_variable(name, bound);
                }
            }
            let (value, value_type) = self.lower_block(&case.body, expected)?;
            if result_local.is_none() {
                result_type = match expected {
                    Some(ty) if !matches!(ty, Type::Void) => ty.clone(),
                    _ => value_type.clone(),
                };
                result_local =
                    Some(self.fresh_local(result_type.clone(), None));
            }
            let target = result_local.unwrap();
            let coerced = self.coerce(value, &value_type, &result_type);
            self.emit(IrStatement::Assign(target, IrRvalue::Use(coerced)));
            self.pop_scope();
            self.set_terminator(IrTerminator::Jump(merge));

            self.switch_to(next_block);
        }

        let target = result_local
            .expect("match has at least one case, so a result exists");
        if !needs_memory(&result_type) {
            let zero = zero_operand(&result_type);
            self.emit(IrStatement::Assign(target, IrRvalue::Use(zero)));
        }
        self.set_terminator(IrTerminator::Jump(merge));

        self.switch_to(merge);
        Ok((IrOperand::Local(target), result_type))
    }

    fn bind_pattern(
        &mut self,
        pattern: &Pattern,
        enum_address: Option<&IrOperand>,
        enum_name: Option<&str>,
        scalar: Option<&(IrOperand, Type)>,
    ) -> Result<()> {
        match pattern {
            Pattern::EnumVariant {
                variant_name,
                bindings,
                ..
            } => {
                let (Some(address), Some(enum_name)) =
                    (enum_address, enum_name)
                else {
                    bail!(
                        "native backend: enum pattern on a non-enum match value"
                    );
                };
                let fields: Vec<(String, usize, Type)> = self
                    .builder
                    .enum_layout(enum_name)
                    .and_then(|layout| layout.variant(variant_name))
                    .map(|variant| {
                        variant
                            .fields
                            .iter()
                            .map(|field| {
                                (
                                    field.name.clone(),
                                    field.offset,
                                    field.ty.clone(),
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                for (field_name, bound_name) in bindings {
                    let Some((_, offset, field_type)) =
                        fields.iter().find(|(name, _, _)| name == field_name)
                    else {
                        bail!(
                            "native backend: variant '{variant_name}' has no field '{field_name}'"
                        );
                    };
                    let field_address = self.fresh_local(
                        Type::Ptr(Box::new(field_type.clone())),
                        None,
                    );
                    self.emit(IrStatement::Assign(
                        field_address,
                        IrRvalue::FieldAddress {
                            base: address.clone(),
                            offset: *offset,
                        },
                    ));
                    let bound = self.fresh_local(
                        field_type.clone(),
                        Some(bound_name.clone()),
                    );
                    if needs_memory(field_type) {
                        let destination =
                            self.address_of_local(bound, field_type);
                        self.emit(IrStatement::Copy {
                            destination,
                            source: IrOperand::Local(field_address),
                            size: self.builder.byte_size(field_type),
                        });
                    } else {
                        self.emit(IrStatement::Assign(
                            bound,
                            IrRvalue::Load {
                                address: IrOperand::Local(field_address),
                                ty: field_type.clone(),
                            },
                        ));
                    }
                    self.define_variable(bound_name, bound);
                }
                Ok(())
            }
            Pattern::Identifier(name) => {
                if let Some((value, value_type)) = scalar {
                    let bound = self
                        .fresh_local(value_type.clone(), Some(name.clone()));
                    self.emit(IrStatement::Assign(
                        bound,
                        IrRvalue::Use(value.clone()),
                    ));
                    self.define_variable(name, bound);
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn type_of_local(&self, local: LocalId) -> Type {
        self.locals[local].ty.clone()
    }

    fn shallow_value_type(&self, expression: &Expression) -> Option<Type> {
        match expression {
            Expression::Literal(Literal::Integer(_)) => Some(Type::I64),
            Expression::Literal(Literal::Float(_)) => Some(Type::F64),
            Expression::Literal(Literal::Float32(_)) => Some(Type::F32),
            Expression::Boolean(_)
            | Expression::Literal(Literal::Boolean(_)) => Some(Type::Bool),
            Expression::Identifier(name) => self
                .resolve_variable(name)
                .map(|local| self.type_of_local(local)),
            Expression::Borrow(inner) => self
                .shallow_value_type(inner)
                .map(|ty| Type::Ref(Box::new(ty))),
            Expression::BorrowMut(inner) => self
                .shallow_value_type(inner)
                .map(|ty| Type::RefMut(Box::new(ty))),
            Expression::StructInit(name, fields) => {
                if self.builder.generic_struct_defs.contains_key(name) {
                    self.generic_instance_of(name, fields).map(Type::Struct)
                } else {
                    Some(Type::Struct(name.clone()))
                }
            }
            _ => None,
        }
    }

    fn generic_instance_of(
        &self,
        struct_name: &str,
        field_inits: &[(String, Expression)],
    ) -> Option<String> {
        let (type_params, fields) =
            self.builder.generic_struct_defs.get(struct_name)?.clone();
        let mut subst: HashMap<String, Type> = HashMap::new();
        for (field_name, value) in field_inits {
            if let Some(field) =
                fields.iter().find(|field| &field.name == field_name)
                && let Some(value_type) = self.shallow_value_type(value)
            {
                infer_subst_into(
                    &field.field_type,
                    &value_type,
                    &type_params,
                    &mut subst,
                );
            }
        }
        let rendered: Vec<String> = type_params
            .iter()
            .map(|type_param| {
                subst
                    .get(type_param)
                    .map(|ty| ty.to_string())
                    .unwrap_or_else(|| type_param.clone())
            })
            .collect();
        Some(format!("{struct_name}<{}>", rendered.join(", ")))
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

fn zero_operand(ty: &Type) -> IrOperand {
    match ty {
        Type::F32 | Type::F64 => {
            IrOperand::Constant(IrConstant::Float(0.0, ty.clone()))
        }
        Type::Bool => IrOperand::Constant(IrConstant::Bool(false)),
        _ if is_integer(ty) => {
            IrOperand::Constant(IrConstant::Integer(0, ty.clone()))
        }
        _ => IrOperand::Constant(IrConstant::Integer(0, Type::I64)),
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
