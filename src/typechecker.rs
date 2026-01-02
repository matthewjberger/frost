use crate::{
    types::Type, Block, Expression, Literal, Operator, Parameter, Statement,
    StructField,
};
use anyhow::{bail, Result};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<StructField>,
}

#[derive(Debug, Clone)]
pub struct FunctionSig {
    pub params: Vec<Type>,
    pub return_type: Type,
}

#[derive(Debug, Default, Clone)]
pub struct TypeEnv {
    bindings: HashMap<String, Type>,
    structs: HashMap<String, StructDef>,
    functions: HashMap<String, FunctionSig>,
    parent: Option<Box<TypeEnv>>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_child(parent: TypeEnv) -> Self {
        Self {
            parent: Some(Box::new(parent)),
            ..Default::default()
        }
    }

    pub fn define(&mut self, name: &str, typ: Type) {
        self.bindings.insert(name.to_string(), typ);
    }

    pub fn define_struct(&mut self, name: &str, def: StructDef) {
        self.structs.insert(name.to_string(), def);
    }

    pub fn define_function(&mut self, name: &str, sig: FunctionSig) {
        self.functions.insert(name.to_string(), sig);
    }

    pub fn lookup(&self, name: &str) -> Option<Type> {
        if let Some(typ) = self.bindings.get(name) {
            return Some(typ.clone());
        }
        if let Some(ref parent) = self.parent {
            return parent.lookup(name);
        }
        None
    }

    pub fn lookup_struct(&self, name: &str) -> Option<StructDef> {
        if let Some(def) = self.structs.get(name) {
            return Some(def.clone());
        }
        if let Some(ref parent) = self.parent {
            return parent.lookup_struct(name);
        }
        None
    }

    pub fn lookup_function(&self, name: &str) -> Option<FunctionSig> {
        if let Some(sig) = self.functions.get(name) {
            return Some(sig.clone());
        }
        if let Some(ref parent) = self.parent {
            return parent.lookup_function(name);
        }
        None
    }
}

pub struct TypeChecker {
    env: TypeEnv,
}

impl TypeChecker {
    pub fn new() -> Self {
        let mut env = TypeEnv::new();
        env.define_function(
            "len",
            FunctionSig {
                params: vec![Type::Unknown],
                return_type: Type::I64,
            },
        );
        env.define_function(
            "first",
            FunctionSig {
                params: vec![Type::Unknown],
                return_type: Type::Unknown,
            },
        );
        env.define_function(
            "last",
            FunctionSig {
                params: vec![Type::Unknown],
                return_type: Type::Unknown,
            },
        );
        env.define_function(
            "rest",
            FunctionSig {
                params: vec![Type::Unknown],
                return_type: Type::Unknown,
            },
        );
        env.define_function(
            "push",
            FunctionSig {
                params: vec![Type::Unknown, Type::Unknown],
                return_type: Type::Unknown,
            },
        );
        env.define_function(
            "print",
            FunctionSig {
                params: vec![Type::Unknown],
                return_type: Type::Void,
            },
        );
        env.define_function(
            "abs",
            FunctionSig {
                params: vec![Type::Unknown],
                return_type: Type::Unknown,
            },
        );
        env.define_function(
            "min",
            FunctionSig {
                params: vec![Type::Unknown, Type::Unknown],
                return_type: Type::Unknown,
            },
        );
        env.define_function(
            "max",
            FunctionSig {
                params: vec![Type::Unknown, Type::Unknown],
                return_type: Type::Unknown,
            },
        );
        env.define_function(
            "substr",
            FunctionSig {
                params: vec![Type::Str, Type::I64, Type::I64],
                return_type: Type::Str,
            },
        );
        env.define_function(
            "contains",
            FunctionSig {
                params: vec![Type::Str, Type::Str],
                return_type: Type::Bool,
            },
        );
        env.define_function(
            "to_string",
            FunctionSig {
                params: vec![Type::Unknown],
                return_type: Type::Str,
            },
        );
        env.define_function(
            "parse_int",
            FunctionSig {
                params: vec![Type::Str],
                return_type: Type::Unknown,
            },
        );
        env.define_function(
            "floor",
            FunctionSig {
                params: vec![Type::Unknown],
                return_type: Type::Unknown,
            },
        );
        env.define_function(
            "ceil",
            FunctionSig {
                params: vec![Type::Unknown],
                return_type: Type::Unknown,
            },
        );
        env.define_function(
            "sqrt",
            FunctionSig {
                params: vec![Type::Unknown],
                return_type: Type::F64,
            },
        );
        env.define_function(
            "read_file",
            FunctionSig {
                params: vec![Type::Str],
                return_type: Type::Unknown,
            },
        );
        env.define_function(
            "write_file",
            FunctionSig {
                params: vec![Type::Str, Type::Str],
                return_type: Type::Bool,
            },
        );
        env.define_function(
            "file_exists",
            FunctionSig {
                params: vec![Type::Str],
                return_type: Type::Bool,
            },
        );
        env.define_function(
            "append_file",
            FunctionSig {
                params: vec![Type::Str, Type::Str],
                return_type: Type::Bool,
            },
        );
        Self { env }
    }

    pub fn check_program(&mut self, statements: &[Statement]) -> Result<()> {
        for statement in statements {
            self.check_statement(statement)?;
        }
        Ok(())
    }

    fn check_statement(
        &mut self,
        statement: &Statement,
    ) -> Result<Option<Type>> {
        match statement {
            Statement::Let {
                name,
                type_annotation,
                value,
                ..
            } => {
                let expr_type = self.infer_expression(value)?;
                if let Some(declared_type) = type_annotation {
                    if !self.types_compatible(declared_type, &expr_type) {
                        bail!(
                            "Type mismatch in declaration '{}': expected {}, got {}",
                            name,
                            declared_type,
                            expr_type
                        );
                    }
                    self.env.define(name, declared_type.clone());
                } else {
                    self.env.define(name, expr_type);
                }
                Ok(None)
            }
            Statement::Constant(name, expression) => {
                let expr_type = self.infer_expression(expression)?;
                self.env.define(name, expr_type);
                Ok(None)
            }
            Statement::Return(expression) => {
                let typ = self.infer_expression(expression)?;
                Ok(Some(typ))
            }
            Statement::Expression(expression) => {
                self.infer_expression(expression)?;
                Ok(None)
            }
            Statement::Struct(name, fields) => {
                self.env.define_struct(
                    name,
                    StructDef {
                        name: name.clone(),
                        fields: fields.clone(),
                    },
                );
                self.env.define(name, Type::Struct(name.clone()));
                Ok(None)
            }
            Statement::Defer(inner) => {
                self.check_statement(inner)?;
                Ok(None)
            }
            Statement::Assignment(lhs, rhs) => {
                let lhs_type = self.infer_expression(lhs)?;
                let rhs_type = self.infer_expression(rhs)?;
                if lhs_type != rhs_type
                    && lhs_type != Type::Unknown
                    && rhs_type != Type::Unknown
                {
                    bail!(
                        "Type mismatch in assignment: expected {}, got {}",
                        lhs_type,
                        rhs_type
                    );
                }
                Ok(None)
            }
            Statement::For(iterator, range, body) => {
                if let Expression::Range(start, end) = range {
                    let start_type = self.infer_expression(start)?;
                    let end_type = self.infer_expression(end)?;
                    if start_type != end_type {
                        bail!("Range start and end must have the same type");
                    }
                    self.env.define(iterator, start_type);
                    for statement in body {
                        self.check_statement(statement)?;
                    }
                } else {
                    bail!("For loop requires a range expression");
                }
                Ok(None)
            }
            Statement::Enum(name, variants) => {
                for variant in variants {
                    let full_name = format!("{}::{}", name, variant.name);
                    self.env.define(&full_name, Type::Enum(name.clone()));
                }
                Ok(None)
            }
            Statement::TypeAlias(_name, _typ) => Ok(None),
            Statement::Break | Statement::Continue => Ok(None),
            Statement::While(condition, body) => {
                let cond_type = self.infer_expression(condition)?;
                if cond_type != Type::Bool && cond_type != Type::Unknown {
                    bail!(
                        "While condition must be boolean, got: {}",
                        cond_type
                    );
                }
                for statement in body {
                    self.check_statement(statement)?;
                }
                Ok(None)
            }
            Statement::Import(_path) => Ok(None),
            Statement::InterpolatedConstant(_, _) => Ok(None),
        }
    }

    fn infer_expression(&mut self, expression: &Expression) -> Result<Type> {
        match expression {
            Expression::Literal(literal) => self.infer_literal(literal),
            Expression::Boolean(_) => Ok(Type::Bool),
            Expression::Identifier(name) => self
                .env
                .lookup(name)
                .ok_or_else(|| anyhow::anyhow!("Undefined variable: {}", name)),
            Expression::Prefix(operator, operand) => {
                let operand_type = self.infer_expression(operand)?;
                match operator {
                    Operator::Negate => {
                        if self.is_numeric(&operand_type) {
                            Ok(operand_type)
                        } else {
                            bail!(
                                "Cannot negate non-numeric type: {}",
                                operand_type
                            )
                        }
                    }
                    Operator::Not => Ok(Type::Bool),
                    _ => bail!("Unknown prefix operator: {:?}", operator),
                }
            }
            Expression::Infix(left, operator, right) => {
                let left_type = self.infer_expression(left)?;
                let right_type = self.infer_expression(right)?;
                self.check_infix_types(&left_type, operator, &right_type)
            }
            Expression::If(condition, consequence, alternative) => {
                let cond_type = self.infer_expression(condition)?;
                if cond_type != Type::Bool && cond_type != Type::Unknown {
                    bail!("Condition must be boolean, got: {}", cond_type);
                }
                let conseq_type = self.check_block(consequence)?;
                if let Some(alt) = alternative {
                    let alt_type = self.check_block(alt)?;
                    if !self.types_compatible(&conseq_type, &alt_type) {
                        bail!(
                            "If branches have different types: {} vs {}",
                            conseq_type,
                            alt_type
                        );
                    }
                }
                Ok(conseq_type)
            }
            Expression::Function(params, return_type, body) => {
                self.check_function(params, return_type, body)
            }
            Expression::Proc(params, return_type, body) => {
                self.check_function(params, return_type, body)
            }
            Expression::Call(function, arguments) => {
                self.check_call(function, arguments)
            }
            Expression::Index(array, index) => {
                let array_type = self.infer_expression(array)?;
                let index_type = self.infer_expression(index)?;
                if !self.is_integer(&index_type) && index_type != Type::Unknown
                {
                    bail!("Array index must be integer, got: {}", index_type);
                }
                match array_type {
                    Type::Array(elem_type, _) => Ok(*elem_type),
                    Type::Slice(elem_type) => Ok(*elem_type),
                    Type::Unknown => Ok(Type::Unknown),
                    _ => bail!("Cannot index into type: {}", array_type),
                }
            }
            Expression::FieldAccess(expr, field) => {
                let expr_type = self.infer_expression(expr)?;
                match expr_type {
                    Type::Struct(struct_name) => {
                        if let Some(struct_def) =
                            self.env.lookup_struct(&struct_name)
                        {
                            for struct_field in &struct_def.fields {
                                if struct_field.name == *field {
                                    return Ok(struct_field.field_type.clone());
                                }
                            }
                            bail!(
                                "Struct {} has no field '{}'",
                                struct_name,
                                field
                            )
                        } else {
                            bail!("Unknown struct type: {}", struct_name)
                        }
                    }
                    Type::Unknown => Ok(Type::Unknown),
                    _ => bail!(
                        "Cannot access field on non-struct type: {}",
                        expr_type
                    ),
                }
            }
            Expression::AddressOf(expr) => {
                let inner_type = self.infer_expression(expr)?;
                Ok(Type::Ptr(Box::new(inner_type)))
            }
            Expression::Borrow(expr) => {
                let inner_type = self.infer_expression(expr)?;
                Ok(Type::Ref(Box::new(inner_type)))
            }
            Expression::BorrowMut(expr) => {
                let inner_type = self.infer_expression(expr)?;
                Ok(Type::RefMut(Box::new(inner_type)))
            }
            Expression::Dereference(expr) => {
                let expr_type = self.infer_expression(expr)?;
                match expr_type {
                    Type::Ptr(inner) => Ok(*inner),
                    Type::Ref(inner) => Ok(*inner),
                    Type::RefMut(inner) => Ok(*inner),
                    Type::Unknown => Ok(Type::Unknown),
                    _ => bail!(
                        "Cannot dereference non-pointer type: {}",
                        expr_type
                    ),
                }
            }
            Expression::StructInit(name, fields) => {
                for (_, value_expr) in fields {
                    self.infer_expression(value_expr)?;
                }
                Ok(Type::Struct(name.clone()))
            }
            Expression::Sizeof(_) => Ok(Type::I64),
            Expression::Range(start, _end) => self.infer_expression(start),
            Expression::Switch(scrutinee, cases) => {
                self.infer_expression(scrutinee)?;
                let mut result_type = Type::Unknown;
                for case in cases {
                    let case_type = self.check_block(&case.body)?;
                    if result_type == Type::Unknown {
                        result_type = case_type;
                    }
                }
                Ok(result_type)
            }
            Expression::Tuple(elements) => {
                for element in elements {
                    self.infer_expression(element)?;
                }
                Ok(Type::Unknown)
            }
            Expression::EnumVariantInit(enum_name, _variant_name, fields) => {
                for (_field_name, value_expr) in fields {
                    self.infer_expression(value_expr)?;
                }
                Ok(Type::Enum(enum_name.clone()))
            }
            Expression::ComptimeBlock(_) => Ok(Type::Void),
            Expression::ComptimeFor { .. } => Ok(Type::Void),
            Expression::TypeValue(typ) => Ok(typ.clone()),
            Expression::Typename(_) => Ok(Type::Str),
            Expression::InterpolatedIdent(_) => Ok(Type::Unknown),
        }
    }

    fn infer_literal(&self, literal: &Literal) -> Result<Type> {
        match literal {
            Literal::Integer(_) => Ok(Type::I64),
            Literal::Float(_) => Ok(Type::F64),
            Literal::Float32(_) => Ok(Type::F32),
            Literal::String(_) => Ok(Type::Str),
            Literal::Array(elements) => {
                if elements.is_empty() {
                    Ok(Type::Array(Box::new(Type::Unknown), 0))
                } else {
                    Ok(Type::Array(Box::new(Type::Unknown), elements.len()))
                }
            }
            Literal::HashMap(_) => Ok(Type::Unknown),
        }
    }

    fn check_infix_types(
        &self,
        left: &Type,
        operator: &Operator,
        right: &Type,
    ) -> Result<Type> {
        if *left == Type::Unknown || *right == Type::Unknown {
            return Ok(Type::Unknown);
        }

        match operator {
            Operator::Add
            | Operator::Subtract
            | Operator::Multiply
            | Operator::Divide
            | Operator::Modulo => {
                if self.is_numeric(left) && self.is_numeric(right) {
                    if !self.types_compatible(left, right) {
                        bail!(
                            "Arithmetic operator requires matching types: {} vs {}",
                            left,
                            right
                        );
                    }
                    Ok(left.clone())
                } else if *operator == Operator::Add
                    && *left == Type::Str
                    && *right == Type::Str
                {
                    Ok(Type::Str)
                } else {
                    bail!(
                        "Invalid types for arithmetic operator: {} {:?} {}",
                        left,
                        operator,
                        right
                    )
                }
            }
            Operator::Equal | Operator::NotEqual => {
                if !self.types_compatible(left, right) {
                    bail!(
                        "Comparison requires matching types: {} vs {}",
                        left,
                        right
                    );
                }
                Ok(Type::Bool)
            }
            Operator::LessThan
            | Operator::GreaterThan
            | Operator::LessThanOrEqual
            | Operator::GreaterThanOrEqual => {
                if !self.is_numeric(left) || !self.is_numeric(right) {
                    bail!(
                        "Comparison requires numeric types: {} vs {}",
                        left,
                        right
                    );
                }
                Ok(Type::Bool)
            }
            _ => bail!("Unknown infix operator: {:?}", operator),
        }
    }

    fn check_function(
        &mut self,
        params: &[Parameter],
        return_type: &Option<Type>,
        body: &Block,
    ) -> Result<Type> {
        let mut child_env = TypeEnv::new_child(self.env.clone());
        std::mem::swap(&mut self.env, &mut child_env);

        for param in params {
            let param_type =
                param.type_annotation.clone().unwrap_or(Type::Unknown);
            self.env.define(&param.name, param_type);
        }

        let body_type = self.check_block(body)?;

        std::mem::swap(&mut self.env, &mut child_env);

        if let Some(declared_return) = return_type {
            if declared_return.is_reference() {
                bail!("functions cannot return references - return an owned value instead");
            }
            if !self.types_compatible(declared_return, &body_type) {
                bail!(
                    "Function return type mismatch: declared {}, body returns {}",
                    declared_return,
                    body_type
                );
            }
        }

        let param_types: Vec<Type> = params
            .iter()
            .map(|p| p.type_annotation.clone().unwrap_or(Type::Unknown))
            .collect();
        let ret = return_type.clone().unwrap_or(body_type.clone());
        if ret.is_reference() {
            bail!("functions cannot return references - return an owned value instead");
        }
        Ok(Type::Proc(param_types, Box::new(ret)))
    }

    fn check_call(
        &mut self,
        function: &Expression,
        arguments: &[Expression],
    ) -> Result<Type> {
        let func_type = self.infer_expression(function)?;

        match func_type {
            Type::Proc(param_types, return_type) => {
                if param_types.len() != arguments.len() {
                    bail!(
                        "Function expects {} arguments, got {}",
                        param_types.len(),
                        arguments.len()
                    );
                }
                for (index, (param_type, arg)) in
                    param_types.iter().zip(arguments).enumerate()
                {
                    let arg_type = self.infer_expression(arg)?;
                    if !self.types_compatible(param_type, &arg_type) {
                        bail!(
                            "Argument {} type mismatch: expected {}, got {}",
                            index + 1,
                            param_type,
                            arg_type
                        );
                    }
                }
                Ok(*return_type)
            }
            Type::Unknown => {
                for arg in arguments {
                    self.infer_expression(arg)?;
                }
                Ok(Type::Unknown)
            }
            _ => bail!("Cannot call non-function type: {}", func_type),
        }
    }

    fn check_block(&mut self, block: &Block) -> Result<Type> {
        let mut last_type = Type::Void;
        for statement in block {
            if let Some(ret_type) = self.check_statement(statement)? {
                return Ok(ret_type);
            }
            if let Statement::Expression(expr) = statement {
                last_type = self.infer_expression(expr)?;
            }
        }
        Ok(last_type)
    }

    fn types_compatible(&self, expected: &Type, actual: &Type) -> bool {
        if *expected == Type::Unknown || *actual == Type::Unknown {
            return true;
        }
        expected == actual
    }

    fn is_numeric(&self, typ: &Type) -> bool {
        matches!(
            typ,
            Type::I8
                | Type::I16
                | Type::I32
                | Type::I64
                | Type::U8
                | Type::U16
                | Type::U32
                | Type::U64
                | Type::F32
                | Type::F64
                | Type::Unknown
        )
    }

    fn is_integer(&self, typ: &Type) -> bool {
        matches!(
            typ,
            Type::I8
                | Type::I16
                | Type::I32
                | Type::I64
                | Type::U8
                | Type::U16
                | Type::U32
                | Type::U64
                | Type::Unknown
        )
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Lexer, Parser};

    fn check(input: &str) -> Result<()> {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;
        let mut checker = TypeChecker::new();
        checker.check_program(&program)
    }

    fn check_fails(input: &str) -> bool {
        check(input).is_err()
    }

    #[test]
    fn infer_integer_literal() {
        assert!(check("5;").is_ok());
    }

    #[test]
    fn infer_bool_literal() {
        assert!(check("true;").is_ok());
        assert!(check("false;").is_ok());
    }

    #[test]
    fn infer_string_literal() {
        assert!(check("\"hello\";").is_ok());
    }

    #[test]
    fn check_let_type_matches() {
        assert!(check("x : i64 = 5;").is_ok());
    }

    #[test]
    fn check_let_type_mismatch() {
        assert!(check_fails("x : i64 = true;"));
    }

    #[test]
    fn check_let_inferred() {
        assert!(check("x := 5; y := x;").is_ok());
    }

    #[test]
    fn check_binary_op_types() {
        assert!(check("5 + 5;").is_ok());
        assert!(check_fails("5 + true;"));
    }

    #[test]
    fn check_comparison_types() {
        assert!(check("5 < 10;").is_ok());
        assert!(check("5 == 5;").is_ok());
        assert!(check("true == false;").is_ok());
    }

    #[test]
    fn check_string_concat() {
        assert!(check("\"hello\" + \" world\";").is_ok());
    }

    #[test]
    fn check_function_param_types() {
        assert!(check(
            "add := fn(a: i64, b: i64) -> i64 { a + b }; add(1, 2);"
        )
        .is_ok());
    }

    #[test]
    fn check_function_return_type() {
        assert!(check_fails("f := fn() -> i64 { true };"));
    }

    #[test]
    fn check_function_arg_mismatch() {
        assert!(check_fails("f := fn(x: i64) { x }; f(true);"));
    }

    #[test]
    fn check_struct_field_access() {
        assert!(check("Point :: struct { x: i64, y: i64 }").is_ok());
    }

    #[test]
    fn check_pointer_operations() {
        assert!(check("x : i64 = 5; p : &i64 = &x;").is_ok());
    }

    #[test]
    fn check_dereference() {
        assert!(check("x : i64 = 5; p : &i64 = &x; p^;").is_ok());
    }

    #[test]
    fn check_borrow_mut() {
        assert!(check("mut x : i64 = 5; p : &mut i64 = &mut x;").is_ok());
    }

    #[test]
    fn check_if_condition_bool() {
        assert!(check("if (true) { 1 }").is_ok());
        assert!(check_fails("if (5) { 1 }"));
    }

    #[test]
    fn check_negate_numeric() {
        assert!(check("-5;").is_ok());
        assert!(check_fails("-true;"));
    }

    #[test]
    fn check_array_index() {
        assert!(check("[1, 2, 3][0];").is_ok());
    }

    #[test]
    fn infer_variable_from_context() {
        assert!(check("x : i64 = 5; y := x;").is_ok());
    }

    #[test]
    fn check_constant_declaration() {
        assert!(check("PI :: 3;").is_ok());
    }

    #[test]
    fn check_proc_literal() {
        assert!(check("f := proc(x: i64) -> i64 { x };").is_ok());
    }

    #[test]
    fn check_modulo_operator() {
        assert!(check("5 % 3;").is_ok());
        assert!(check("x : i64 = 10; x % 3;").is_ok());
        assert!(check_fails("5 % true;"));
        assert!(check_fails("\"hello\" % 2;"));
    }

    #[test]
    fn check_less_than_or_equal() {
        assert!(check("5 <= 10;").is_ok());
        assert!(check("x : i64 = 5; x <= 10;").is_ok());
        assert!(check_fails("true <= 5;"));
    }

    #[test]
    fn check_greater_than_or_equal() {
        assert!(check("10 >= 5;").is_ok());
        assert!(check("x : i64 = 10; x >= 5;").is_ok());
        assert!(check_fails("5 >= true;"));
    }

    #[test]
    fn check_cannot_return_reference() {
        assert!(check_fails("f :: proc() -> &i64 { x := 1; &x };"));
        assert!(check_fails(
            "f :: proc() -> &mut i64 { mut x := 1; &mut x };"
        ));
    }

    #[test]
    fn check_can_accept_reference_param() {
        assert!(check("f :: proc(x: &i64) -> i64 { x^ };").is_ok());
        assert!(check("f :: proc(x: &mut i64) { x^ = 5; };").is_ok());
    }
}
