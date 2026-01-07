use crate::{
    parser::{IdentPart, ReturnSignature}, types::Type, Expression, HeapObject, Literal, Operator,
    Parameter, Statement, StructField, Value64,
};
use anyhow::{bail, Result};
use std::{
    collections::{HashMap, HashSet},
    fmt,
    path::PathBuf,
    slice::Iter,
};

#[derive(Debug, Clone)]
pub struct CompiledStruct {
    pub name: String,
    pub fields: Vec<StructField>,
    pub field_offsets: HashMap<String, usize>,
    pub size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SymbolScope {
    Global,
    Local,
    Builtin,
    Free,
    Function,
    Native,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OwnershipState {
    Owned,
    Moved,
    Borrowed,
    BorrowedMut,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    pub name: String,
    pub scope: SymbolScope,
    pub index: usize,
    pub symbol_type: Option<Type>,
    pub mutable: bool,
    pub ownership_state: OwnershipState,
}

#[derive(Debug, Default, Clone)]
pub struct SymbolTable {
    pub store: HashMap<String, Symbol>,
    pub num_definitions: usize,
    pub outer: Option<Box<SymbolTable>>,
    pub free_symbols: Vec<Symbol>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledFunction {
    pub instructions: Vec<Instruction>,
    pub num_locals: usize,
    pub num_parameters: usize,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_enclosed(outer: SymbolTable) -> Self {
        Self {
            outer: Some(Box::new(outer)),
            ..Default::default()
        }
    }

    pub fn define(&mut self, name: &str) -> Symbol {
        self.define_with_type_and_mutability(name, None, false)
    }

    pub fn define_with_type(
        &mut self,
        name: &str,
        symbol_type: Option<Type>,
    ) -> Symbol {
        self.define_with_type_and_mutability(name, symbol_type, false)
    }

    pub fn define_with_type_and_mutability(
        &mut self,
        name: &str,
        symbol_type: Option<Type>,
        mutable: bool,
    ) -> Symbol {
        let scope = if self.outer.is_some() {
            SymbolScope::Local
        } else {
            SymbolScope::Global
        };
        let symbol = Symbol {
            name: name.to_string(),
            scope,
            index: self.num_definitions,
            symbol_type,
            mutable,
            ownership_state: OwnershipState::Owned,
        };
        self.store.insert(name.to_string(), symbol.clone());
        self.num_definitions += 1;
        symbol
    }

    pub fn define_builtin(&mut self, index: usize, name: &str) -> Symbol {
        let symbol = Symbol {
            name: name.to_string(),
            scope: SymbolScope::Builtin,
            index,
            symbol_type: None,
            mutable: false,
            ownership_state: OwnershipState::Owned,
        };
        self.store.insert(name.to_string(), symbol.clone());
        symbol
    }

    pub fn define_native(&mut self, index: usize, name: &str) -> Symbol {
        let symbol = Symbol {
            name: name.to_string(),
            scope: SymbolScope::Native,
            index,
            symbol_type: None,
            mutable: false,
            ownership_state: OwnershipState::Owned,
        };
        self.store.insert(name.to_string(), symbol.clone());
        symbol
    }

    pub fn define_function_name(&mut self, name: &str) -> Symbol {
        let symbol = Symbol {
            name: name.to_string(),
            scope: SymbolScope::Function,
            index: 0,
            symbol_type: None,
            mutable: false,
            ownership_state: OwnershipState::Owned,
        };
        self.store.insert(name.to_string(), symbol.clone());
        symbol
    }

    pub fn resolve(&mut self, name: &str) -> Option<Symbol> {
        if let Some(symbol) = self.store.get(name) {
            return Some(symbol.clone());
        }
        if let Some(ref mut outer) = self.outer {
            if let Some(symbol) = outer.resolve(name) {
                if symbol.scope == SymbolScope::Global
                    || symbol.scope == SymbolScope::Builtin
                    || symbol.scope == SymbolScope::Native
                {
                    return Some(symbol);
                }
                return Some(self.define_free(symbol));
            }
        }
        None
    }

    fn define_free(&mut self, original: Symbol) -> Symbol {
        self.free_symbols.push(original.clone());
        let symbol = Symbol {
            name: original.name,
            scope: SymbolScope::Free,
            index: self.free_symbols.len() - 1,
            symbol_type: original.symbol_type,
            mutable: original.mutable,
            ownership_state: original.ownership_state,
        };
        self.store.insert(symbol.name.clone(), symbol.clone());
        symbol
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Opcode {
    Constant,
    Pop,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    AddI64,
    SubI64,
    MulI64,
    DivI64,
    ModI64,
    AddF64,
    SubF64,
    MulF64,
    DivF64,
    AddF32,
    SubF32,
    MulF32,
    DivF32,
    NegateF32,
    True,
    False,
    Equal,
    NotEqual,
    GreaterThan,
    EqualI64,
    NotEqualI64,
    LessThanI64,
    LessThanOrEqualI64,
    GreaterThanI64,
    GreaterThanOrEqualI64,
    EqualBool,
    NotEqualBool,
    Minus,
    NegateI64,
    NegateF64,
    Bang,
    JumpNotTruthy,
    Jump,
    Null,
    GetGlobal,
    SetGlobal,
    Array,
    Hash,
    Index,
    IndexSet,
    Call,
    ReturnValue,
    Return,
    GetLocal,
    SetLocal,
    GetBuiltin,
    GetNative,
    Closure,
    GetFree,
    CurrentClosure,
    LoadPtr,
    StorePtr,
    AddressOfLocal,
    AddressOfGlobal,
    Alloc,
    Free,
    StructAlloc,
    StructGet,
    StructSet,
    TaggedUnionAlloc,
    TaggedUnionSetTag,
    TaggedUnionGetTag,
    TaggedUnionGetField,
    TaggedUnionSetField,
    TupleAlloc,
    TupleGet,
    Dup,
    Drop,
    ShiftLeft,
    ShiftRight,
    BitwiseAnd,
    BitwiseOr,
    GetContext,
    SetContext,
    GetContextField,
    SetContextField,
    PushContextScope,
    PopContextScope,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Instruction {
    pub opcode: Opcode,
    pub operands: Vec<u16>,
}

impl Instruction {
    pub fn new(opcode: Opcode, operands: Vec<u16>) -> Self {
        Self { opcode, operands }
    }

    pub fn as_bytes(&self) -> Vec<u8> {
        let operands: Vec<u8> =
            self.operands.iter().flat_map(|x| x.to_be_bytes()).collect();
        [vec![self.opcode as u8], operands].concat()
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let operands = self
            .operands
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(" ");
        write!(f, "Op{:?} {}", self.opcode, operands,)
    }
}

#[derive(Default, Debug)]
pub struct Bytecode {
    pub instructions: Vec<Instruction>,
    pub constants: Vec<Value64>,
    pub functions: Vec<CompiledFunction>,
    pub heap: Vec<HeapObject>,
    pub native_names: Vec<String>,
    pub global_symbols: HashMap<String, usize>,
}

impl Bytecode {
    pub fn assemble(&self) -> Vec<u8> {
        self.instructions
            .iter()
            .flat_map(|instruction| instruction.as_bytes())
            .collect::<Vec<_>>()
    }

    pub fn disassemble(&self) -> String {
        let mut byte_counter = 0;
        self.instructions
            .iter()
            .map(|instruction| {
                let result = format!("{:0>4} {}", byte_counter, instruction);
                byte_counter += 1 + (instruction.operands.len() * 2); // opcode (1 byte) + operands (2 bytes)
                result
            })
            .collect::<Vec<String>>()
            .join("\n")
    }
}

#[derive(Debug, Clone)]
pub struct CompiledEnumVariant {
    pub name: String,
    pub tag: usize,
    pub fields: Vec<StructField>,
}

#[derive(Debug, Clone)]
pub struct CompiledEnum {
    pub name: String,
    pub variants: Vec<CompiledEnumVariant>,
}

#[derive(Debug, Clone)]
pub struct ActiveBorrow {
    pub borrowed_symbol: String,
    pub scope_depth: usize,
    pub is_mutable: bool,
}

#[derive(Debug, Default, Clone)]
pub struct BorrowChecker {
    pub active_borrows: Vec<ActiveBorrow>,
    pub scope_depth: usize,
}

impl BorrowChecker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enter_scope(&mut self) {
        self.scope_depth += 1;
    }

    pub fn exit_scope(&mut self) {
        let current_depth = self.scope_depth;
        self.active_borrows
            .retain(|borrow| borrow.scope_depth < current_depth);
        self.scope_depth = self.scope_depth.saturating_sub(1);
    }

    pub fn try_borrow(&mut self, name: &str, is_mutable: bool) -> Result<()> {
        for borrow in &self.active_borrows {
            if borrow.borrowed_symbol == name
                && (is_mutable || borrow.is_mutable)
            {
                anyhow::bail!(
                    "cannot borrow '{}' as {} because it is already borrowed as {}",
                    name,
                    if is_mutable { "mutable" } else { "immutable" },
                    if borrow.is_mutable { "mutable" } else { "immutable" }
                );
            }
        }
        self.active_borrows.push(ActiveBorrow {
            borrowed_symbol: name.to_string(),
            scope_depth: self.scope_depth,
            is_mutable,
        });
        Ok(())
    }

    pub fn try_move(&self, name: &str) -> Result<()> {
        for borrow in &self.active_borrows {
            if borrow.borrowed_symbol == name {
                anyhow::bail!("cannot move '{}' because it is borrowed", name);
            }
        }
        Ok(())
    }

    pub fn try_mutate(&self, name: &str) -> Result<()> {
        for borrow in &self.active_borrows {
            if borrow.borrowed_symbol == name {
                anyhow::bail!("cannot mutate '{}' while it is borrowed", name);
            }
        }
        Ok(())
    }
}

struct SubstitutionContext<'a> {
    index_var: Option<&'a str>,
    type_var: &'a str,
    index: usize,
    type_name: &'a str,
}

struct ComptimeForParams<'a> {
    index_var: &'a Option<String>,
    type_var: &'a str,
    types: &'a [Type],
    body: &'a [Statement],
}

#[derive(Debug, Clone)]
pub struct GenericFunctionDef {
    pub type_params: Vec<String>,
    pub parameters: Vec<Parameter>,
    pub return_sig: ReturnSignature,
    pub body: Vec<Statement>,
}

struct MonoSubstitution {
    type_bindings: HashMap<String, Type>,
}

impl MonoSubstitution {
    fn new() -> Self {
        Self {
            type_bindings: HashMap::new(),
        }
    }

    fn bind(&mut self, param: &str, concrete: Type) {
        self.type_bindings.insert(param.to_string(), concrete);
    }

    fn substitute_type(&self, typ: &Type) -> Type {
        match typ {
            Type::TypeParam(name) => {
                if let Some(concrete) = self.type_bindings.get(name) {
                    concrete.clone()
                } else {
                    typ.clone()
                }
            }
            Type::Ptr(inner) => Type::Ptr(Box::new(self.substitute_type(inner))),
            Type::Ref(inner) => Type::Ref(Box::new(self.substitute_type(inner))),
            Type::RefMut(inner) => Type::RefMut(Box::new(self.substitute_type(inner))),
            Type::Array(inner, size) => Type::Array(Box::new(self.substitute_type(inner)), *size),
            Type::Slice(inner) => Type::Slice(Box::new(self.substitute_type(inner))),
            Type::Proc(params, ret) => Type::Proc(
                params.iter().map(|p| self.substitute_type(p)).collect(),
                Box::new(self.substitute_type(ret)),
            ),
            Type::Distinct(inner) => Type::Distinct(Box::new(self.substitute_type(inner))),
            Type::Optional(inner) => Type::Optional(Box::new(self.substitute_type(inner))),
            Type::Handle(inner) => Type::Handle(Box::new(self.substitute_type(inner))),
            other => other.clone(),
        }
    }

    fn substitute_return_sig(&self, sig: &ReturnSignature) -> ReturnSignature {
        match sig {
            ReturnSignature::None => ReturnSignature::None,
            ReturnSignature::Single(t) => ReturnSignature::Single(self.substitute_type(t)),
            ReturnSignature::Named(params) => {
                ReturnSignature::Named(
                    params
                        .iter()
                        .map(|p| crate::parser::ReturnParam {
                            name: p.name.clone(),
                            param_type: self.substitute_type(&p.param_type),
                        })
                        .collect(),
                )
            }
        }
    }

    fn substitute_param(&self, param: &Parameter) -> Parameter {
        Parameter {
            name: param.name.clone(),
            type_annotation: param.type_annotation.as_ref().map(|t| self.substitute_type(t)),
            mutable: param.mutable,
        }
    }

    fn substitute_expr(&self, expr: &Expression) -> Expression {
        match expr {
            Expression::Identifier(name) => {
                if self.type_bindings.contains_key(name) {
                    Expression::TypeValue(self.type_bindings.get(name).unwrap().clone())
                } else {
                    Expression::Identifier(name.clone())
                }
            }
            Expression::Infix(left, op, right) => Expression::Infix(
                Box::new(self.substitute_expr(left)),
                *op,
                Box::new(self.substitute_expr(right)),
            ),
            Expression::Call(func, args) => Expression::Call(
                Box::new(self.substitute_expr(func)),
                args.iter().map(|a| self.substitute_expr(a)).collect(),
            ),
            Expression::Index(base, index) => Expression::Index(
                Box::new(self.substitute_expr(base)),
                Box::new(self.substitute_expr(index)),
            ),
            Expression::Prefix(op, operand) => Expression::Prefix(
                *op,
                Box::new(self.substitute_expr(operand)),
            ),
            Expression::If(cond, then_block, else_block) => Expression::If(
                Box::new(self.substitute_expr(cond)),
                then_block.iter().map(|s| self.substitute_stmt(s)).collect(),
                else_block.as_ref().map(|b| b.iter().map(|s| self.substitute_stmt(s)).collect()),
            ),
            Expression::StructInit(name, fields) => Expression::StructInit(
                name.clone(),
                fields.iter().map(|(n, e)| (n.clone(), self.substitute_expr(e))).collect(),
            ),
            Expression::FieldAccess(base, field) => Expression::FieldAccess(
                Box::new(self.substitute_expr(base)),
                field.clone(),
            ),
            Expression::Function(params, ret_sig, body) => Expression::Function(
                params.iter().map(|p| self.substitute_param(p)).collect(),
                self.substitute_return_sig(ret_sig),
                body.iter().map(|s| self.substitute_stmt(s)).collect(),
            ),
            Expression::Proc(params, ret_sig, body) => Expression::Proc(
                params.iter().map(|p| self.substitute_param(p)).collect(),
                self.substitute_return_sig(ret_sig),
                body.iter().map(|s| self.substitute_stmt(s)).collect(),
            ),
            Expression::Tuple(elems) => Expression::Tuple(
                elems.iter().map(|e| self.substitute_expr(e)).collect(),
            ),
            Expression::Dereference(inner) => Expression::Dereference(
                Box::new(self.substitute_expr(inner)),
            ),
            Expression::AddressOf(inner) => Expression::AddressOf(
                Box::new(self.substitute_expr(inner)),
            ),
            Expression::Borrow(inner) => Expression::Borrow(
                Box::new(self.substitute_expr(inner)),
            ),
            Expression::BorrowMut(inner) => Expression::BorrowMut(
                Box::new(self.substitute_expr(inner)),
            ),
            Expression::Sizeof(t) => Expression::Sizeof(self.substitute_type(t)),
            Expression::Typename(t) => Expression::Typename(self.substitute_type(t)),
            Expression::Range(start, end, inclusive) => Expression::Range(
                Box::new(self.substitute_expr(start)),
                Box::new(self.substitute_expr(end)),
                *inclusive,
            ),
            Expression::Switch(scrutinee, cases) => Expression::Switch(
                Box::new(self.substitute_expr(scrutinee)),
                cases.iter().map(|case| crate::parser::SwitchCase {
                    pattern: case.pattern.clone(),
                    body: case.body.iter().map(|s| self.substitute_stmt(s)).collect(),
                }).collect(),
            ),
            Expression::EnumVariantInit(enum_name, variant, fields) => Expression::EnumVariantInit(
                enum_name.clone(),
                variant.clone(),
                fields.iter().map(|(n, e)| (n.clone(), self.substitute_expr(e))).collect(),
            ),
            Expression::IfLet(pattern, value, consequence, alternative) => Expression::IfLet(
                pattern.clone(),
                Box::new(self.substitute_expr(value)),
                consequence.iter().map(|s| self.substitute_stmt(s)).collect(),
                alternative.as_ref().map(|b| b.iter().map(|s| self.substitute_stmt(s)).collect()),
            ),
            Expression::Unsafe(body) => Expression::Unsafe(
                body.iter().map(|s| self.substitute_stmt(s)).collect(),
            ),
            Expression::ComptimeBlock(body) => Expression::ComptimeBlock(
                body.iter().map(|s| self.substitute_stmt(s)).collect(),
            ),
            other => other.clone(),
        }
    }

    fn substitute_stmt(&self, stmt: &Statement) -> Statement {
        match stmt {
            Statement::Constant(name, expr) => {
                Statement::Constant(name.clone(), self.substitute_expr(expr))
            }
            Statement::Let { name, mutable, type_annotation, value } => Statement::Let {
                name: name.clone(),
                mutable: *mutable,
                type_annotation: type_annotation.as_ref().map(|t| self.substitute_type(t)),
                value: self.substitute_expr(value),
            },
            Statement::Expression(expr) => {
                Statement::Expression(self.substitute_expr(expr))
            }
            Statement::Return(expr) => {
                Statement::Return(self.substitute_expr(expr))
            }
            Statement::Defer(inner) => {
                Statement::Defer(Box::new(self.substitute_stmt(inner)))
            }
            Statement::Assignment(lhs, rhs) => {
                Statement::Assignment(self.substitute_expr(lhs), self.substitute_expr(rhs))
            }
            Statement::For(iterator, iterable, body) => {
                Statement::For(
                    iterator.clone(),
                    self.substitute_expr(iterable),
                    body.iter().map(|s| self.substitute_stmt(s)).collect(),
                )
            }
            Statement::While(cond, body) => {
                Statement::While(
                    self.substitute_expr(cond),
                    body.iter().map(|s| self.substitute_stmt(s)).collect(),
                )
            }
            other => other.clone(),
        }
    }

    fn mangle_name(&self, base_name: &str, type_params: &[String]) -> String {
        let mut mangled = base_name.to_string();
        for param in type_params {
            if let Some(concrete) = self.type_bindings.get(param) {
                mangled.push('_');
                mangled.push_str(&Self::type_to_mangle_string(concrete));
            }
        }
        mangled
    }

    fn type_to_mangle_string(typ: &Type) -> String {
        match typ {
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
            Type::Str => "str".to_string(),
            Type::Struct(name) | Type::Enum(name) => name.clone(),
            Type::Ptr(inner) => format!("ptr_{}", Self::type_to_mangle_string(inner)),
            Type::Ref(inner) => format!("ref_{}", Self::type_to_mangle_string(inner)),
            Type::RefMut(inner) => format!("refmut_{}", Self::type_to_mangle_string(inner)),
            Type::Array(inner, size) => format!("arr_{}_{}", Self::type_to_mangle_string(inner), size),
            Type::Slice(inner) => format!("slice_{}", Self::type_to_mangle_string(inner)),
            Type::Optional(inner) => format!("opt_{}", Self::type_to_mangle_string(inner)),
            _ => "unknown".to_string(),
        }
    }
}

pub struct Compiler<'a> {
    pub statements: Iter<'a, Statement>,
    pub symbol_table: SymbolTable,
    pub typed_mode: bool,
    pub struct_defs: HashMap<String, CompiledStruct>,
    pub enum_defs: HashMap<String, CompiledEnum>,
    pub defer_stack: Vec<Statement>,
    pub loop_breaks: Vec<Vec<usize>>,
    pub loop_continues: Vec<usize>,
    pub borrow_checker: BorrowChecker,
    pub scope_owned_values: Vec<Vec<String>>,
    pub moved_symbols: HashSet<String>,
    pub suppress_move_marking: bool,
    pub native_names: Vec<String>,
    pub comptime_constants: HashMap<String, i64>,
    pub function_return_types: HashMap<String, Type>,
    pub generic_functions: HashMap<String, GenericFunctionDef>,
    pub monomorphized_functions: HashSet<String>,
    pub base_path: Option<PathBuf>,
    pub imported_files: HashSet<PathBuf>,
}

impl<'a> Compiler<'a> {
    pub fn new(statements: &'a [Statement]) -> Self {
        let mut symbol_table = SymbolTable::new();
        let builtins = [
            "len",
            "first",
            "last",
            "rest",
            "push",
            "print",
            "abs",
            "min",
            "max",
            "substr",
            "contains",
            "to_string",
            "parse_int",
            "floor",
            "ceil",
            "sqrt",
            "read_file",
            "write_file",
            "file_exists",
            "append_file",
            "vec2",
            "vec3",
            "vec4",
            "quat",
            "quat_identity",
            "vec2_add",
            "vec2_sub",
            "vec2_mul",
            "vec2_div",
            "vec2_dot",
            "vec2_len",
            "vec2_normalize",
            "vec3_add",
            "vec3_sub",
            "vec3_mul",
            "vec3_div",
            "vec3_dot",
            "vec3_cross",
            "vec3_len",
            "vec3_normalize",
            "vec4_add",
            "vec4_sub",
            "vec4_mul",
            "vec4_div",
            "vec4_dot",
            "vec4_len",
            "vec4_normalize",
            "quat_mul",
            "quat_normalize",
            "quat_from_euler",
            "quat_rotate_vec3",
            "arena_new",
            "arena_alloc",
            "arena_reset",
            "arena_get",
            "pool_new",
            "pool_alloc",
            "pool_get",
            "pool_free",
            "handle_index",
            "handle_generation",
            "char_at",
            "assert",
            "alloc",
            "temp_alloc",
        ];
        for (index, name) in builtins.iter().enumerate() {
            symbol_table.define_builtin(index, name);
        }
        Self {
            statements: statements.iter(),
            symbol_table,
            typed_mode: true,
            struct_defs: HashMap::new(),
            enum_defs: HashMap::new(),
            defer_stack: Vec::new(),
            loop_breaks: Vec::new(),
            loop_continues: Vec::new(),
            borrow_checker: BorrowChecker::new(),
            scope_owned_values: vec![Vec::new()],
            moved_symbols: HashSet::new(),
            suppress_move_marking: false,
            native_names: Vec::new(),
            comptime_constants: HashMap::new(),
            function_return_types: HashMap::new(),
            generic_functions: HashMap::new(),
            monomorphized_functions: HashSet::new(),
            base_path: None,
            imported_files: HashSet::new(),
        }
    }

    pub fn new_with_path(statements: &'a [Statement], base_path: PathBuf) -> Self {
        let mut compiler = Self::new(statements);
        compiler.base_path = Some(base_path);
        compiler
    }

    pub fn new_with_natives(
        statements: &'a [Statement],
        native_names: &[&str],
    ) -> Self {
        let mut compiler = Self::new(statements);
        for (index, name) in native_names.iter().enumerate() {
            compiler.symbol_table.define_native(index, name);
            compiler.native_names.push(name.to_string());
        }
        compiler
    }

    #[deprecated(
        note = "typed mode is now the default, use Compiler::new() instead"
    )]
    pub fn new_typed(statements: &'a [Statement]) -> Self {
        Self::new(statements)
    }

    pub fn new_untyped(statements: &'a [Statement]) -> Self {
        let mut symbol_table = SymbolTable::new();
        let builtins = [
            "len",
            "first",
            "last",
            "rest",
            "push",
            "print",
            "abs",
            "min",
            "max",
            "substr",
            "contains",
            "to_string",
            "parse_int",
            "floor",
            "ceil",
            "sqrt",
            "read_file",
            "write_file",
            "file_exists",
            "append_file",
            "vec2",
            "vec3",
            "vec4",
            "quat",
            "quat_identity",
            "vec2_add",
            "vec2_sub",
            "vec2_mul",
            "vec2_div",
            "vec2_dot",
            "vec2_len",
            "vec2_normalize",
            "vec3_add",
            "vec3_sub",
            "vec3_mul",
            "vec3_div",
            "vec3_dot",
            "vec3_cross",
            "vec3_len",
            "vec3_normalize",
            "vec4_add",
            "vec4_sub",
            "vec4_mul",
            "vec4_div",
            "vec4_dot",
            "vec4_len",
            "vec4_normalize",
            "quat_mul",
            "quat_normalize",
            "quat_from_euler",
            "quat_rotate_vec3",
            "arena_new",
            "arena_alloc",
            "arena_reset",
            "arena_get",
            "pool_new",
            "pool_alloc",
            "pool_get",
            "pool_free",
            "handle_index",
            "handle_generation",
            "char_at",
            "assert",
            "alloc",
            "temp_alloc",
        ];
        for (index, name) in builtins.iter().enumerate() {
            symbol_table.define_builtin(index, name);
        }
        Self {
            statements: statements.iter(),
            symbol_table,
            typed_mode: false,
            struct_defs: HashMap::new(),
            enum_defs: HashMap::new(),
            defer_stack: Vec::new(),
            loop_breaks: Vec::new(),
            loop_continues: Vec::new(),
            borrow_checker: BorrowChecker::new(),
            scope_owned_values: vec![Vec::new()],
            moved_symbols: HashSet::new(),
            suppress_move_marking: false,
            native_names: Vec::new(),
            comptime_constants: HashMap::new(),
            function_return_types: HashMap::new(),
            generic_functions: HashMap::new(),
            monomorphized_functions: HashSet::new(),
            base_path: None,
            imported_files: HashSet::new(),
        }
    }

    pub fn new_with_state(
        statements: &'a [Statement],
        symbol_table: SymbolTable,
    ) -> Self {
        Self {
            statements: statements.iter(),
            symbol_table,
            typed_mode: true,
            struct_defs: HashMap::new(),
            enum_defs: HashMap::new(),
            defer_stack: Vec::new(),
            loop_breaks: Vec::new(),
            loop_continues: Vec::new(),
            borrow_checker: BorrowChecker::new(),
            scope_owned_values: vec![Vec::new()],
            moved_symbols: HashSet::new(),
            suppress_move_marking: false,
            native_names: Vec::new(),
            comptime_constants: HashMap::new(),
            function_return_types: HashMap::new(),
            generic_functions: HashMap::new(),
            monomorphized_functions: HashSet::new(),
            base_path: None,
            imported_files: HashSet::new(),
        }
    }

    pub fn compile(&mut self) -> Result<Bytecode> {
        let mut bytecode = Bytecode::default();
        while let Some(statement) = self.statements.next() {
            self.compile_statement(statement, &mut bytecode)?;
        }
        Ok(bytecode)
    }

    fn resolve_type(&self, typ: &Type) -> Type {
        match typ {
            Type::Struct(name) if self.enum_defs.contains_key(name) => {
                Type::Enum(name.clone())
            }
            other => other.clone(),
        }
    }

    fn type_contains_type_param(typ: &Type) -> bool {
        match typ {
            Type::TypeParam(_) => true,
            Type::Ptr(inner) | Type::Ref(inner) | Type::RefMut(inner)
            | Type::Slice(inner) | Type::Distinct(inner)
            | Type::Handle(inner) | Type::Optional(inner) => {
                Self::type_contains_type_param(inner)
            }
            Type::Array(inner, _) => Self::type_contains_type_param(inner),
            Type::Proc(params, ret) => {
                params.iter().any(Self::type_contains_type_param)
                    || Self::type_contains_type_param(ret)
            }
            _ => false,
        }
    }

    fn function_is_generic(parameters: &[Parameter], return_sig: &ReturnSignature) -> bool {
        for param in parameters {
            if let Some(ref typ) = param.type_annotation {
                if Self::type_contains_type_param(typ) {
                    return true;
                }
            }
        }
        if let ReturnSignature::Single(typ) = return_sig {
            if Self::type_contains_type_param(typ) {
                return true;
            }
        }
        if let ReturnSignature::Named(params) = return_sig {
            for param in params {
                if Self::type_contains_type_param(&param.param_type) {
                    return true;
                }
            }
        }
        false
    }

    fn extract_type_params(parameters: &[Parameter]) -> Vec<String> {
        let mut type_params = Vec::new();
        for param in parameters {
            if let Some(Type::TypeParam(name)) = &param.type_annotation {
                if !type_params.contains(name) {
                    type_params.push(name.clone());
                }
            }
        }
        type_params
    }

    fn validate_and_store_return_type(&mut self, name: &str, return_sig: &ReturnSignature) -> Result<()> {
        if let Some(bad_type) = return_sig.has_second_class() {
            bail!(
                "function '{}' cannot return reference type '{}' - references are second-class and cannot be returned",
                name, bad_type
            );
        }
        if let Some(bad_type) = return_sig.contains_reference() {
            bail!(
                "function '{}' cannot return type '{}' which contains a reference - references are second-class",
                name, bad_type
            );
        }
        if let Some(ret_type) = return_sig.to_type() {
            self.function_return_types.insert(name.to_string(), self.resolve_type(&ret_type));
        }
        Ok(())
    }

    fn enter_scope(&mut self) {
        self.borrow_checker.enter_scope();
        self.scope_owned_values.push(Vec::new());
    }

    fn exit_scope(&mut self, bytecode: &mut Bytecode) {
        self.borrow_checker.exit_scope();
        if let Some(owned) = self.scope_owned_values.pop() {
            for name in owned.iter().rev() {
                if !self.moved_symbols.contains(name) {
                    if let Some(symbol) = self.symbol_table.resolve(name) {
                        let should_drop = match &symbol.symbol_type {
                            Some(Type::Slice(_)) => false,
                            Some(Type::Enum(_)) => false,
                            _ => true,
                        };
                        if should_drop {
                            self.load_symbol(&symbol, bytecode);
                            bytecode
                                .instructions
                                .push(Instruction::new(Opcode::Drop, vec![]));
                        }
                    }
                }
            }
            for name in owned {
                self.moved_symbols.remove(&name);
            }
        }
    }

    fn compile_statement(
        &mut self,
        statement: &Statement,
        bytecode: &mut Bytecode,
    ) -> Result<()> {
        match statement {
            Statement::Expression(expression) => {
                let needs_pop = !matches!(
                    expression,
                    Expression::ComptimeBlock(_)
                        | Expression::ComptimeFor { .. }
                );
                self.compile_expression(expression, bytecode)?;
                if needs_pop {
                    bytecode
                        .instructions
                        .push(Instruction::new(Opcode::Pop, vec![]));
                }
                Ok(())
            }
            Statement::Let {
                name,
                type_annotation,
                value,
                mutable,
            } => {
                let inferred_type =
                    if let Expression::StructInit(struct_name, _) = value {
                        Some(Type::Struct(struct_name.clone()))
                    } else if let Expression::EnumVariantInit(enum_name, _, _) =
                        value
                    {
                        Some(Type::Enum(enum_name.clone()))
                    } else if let Expression::Literal(Literal::Array(elements)) = value {
                        if let Some(first) = elements.first() {
                            if let Expression::StructInit(struct_name, _) = first {
                                Some(Type::Slice(Box::new(Type::Struct(struct_name.clone()))))
                            } else {
                                type_annotation.clone()
                            }
                        } else {
                            type_annotation.clone()
                        }
                    } else if let Expression::Index(array_expr, _) = value {
                        if let Expression::Identifier(array_name) = array_expr.as_ref() {
                            if let Some(symbol) = self.symbol_table.store.get(array_name) {
                                if let Some(symbol_type) = &symbol.symbol_type {
                                    match symbol_type {
                                        Type::Array(inner, _) | Type::Slice(inner) => {
                                            Some(inner.as_ref().clone())
                                        }
                                        _ => type_annotation.clone(),
                                    }
                                } else {
                                    type_annotation.clone()
                                }
                            } else {
                                type_annotation.clone()
                            }
                        } else if let Expression::FieldAccess(base_expr, field_name) = array_expr.as_ref() {
                            self.infer_field_access_element_type(base_expr, field_name)
                                .or_else(|| type_annotation.clone())
                        } else {
                            type_annotation.clone()
                        }
                    } else if let Expression::Call(func_expr, _) = value {
                        if let Expression::Identifier(func_name) = func_expr.as_ref() {
                            self.function_return_types.get(func_name).cloned()
                                .or_else(|| type_annotation.clone())
                        } else {
                            type_annotation.clone()
                        }
                    } else if let Expression::FieldAccess(base_expr, field_name) = value {
                        self.infer_field_access_type(base_expr, field_name)
                            .or_else(|| type_annotation.clone())
                    } else {
                        type_annotation.clone()
                    };
                let needs_drop =
                    inferred_type.as_ref().is_some_and(|t| t.needs_drop());
                let symbol = self.symbol_table.define_with_type_and_mutability(
                    name,
                    inferred_type,
                    *mutable,
                );
                if let Expression::Function(parameters, return_sig, body) = value {
                    self.validate_and_store_return_type(name, return_sig)?;
                    self.compile_function_with_name(
                        name, parameters, return_sig, body, bytecode,
                    )?;
                } else if let Expression::Proc(parameters, return_sig, body) = value {
                    self.validate_and_store_return_type(name, return_sig)?;
                    self.compile_function_with_name(
                        name, parameters, return_sig, body, bytecode,
                    )?;
                } else {
                    self.compile_expression(value, bytecode)?;
                }
                let opcode = if symbol.scope == SymbolScope::Global {
                    bytecode.global_symbols.insert(name.clone(), symbol.index);
                    Opcode::SetGlobal
                } else {
                    Opcode::SetLocal
                };
                bytecode
                    .instructions
                    .push(Instruction::new(opcode, vec![symbol.index as u16]));
                if needs_drop && symbol.scope == SymbolScope::Local {
                    if let Some(owned) = self.scope_owned_values.last_mut() {
                        owned.push(name.clone());
                    }
                }
                Ok(())
            }
            Statement::Constant(name, expression) => {
                if let Expression::Function(parameters, return_sig, body)
                    | Expression::Proc(parameters, return_sig, body) = expression
                {
                    if Self::function_is_generic(parameters, return_sig) {
                        let type_params = Self::extract_type_params(parameters);
                        self.generic_functions.insert(
                            name.clone(),
                            GenericFunctionDef {
                                type_params,
                                parameters: parameters.clone(),
                                return_sig: return_sig.clone(),
                                body: body.clone(),
                            },
                        );
                        return Ok(());
                    }
                    let symbol = self.symbol_table.define(name);
                    self.validate_and_store_return_type(name, return_sig)?;
                    self.compile_function_with_name(
                        name, parameters, return_sig, body, bytecode,
                    )?;
                    let opcode = if symbol.scope == SymbolScope::Global {
                        bytecode.global_symbols.insert(name.clone(), symbol.index);
                        Opcode::SetGlobal
                    } else {
                        Opcode::SetLocal
                    };
                    bytecode
                        .instructions
                        .push(Instruction::new(opcode, vec![symbol.index as u16]));
                    return Ok(());
                }
                let symbol = self.symbol_table.define(name);
                self.compile_expression(expression, bytecode)?;
                let opcode = if symbol.scope == SymbolScope::Global {
                    bytecode.global_symbols.insert(name.clone(), symbol.index);
                    Opcode::SetGlobal
                } else {
                    Opcode::SetLocal
                };
                bytecode
                    .instructions
                    .push(Instruction::new(opcode, vec![symbol.index as u16]));
                Ok(())
            }
            Statement::Struct(name, _type_params, fields) => {
                for field in fields.iter() {
                    if field.field_type.is_second_class() {
                        bail!(
                            "struct '{}' field '{}' has reference type '{}' - references are second-class and cannot be stored in structs",
                            name, field.name, field.field_type
                        );
                    }
                    if field.field_type.contains_reference() {
                        bail!(
                            "struct '{}' field '{}' has type '{}' which contains a reference - references are second-class and cannot be stored",
                            name, field.name, field.field_type
                        );
                    }
                }
                let mut field_offsets = HashMap::new();
                let mut byte_size = 0usize;
                for (index, field) in fields.iter().enumerate() {
                    field_offsets.insert(field.name.clone(), index);
                    byte_size += field.field_type.size_of();
                }
                let compiled_struct = CompiledStruct {
                    name: name.clone(),
                    fields: fields.clone(),
                    field_offsets,
                    size: byte_size,
                };
                self.struct_defs.insert(name.clone(), compiled_struct);
                Ok(())
            }
            Statement::Enum(name, variants) => {
                for variant in variants.iter() {
                    if let Some(fields) = &variant.fields {
                        for field in fields {
                            if field.field_type.is_second_class() {
                                bail!(
                                    "enum '{}' variant '{}' field '{}' has reference type '{}' - references are second-class and cannot be stored",
                                    name, variant.name, field.name, field.field_type
                                );
                            }
                            if field.field_type.contains_reference() {
                                bail!(
                                    "enum '{}' variant '{}' field '{}' has type '{}' which contains a reference - references are second-class and cannot be stored",
                                    name, variant.name, field.name, field.field_type
                                );
                            }
                        }
                    }
                }
                let mut compiled_variants = Vec::new();
                for (index, variant) in variants.iter().enumerate() {
                    let full_name = format!("{}::{}", name, variant.name);
                    let fields = variant.fields.clone().unwrap_or_default();
                    let is_unit = fields.is_empty();

                    if !is_unit {
                        self.symbol_table.define_with_type(
                            &full_name,
                            Some(Type::Enum(name.clone())),
                        );
                    }

                    compiled_variants.push(CompiledEnumVariant {
                        name: variant.name.clone(),
                        tag: index,
                        fields: fields.clone(),
                    });

                }
                self.enum_defs.insert(
                    name.clone(),
                    CompiledEnum {
                        name: name.clone(),
                        variants: compiled_variants,
                    },
                );
                Ok(())
            }
            Statement::TypeAlias(_name, _typ) => Ok(()),
            Statement::Defer(statement) => {
                self.defer_stack.push(statement.as_ref().clone());
                Ok(())
            }
            Statement::Return(expression) => {
                self.compile_expression(expression, bytecode)?;
                let deferred = self.defer_stack.clone();
                for deferred_stmt in deferred.iter().rev() {
                    self.compile_statement(deferred_stmt, bytecode)?;
                }
                bytecode
                    .instructions
                    .push(Instruction::new(Opcode::ReturnValue, vec![]));
                Ok(())
            }
            Statement::Assignment(lhs, rhs) => {
                match lhs {
                    Expression::Dereference(ptr_expr) => {
                        self.compile_expression(ptr_expr, bytecode)?;
                        self.compile_expression(rhs, bytecode)?;
                        bytecode
                            .instructions
                            .push(Instruction::new(Opcode::StorePtr, vec![]));
                    }
                    Expression::Identifier(name) => {
                        let symbol = self
                            .symbol_table
                            .resolve(name)
                            .ok_or_else(|| {
                                anyhow::anyhow!("undefined variable: {}", name)
                            })?;
                        if !symbol.mutable {
                            anyhow::bail!("cannot assign to immutable variable '{}'. Use 'mut' to declare mutable variables.", name);
                        }
                        self.compile_expression(rhs, bytecode)?;
                        let opcode = match symbol.scope {
                            SymbolScope::Local => Opcode::SetLocal,
                            SymbolScope::Global => Opcode::SetGlobal,
                            _ => anyhow::bail!(
                                "cannot assign to {:?}",
                                symbol.scope
                            ),
                        };
                        bytecode.instructions.push(Instruction::new(
                            opcode,
                            vec![symbol.index as u16],
                        ));
                        self.moved_symbols.remove(name);
                        if let Some(ref symbol_type) = symbol.symbol_type {
                            if symbol_type.needs_drop() {
                                if let Some(scope) = self.scope_owned_values.last_mut() {
                                    if !scope.contains(&name.to_string()) {
                                        scope.push(name.clone());
                                    }
                                }
                            }
                        }
                    }
                    Expression::FieldAccess(expr, field) => {
                        if let Expression::Identifier(name) = expr.as_ref() {
                            let symbol = self
                                .symbol_table
                                .resolve(name)
                                .ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "undefined variable: {}",
                                        name
                                    )
                                })?;
                            let get_opcode = match symbol.scope {
                                SymbolScope::Local => Opcode::GetLocal,
                                SymbolScope::Global => Opcode::GetGlobal,
                                _ => anyhow::bail!(
                                    "cannot get field of {:?}",
                                    symbol.scope
                                ),
                            };
                            let set_opcode = match symbol.scope {
                                SymbolScope::Local => Opcode::SetLocal,
                                SymbolScope::Global => Opcode::SetGlobal,
                                _ => anyhow::bail!(
                                    "cannot set field of {:?}",
                                    symbol.scope
                                ),
                            };
                            let (struct_name, is_ref) = match &symbol.symbol_type {
                                Some(Type::Struct(name)) => (Some(name.clone()), false),
                                Some(Type::RefMut(inner)) => {
                                    if let Type::Struct(name) = inner.as_ref() {
                                        (Some(name.clone()), true)
                                    } else {
                                        (None, false)
                                    }
                                }
                                _ => (None, false),
                            };
                            let offset = if let Some(struct_name) = &struct_name {
                                if let Some(struct_def) =
                                    self.struct_defs.get(struct_name)
                                {
                                    *struct_def
                                        .field_offsets
                                        .get(field)
                                        .ok_or_else(|| {
                                            anyhow::anyhow!(
                                                "unknown field: {}",
                                                field
                                            )
                                        })?
                                } else {
                                    0
                                }
                            } else {
                                0
                            };
                            bytecode.instructions.push(Instruction::new(
                                get_opcode,
                                vec![symbol.index as u16],
                            ));
                            if is_ref {
                                bytecode.instructions.push(Instruction::new(
                                    Opcode::LoadPtr,
                                    vec![],
                                ));
                            }
                            self.compile_expression(rhs, bytecode)?;
                            bytecode.instructions.push(Instruction::new(
                                Opcode::StructSet,
                                vec![offset as u16],
                            ));
                            if is_ref {
                                bytecode.instructions.push(Instruction::new(
                                    Opcode::Pop,
                                    vec![],
                                ));
                            } else {
                                bytecode.instructions.push(Instruction::new(
                                    set_opcode,
                                    vec![symbol.index as u16],
                                ));
                            }
                        } else {
                            anyhow::bail!(
                                "can only assign to fields of identifiers"
                            );
                        }
                    }
                    Expression::Index(arr_expr, index_expr) => {
                        if let Expression::Identifier(name) = arr_expr.as_ref() {
                            let symbol = self
                                .symbol_table
                                .resolve(name)
                                .ok_or_else(|| {
                                    anyhow::anyhow!("undefined variable: {}", name)
                                })?;
                            if !symbol.mutable {
                                anyhow::bail!(
                                    "cannot assign to index of immutable array '{}'. Use 'mut' to declare mutable arrays.",
                                    name
                                );
                            }
                            let get_opcode = match symbol.scope {
                                SymbolScope::Local => Opcode::GetLocal,
                                SymbolScope::Global => Opcode::GetGlobal,
                                _ => anyhow::bail!(
                                    "cannot index {:?}",
                                    symbol.scope
                                ),
                            };
                            bytecode.instructions.push(Instruction::new(
                                get_opcode,
                                vec![symbol.index as u16],
                            ));
                            self.compile_expression(index_expr, bytecode)?;
                            self.compile_expression(rhs, bytecode)?;
                            bytecode
                                .instructions
                                .push(Instruction::new(Opcode::IndexSet, vec![]));
                        } else {
                            self.compile_expression(arr_expr, bytecode)?;
                            self.compile_expression(index_expr, bytecode)?;
                            self.compile_expression(rhs, bytecode)?;
                            bytecode
                                .instructions
                                .push(Instruction::new(Opcode::IndexSet, vec![]));
                        }
                    }
                    _ => anyhow::bail!("invalid assignment target"),
                }
                Ok(())
            }
            Statement::For(iterator, range, body) => {
                if let Expression::Range(start, end, inclusive) = range {
                    self.compile_expression(start, bytecode)?;
                    let symbol = self
                        .symbol_table
                        .define_with_type(iterator, Some(Type::I64));
                    let (set_opcode, get_opcode) = match symbol.scope {
                        SymbolScope::Local => {
                            (Opcode::SetLocal, Opcode::GetLocal)
                        }
                        SymbolScope::Global => {
                            (Opcode::SetGlobal, Opcode::GetGlobal)
                        }
                        _ => anyhow::bail!(
                            "unexpected scope for for loop iterator"
                        ),
                    };
                    bytecode.instructions.push(Instruction::new(
                        set_opcode,
                        vec![symbol.index as u16],
                    ));

                    let loop_start = bytecode.instructions.len();
                    self.loop_breaks.push(Vec::new());
                    self.loop_continues.push(loop_start);

                    self.compile_expression(end, bytecode)?;
                    bytecode.instructions.push(Instruction::new(
                        get_opcode,
                        vec![symbol.index as u16],
                    ));
                    let compare_opcode = if *inclusive {
                        Opcode::GreaterThanOrEqualI64
                    } else {
                        Opcode::GreaterThanI64
                    };
                    bytecode
                        .instructions
                        .push(Instruction::new(compare_opcode, vec![]));

                    let jump_end_pos = bytecode.instructions.len();
                    bytecode.instructions.push(Instruction::new(
                        Opcode::JumpNotTruthy,
                        vec![9999],
                    ));

                    self.enter_scope();
                    for statement in body {
                        self.compile_statement(statement, bytecode)?;
                    }
                    self.exit_scope(bytecode);

                    bytecode.instructions.push(Instruction::new(
                        get_opcode,
                        vec![symbol.index as u16],
                    ));
                    let one_index = bytecode.constants.len() as u16;
                    bytecode.constants.push(Value64::Integer(1));
                    bytecode.instructions.push(Instruction::new(
                        Opcode::Constant,
                        vec![one_index],
                    ));
                    bytecode
                        .instructions
                        .push(Instruction::new(Opcode::AddI64, vec![]));
                    bytecode.instructions.push(Instruction::new(
                        set_opcode,
                        vec![symbol.index as u16],
                    ));

                    bytecode.instructions.push(Instruction::new(
                        Opcode::Jump,
                        vec![loop_start as u16],
                    ));

                    let after_loop = bytecode.instructions.len();
                    bytecode.instructions[jump_end_pos].operands[0] =
                        after_loop as u16;

                    let break_positions =
                        self.loop_breaks.pop().unwrap_or_default();
                    for break_pos in break_positions {
                        bytecode.instructions[break_pos].operands[0] =
                            after_loop as u16;
                    }
                    self.loop_continues.pop();

                    Ok(())
                } else {
                    anyhow::bail!("for loop requires a range expression")
                }
            }
            Statement::While(condition, body) => {
                let loop_start = bytecode.instructions.len();
                self.loop_breaks.push(Vec::new());
                self.loop_continues.push(loop_start);

                self.compile_expression(condition, bytecode)?;

                let jump_end_pos = bytecode.instructions.len();
                bytecode
                    .instructions
                    .push(Instruction::new(Opcode::JumpNotTruthy, vec![9999]));

                self.enter_scope();
                for statement in body {
                    self.compile_statement(statement, bytecode)?;
                }
                self.exit_scope(bytecode);

                bytecode.instructions.push(Instruction::new(
                    Opcode::Jump,
                    vec![loop_start as u16],
                ));

                let after_loop = bytecode.instructions.len();
                bytecode.instructions[jump_end_pos].operands[0] =
                    after_loop as u16;

                let break_positions =
                    self.loop_breaks.pop().unwrap_or_default();
                for break_pos in break_positions {
                    bytecode.instructions[break_pos].operands[0] =
                        after_loop as u16;
                }
                self.loop_continues.pop();

                Ok(())
            }
            Statement::Break => {
                if self.loop_breaks.is_empty() {
                    anyhow::bail!("break outside of loop");
                }
                let break_pos = bytecode.instructions.len();
                bytecode
                    .instructions
                    .push(Instruction::new(Opcode::Jump, vec![9999]));
                self.loop_breaks.last_mut().unwrap().push(break_pos);
                Ok(())
            }
            Statement::Continue => {
                if self.loop_continues.is_empty() {
                    anyhow::bail!("continue outside of loop");
                }
                let continue_target = *self.loop_continues.last().unwrap();
                bytecode.instructions.push(Instruction::new(
                    Opcode::Jump,
                    vec![continue_target as u16],
                ));
                Ok(())
            }
            Statement::Import(path) => {
                let resolved_path = if let Some(base) = &self.base_path {
                    base.join(path)
                } else {
                    PathBuf::from(path)
                };
                let canonical_path = resolved_path.canonicalize().map_err(|e| {
                    anyhow::anyhow!("failed to resolve import '{}': {}", path, e)
                })?;
                if self.imported_files.contains(&canonical_path) {
                    return Ok(());
                }
                self.imported_files.insert(canonical_path.clone());
                let source = std::fs::read_to_string(&canonical_path).map_err(|e| {
                    anyhow::anyhow!("failed to read import '{}': {}", path, e)
                })?;
                let mut lexer = crate::Lexer::new(&source);
                let tokens = lexer.tokenize().map_err(|e| {
                    anyhow::anyhow!(
                        "failed to tokenize import '{}': {}",
                        path,
                        e
                    )
                })?;
                let mut parser = crate::Parser::new(&tokens);
                let statements = parser.parse().map_err(|e| {
                    anyhow::anyhow!("failed to parse import '{}': {}", path, e)
                })?;
                let old_base = self.base_path.take();
                self.base_path = canonical_path.parent().map(|p| p.to_path_buf());
                for statement in &statements {
                    self.compile_statement(statement, bytecode)?;
                }
                self.base_path = old_base;
                Ok(())
            }
            Statement::InterpolatedConstant(_, _) => {
                anyhow::bail!("InterpolatedConstant can only be used inside comptime blocks")
            }
            Statement::Extern { .. } => {
                Ok(())
            }
            Statement::PushContext { context_expr, body } => {
                bytecode.instructions.push(Instruction::new(Opcode::PushContextScope, vec![]));
                self.compile_expression(context_expr, bytecode)?;
                bytecode.instructions.push(Instruction::new(Opcode::SetContext, vec![]));
                for statement in body {
                    self.compile_statement(statement, bytecode)?;
                }
                bytecode.instructions.push(Instruction::new(Opcode::PopContextScope, vec![]));
                Ok(())
            }
            Statement::PushAllocator { allocator_expr, body } => {
                bytecode.instructions.push(Instruction::new(Opcode::PushContextScope, vec![]));
                self.compile_expression(allocator_expr, bytecode)?;
                bytecode.instructions.push(Instruction::new(Opcode::SetContextField, vec![0]));
                for statement in body {
                    self.compile_statement(statement, bytecode)?;
                }
                bytecode.instructions.push(Instruction::new(Opcode::PopContextScope, vec![]));
                Ok(())
            }
        }
    }

    fn compile_expression(
        &mut self,
        expression: &Expression,
        bytecode: &mut Bytecode,
    ) -> Result<()> {
        match expression {
            Expression::Identifier(name) => {
                if self.moved_symbols.contains(name) {
                    anyhow::bail!("use of moved value: '{}'", name);
                }
                let symbol =
                    self.symbol_table.resolve(name).ok_or_else(|| {
                        anyhow::anyhow!("undefined variable: {}", name)
                    })?;
                if let Some(ref symbol_type) = symbol.symbol_type {
                    if !symbol_type.is_copy() {
                        let is_unit_variant = if name.contains("::") {
                            if let Some((enum_name, variant_name)) = name.split_once("::") {
                                self.enum_defs.get(enum_name).map_or(false, |enum_def| {
                                    enum_def.variants.iter()
                                        .find(|v| v.name == variant_name)
                                        .map_or(false, |v| v.fields.is_empty())
                                })
                            } else {
                                false
                            }
                        } else {
                            false
                        };
                        if !is_unit_variant && !self.suppress_move_marking {
                            self.moved_symbols.insert(name.clone());
                        }
                    }
                }
                self.load_symbol(&symbol, bytecode);
                Ok(())
            }
            Expression::Literal(literal) => {
                self.compile_literal(literal, bytecode)
            }
            Expression::Infix(left, operator, right) => {
                self.compile_infix(left, operator, right, bytecode)
            }
            Expression::Boolean(value) => {
                let opcode = if *value { Opcode::True } else { Opcode::False };
                bytecode.instructions.push(Instruction::new(opcode, vec![]));
                Ok(())
            }
            Expression::Prefix(operator, operand) => {
                self.compile_expression(operand, bytecode)?;
                let opcode = match operator {
                    Operator::Negate => Opcode::Minus,
                    Operator::Not => Opcode::Bang,
                    _ => unimplemented!(
                        "Prefix operator {:?} not implemented",
                        operator
                    ),
                };
                bytecode.instructions.push(Instruction::new(opcode, vec![]));
                Ok(())
            }
            Expression::Index(left, index) => {
                let was_suppressing = self.suppress_move_marking;
                self.suppress_move_marking = true;
                self.compile_expression(left, bytecode)?;
                self.suppress_move_marking = was_suppressing;
                self.compile_expression(index, bytecode)?;
                bytecode
                    .instructions
                    .push(Instruction::new(Opcode::Index, vec![]));
                Ok(())
            }
            Expression::If(condition, consequence, alternative) => {
                self.compile_expression(condition, bytecode)?;

                let jump_not_truthy_pos = bytecode.instructions.len();
                bytecode
                    .instructions
                    .push(Instruction::new(Opcode::JumpNotTruthy, vec![9999]));

                let moved_before = self.moved_symbols.clone();
                for statement in consequence {
                    self.compile_statement(statement, bytecode)?;
                }
                self.remove_last_pop(bytecode);
                let moved_in_consequence = self.moved_symbols.clone();

                let jump_pos = bytecode.instructions.len();
                bytecode
                    .instructions
                    .push(Instruction::new(Opcode::Jump, vec![9999]));

                let after_consequence = bytecode.instructions.len();
                bytecode.instructions[jump_not_truthy_pos].operands[0] =
                    after_consequence as u16;

                self.moved_symbols = moved_before.clone();
                if let Some(alt) = alternative {
                    for statement in alt {
                        self.compile_statement(statement, bytecode)?;
                    }
                    self.remove_last_pop(bytecode);
                } else {
                    bytecode
                        .instructions
                        .push(Instruction::new(Opcode::Null, vec![]));
                }
                let moved_in_alternative = self.moved_symbols.clone();

                self.moved_symbols = moved_before;
                for sym in moved_in_consequence.intersection(&moved_in_alternative) {
                    self.moved_symbols.insert(sym.clone());
                }

                let after_alternative = bytecode.instructions.len();
                bytecode.instructions[jump_pos].operands[0] =
                    after_alternative as u16;

                Ok(())
            }
            Expression::Function(parameters, return_sig, body)
            | Expression::Proc(parameters, return_sig, body) => {
                if let Some(bad_type) = return_sig.has_second_class() {
                    bail!(
                        "function cannot return reference type '{}' - references are second-class and cannot be returned",
                        bad_type
                    );
                }
                if let Some(bad_type) = return_sig.contains_reference() {
                    bail!(
                        "function cannot return type '{}' which contains a reference - references are second-class",
                        bad_type
                    );
                }
                let outer_symbol_table = std::mem::take(&mut self.symbol_table);
                let outer_moved_symbols = std::mem::take(&mut self.moved_symbols);
                self.symbol_table =
                    SymbolTable::new_enclosed(outer_symbol_table);

                for param in parameters {
                    self.symbol_table.define_with_type_and_mutability(
                        &param.name,
                        param.type_annotation.clone(),
                        param.mutable,
                    );
                }

                if let Some(named_params) = return_sig.named_params() {
                    for return_param in named_params {
                        self.symbol_table.define_with_type_and_mutability(
                            &return_param.name,
                            Some(return_param.param_type.clone()),
                            true,
                        );
                    }
                }

                let mut fn_bytecode = Bytecode {
                    instructions: vec![],
                    constants: std::mem::take(&mut bytecode.constants),
                    functions: std::mem::take(&mut bytecode.functions),
                    heap: std::mem::take(&mut bytecode.heap),
                    native_names: std::mem::take(&mut bytecode.native_names),
                    global_symbols: std::mem::take(&mut bytecode.global_symbols),
                };
                for statement in body {
                    self.compile_statement(statement, &mut fn_bytecode)?;
                }

                if self.last_instruction_is(&fn_bytecode, Opcode::Pop) {
                    self.replace_last_pop_with_return(&mut fn_bytecode);
                }

                if fn_bytecode.instructions.is_empty()
                    || !self
                        .last_instruction_is(&fn_bytecode, Opcode::ReturnValue)
                {
                    if let Some(named_params) = return_sig.named_params() {
                        if named_params.len() == 1 {
                            if let Some(symbol) = self.symbol_table.resolve(&named_params[0].name) {
                                fn_bytecode.instructions.push(Instruction::new(
                                    Opcode::GetLocal,
                                    vec![symbol.index as u16],
                                ));
                                fn_bytecode
                                    .instructions
                                    .push(Instruction::new(Opcode::ReturnValue, vec![]));
                            } else {
                                fn_bytecode
                                    .instructions
                                    .push(Instruction::new(Opcode::Return, vec![]));
                            }
                        } else {
                            fn_bytecode
                                .instructions
                                .push(Instruction::new(Opcode::Return, vec![]));
                        }
                    } else {
                        fn_bytecode
                            .instructions
                            .push(Instruction::new(Opcode::Return, vec![]));
                    }
                }

                bytecode.constants = fn_bytecode.constants;
                bytecode.functions = fn_bytecode.functions;
                bytecode.heap = fn_bytecode.heap;
                bytecode.global_symbols = fn_bytecode.global_symbols;

                let free_symbols = self.symbol_table.free_symbols.clone();
                let num_locals = self.symbol_table.num_definitions;
                let num_parameters = parameters.len();

                if let Some(outer) = self.symbol_table.outer.take() {
                    self.symbol_table = *outer;
                }
                self.moved_symbols = outer_moved_symbols;

                for sym in &free_symbols {
                    self.load_symbol(sym, bytecode);
                }

                let compiled_fn = CompiledFunction {
                    instructions: fn_bytecode.instructions,
                    num_locals,
                    num_parameters,
                };
                let function_index = bytecode.functions.len() as u16;
                bytecode.functions.push(compiled_fn);
                bytecode.instructions.push(Instruction::new(
                    Opcode::Closure,
                    vec![function_index, free_symbols.len() as u16],
                ));

                Ok(())
            }
            Expression::FieldAccess(expr, field) => {
                if let Expression::Identifier(name) = expr.as_ref() {
                    if self.moved_symbols.contains(name) {
                        anyhow::bail!("use of moved value: '{}'", name);
                    }
                    let symbol =
                        self.symbol_table.resolve(name).ok_or_else(|| {
                            anyhow::anyhow!("undefined variable: {}", name)
                        })?;
                    self.load_symbol(&symbol, bytecode);
                    let (struct_name, is_ref) = match &symbol.symbol_type {
                        Some(Type::Struct(name)) => (Some(name.clone()), false),
                        Some(Type::Ref(inner)) => {
                            if let Type::Struct(name) = inner.as_ref() {
                                (Some(name.clone()), true)
                            } else {
                                (None, false)
                            }
                        }
                        Some(Type::RefMut(inner)) => {
                            if let Type::Struct(name) = inner.as_ref() {
                                (Some(name.clone()), true)
                            } else {
                                (None, false)
                            }
                        }
                        _ => (None, false),
                    };
                    if is_ref {
                        bytecode.instructions.push(Instruction::new(
                            Opcode::LoadPtr,
                            vec![],
                        ));
                    }
                    if let Some(struct_name) = struct_name {
                        if let Some(struct_def) =
                            self.struct_defs.get(&struct_name)
                        {
                            if let Some(&offset) =
                                struct_def.field_offsets.get(field)
                            {
                                bytecode.instructions.push(Instruction::new(
                                    Opcode::StructGet,
                                    vec![offset as u16],
                                ));
                                return Ok(());
                            }
                        }
                    }
                } else {
                    let was_suppressing = self.suppress_move_marking;
                    self.suppress_move_marking = true;
                    self.compile_expression(expr, bytecode)?;
                    self.suppress_move_marking = was_suppressing;
                    let struct_type = self.infer_expression_struct_type(expr);
                    if let Some(struct_name) = struct_type {
                        if let Some(struct_def) = self.struct_defs.get(&struct_name) {
                            if let Some(&offset) = struct_def.field_offsets.get(field) {
                                bytecode.instructions.push(Instruction::new(
                                    Opcode::StructGet,
                                    vec![offset as u16],
                                ));
                                return Ok(());
                            }
                        }
                    }
                }
                bytecode
                    .instructions
                    .push(Instruction::new(Opcode::StructGet, vec![0]));
                Ok(())
            }
            Expression::StructInit(struct_name, fields) => {
                if let Some(struct_def) =
                    self.struct_defs.get(struct_name).cloned()
                {
                    bytecode.instructions.push(Instruction::new(
                        Opcode::StructAlloc,
                        vec![struct_def.fields.len() as u16],
                    ));
                    for (field_name, value_expr) in fields {
                        if let Some(&offset) =
                            struct_def.field_offsets.get(field_name)
                        {
                            self.compile_expression(value_expr, bytecode)?;
                            bytecode.instructions.push(Instruction::new(
                                Opcode::StructSet,
                                vec![offset as u16],
                            ));
                        }
                    }
                } else {
                    anyhow::bail!("undefined struct: {}", struct_name);
                }
                Ok(())
            }
            Expression::AddressOf(expr) => {
                match expr.as_ref() {
                    Expression::Identifier(name) => {
                        let symbol = self
                            .symbol_table
                            .resolve(name)
                            .ok_or_else(|| {
                                anyhow::anyhow!("undefined variable: {}", name)
                            })?;
                        let opcode = match symbol.scope {
                            SymbolScope::Local => Opcode::AddressOfLocal,
                            SymbolScope::Global => Opcode::AddressOfGlobal,
                            _ => anyhow::bail!(
                                "cannot take address of {:?}",
                                symbol.scope
                            ),
                        };
                        bytecode.instructions.push(Instruction::new(
                            opcode,
                            vec![symbol.index as u16],
                        ));
                    }
                    _ => anyhow::bail!("can only take address of identifiers"),
                }
                Ok(())
            }
            Expression::Borrow(expr) => {
                match expr.as_ref() {
                    Expression::Identifier(name) => {
                        if self.moved_symbols.contains(name) {
                            anyhow::bail!(
                                "cannot borrow moved value: '{}'",
                                name
                            );
                        }
                        self.borrow_checker.try_borrow(name, false)?;
                        let symbol = self
                            .symbol_table
                            .resolve(name)
                            .ok_or_else(|| {
                                anyhow::anyhow!("undefined variable: {}", name)
                            })?;
                        let opcode = match symbol.scope {
                            SymbolScope::Local => Opcode::AddressOfLocal,
                            SymbolScope::Global => Opcode::AddressOfGlobal,
                            _ => anyhow::bail!(
                                "cannot borrow {:?}",
                                symbol.scope
                            ),
                        };
                        bytecode.instructions.push(Instruction::new(
                            opcode,
                            vec![symbol.index as u16],
                        ));
                    }
                    Expression::FieldAccess(base_expr, field) => {
                        let (base_name, total_offset) = self.compute_field_access_offset(base_expr, field)?;
                        if self.moved_symbols.contains(&base_name) {
                            anyhow::bail!("cannot borrow moved value: '{}'", base_name);
                        }
                        self.borrow_checker.try_borrow(&base_name, false)?;
                        let symbol = self.symbol_table.resolve(&base_name).ok_or_else(|| {
                            anyhow::anyhow!("undefined variable: {}", base_name)
                        })?;
                        let opcode = match symbol.scope {
                            SymbolScope::Local => Opcode::AddressOfLocal,
                            SymbolScope::Global => Opcode::AddressOfGlobal,
                            _ => anyhow::bail!("cannot borrow {:?}", symbol.scope),
                        };
                        bytecode.instructions.push(Instruction::new(opcode, vec![symbol.index as u16]));
                        if total_offset > 0 {
                            let constant_index = bytecode.constants.len() as u16;
                            bytecode.constants.push(Value64::Integer(total_offset as i64));
                            bytecode.instructions.push(Instruction::new(Opcode::Constant, vec![constant_index]));
                            bytecode.instructions.push(Instruction::new(Opcode::AddI64, vec![]));
                        }
                    }
                    _ => anyhow::bail!("can only borrow identifiers or field access"),
                }
                Ok(())
            }
            Expression::BorrowMut(expr) => {
                match expr.as_ref() {
                    Expression::Identifier(name) => {
                        if self.moved_symbols.contains(name) {
                            anyhow::bail!(
                                "cannot borrow moved value: '{}'",
                                name
                            );
                        }
                        let symbol = self
                            .symbol_table
                            .resolve(name)
                            .ok_or_else(|| {
                                anyhow::anyhow!("undefined variable: {}", name)
                            })?;
                        if !symbol.mutable {
                            anyhow::bail!(
                                "cannot mutably borrow immutable variable '{}'",
                                name
                            );
                        }
                        self.borrow_checker.try_borrow(name, true)?;
                        let opcode = match symbol.scope {
                            SymbolScope::Local => Opcode::AddressOfLocal,
                            SymbolScope::Global => Opcode::AddressOfGlobal,
                            _ => anyhow::bail!(
                                "cannot borrow {:?}",
                                symbol.scope
                            ),
                        };
                        bytecode.instructions.push(Instruction::new(
                            opcode,
                            vec![symbol.index as u16],
                        ));
                    }
                    Expression::FieldAccess(base_expr, field) => {
                        let (base_name, total_offset) = self.compute_field_access_offset(base_expr, field)?;
                        if self.moved_symbols.contains(&base_name) {
                            anyhow::bail!("cannot borrow moved value: '{}'", base_name);
                        }
                        let symbol = self.symbol_table.resolve(&base_name).ok_or_else(|| {
                            anyhow::anyhow!("undefined variable: {}", base_name)
                        })?;
                        if !symbol.mutable {
                            anyhow::bail!("cannot mutably borrow immutable variable '{}'", base_name);
                        }
                        self.borrow_checker.try_borrow(&base_name, true)?;
                        let opcode = match symbol.scope {
                            SymbolScope::Local => Opcode::AddressOfLocal,
                            SymbolScope::Global => Opcode::AddressOfGlobal,
                            _ => anyhow::bail!("cannot borrow {:?}", symbol.scope),
                        };
                        bytecode.instructions.push(Instruction::new(opcode, vec![symbol.index as u16]));
                        if total_offset > 0 {
                            let constant_index = bytecode.constants.len() as u16;
                            bytecode.constants.push(Value64::Integer(total_offset as i64));
                            bytecode.instructions.push(Instruction::new(Opcode::Constant, vec![constant_index]));
                            bytecode.instructions.push(Instruction::new(Opcode::AddI64, vec![]));
                        }
                    }
                    _ => anyhow::bail!("can only borrow identifiers or field access"),
                }
                Ok(())
            }
            Expression::Dereference(expr) => {
                self.compile_expression(expr, bytecode)?;
                bytecode
                    .instructions
                    .push(Instruction::new(Opcode::LoadPtr, vec![]));
                Ok(())
            }
            Expression::Call(function, arguments) => {
                if let Expression::Identifier(name) = function.as_ref() {
                    if let Some(generic_def) = self.generic_functions.get(name).cloned() {
                        let mut substitution = MonoSubstitution::new();
                        for (index, param) in generic_def.parameters.iter().enumerate() {
                            if let Some(Type::TypeParam(type_param_name)) = &param.type_annotation {
                                if index < arguments.len() {
                                    if let Some(inferred_type) = self.infer_expression_type(&arguments[index]) {
                                        substitution.bind(type_param_name, inferred_type);
                                    }
                                }
                            }
                        }
                        let mangled_name = substitution.mangle_name(name, &generic_def.type_params);
                        if !self.monomorphized_functions.contains(&mangled_name) {
                            self.monomorphized_functions.insert(mangled_name.clone());
                            let mono_params: Vec<Parameter> = generic_def.parameters.iter()
                                .map(|p| substitution.substitute_param(p))
                                .collect();
                            let mono_return_sig = substitution.substitute_return_sig(&generic_def.return_sig);
                            let mono_body: Vec<Statement> = generic_def.body.iter()
                                .map(|s| substitution.substitute_stmt(s))
                                .collect();
                            let symbol = self.symbol_table.define(&mangled_name);
                            self.validate_and_store_return_type(&mangled_name, &mono_return_sig)?;
                            self.compile_function_with_name(
                                &mangled_name, &mono_params, &mono_return_sig, &mono_body, bytecode,
                            )?;
                            let opcode = if symbol.scope == SymbolScope::Global {
                                bytecode.global_symbols.insert(mangled_name.clone(), symbol.index);
                                Opcode::SetGlobal
                            } else {
                                Opcode::SetLocal
                            };
                            bytecode.instructions.push(Instruction::new(opcode, vec![symbol.index as u16]));
                        }
                        if let Some(symbol) = self.symbol_table.resolve(&mangled_name) {
                            self.load_symbol(&symbol, bytecode);
                        }
                        let is_constructor = mangled_name.ends_with("_new") || mangled_name.starts_with("new_");
                        let was_suppressing = self.suppress_move_marking;
                        self.suppress_move_marking = true;
                        for arg in arguments {
                            self.compile_expression(arg, bytecode)?;
                            if is_constructor {
                                if let Expression::Identifier(arg_name) = arg {
                                    if let Some(sym) = self.symbol_table.resolve(arg_name) {
                                        if let Some(ref symbol_type) = sym.symbol_type {
                                            if !symbol_type.is_copy() {
                                                if let Some(scope) = self.scope_owned_values.last_mut() {
                                                    scope.retain(|n| n != arg_name);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        self.suppress_move_marking = was_suppressing;
                        bytecode.instructions.push(Instruction::new(
                            Opcode::Call,
                            vec![arguments.len() as u16],
                        ));
                        self.borrow_checker.active_borrows.clear();
                        return Ok(());
                    }
                }
                let is_constructor = if let Expression::Identifier(func_name) = function.as_ref() {
                    func_name.ends_with("_new") || func_name.starts_with("new_")
                } else {
                    false
                };
                self.compile_expression(function, bytecode)?;
                let was_suppressing = self.suppress_move_marking;
                self.suppress_move_marking = true;
                for arg in arguments {
                    self.compile_expression(arg, bytecode)?;
                    if is_constructor {
                        if let Expression::Identifier(name) = arg {
                            if let Some(symbol) = self.symbol_table.resolve(name) {
                                if let Some(ref symbol_type) = symbol.symbol_type {
                                    if !symbol_type.is_copy() {
                                        if let Some(scope) = self.scope_owned_values.last_mut() {
                                            scope.retain(|n| n != name);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                self.suppress_move_marking = was_suppressing;
                bytecode.instructions.push(Instruction::new(
                    Opcode::Call,
                    vec![arguments.len() as u16],
                ));
                self.borrow_checker.active_borrows.clear();
                Ok(())
            }
            Expression::Sizeof(typ) => {
                let size = if let Type::Struct(name) = typ {
                    self.struct_defs.get(name).map(|s| s.size).unwrap_or(0)
                } else {
                    typ.size_of()
                };
                let constant_index = bytecode.constants.len() as u16;
                bytecode.constants.push(Value64::Integer(size as i64));
                bytecode.instructions.push(Instruction::new(
                    Opcode::Constant,
                    vec![constant_index],
                ));
                Ok(())
            }
            Expression::Range(_, _, _) => {
                anyhow::bail!("range expressions can only be used in for loops")
            }
            Expression::Tuple(elements) => {
                for element in elements {
                    self.compile_expression(element, bytecode)?;
                }
                bytecode.instructions.push(Instruction::new(
                    Opcode::TupleAlloc,
                    vec![elements.len() as u16],
                ));
                Ok(())
            }
            Expression::EnumVariantInit(enum_name, variant_name, fields) => {
                let full_name = format!("{}::{}", enum_name, variant_name);
                if fields.is_empty() {
                    if let Some(symbol) = self.symbol_table.resolve(&full_name) {
                        self.load_symbol(&symbol, bytecode);
                    } else {
                        let tag = self.get_variant_tag(enum_name, variant_name);
                        let constant_index = bytecode.constants.len() as u16;
                        bytecode.constants.push(Value64::Integer(tag as i64));
                        bytecode.instructions.push(Instruction::new(
                            Opcode::Constant,
                            vec![constant_index],
                        ));
                    }
                } else {
                    let tag = self.get_variant_tag(enum_name, variant_name);
                    bytecode.instructions.push(Instruction::new(
                        Opcode::TaggedUnionAlloc,
                        vec![fields.len() as u16],
                    ));
                    bytecode.instructions.push(Instruction::new(
                        Opcode::TaggedUnionSetTag,
                        vec![tag as u16],
                    ));
                    let field_order =
                        self.get_variant_field_order(enum_name, variant_name);
                    for (field_name, value_expr) in fields {
                        self.compile_expression(value_expr, bytecode)?;
                        let offset = field_order
                            .iter()
                            .position(|f| f == field_name)
                            .unwrap_or(0);
                        bytecode.instructions.push(Instruction::new(
                            Opcode::TaggedUnionSetField,
                            vec![offset as u16],
                        ));
                    }
                }
                let _ = full_name;
                Ok(())
            }
            Expression::Switch(scrutinee, cases) => {
                self.compile_switch(scrutinee, cases, bytecode)
            }
            Expression::ComptimeBlock(body) => {
                self.compile_comptime_block(body, bytecode)
            }
            Expression::ComptimeFor {
                index_var,
                type_var,
                types,
                body,
            } => {
                let params = ComptimeForParams {
                    index_var,
                    type_var,
                    types,
                    body,
                };
                self.compile_comptime_for(&params, bytecode)
            }
            Expression::TypeValue(_) => {
                bail!("TypeValue can only be used in comptime context")
            }
            Expression::Typename(typ) => {
                let name = format!("{}", typ);
                self.compile_expression(
                    &Expression::Literal(Literal::String(name)),
                    bytecode,
                )
            }
            Expression::InterpolatedIdent(_parts) => {
                bail!("InterpolatedIdent can only be used in comptime context")
            }
            Expression::Unsafe(body) => {
                for statement in body {
                    self.compile_statement(statement, bytecode)?;
                }
                self.remove_last_pop(bytecode);
                Ok(())
            }
            Expression::ContextAccess => {
                bytecode.instructions.push(Instruction::new(Opcode::GetContext, vec![]));
                Ok(())
            }
            Expression::IfLet(pattern, value, consequence, alternative) => {
                self.compile_if_let(pattern, value, consequence, alternative.as_ref(), bytecode)
            }
        }
    }

    fn compile_if_let(
        &mut self,
        pattern: &crate::parser::Pattern,
        value: &Expression,
        consequence: &[Statement],
        alternative: Option<&Vec<Statement>>,
        bytecode: &mut Bytecode,
    ) -> Result<()> {
        let value_enum_name = match self.infer_expression_type(value) {
            Some(Type::Enum(name)) => Some(name),
            Some(Type::Struct(name)) if self.enum_defs.contains_key(&name) => Some(name),
            _ => None,
        };
        self.compile_expression(value, bytecode)?;

        let mut jump_to_else: Vec<usize> = Vec::new();

        match pattern {
            crate::parser::Pattern::Wildcard => {}
            crate::parser::Pattern::Literal(lit) => {
                bytecode.instructions.push(Instruction::new(Opcode::Dup, vec![]));
                self.compile_literal(lit, bytecode)?;
                let eq_opcode = match lit {
                    Literal::Integer(_) => Opcode::EqualI64,
                    _ => Opcode::Equal,
                };
                bytecode.instructions.push(Instruction::new(eq_opcode, vec![]));
                let jump_pos = bytecode.instructions.len();
                bytecode.instructions.push(Instruction::new(Opcode::JumpNotTruthy, vec![9999]));
                jump_to_else.push(jump_pos);
            }
            crate::parser::Pattern::Identifier(name) => {
                bytecode.instructions.push(Instruction::new(Opcode::Dup, vec![]));
                let symbol = self.symbol_table.define(name);
                let opcode = match symbol.scope {
                    SymbolScope::Local => Opcode::SetLocal,
                    SymbolScope::Global => Opcode::SetGlobal,
                    _ => anyhow::bail!("unexpected scope"),
                };
                bytecode.instructions.push(Instruction::new(opcode, vec![symbol.index as u16]));
            }
            crate::parser::Pattern::Tuple(patterns) => {
                for (index, pat) in patterns.iter().enumerate() {
                    match pat {
                        crate::parser::Pattern::Wildcard => {}
                        crate::parser::Pattern::Literal(lit) => {
                            bytecode.instructions.push(Instruction::new(Opcode::Dup, vec![]));
                            bytecode.instructions.push(Instruction::new(Opcode::TupleGet, vec![index as u16]));
                            self.compile_literal(lit, bytecode)?;
                            bytecode.instructions.push(Instruction::new(Opcode::EqualI64, vec![]));
                            let jump_pos = bytecode.instructions.len();
                            bytecode.instructions.push(Instruction::new(Opcode::JumpNotTruthy, vec![9999]));
                            jump_to_else.push(jump_pos);
                        }
                        crate::parser::Pattern::Identifier(name) => {
                            bytecode.instructions.push(Instruction::new(Opcode::Dup, vec![]));
                            bytecode.instructions.push(Instruction::new(Opcode::TupleGet, vec![index as u16]));
                            let symbol = self.symbol_table.define(name);
                            let opcode = match symbol.scope {
                                SymbolScope::Local => Opcode::SetLocal,
                                SymbolScope::Global => Opcode::SetGlobal,
                                _ => anyhow::bail!("unexpected scope"),
                            };
                            bytecode.instructions.push(Instruction::new(opcode, vec![symbol.index as u16]));
                        }
                        _ => anyhow::bail!("nested patterns not supported in if let tuple"),
                    }
                }
            }
            crate::parser::Pattern::EnumVariant { enum_name, variant_name, bindings } => {
                let effective_enum_name = enum_name.as_ref().cloned().or_else(|| value_enum_name.clone());
                let tag = if let Some(en) = &effective_enum_name {
                    self.get_variant_tag(en, variant_name)
                } else {
                    0
                };
                bytecode.instructions.push(Instruction::new(Opcode::Dup, vec![]));
                bytecode.instructions.push(Instruction::new(Opcode::TaggedUnionGetTag, vec![]));
                let constant_index = bytecode.constants.len() as u16;
                bytecode.constants.push(Value64::Integer(tag as i64));
                bytecode.instructions.push(Instruction::new(Opcode::Constant, vec![constant_index]));
                bytecode.instructions.push(Instruction::new(Opcode::EqualI64, vec![]));
                let jump_pos = bytecode.instructions.len();
                bytecode.instructions.push(Instruction::new(Opcode::JumpNotTruthy, vec![9999]));
                jump_to_else.push(jump_pos);

                for (offset, (_field_name, binding_name)) in bindings.iter().enumerate() {
                    bytecode.instructions.push(Instruction::new(Opcode::Dup, vec![]));
                    bytecode.instructions.push(Instruction::new(Opcode::TaggedUnionGetField, vec![offset as u16]));
                    let symbol = self.symbol_table.define(binding_name);
                    let opcode = match symbol.scope {
                        SymbolScope::Local => Opcode::SetLocal,
                        SymbolScope::Global => Opcode::SetGlobal,
                        _ => anyhow::bail!("unexpected scope"),
                    };
                    bytecode.instructions.push(Instruction::new(opcode, vec![symbol.index as u16]));
                }
            }
        }

        bytecode.instructions.push(Instruction::new(Opcode::Pop, vec![]));

        let moved_before = self.moved_symbols.clone();
        for statement in consequence {
            self.compile_statement(statement, bytecode)?;
        }
        self.remove_last_pop(bytecode);
        let moved_in_consequence = self.moved_symbols.clone();

        let jump_over_else = bytecode.instructions.len();
        bytecode.instructions.push(Instruction::new(Opcode::Jump, vec![9999]));

        let else_start = bytecode.instructions.len();
        for jump_pos in jump_to_else {
            bytecode.instructions[jump_pos].operands[0] = else_start as u16;
        }

        bytecode.instructions.push(Instruction::new(Opcode::Pop, vec![]));

        self.moved_symbols = moved_before.clone();
        if let Some(alt) = alternative {
            for statement in alt {
                self.compile_statement(statement, bytecode)?;
            }
            self.remove_last_pop(bytecode);
        } else {
            bytecode.instructions.push(Instruction::new(Opcode::Null, vec![]));
        }
        let moved_in_alternative = self.moved_symbols.clone();

        self.moved_symbols = moved_before;
        for sym in moved_in_consequence.intersection(&moved_in_alternative) {
            self.moved_symbols.insert(sym.clone());
        }

        let end_pos = bytecode.instructions.len();
        bytecode.instructions[jump_over_else].operands[0] = end_pos as u16;

        Ok(())
    }

    fn get_variant_tag(&self, enum_name: &str, variant_name: &str) -> usize {
        if let Some(enum_def) = self.enum_defs.get(enum_name) {
            for variant in &enum_def.variants {
                if variant.name == variant_name {
                    return variant.tag;
                }
            }
        }
        0
    }

    fn get_variant_field_order(
        &self,
        enum_name: &str,
        variant_name: &str,
    ) -> Vec<String> {
        if let Some(enum_def) = self.enum_defs.get(enum_name) {
            for variant in &enum_def.variants {
                if variant.name == variant_name {
                    return variant
                        .fields
                        .iter()
                        .map(|f| f.name.clone())
                        .collect();
                }
            }
        }
        Vec::new()
    }

    fn compile_switch(
        &mut self,
        scrutinee: &Expression,
        cases: &[crate::parser::SwitchCase],
        bytecode: &mut Bytecode,
    ) -> Result<()> {
        let scrutinee_enum_name = match self.infer_expression_type(scrutinee) {
            Some(Type::Enum(name)) => Some(name),
            Some(Type::Struct(name)) if self.enum_defs.contains_key(&name) => {
                Some(name)
            }
            _ => None,
        };
        self.compile_expression(scrutinee, bytecode)?;

        let mut jump_ends: Vec<usize> = Vec::new();
        let mut next_case_jumps: Vec<usize> = Vec::new();

        for (_case_index, case) in cases.iter().enumerate() {
            for jump_pos in next_case_jumps.drain(..) {
                bytecode.instructions[jump_pos].operands[0] =
                    bytecode.instructions.len() as u16;
            }

            match &case.pattern {
                crate::parser::Pattern::Wildcard => {}
                crate::parser::Pattern::Literal(lit) => {
                    bytecode
                        .instructions
                        .push(Instruction::new(Opcode::Dup, vec![]));
                    self.compile_literal(lit, bytecode)?;
                    let eq_opcode = match lit {
                        Literal::Integer(_) => Opcode::EqualI64,
                        _ => Opcode::Equal,
                    };
                    bytecode
                        .instructions
                        .push(Instruction::new(eq_opcode, vec![]));
                    let jump_pos = bytecode.instructions.len();
                    bytecode.instructions.push(Instruction::new(
                        Opcode::JumpNotTruthy,
                        vec![9999],
                    ));
                    next_case_jumps.push(jump_pos);
                }
                crate::parser::Pattern::Identifier(name) => {
                    bytecode
                        .instructions
                        .push(Instruction::new(Opcode::Dup, vec![]));
                    let symbol = self.symbol_table.define(name);
                    let opcode = match symbol.scope {
                        SymbolScope::Local => Opcode::SetLocal,
                        SymbolScope::Global => Opcode::SetGlobal,
                        _ => anyhow::bail!("unexpected scope"),
                    };
                    bytecode.instructions.push(Instruction::new(
                        opcode,
                        vec![symbol.index as u16],
                    ));
                }
                crate::parser::Pattern::Tuple(patterns) => {
                    for (index, pattern) in patterns.iter().enumerate() {
                        match pattern {
                            crate::parser::Pattern::Wildcard => {}
                            crate::parser::Pattern::Literal(lit) => {
                                bytecode.instructions.push(Instruction::new(
                                    Opcode::Dup,
                                    vec![],
                                ));
                                bytecode.instructions.push(Instruction::new(
                                    Opcode::TupleGet,
                                    vec![index as u16],
                                ));
                                self.compile_literal(lit, bytecode)?;
                                bytecode.instructions.push(Instruction::new(
                                    Opcode::EqualI64,
                                    vec![],
                                ));
                                let jump_pos = bytecode.instructions.len();
                                bytecode.instructions.push(Instruction::new(
                                    Opcode::JumpNotTruthy,
                                    vec![9999],
                                ));
                                next_case_jumps.push(jump_pos);
                            }
                            crate::parser::Pattern::Identifier(name) => {
                                bytecode.instructions.push(Instruction::new(
                                    Opcode::Dup,
                                    vec![],
                                ));
                                bytecode.instructions.push(Instruction::new(
                                    Opcode::TupleGet,
                                    vec![index as u16],
                                ));
                                let symbol = self.symbol_table.define(name);
                                let opcode = match symbol.scope {
                                    SymbolScope::Local => Opcode::SetLocal,
                                    SymbolScope::Global => Opcode::SetGlobal,
                                    _ => anyhow::bail!("unexpected scope"),
                                };
                                bytecode.instructions.push(Instruction::new(
                                    opcode,
                                    vec![symbol.index as u16],
                                ));
                            }
                            _ => anyhow::bail!(
                                "nested patterns not supported in tuples"
                            ),
                        }
                    }
                }
                crate::parser::Pattern::EnumVariant {
                    enum_name,
                    variant_name,
                    bindings,
                } => {
                    let effective_enum_name = enum_name
                        .as_ref()
                        .cloned()
                        .or_else(|| scrutinee_enum_name.clone());
                    let tag = if let Some(en) = &effective_enum_name {
                        self.get_variant_tag(en, variant_name)
                    } else {
                        0
                    };
                    bytecode
                        .instructions
                        .push(Instruction::new(Opcode::Dup, vec![]));
                    bytecode.instructions.push(Instruction::new(
                        Opcode::TaggedUnionGetTag,
                        vec![],
                    ));
                    let constant_index = bytecode.constants.len() as u16;
                    bytecode.constants.push(Value64::Integer(tag as i64));
                    bytecode.instructions.push(Instruction::new(
                        Opcode::Constant,
                        vec![constant_index],
                    ));
                    bytecode
                        .instructions
                        .push(Instruction::new(Opcode::EqualI64, vec![]));
                    let jump_pos = bytecode.instructions.len();
                    bytecode.instructions.push(Instruction::new(
                        Opcode::JumpNotTruthy,
                        vec![9999],
                    ));
                    next_case_jumps.push(jump_pos);

                    for (offset, (field_name, binding_name)) in
                        bindings.iter().enumerate()
                    {
                        bytecode
                            .instructions
                            .push(Instruction::new(Opcode::Dup, vec![]));
                        bytecode.instructions.push(Instruction::new(
                            Opcode::TaggedUnionGetField,
                            vec![offset as u16],
                        ));
                        let field_type = effective_enum_name.as_ref().and_then(|en| {
                            self.enum_defs.get(en).and_then(|enum_def| {
                                enum_def.variants.iter()
                                    .find(|v| v.name == *variant_name)
                                    .and_then(|variant| {
                                        variant.fields.iter()
                                            .find(|f| f.name == *field_name)
                                            .map(|f| f.field_type.clone())
                                    })
                            })
                        });
                        self.moved_symbols.remove(binding_name);
                        let symbol = self.symbol_table.define_with_type(binding_name, field_type);
                        let opcode = match symbol.scope {
                            SymbolScope::Local => Opcode::SetLocal,
                            SymbolScope::Global => Opcode::SetGlobal,
                            _ => anyhow::bail!("unexpected scope"),
                        };
                        bytecode.instructions.push(Instruction::new(
                            opcode,
                            vec![symbol.index as u16],
                        ));
                    }
                }
            }

            bytecode
                .instructions
                .push(Instruction::new(Opcode::Pop, vec![]));

            for statement in &case.body {
                self.compile_statement(statement, bytecode)?;
            }
            self.remove_last_pop(bytecode);

            let jump_pos = bytecode.instructions.len();
            bytecode
                .instructions
                .push(Instruction::new(Opcode::Jump, vec![9999]));
            jump_ends.push(jump_pos);
        }

        if !next_case_jumps.is_empty() {
            let fallthrough_pos = bytecode.instructions.len();
            for jump_pos in next_case_jumps {
                bytecode.instructions[jump_pos].operands[0] =
                    fallthrough_pos as u16;
            }
            bytecode
                .instructions
                .push(Instruction::new(Opcode::Pop, vec![]));
            bytecode
                .instructions
                .push(Instruction::new(Opcode::Null, vec![]));
        }

        let end_pos = bytecode.instructions.len();
        for jump_pos in jump_ends {
            bytecode.instructions[jump_pos].operands[0] = end_pos as u16;
        }

        Ok(())
    }

    fn compile_literal(
        &mut self,
        literal: &Literal,
        bytecode: &mut Bytecode,
    ) -> Result<()> {
        match literal {
            Literal::Integer(value) => {
                let constant_index = bytecode.constants.len() as u16;
                bytecode.constants.push(Value64::Integer(*value));
                bytecode.instructions.push(Instruction::new(
                    Opcode::Constant,
                    vec![constant_index],
                ));
            }
            Literal::Float(value) => {
                let constant_index = bytecode.constants.len() as u16;
                bytecode.constants.push(Value64::Float(*value));
                bytecode.instructions.push(Instruction::new(
                    Opcode::Constant,
                    vec![constant_index],
                ));
            }
            Literal::Float32(value) => {
                let constant_index = bytecode.constants.len() as u16;
                bytecode.constants.push(Value64::Float32(*value));
                bytecode.instructions.push(Instruction::new(
                    Opcode::Constant,
                    vec![constant_index],
                ));
            }
            Literal::Boolean(value) => {
                let constant_index = bytecode.constants.len() as u16;
                bytecode.constants.push(Value64::Bool(*value));
                bytecode.instructions.push(Instruction::new(
                    Opcode::Constant,
                    vec![constant_index],
                ));
            }
            Literal::String(value) => {
                let heap_index = bytecode.heap.len() as u32;
                bytecode.heap.push(HeapObject::String(value.clone()));
                let constant_index = bytecode.constants.len() as u16;
                bytecode.constants.push(Value64::HeapRef(heap_index));
                bytecode.instructions.push(Instruction::new(
                    Opcode::Constant,
                    vec![constant_index],
                ));
            }
            Literal::Array(elements) => {
                for element in elements {
                    self.compile_expression(element, bytecode)?;
                }
                bytecode.instructions.push(Instruction::new(
                    Opcode::Array,
                    vec![elements.len() as u16],
                ));
            }
            Literal::HashMap(pairs) => {
                for (key, value) in pairs {
                    self.compile_expression(key, bytecode)?;
                    self.compile_expression(value, bytecode)?;
                }
                bytecode.instructions.push(Instruction::new(
                    Opcode::Hash,
                    vec![(pairs.len() * 2) as u16],
                ));
            }
        }
        Ok(())
    }

    fn infer_expression_type(
        &mut self,
        expression: &Expression,
    ) -> Option<Type> {
        match expression {
            Expression::Literal(Literal::Integer(_)) => Some(Type::I64),
            Expression::Literal(Literal::Float(_)) => Some(Type::F64),
            Expression::Literal(Literal::Float32(_)) => Some(Type::F32),
            Expression::Literal(Literal::String(_)) => Some(Type::Str),
            Expression::Boolean(_) => Some(Type::Bool),
            Expression::Identifier(name) => {
                self.symbol_table.resolve(name).and_then(|s| s.symbol_type)
            }
            Expression::Infix(left, operator, _right) => {
                let left_type = self.infer_expression_type(left)?;
                match operator {
                    Operator::Add
                    | Operator::Subtract
                    | Operator::Multiply
                    | Operator::Divide
                    | Operator::Modulo => Some(left_type),
                    Operator::Equal
                    | Operator::NotEqual
                    | Operator::LessThan
                    | Operator::LessThanOrEqual
                    | Operator::GreaterThan
                    | Operator::GreaterThanOrEqual => Some(Type::Bool),
                    _ => None,
                }
            }
            Expression::Prefix(Operator::Negate, operand) => {
                self.infer_expression_type(operand)
            }
            Expression::Prefix(Operator::Not, _) => Some(Type::Bool),
            Expression::EnumVariantInit(enum_name, _, _) => {
                Some(Type::Enum(enum_name.clone()))
            }
            _ => None,
        }
    }

    fn compile_infix(
        &mut self,
        left: &Expression,
        operator: &Operator,
        right: &Expression,
        bytecode: &mut Bytecode,
    ) -> Result<()> {
        let left_type = if self.typed_mode {
            self.infer_expression_type(left)
        } else {
            None
        };

        if *operator == Operator::LessThan {
            self.compile_expression(right, bytecode)?;
            self.compile_expression(left, bytecode)?;
            let opcode = if self.typed_mode && left_type == Some(Type::I64) {
                Opcode::GreaterThanI64
            } else {
                Opcode::GreaterThan
            };
            bytecode.instructions.push(Instruction::new(opcode, vec![]));
            return Ok(());
        }

        if *operator == Operator::LessThanOrEqual {
            self.compile_expression(left, bytecode)?;
            self.compile_expression(right, bytecode)?;
            let opcode = if self.typed_mode && left_type == Some(Type::I64) {
                Opcode::GreaterThanI64
            } else {
                Opcode::GreaterThan
            };
            bytecode.instructions.push(Instruction::new(opcode, vec![]));
            bytecode
                .instructions
                .push(Instruction::new(Opcode::Bang, vec![]));
            return Ok(());
        }

        if *operator == Operator::GreaterThanOrEqual {
            self.compile_expression(right, bytecode)?;
            self.compile_expression(left, bytecode)?;
            let opcode = if self.typed_mode && left_type == Some(Type::I64) {
                Opcode::GreaterThanI64
            } else {
                Opcode::GreaterThan
            };
            bytecode.instructions.push(Instruction::new(opcode, vec![]));
            bytecode
                .instructions
                .push(Instruction::new(Opcode::Bang, vec![]));
            return Ok(());
        }

        if *operator == Operator::And {
            self.compile_expression(left, bytecode)?;
            let jump_false_pos = bytecode.instructions.len();
            bytecode
                .instructions
                .push(Instruction::new(Opcode::JumpNotTruthy, vec![9999]));
            self.compile_expression(right, bytecode)?;
            let jump_end_pos = bytecode.instructions.len();
            bytecode
                .instructions
                .push(Instruction::new(Opcode::Jump, vec![9999]));
            let false_pos = bytecode.instructions.len();
            bytecode.instructions[jump_false_pos].operands[0] =
                false_pos as u16;
            bytecode
                .instructions
                .push(Instruction::new(Opcode::False, vec![]));
            let end_pos = bytecode.instructions.len();
            bytecode.instructions[jump_end_pos].operands[0] = end_pos as u16;
            return Ok(());
        }

        if *operator == Operator::Or {
            self.compile_expression(left, bytecode)?;
            let jump_true_pos = bytecode.instructions.len();
            bytecode
                .instructions
                .push(Instruction::new(Opcode::JumpNotTruthy, vec![9999]));
            bytecode
                .instructions
                .push(Instruction::new(Opcode::True, vec![]));
            let jump_end_pos = bytecode.instructions.len();
            bytecode
                .instructions
                .push(Instruction::new(Opcode::Jump, vec![9999]));
            let false_pos = bytecode.instructions.len();
            bytecode.instructions[jump_true_pos].operands[0] = false_pos as u16;
            self.compile_expression(right, bytecode)?;
            let end_pos = bytecode.instructions.len();
            bytecode.instructions[jump_end_pos].operands[0] = end_pos as u16;
            return Ok(());
        }

        self.compile_expression(left, bytecode)?;
        self.compile_expression(right, bytecode)?;

        let opcode = if self.typed_mode {
            match (operator, &left_type) {
                (Operator::Add, Some(Type::I64)) => Opcode::AddI64,
                (Operator::Subtract, Some(Type::I64)) => Opcode::SubI64,
                (Operator::Multiply, Some(Type::I64)) => Opcode::MulI64,
                (Operator::Divide, Some(Type::I64)) => Opcode::DivI64,
                (Operator::Modulo, Some(Type::I64)) => Opcode::ModI64,
                (Operator::Equal, Some(Type::I64)) => Opcode::EqualI64,
                (Operator::NotEqual, Some(Type::I64)) => Opcode::NotEqualI64,
                (Operator::GreaterThan, Some(Type::I64)) => {
                    Opcode::GreaterThanI64
                }
                (Operator::Add, Some(Type::F64)) => Opcode::AddF64,
                (Operator::Subtract, Some(Type::F64)) => Opcode::SubF64,
                (Operator::Multiply, Some(Type::F64)) => Opcode::MulF64,
                (Operator::Divide, Some(Type::F64)) => Opcode::DivF64,
                (Operator::Add, Some(Type::F32)) => Opcode::AddF32,
                (Operator::Subtract, Some(Type::F32)) => Opcode::SubF32,
                (Operator::Multiply, Some(Type::F32)) => Opcode::MulF32,
                (Operator::Divide, Some(Type::F32)) => Opcode::DivF32,
                (Operator::Equal, Some(Type::Bool)) => Opcode::EqualBool,
                (Operator::NotEqual, Some(Type::Bool)) => Opcode::NotEqualBool,
                _ => match operator {
                    Operator::Add => Opcode::Add,
                    Operator::Subtract => Opcode::Sub,
                    Operator::Multiply => Opcode::Mul,
                    Operator::Divide => Opcode::Div,
                    Operator::Modulo => Opcode::Mod,
                    Operator::Equal => Opcode::Equal,
                    Operator::NotEqual => Opcode::NotEqual,
                    Operator::GreaterThan => Opcode::GreaterThan,
                    Operator::ShiftLeft => Opcode::ShiftLeft,
                    Operator::ShiftRight => Opcode::ShiftRight,
                    Operator::BitwiseAnd => Opcode::BitwiseAnd,
                    Operator::BitwiseOr => Opcode::BitwiseOr,
                    _ => unimplemented!(
                        "Operator {:?} not implemented for infix",
                        operator
                    ),
                },
            }
        } else {
            match operator {
                Operator::Add => Opcode::Add,
                Operator::Subtract => Opcode::Sub,
                Operator::Multiply => Opcode::Mul,
                Operator::Divide => Opcode::Div,
                Operator::Modulo => Opcode::Mod,
                Operator::Equal => Opcode::Equal,
                Operator::NotEqual => Opcode::NotEqual,
                Operator::GreaterThan => Opcode::GreaterThan,
                Operator::ShiftLeft => Opcode::ShiftLeft,
                Operator::ShiftRight => Opcode::ShiftRight,
                Operator::BitwiseAnd => Opcode::BitwiseAnd,
                Operator::BitwiseOr => Opcode::BitwiseOr,
                _ => unimplemented!(
                    "Operator {:?} not implemented for infix",
                    operator
                ),
            }
        };
        bytecode.instructions.push(Instruction::new(opcode, vec![]));
        Ok(())
    }

    fn load_symbol(&self, symbol: &Symbol, bytecode: &mut Bytecode) {
        let (opcode, operands) = match symbol.scope {
            SymbolScope::Global => {
                (Opcode::GetGlobal, vec![symbol.index as u16])
            }
            SymbolScope::Local => (Opcode::GetLocal, vec![symbol.index as u16]),
            SymbolScope::Builtin => {
                (Opcode::GetBuiltin, vec![symbol.index as u16])
            }
            SymbolScope::Native => {
                (Opcode::GetNative, vec![symbol.index as u16])
            }
            SymbolScope::Free => (Opcode::GetFree, vec![symbol.index as u16]),
            SymbolScope::Function => (Opcode::CurrentClosure, vec![]),
        };
        bytecode
            .instructions
            .push(Instruction::new(opcode, operands));
    }

    fn compile_function_with_name(
        &mut self,
        name: &str,
        parameters: &[Parameter],
        return_sig: &ReturnSignature,
        body: &[Statement],
        bytecode: &mut Bytecode,
    ) -> Result<()> {
        let outer_symbol_table = std::mem::take(&mut self.symbol_table);
        let outer_moved_symbols = std::mem::take(&mut self.moved_symbols);
        self.symbol_table = SymbolTable::new_enclosed(outer_symbol_table);

        self.symbol_table.define_function_name(name);

        for param in parameters {
            self.symbol_table.define_with_type_and_mutability(
                &param.name,
                param.type_annotation.clone(),
                param.mutable,
            );
        }

        if let Some(named_params) = return_sig.named_params() {
            for return_param in named_params {
                self.symbol_table.define_with_type_and_mutability(
                    &return_param.name,
                    Some(return_param.param_type.clone()),
                    true,
                );
            }
        }

        let mut fn_bytecode = Bytecode {
            instructions: vec![],
            constants: std::mem::take(&mut bytecode.constants),
            functions: std::mem::take(&mut bytecode.functions),
            heap: std::mem::take(&mut bytecode.heap),
            native_names: std::mem::take(&mut bytecode.native_names),
            global_symbols: std::mem::take(&mut bytecode.global_symbols),
        };
        for statement in body {
            self.compile_statement(statement, &mut fn_bytecode)?;
        }

        if self.last_instruction_is(&fn_bytecode, Opcode::Pop) {
            self.replace_last_pop_with_return(&mut fn_bytecode);
        }

        if fn_bytecode.instructions.is_empty()
            || !self.last_instruction_is(&fn_bytecode, Opcode::ReturnValue)
        {
            if let Some(named_params) = return_sig.named_params() {
                if named_params.len() == 1 {
                    if let Some(symbol) = self.symbol_table.resolve(&named_params[0].name) {
                        fn_bytecode.instructions.push(Instruction::new(
                            Opcode::GetLocal,
                            vec![symbol.index as u16],
                        ));
                        fn_bytecode
                            .instructions
                            .push(Instruction::new(Opcode::ReturnValue, vec![]));
                    } else {
                        fn_bytecode
                            .instructions
                            .push(Instruction::new(Opcode::Return, vec![]));
                    }
                } else {
                    fn_bytecode
                        .instructions
                        .push(Instruction::new(Opcode::Return, vec![]));
                }
            } else {
                fn_bytecode
                    .instructions
                    .push(Instruction::new(Opcode::Return, vec![]));
            }
        }

        bytecode.constants = fn_bytecode.constants;
        bytecode.functions = fn_bytecode.functions;
        bytecode.heap = fn_bytecode.heap;
        bytecode.global_symbols = fn_bytecode.global_symbols;

        let free_symbols = self.symbol_table.free_symbols.clone();
        let num_locals = self.symbol_table.num_definitions;
        let num_parameters = parameters.len();

        if let Some(outer) = self.symbol_table.outer.take() {
            self.symbol_table = *outer;
        }
        self.moved_symbols = outer_moved_symbols;

        for sym in &free_symbols {
            self.load_symbol(sym, bytecode);
        }

        let compiled_fn = CompiledFunction {
            instructions: fn_bytecode.instructions,
            num_locals,
            num_parameters,
        };
        let function_index = bytecode.functions.len() as u16;
        bytecode.functions.push(compiled_fn);
        bytecode.instructions.push(Instruction::new(
            Opcode::Closure,
            vec![function_index, free_symbols.len() as u16],
        ));

        Ok(())
    }

    fn last_instruction_is(&self, bytecode: &Bytecode, opcode: Opcode) -> bool {
        bytecode
            .instructions
            .last()
            .map(|i| i.opcode == opcode)
            .unwrap_or(false)
    }

    fn remove_last_pop(&self, bytecode: &mut Bytecode) {
        if self.last_instruction_is(bytecode, Opcode::Pop) {
            bytecode.instructions.pop();
        }
    }

    fn infer_expression_struct_type(&self, expr: &Expression) -> Option<String> {
        match expr {
            Expression::Identifier(name) => {
                if let Some(symbol) = self.symbol_table.store.get(name) {
                    if let Some(Type::Struct(struct_name)) = &symbol.symbol_type {
                        return Some(struct_name.clone());
                    }
                }
                None
            }
            Expression::Index(array_expr, _) => {
                if let Expression::Identifier(name) = array_expr.as_ref() {
                    if let Some(symbol) = self.symbol_table.store.get(name) {
                        if let Some(symbol_type) = &symbol.symbol_type {
                            match symbol_type {
                                Type::Array(inner, _) | Type::Slice(inner) => {
                                    if let Type::Struct(struct_name) = inner.as_ref() {
                                        return Some(struct_name.clone());
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                None
            }
            Expression::FieldAccess(inner_expr, field_name) => {
                let parent_type = self.infer_expression_struct_type(inner_expr)?;
                let struct_def = self.struct_defs.get(&parent_type)?;
                let field_def = struct_def.fields.iter().find(|f| &f.name == field_name)?;
                if let Type::Struct(struct_name) = &field_def.field_type {
                    Some(struct_name.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn infer_field_access_type(&self, base_expr: &Expression, field_name: &str) -> Option<Type> {
        let struct_name = match base_expr {
            Expression::Identifier(name) => {
                if let Some(symbol) = self.symbol_table.store.get(name) {
                    match &symbol.symbol_type {
                        Some(Type::Struct(s)) => Some(s.clone()),
                        Some(Type::Ref(inner)) | Some(Type::RefMut(inner)) => {
                            if let Type::Struct(s) = inner.as_ref() {
                                Some(s.clone())
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        }?;
        let struct_def = self.struct_defs.get(&struct_name)?;
        let field_def = struct_def.fields.iter().find(|f| f.name == field_name)?;
        Some(field_def.field_type.clone())
    }

    fn infer_field_access_element_type(&self, base_expr: &Expression, field_name: &str) -> Option<Type> {
        let struct_name = match base_expr {
            Expression::Identifier(name) => {
                if let Some(symbol) = self.symbol_table.store.get(name) {
                    match &symbol.symbol_type {
                        Some(Type::Struct(s)) => Some(s.clone()),
                        Some(Type::Ref(inner)) | Some(Type::RefMut(inner)) => {
                            if let Type::Struct(s) = inner.as_ref() {
                                Some(s.clone())
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        }?;
        let struct_def = self.struct_defs.get(&struct_name)?;
        let field_def = struct_def.fields.iter().find(|f| f.name == field_name)?;
        match &field_def.field_type {
            Type::Array(inner, _) | Type::Slice(inner) => Some(inner.as_ref().clone()),
            _ => None,
        }
    }

    fn compute_field_access_offset(&mut self, expr: &Expression, field: &str) -> Result<(String, usize)> {
        match expr {
            Expression::Identifier(name) => {
                let symbol = self.symbol_table.resolve(name).ok_or_else(|| {
                    anyhow::anyhow!("undefined variable: {}", name)
                })?;
                let struct_name = match &symbol.symbol_type {
                    Some(Type::Struct(s)) => s.clone(),
                    Some(Type::RefMut(inner)) | Some(Type::Ref(inner)) => {
                        if let Type::Struct(s) = inner.as_ref() {
                            s.clone()
                        } else {
                            anyhow::bail!("expected struct type for field access")
                        }
                    }
                    _ => anyhow::bail!("expected struct type for field access"),
                };
                let struct_def = self.struct_defs.get(&struct_name).ok_or_else(|| {
                    anyhow::anyhow!("undefined struct: {}", struct_name)
                })?;
                let offset = *struct_def.field_offsets.get(field).ok_or_else(|| {
                    anyhow::anyhow!("unknown field: {}", field)
                })?;
                Ok((name.clone(), offset))
            }
            Expression::FieldAccess(inner_expr, inner_field) => {
                let (base_name, base_offset) = self.compute_field_access_offset(inner_expr, inner_field)?;
                let inner_struct_type = self.get_field_struct_type(inner_expr, inner_field)?;
                let struct_def = self.struct_defs.get(&inner_struct_type).ok_or_else(|| {
                    anyhow::anyhow!("undefined struct: {}", inner_struct_type)
                })?;
                let field_offset = *struct_def.field_offsets.get(field).ok_or_else(|| {
                    anyhow::anyhow!("unknown field: {}", field)
                })?;
                Ok((base_name, base_offset + field_offset))
            }
            _ => anyhow::bail!("expected identifier or field access for borrowing"),
        }
    }

    fn get_field_struct_type(&mut self, expr: &Expression, field: &str) -> Result<String> {
        match expr {
            Expression::Identifier(name) => {
                let symbol = self.symbol_table.resolve(name).ok_or_else(|| {
                    anyhow::anyhow!("undefined variable: {}", name)
                })?;
                let struct_name = match &symbol.symbol_type {
                    Some(Type::Struct(s)) => s.clone(),
                    Some(Type::RefMut(inner)) | Some(Type::Ref(inner)) => {
                        if let Type::Struct(s) = inner.as_ref() {
                            s.clone()
                        } else {
                            anyhow::bail!("expected struct type")
                        }
                    }
                    _ => anyhow::bail!("expected struct type"),
                };
                let struct_def = self.struct_defs.get(&struct_name).ok_or_else(|| {
                    anyhow::anyhow!("undefined struct: {}", struct_name)
                })?;
                let field_def = struct_def.fields.iter().find(|f| f.name == field).ok_or_else(|| {
                    anyhow::anyhow!("unknown field: {}", field)
                })?;
                if let Type::Struct(s) = &field_def.field_type {
                    Ok(s.clone())
                } else {
                    anyhow::bail!("field {} is not a struct", field)
                }
            }
            Expression::FieldAccess(inner_expr, inner_field) => {
                let inner_type = self.get_field_struct_type(inner_expr, inner_field)?;
                let struct_def = self.struct_defs.get(&inner_type).ok_or_else(|| {
                    anyhow::anyhow!("undefined struct: {}", inner_type)
                })?;
                let field_def = struct_def.fields.iter().find(|f| f.name == field).ok_or_else(|| {
                    anyhow::anyhow!("unknown field: {}", field)
                })?;
                if let Type::Struct(s) = &field_def.field_type {
                    Ok(s.clone())
                } else {
                    anyhow::bail!("field {} is not a struct", field)
                }
            }
            _ => anyhow::bail!("expected identifier or field access"),
        }
    }

    fn replace_last_pop_with_return(&self, bytecode: &mut Bytecode) {
        if self.last_instruction_is(bytecode, Opcode::Pop) {
            bytecode.instructions.pop();
            bytecode
                .instructions
                .push(Instruction::new(Opcode::ReturnValue, vec![]));
        }
    }

    fn compile_comptime_block(
        &mut self,
        body: &[Statement],
        bytecode: &mut Bytecode,
    ) -> Result<()> {
        for statement in body {
            match statement {
                Statement::Constant(name, expr) => {
                    let value = self.evaluate_comptime_expr(expr)?;
                    self.comptime_constants.insert(name.clone(), value);
                }
                Statement::Expression(Expression::ComptimeFor {
                    index_var,
                    type_var,
                    types,
                    body,
                }) => {
                    let params = ComptimeForParams {
                        index_var,
                        type_var,
                        types,
                        body,
                    };
                    self.compile_comptime_for(&params, bytecode)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn compile_comptime_for(
        &mut self,
        params: &ComptimeForParams,
        bytecode: &mut Bytecode,
    ) -> Result<()> {
        for (index, typ) in params.types.iter().enumerate() {
            let type_name = format!("{}", typ);
            let ctx = SubstitutionContext {
                index_var: params.index_var.as_deref(),
                type_var: params.type_var,
                index,
                type_name: &type_name,
            };
            let expanded_body: Vec<Statement> = params
                .body
                .iter()
                .map(|stmt| Self::substitute_in_statement(stmt, &ctx))
                .collect();
            for statement in &expanded_body {
                self.compile_statement(statement, bytecode)?;
            }
        }
        Ok(())
    }

    fn substitute_in_statement(
        stmt: &Statement,
        ctx: &SubstitutionContext,
    ) -> Statement {
        match stmt {
            Statement::Constant(name, expr) => {
                let new_name = Self::substitute_identifier(
                    name,
                    ctx.type_var,
                    ctx.type_name,
                );
                let new_expr = Self::substitute_in_expr(expr, ctx);
                Statement::Constant(new_name, new_expr)
            }
            Statement::Let {
                name,
                mutable,
                type_annotation,
                value,
            } => Statement::Let {
                name: Self::substitute_identifier(
                    name,
                    ctx.type_var,
                    ctx.type_name,
                ),
                mutable: *mutable,
                type_annotation: type_annotation
                    .as_ref()
                    .map(|t| Self::substitute_in_type(t, ctx)),
                value: Self::substitute_in_expr(value, ctx),
            },
            Statement::Expression(expr) => {
                Statement::Expression(Self::substitute_in_expr(expr, ctx))
            }
            Statement::InterpolatedConstant(parts, expr) => {
                let resolved_name: String = parts
                    .iter()
                    .map(|part| match part {
                        IdentPart::Literal(s) => s.clone(),
                        IdentPart::TypeVar(var) => {
                            if var == ctx.type_var {
                                ctx.type_name.to_string()
                            } else {
                                format!("#{}", var)
                            }
                        }
                    })
                    .collect();
                let new_expr = Self::substitute_in_expr(expr, ctx);
                Statement::Constant(resolved_name, new_expr)
            }
            other => other.clone(),
        }
    }

    fn substitute_in_expr(
        expr: &Expression,
        ctx: &SubstitutionContext,
    ) -> Expression {
        match expr {
            Expression::InterpolatedIdent(parts) => {
                let resolved: String = parts
                    .iter()
                    .map(|part| match part {
                        IdentPart::Literal(s) => s.clone(),
                        IdentPart::TypeVar(var) => {
                            if var == ctx.type_var {
                                ctx.type_name.to_string()
                            } else {
                                format!("#{}", var)
                            }
                        }
                    })
                    .collect();
                Expression::Identifier(resolved)
            }
            Expression::Identifier(name) => {
                if Some(name.as_str()) == ctx.index_var {
                    Expression::Literal(Literal::Integer(ctx.index as i64))
                } else if name == ctx.type_var {
                    Expression::TypeValue(Type::Struct(
                        ctx.type_name.to_string(),
                    ))
                } else {
                    Expression::Identifier(name.clone())
                }
            }
            Expression::Infix(left, op, right) => Expression::Infix(
                Box::new(Self::substitute_in_expr(left, ctx)),
                *op,
                Box::new(Self::substitute_in_expr(right, ctx)),
            ),
            Expression::Call(func, args) => Expression::Call(
                Box::new(Self::substitute_in_expr(func, ctx)),
                args.iter()
                    .map(|a| Self::substitute_in_expr(a, ctx))
                    .collect(),
            ),
            Expression::Function(params, return_sig, body) => {
                Expression::Function(
                    params
                        .iter()
                        .map(|p| Self::substitute_in_param(p, ctx))
                        .collect(),
                    Self::substitute_in_return_sig(return_sig, ctx),
                    body.iter()
                        .map(|s| Self::substitute_in_statement(s, ctx))
                        .collect(),
                )
            }
            Expression::Proc(params, return_sig, body) => Expression::Proc(
                params
                    .iter()
                    .map(|p| Self::substitute_in_param(p, ctx))
                    .collect(),
                Self::substitute_in_return_sig(return_sig, ctx),
                body.iter()
                    .map(|s| Self::substitute_in_statement(s, ctx))
                    .collect(),
            ),
            Expression::FieldAccess(expr, field) => {
                let new_field = Self::substitute_identifier(
                    field,
                    ctx.type_var,
                    ctx.type_name,
                );
                Expression::FieldAccess(
                    Box::new(Self::substitute_in_expr(expr, ctx)),
                    new_field,
                )
            }
            Expression::Sizeof(typ) => {
                if format!("{}", typ) == ctx.type_var {
                    Expression::Sizeof(Type::Struct(ctx.type_name.to_string()))
                } else {
                    expr.clone()
                }
            }
            Expression::Typename(typ) => {
                if format!("{}", typ) == ctx.type_var {
                    Expression::Literal(Literal::String(
                        ctx.type_name.to_string(),
                    ))
                } else {
                    Expression::Literal(Literal::String(format!("{}", typ)))
                }
            }
            other => other.clone(),
        }
    }

    fn substitute_identifier(
        name: &str,
        type_var: &str,
        type_name: &str,
    ) -> String {
        if name.contains(&format!("#{}", type_var)) {
            name.replace(&format!("#{}", type_var), type_name)
        } else {
            name.to_string()
        }
    }

    fn substitute_in_type(typ: &Type, ctx: &SubstitutionContext) -> Type {
        match typ {
            Type::TypeParam(name) if name == ctx.type_var => {
                Type::Struct(ctx.type_name.to_string())
            }
            Type::Struct(name) if name == ctx.type_var => {
                Type::Struct(ctx.type_name.to_string())
            }
            Type::Ptr(inner) => {
                Type::Ptr(Box::new(Self::substitute_in_type(inner, ctx)))
            }
            Type::Ref(inner) => {
                Type::Ref(Box::new(Self::substitute_in_type(inner, ctx)))
            }
            Type::RefMut(inner) => {
                Type::RefMut(Box::new(Self::substitute_in_type(inner, ctx)))
            }
            Type::Array(inner, size) => Type::Array(
                Box::new(Self::substitute_in_type(inner, ctx)),
                *size,
            ),
            Type::Slice(inner) => {
                Type::Slice(Box::new(Self::substitute_in_type(inner, ctx)))
            }
            Type::Proc(params, ret) => Type::Proc(
                params
                    .iter()
                    .map(|p| Self::substitute_in_type(p, ctx))
                    .collect(),
                Box::new(Self::substitute_in_type(ret, ctx)),
            ),
            Type::Distinct(inner) => {
                Type::Distinct(Box::new(Self::substitute_in_type(inner, ctx)))
            }
            other => other.clone(),
        }
    }

    fn substitute_in_return_sig(
        sig: &ReturnSignature,
        ctx: &SubstitutionContext,
    ) -> ReturnSignature {
        match sig {
            ReturnSignature::None => ReturnSignature::None,
            ReturnSignature::Single(t) => {
                ReturnSignature::Single(Self::substitute_in_type(t, ctx))
            }
            ReturnSignature::Named(params) => {
                ReturnSignature::Named(
                    params
                        .iter()
                        .map(|p| crate::parser::ReturnParam {
                            name: Self::substitute_identifier(&p.name, ctx.type_var, ctx.type_name),
                            param_type: Self::substitute_in_type(&p.param_type, ctx),
                        })
                        .collect(),
                )
            }
        }
    }

    fn substitute_in_param(
        param: &Parameter,
        ctx: &SubstitutionContext,
    ) -> Parameter {
        Parameter {
            name: Self::substitute_identifier(
                &param.name,
                ctx.type_var,
                ctx.type_name,
            ),
            type_annotation: param
                .type_annotation
                .as_ref()
                .map(|t| Self::substitute_in_type(t, ctx)),
            mutable: param.mutable,
        }
    }

    fn evaluate_comptime_expr(&self, expr: &Expression) -> Result<i64> {
        match expr {
            Expression::Literal(Literal::Integer(n)) => Ok(*n),
            Expression::Infix(left, Operator::ShiftLeft, right) => {
                let l = self.evaluate_comptime_expr(left)?;
                let r = self.evaluate_comptime_expr(right)?;
                Ok(l << r)
            }
            Expression::Infix(left, Operator::ShiftRight, right) => {
                let l = self.evaluate_comptime_expr(left)?;
                let r = self.evaluate_comptime_expr(right)?;
                Ok(l >> r)
            }
            Expression::Infix(left, Operator::BitwiseAnd, right) => {
                let l = self.evaluate_comptime_expr(left)?;
                let r = self.evaluate_comptime_expr(right)?;
                Ok(l & r)
            }
            Expression::Infix(left, Operator::BitwiseOr, right) => {
                let l = self.evaluate_comptime_expr(left)?;
                let r = self.evaluate_comptime_expr(right)?;
                Ok(l | r)
            }
            Expression::Infix(left, Operator::Add, right) => {
                let l = self.evaluate_comptime_expr(left)?;
                let r = self.evaluate_comptime_expr(right)?;
                Ok(l + r)
            }
            Expression::Infix(left, Operator::Multiply, right) => {
                let l = self.evaluate_comptime_expr(left)?;
                let r = self.evaluate_comptime_expr(right)?;
                Ok(l * r)
            }
            Expression::Identifier(name) => {
                if let Some(&value) = self.comptime_constants.get(name) {
                    Ok(value)
                } else {
                    anyhow::bail!("Unknown comptime constant: {}", name)
                }
            }
            _ => anyhow::bail!(
                "Cannot evaluate expression at compile time: {:?}",
                expr
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Compiler, Instruction, Lexer, Opcode, Parser, Value64};
    use anyhow::Result;

    #[test]
    fn test_instruction_as_bytes() -> Result<()> {
        let tests = [(
            Instruction::new(Opcode::Constant, vec![u16::MAX]),
            vec![
                vec![Opcode::Constant as u8],
                u16::MAX.to_be_bytes().to_vec(),
            ]
            .concat(),
        )];

        for (instruction, expected_result) in tests {
            assert_eq!(instruction.as_bytes(), *expected_result);
        }

        Ok(())
    }

    #[test]
    fn test_compiler() -> Result<()> {
        let input = "1 + 2";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        assert_eq!(
            bytecode.constants,
            vec![Value64::Integer(1), Value64::Integer(2)]
        );
        assert_eq!(
            bytecode.instructions,
            vec![
                Instruction::new(Opcode::Constant, vec![0]),
                Instruction::new(Opcode::Constant, vec![1]),
                Instruction::new(Opcode::AddI64, vec![]),
                Instruction::new(Opcode::Pop, vec![]),
            ]
        );
        Ok(())
    }

    #[test]
    fn test_typed_compiler_integer_comparison() -> Result<()> {
        let input = "5 < 10";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        assert!(bytecode
            .instructions
            .iter()
            .any(|i| i.opcode == Opcode::GreaterThanI64));
        Ok(())
    }

    #[test]
    fn test_typed_compiler_bool_comparison() -> Result<()> {
        let input = "true == false";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        assert!(bytecode
            .instructions
            .iter()
            .any(|i| i.opcode == Opcode::EqualBool));
        Ok(())
    }

    #[test]
    fn test_typed_vm_execution() -> Result<()> {
        use crate::{Bytecode, VirtualMachine};

        let tests = [
            ("1 + 2", Value64::Integer(3)),
            ("5 - 3", Value64::Integer(2)),
            ("4 * 5", Value64::Integer(20)),
            ("10 / 2", Value64::Integer(5)),
            ("10 % 3", Value64::Integer(1)),
            ("17 % 5", Value64::Integer(2)),
            ("5 < 10", Value64::Bool(true)),
            ("5 > 10", Value64::Bool(false)),
            ("5 <= 10", Value64::Bool(true)),
            ("5 <= 5", Value64::Bool(true)),
            ("10 >= 5", Value64::Bool(true)),
            ("5 >= 5", Value64::Bool(true)),
            ("5 == 5", Value64::Bool(true)),
            ("5 != 5", Value64::Bool(false)),
            ("true == true", Value64::Bool(true)),
            ("true != false", Value64::Bool(true)),
        ];

        for (input, expected) in tests {
            let mut lexer = Lexer::new(input);
            let tokens = lexer.tokenize()?;
            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            let mut compiler = Compiler::new(&program);
            let Bytecode {
                instructions,
                constants,
                functions,
                heap,
                ..
            } = compiler.compile()?;

            let mut vm = VirtualMachine::new(constants, functions, heap);
            vm.run(&instructions)?;

            assert_eq!(vm.last_popped()?, expected, "Failed for: {}", input);
        }

        Ok(())
    }

    #[test]
    fn test_address_of_global() -> Result<()> {
        let input = "x := 5; &x";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        assert!(bytecode
            .instructions
            .iter()
            .any(|i| i.opcode == Opcode::AddressOfGlobal));
        Ok(())
    }

    #[test]
    fn test_dereference() -> Result<()> {
        let input = "x := 5; p := &x; p^";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        assert!(bytecode
            .instructions
            .iter()
            .any(|i| i.opcode == Opcode::AddressOfGlobal));
        assert!(bytecode
            .instructions
            .iter()
            .any(|i| i.opcode == Opcode::LoadPtr));
        Ok(())
    }

    #[test]
    fn test_comptime_for_produces_no_loop_instructions() -> Result<()> {
        let input = r#"
            Position :: struct { x: f64, y: f64 }
            Velocity :: struct { dx: f64, dy: f64 }
            comptime for index, T in [Position, Velocity] {
                BIT_#T :: 1 << index
            }
            BIT_Position
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        let has_jump = bytecode
            .instructions
            .iter()
            .any(|i| matches!(i.opcode, Opcode::Jump | Opcode::JumpNotTruthy));
        assert!(
            !has_jump,
            "Comptime for should not produce any loop/jump instructions"
        );

        let has_get_global = bytecode
            .instructions
            .iter()
            .any(|i| i.opcode == Opcode::GetGlobal);
        assert!(
            has_get_global,
            "Should be able to access BIT_Position at runtime"
        );

        Ok(())
    }

    #[test]
    fn test_comptime_for_expands_multiple_constants() -> Result<()> {
        let input = r#"
            A :: struct { a: i64 }
            B :: struct { b: i64 }
            C :: struct { c: i64 }
            comptime for index, T in [A, B, C] {
                MASK_#T :: 1 << index
            }
            MASK_A | MASK_B | MASK_C
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        let get_global_count = bytecode
            .instructions
            .iter()
            .filter(|i| i.opcode == Opcode::GetGlobal)
            .count();
        assert!(
            get_global_count >= 3,
            "Should have at least 3 GetGlobal for MASK_A, MASK_B, MASK_C"
        );

        let bitwise_or_count = bytecode
            .instructions
            .iter()
            .filter(|i| i.opcode == Opcode::BitwiseOr)
            .count();
        assert_eq!(bitwise_or_count, 2, "Should have 2 BitwiseOr operations");

        Ok(())
    }

    #[test]
    fn test_second_class_ref_in_struct_rejected() {
        let input = r#"
            BadStruct :: struct {
                value: &i64
            }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let program = parser.parse().unwrap();

        let mut compiler = Compiler::new(&program);
        let result = compiler.compile();

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("second-class"), "Error should mention second-class: {}", err);
    }

    #[test]
    fn test_second_class_ref_mut_in_struct_rejected() {
        let input = r#"
            BadStruct :: struct {
                value: &mut i64
            }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let program = parser.parse().unwrap();

        let mut compiler = Compiler::new(&program);
        let result = compiler.compile();

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("second-class"), "Error should mention second-class: {}", err);
    }

    #[test]
    fn test_second_class_ref_in_enum_rejected() {
        let input = r#"
            BadEnum :: enum {
                Variant { value: &i64 }
            }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let program = parser.parse().unwrap();

        let mut compiler = Compiler::new(&program);
        let result = compiler.compile();

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("second-class"), "Error should mention second-class: {}", err);
    }

    #[test]
    fn test_second_class_ref_return_type_rejected() {
        let input = r#"
            get_ref :: fn(x: i64) -> &i64 {
                &x
            }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let program = parser.parse().unwrap();

        let mut compiler = Compiler::new(&program);
        let result = compiler.compile();

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("second-class"), "Error should mention second-class: {}", err);
    }

    #[test]
    fn test_second_class_ref_param_allowed() -> Result<()> {
        let input = r#"
            update :: fn(p: &mut i64) -> void {
                p^ = 42
            }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        let mut compiler = Compiler::new(&program);
        let result = compiler.compile();

        assert!(result.is_ok(), "Reference parameters should be allowed: {:?}", result);
        Ok(())
    }

    #[test]
    fn test_second_class_local_ref_allowed() -> Result<()> {
        let input = r#"
            x := 42;
            r := &x;
            r^
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        let mut compiler = Compiler::new(&program);
        let result = compiler.compile();

        assert!(result.is_ok(), "Local reference variables should be allowed: {:?}", result);
        Ok(())
    }

    #[test]
    fn test_pointer_in_struct_allowed() -> Result<()> {
        let input = r#"
            Node :: struct {
                value: i64,
                next: ^Node
            }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        let mut compiler = Compiler::new(&program);
        let result = compiler.compile();

        assert!(result.is_ok(), "Raw pointers in structs should be allowed: {:?}", result);
        Ok(())
    }

    #[test]
    fn test_function_return_type_tracking() -> Result<()> {
        use crate::types::Type;

        let input = r#"
            Point :: struct { x: i64, y: i64 }
            make_point :: fn(a: i64, b: i64) -> Point {
                Point { x = a, y = b }
            }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        let mut compiler = Compiler::new(&program);
        compiler.compile()?;

        assert!(
            compiler.function_return_types.contains_key("make_point"),
            "function_return_types should contain 'make_point'"
        );
        assert_eq!(
            compiler.function_return_types.get("make_point"),
            Some(&Type::Struct("Point".to_string())),
            "make_point should have return type Point"
        );
        Ok(())
    }

    #[test]
    fn test_struct_field_access_on_function_return_emits_correct_offset() -> Result<()> {
        let input = r#"
            Point :: struct { x: i64, y: i64 }
            make_point :: fn() -> Point {
                Point { x = 10, y = 20 }
            }
            p := make_point();
            p.y
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        let struct_get_instructions: Vec<_> = bytecode
            .instructions
            .iter()
            .filter(|i| i.opcode == Opcode::StructGet)
            .collect();

        assert!(
            !struct_get_instructions.is_empty(),
            "Should have StructGet instructions"
        );

        let has_offset_1 = struct_get_instructions
            .iter()
            .any(|i| !i.operands.is_empty() && i.operands[0] == 1);

        assert!(
            has_offset_1,
            "Accessing p.y should emit StructGet with offset 1, found: {:?}",
            struct_get_instructions
        );
        Ok(())
    }

    #[test]
    fn test_struct_multiple_fields_access_correct_offsets() -> Result<()> {
        let input = r#"
            Vec3 :: struct { x: i64, y: i64, z: i64 }
            make_vec :: fn() -> Vec3 {
                Vec3 { x = 1, y = 2, z = 3 }
            }
            v := make_vec();
            v.x + v.y + v.z
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        let struct_get_offsets: Vec<u16> = bytecode
            .instructions
            .iter()
            .filter(|i| i.opcode == Opcode::StructGet)
            .filter_map(|i| i.operands.first().copied())
            .collect();

        assert!(
            struct_get_offsets.contains(&0),
            "Should access field at offset 0 (x)"
        );
        assert!(
            struct_get_offsets.contains(&1),
            "Should access field at offset 1 (y)"
        );
        assert!(
            struct_get_offsets.contains(&2),
            "Should access field at offset 2 (z)"
        );
        Ok(())
    }

    #[test]
    fn test_enum_return_type_tracking() -> Result<()> {
        use crate::types::Type;

        let input = r#"
            Color :: enum { Red, Green, Blue }
            get_color :: fn() -> Color {
                Color::Blue
            }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        let mut compiler = Compiler::new(&program);
        compiler.compile()?;

        assert!(
            compiler.function_return_types.contains_key("get_color"),
            "function_return_types should contain 'get_color'"
        );
        assert_eq!(
            compiler.function_return_types.get("get_color"),
            Some(&Type::Enum("Color".to_string())),
            "get_color should have return type Color"
        );
        Ok(())
    }
}
