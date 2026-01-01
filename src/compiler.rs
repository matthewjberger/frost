use crate::{
    types::Type, Expression, HeapObject, Literal, Operator, Parameter,
    Statement, StructField, Value64,
};
use anyhow::Result;
use std::{
    collections::{HashMap, HashSet},
    fmt,
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
    GreaterThanI64,
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
    pub native_names: Vec<String>,
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
            native_names: Vec::new(),
        }
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
            native_names: Vec::new(),
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
            native_names: Vec::new(),
        }
    }

    pub fn compile(&mut self) -> Result<Bytecode> {
        let mut bytecode = Bytecode::default();
        while let Some(statement) = self.statements.next() {
            self.compile_statement(statement, &mut bytecode)?;
        }
        Ok(bytecode)
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
                        self.load_symbol(&symbol, bytecode);
                        bytecode
                            .instructions
                            .push(Instruction::new(Opcode::Drop, vec![]));
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
                self.compile_expression(expression, bytecode)?;
                bytecode
                    .instructions
                    .push(Instruction::new(Opcode::Pop, vec![]));
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
                if let Expression::Function(parameters, _return_type, body) =
                    value
                {
                    self.compile_function_with_name(
                        name, parameters, body, bytecode,
                    )?;
                } else {
                    self.compile_expression(value, bytecode)?;
                }
                let opcode = if symbol.scope == SymbolScope::Global {
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
                let symbol = self.symbol_table.define(name);
                if let Expression::Function(parameters, _return_type, body) =
                    expression
                {
                    self.compile_function_with_name(
                        name, parameters, body, bytecode,
                    )?;
                } else if let Expression::Proc(parameters, _return_type, body) =
                    expression
                {
                    self.compile_function_with_name(
                        name, parameters, body, bytecode,
                    )?;
                } else {
                    self.compile_expression(expression, bytecode)?;
                }
                let opcode = if symbol.scope == SymbolScope::Global {
                    Opcode::SetGlobal
                } else {
                    Opcode::SetLocal
                };
                bytecode
                    .instructions
                    .push(Instruction::new(opcode, vec![symbol.index as u16]));
                Ok(())
            }
            Statement::Struct(name, fields) => {
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
                let mut compiled_variants = Vec::new();
                for (index, variant) in variants.iter().enumerate() {
                    let full_name = format!("{}::{}", name, variant.name);
                    self.symbol_table.define_with_type(
                        &full_name,
                        Some(Type::Enum(name.clone())),
                    );

                    let fields = variant.fields.clone().unwrap_or_default();
                    compiled_variants.push(CompiledEnumVariant {
                        name: variant.name.clone(),
                        tag: index,
                        fields: fields.clone(),
                    });

                    if variant.fields.is_none() {
                        let constant_index = bytecode.constants.len() as u16;
                        bytecode.constants.push(Value64::Integer(index as i64));
                        bytecode.instructions.push(Instruction::new(
                            Opcode::Constant,
                            vec![constant_index],
                        ));
                        let symbol =
                            self.symbol_table.resolve(&full_name).unwrap();
                        let opcode = match symbol.scope {
                            SymbolScope::Global => Opcode::SetGlobal,
                            SymbolScope::Local => Opcode::SetLocal,
                            _ => anyhow::bail!(
                                "unexpected scope for enum variant"
                            ),
                        };
                        bytecode.instructions.push(Instruction::new(
                            opcode,
                            vec![symbol.index as u16],
                        ));
                    }
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
                            let offset =
                                if let Some(Type::Struct(struct_name)) =
                                    &symbol.symbol_type
                                {
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
                            self.compile_expression(rhs, bytecode)?;
                            bytecode.instructions.push(Instruction::new(
                                Opcode::StructSet,
                                vec![offset as u16],
                            ));
                            bytecode.instructions.push(Instruction::new(
                                set_opcode,
                                vec![symbol.index as u16],
                            ));
                        } else {
                            anyhow::bail!(
                                "can only assign to fields of identifiers"
                            );
                        }
                    }
                    _ => anyhow::bail!("invalid assignment target"),
                }
                Ok(())
            }
            Statement::For(iterator, range, body) => {
                if let Expression::Range(start, end) = range {
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
                    bytecode
                        .instructions
                        .push(Instruction::new(Opcode::GreaterThanI64, vec![]));

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
                let source = std::fs::read_to_string(path)
                    .map_err(|e| anyhow::anyhow!("failed to read import '{}': {}", path, e))?;
                let mut lexer = crate::Lexer::new(&source);
                let tokens = lexer.tokenize()
                    .map_err(|e| anyhow::anyhow!("failed to tokenize import '{}': {}", path, e))?;
                let mut parser = crate::Parser::new(&tokens);
                let statements = parser.parse()
                    .map_err(|e| anyhow::anyhow!("failed to parse import '{}': {}", path, e))?;
                for statement in &statements {
                    self.compile_statement(statement, bytecode)?;
                }
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
                        self.moved_symbols.insert(name.clone());
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
                self.compile_expression(left, bytecode)?;
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

                for statement in consequence {
                    self.compile_statement(statement, bytecode)?;
                }
                self.remove_last_pop(bytecode);

                let jump_pos = bytecode.instructions.len();
                bytecode
                    .instructions
                    .push(Instruction::new(Opcode::Jump, vec![9999]));

                let after_consequence = bytecode.instructions.len();
                bytecode.instructions[jump_not_truthy_pos].operands[0] =
                    after_consequence as u16;

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

                let after_alternative = bytecode.instructions.len();
                bytecode.instructions[jump_pos].operands[0] =
                    after_alternative as u16;

                Ok(())
            }
            Expression::Function(parameters, _return_type, body)
            | Expression::Proc(parameters, _return_type, body) => {
                let outer_symbol_table = std::mem::take(&mut self.symbol_table);
                self.symbol_table =
                    SymbolTable::new_enclosed(outer_symbol_table);

                for param in parameters {
                    self.symbol_table.define(&param.name);
                }

                let mut fn_bytecode = Bytecode {
                    instructions: vec![],
                    constants: std::mem::take(&mut bytecode.constants),
                    functions: std::mem::take(&mut bytecode.functions),
                    heap: std::mem::take(&mut bytecode.heap),
                    native_names: std::mem::take(&mut bytecode.native_names),
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
                    fn_bytecode
                        .instructions
                        .push(Instruction::new(Opcode::Return, vec![]));
                }

                bytecode.constants = fn_bytecode.constants;
                bytecode.functions = fn_bytecode.functions;
                bytecode.heap = fn_bytecode.heap;

                let free_symbols = self.symbol_table.free_symbols.clone();
                let num_locals = self.symbol_table.num_definitions;
                let num_parameters = parameters.len();

                if let Some(outer) = self.symbol_table.outer.take() {
                    self.symbol_table = *outer;
                }

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
                    if let Some(Type::Struct(struct_name)) = symbol.symbol_type
                    {
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
                    self.compile_expression(expr, bytecode)?;
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
                        vec![struct_def.size as u16],
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
                    _ => anyhow::bail!("can only borrow identifiers"),
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
                    _ => anyhow::bail!("can only borrow identifiers"),
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
                self.compile_expression(function, bytecode)?;
                for arg in arguments {
                    self.compile_expression(arg, bytecode)?;
                }
                bytecode.instructions.push(Instruction::new(
                    Opcode::Call,
                    vec![arguments.len() as u16],
                ));
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
            Expression::Range(_, _) => {
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
                let _ = full_name;
                Ok(())
            }
            Expression::Switch(scrutinee, cases) => {
                self.compile_switch(scrutinee, cases, bytecode)
            }
        }
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
        self.compile_expression(scrutinee, bytecode)?;

        let mut jump_ends: Vec<usize> = Vec::new();
        let mut next_case_jumps: Vec<usize> = Vec::new();

        for (case_index, case) in cases.iter().enumerate() {
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
                    bytecode
                        .instructions
                        .push(Instruction::new(Opcode::EqualI64, vec![]));
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
                    let tag = if let Some(en) = enum_name {
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
                        let symbol = self.symbol_table.define(binding_name);
                        let opcode = match symbol.scope {
                            SymbolScope::Local => Opcode::SetLocal,
                            SymbolScope::Global => Opcode::SetGlobal,
                            _ => anyhow::bail!("unexpected scope"),
                        };
                        bytecode.instructions.push(Instruction::new(
                            opcode,
                            vec![symbol.index as u16],
                        ));
                        let _ = field_name;
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

            if case_index < cases.len() - 1 {
                let jump_pos = bytecode.instructions.len();
                bytecode
                    .instructions
                    .push(Instruction::new(Opcode::Jump, vec![9999]));
                jump_ends.push(jump_pos);
            }
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
            bytecode
                .instructions
                .push(Instruction::new(Opcode::Pop, vec![]));
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
                .push(Instruction::new(Opcode::Pop, vec![]));
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
                .push(Instruction::new(Opcode::Pop, vec![]));
            bytecode
                .instructions
                .push(Instruction::new(Opcode::True, vec![]));
            let jump_end_pos = bytecode.instructions.len();
            bytecode
                .instructions
                .push(Instruction::new(Opcode::Jump, vec![9999]));
            let false_pos = bytecode.instructions.len();
            bytecode.instructions[jump_true_pos].operands[0] = false_pos as u16;
            bytecode
                .instructions
                .push(Instruction::new(Opcode::Pop, vec![]));
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
        body: &[Statement],
        bytecode: &mut Bytecode,
    ) -> Result<()> {
        let outer_symbol_table = std::mem::take(&mut self.symbol_table);
        self.symbol_table = SymbolTable::new_enclosed(outer_symbol_table);

        self.symbol_table.define_function_name(name);

        for param in parameters {
            self.symbol_table
                .define_with_type(&param.name, param.type_annotation.clone());
        }

        let mut fn_bytecode = Bytecode {
            instructions: vec![],
            constants: std::mem::take(&mut bytecode.constants),
            functions: std::mem::take(&mut bytecode.functions),
            heap: std::mem::take(&mut bytecode.heap),
            native_names: std::mem::take(&mut bytecode.native_names),
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
            fn_bytecode
                .instructions
                .push(Instruction::new(Opcode::Return, vec![]));
        }

        bytecode.constants = fn_bytecode.constants;
        bytecode.functions = fn_bytecode.functions;
        bytecode.heap = fn_bytecode.heap;

        let free_symbols = self.symbol_table.free_symbols.clone();
        let num_locals = self.symbol_table.num_definitions;
        let num_parameters = parameters.len();

        if let Some(outer) = self.symbol_table.outer.take() {
            self.symbol_table = *outer;
        }

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

    fn replace_last_pop_with_return(&self, bytecode: &mut Bytecode) {
        if self.last_instruction_is(bytecode, Opcode::Pop) {
            bytecode.instructions.pop();
            bytecode
                .instructions
                .push(Instruction::new(Opcode::ReturnValue, vec![]));
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
}
