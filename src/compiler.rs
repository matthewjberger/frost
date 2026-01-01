use crate::{Expression, Literal, Object, Operator, Statement};
use anyhow::Result;
use std::{collections::HashMap, fmt, slice::Iter};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SymbolScope {
    Global,
    Local,
    Builtin,
    Free,
    Function,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    pub name: String,
    pub scope: SymbolScope,
    pub index: usize,
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
        let scope = if self.outer.is_some() {
            SymbolScope::Local
        } else {
            SymbolScope::Global
        };
        let symbol = Symbol {
            name: name.to_string(),
            scope,
            index: self.num_definitions,
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
        };
        self.store.insert(name.to_string(), symbol.clone());
        symbol
    }

    pub fn define_function_name(&mut self, name: &str) -> Symbol {
        let symbol = Symbol {
            name: name.to_string(),
            scope: SymbolScope::Function,
            index: 0,
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
                if symbol.scope == SymbolScope::Global || symbol.scope == SymbolScope::Builtin {
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
    True,
    False,
    Equal,
    NotEqual,
    GreaterThan,
    Minus,
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
    Closure,
    GetFree,
    CurrentClosure,
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

#[derive(Default, Debug, PartialEq)]
pub struct Bytecode {
    pub instructions: Vec<Instruction>,
    pub constants: Vec<Object>,
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

pub struct Compiler<'a> {
    pub statements: Iter<'a, Statement>,
    pub symbol_table: SymbolTable,
}

impl<'a> Compiler<'a> {
    pub fn new(statements: &'a [Statement]) -> Self {
        let mut symbol_table = SymbolTable::new();
        let builtins = ["len", "first", "last", "rest", "push", "print"];
        for (index, name) in builtins.iter().enumerate() {
            symbol_table.define_builtin(index, name);
        }
        Self {
            statements: statements.iter(),
            symbol_table,
        }
    }

    pub fn new_with_state(statements: &'a [Statement], symbol_table: SymbolTable) -> Self {
        Self {
            statements: statements.iter(),
            symbol_table,
        }
    }

    pub fn compile(&mut self) -> Result<Bytecode> {
        let mut bytecode = Bytecode::default();
        while let Some(statement) = self.statements.next() {
            self.compile_statement(statement, &mut bytecode)?;
        }
        Ok(bytecode)
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
            Statement::Let(name, expression) => {
                let symbol = self.symbol_table.define(name);
                if let Expression::Function(parameters, body) = expression {
                    self.compile_function_with_name(name, parameters, body, bytecode)?;
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
            Statement::Return(expression) => {
                self.compile_expression(expression, bytecode)?;
                bytecode
                    .instructions
                    .push(Instruction::new(Opcode::ReturnValue, vec![]));
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
                let symbol = self
                    .symbol_table
                    .resolve(name)
                    .ok_or_else(|| anyhow::anyhow!("undefined variable: {}", name))?;
                self.load_symbol(&symbol, bytecode);
                Ok(())
            }
            Expression::Literal(literal) => self.compile_literal(literal, bytecode),
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
                    _ => unimplemented!("Prefix operator {:?} not implemented", operator),
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
                bytecode.instructions[jump_pos].operands[0] = after_alternative as u16;

                Ok(())
            }
            Expression::Function(parameters, body) => {
                let outer_symbol_table = std::mem::take(&mut self.symbol_table);
                self.symbol_table = SymbolTable::new_enclosed(outer_symbol_table);

                for param in parameters {
                    self.symbol_table.define(param);
                }

                let mut fn_bytecode = Bytecode {
                    instructions: vec![],
                    constants: std::mem::take(&mut bytecode.constants),
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
                let constant_index = bytecode.constants.len() as u16;
                bytecode.constants.push(Object::CompiledFunction(compiled_fn));
                bytecode.instructions.push(Instruction::new(
                    Opcode::Closure,
                    vec![constant_index, free_symbols.len() as u16],
                ));

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
        }
    }

    fn compile_literal(
        &mut self,
        literal: &Literal,
        bytecode: &mut Bytecode,
    ) -> Result<()> {
        match literal {
            Literal::Integer(value) => {
                let constant_index = bytecode.constants.len() as u16;
                bytecode.constants.push(Object::Integer(*value));
                bytecode
                    .instructions
                    .push(Instruction::new(Opcode::Constant, vec![constant_index]));
            }
            Literal::String(value) => {
                let constant_index = bytecode.constants.len() as u16;
                bytecode.constants.push(Object::String(value.clone()));
                bytecode
                    .instructions
                    .push(Instruction::new(Opcode::Constant, vec![constant_index]));
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

    fn compile_infix(
        &mut self,
        left: &Expression,
        operator: &Operator,
        right: &Expression,
        bytecode: &mut Bytecode,
    ) -> Result<()> {
        if *operator == Operator::LessThan {
            self.compile_expression(right, bytecode)?;
            self.compile_expression(left, bytecode)?;
            bytecode
                .instructions
                .push(Instruction::new(Opcode::GreaterThan, vec![]));
            return Ok(());
        }

        self.compile_expression(left, bytecode)?;
        self.compile_expression(right, bytecode)?;

        let opcode = match operator {
            Operator::Add => Opcode::Add,
            Operator::Subtract => Opcode::Sub,
            Operator::Multiply => Opcode::Mul,
            Operator::Divide => Opcode::Div,
            Operator::Equal => Opcode::Equal,
            Operator::NotEqual => Opcode::NotEqual,
            Operator::GreaterThan => Opcode::GreaterThan,
            _ => unimplemented!("Operator {:?} not implemented for infix", operator),
        };
        bytecode.instructions.push(Instruction::new(opcode, vec![]));
        Ok(())
    }

    fn load_symbol(&self, symbol: &Symbol, bytecode: &mut Bytecode) {
        let (opcode, operands) = match symbol.scope {
            SymbolScope::Global => (Opcode::GetGlobal, vec![symbol.index as u16]),
            SymbolScope::Local => (Opcode::GetLocal, vec![symbol.index as u16]),
            SymbolScope::Builtin => (Opcode::GetBuiltin, vec![symbol.index as u16]),
            SymbolScope::Free => (Opcode::GetFree, vec![symbol.index as u16]),
            SymbolScope::Function => (Opcode::CurrentClosure, vec![]),
        };
        bytecode.instructions.push(Instruction::new(opcode, operands));
    }

    fn compile_function_with_name(
        &mut self,
        name: &str,
        parameters: &[String],
        body: &[Statement],
        bytecode: &mut Bytecode,
    ) -> Result<()> {
        let outer_symbol_table = std::mem::take(&mut self.symbol_table);
        self.symbol_table = SymbolTable::new_enclosed(outer_symbol_table);

        self.symbol_table.define_function_name(name);

        for param in parameters {
            self.symbol_table.define(param);
        }

        let mut fn_bytecode = Bytecode {
            instructions: vec![],
            constants: std::mem::take(&mut bytecode.constants),
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
        let constant_index = bytecode.constants.len() as u16;
        bytecode.constants.push(Object::CompiledFunction(compiled_fn));
        bytecode.instructions.push(Instruction::new(
            Opcode::Closure,
            vec![constant_index, free_symbols.len() as u16],
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
    use crate::{
        Bytecode, Compiler, Instruction, Lexer, Object, Opcode, Parser,
    };
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
    fn test_instruction_strings() -> Result<()> {
        let instructions = vec![
            Instruction::new(Opcode::Constant, vec![1]),
            Instruction::new(Opcode::Constant, vec![2]),
            Instruction::new(Opcode::Constant, vec![65535]),
        ];

        let expected = [
            "0000 OpConstant 1",
            "0003 OpConstant 2",
            "0006 OpConstant 65535",
        ]
        .join("\n");

        let bytecode = Bytecode {
            instructions,
            ..Default::default()
        };

        assert_eq!(bytecode.disassemble(), expected);

        Ok(())
    }

    #[test]
    fn test_compiler() -> Result<()> {
        let tests = [(
            "1 + 2",
            Bytecode {
                constants: vec![Object::Integer(1), Object::Integer(2)],
                instructions: vec![
                    Instruction::new(Opcode::Constant, vec![0]),
                    Instruction::new(Opcode::Constant, vec![1]),
                    Instruction::new(Opcode::Add, vec![]),
                    Instruction::new(Opcode::Pop, vec![]),
                ],
            },
        )];

        for (input, expected_result) in tests {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            let mut compiler = Compiler::new(&program);
            let bytecode = compiler.compile()?;

            assert_eq!(bytecode, expected_result);
        }

        Ok(())
    }
}
