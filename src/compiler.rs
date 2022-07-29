use crate::{Expression, Literal, Object, Operator, Statement};
use anyhow::Result;
use std::{fmt, slice::Iter};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Opcode {
    Null,
    Constant,
}

#[derive(Debug, PartialEq)]
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
    pub bytecode: Bytecode,
    pub statements: Iter<'a, Statement>,
}

impl<'a> Compiler<'a> {
    pub fn new(statements: &'a [Statement]) -> Self {
        Self {
            bytecode: Bytecode::default(),
            statements: statements.iter(),
        }
    }

    pub fn compile(&mut self) -> Result<&Bytecode> {
        while let Some(statement) = self.statements.next() {
            self.compile_statement(statement)?;
        }
        Ok(&self.bytecode)
    }

    fn add_constant(&mut self, constant: Object) {
        self.bytecode.constants.push(constant);
    }

    fn emit(&mut self, instruction: Instruction) {
        self.bytecode.instructions.push(instruction);
    }

    fn compile_statement(&mut self, statement: &Statement) -> Result<()> {
        match statement {
            Statement::Expression(expression) => {
                self.compile_expression(expression)
            }
            _ => unimplemented!(),
        }
    }

    fn compile_expression(&mut self, expression: &Expression) -> Result<()> {
        match expression {
            Expression::Literal(literal) => self.compile_literal(literal),
            Expression::Infix(left, operator, right) => {
                self.compile_infix(left, operator, right)
            }
            _ => unimplemented!(),
        }
    }

    fn compile_literal(&mut self, literal: &Literal) -> Result<()> {
        let instruction = match literal {
            Literal::Integer(value) => {
                self.add_constant(Object::Integer(*value));
                Instruction::new(Opcode::Constant, vec![*value as u16])
            }
            _ => unimplemented!(),
        };
        self.emit(instruction);
        Ok(())
    }

    fn compile_infix(
        &mut self,
        left: &Expression,
        _operator: &Operator,
        right: &Expression,
    ) -> Result<()> {
        self.compile_expression(left)?;
        self.compile_expression(right)?;
        Ok(())
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
                    Instruction::new(Opcode::Constant, vec![1]),
                    Instruction::new(Opcode::Constant, vec![2]),
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

            assert_eq!(*bytecode, expected_result);
        }

        Ok(())
    }
}
