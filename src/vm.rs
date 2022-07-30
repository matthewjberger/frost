use crate::{Instruction, Object, Opcode};
use anyhow::{Context, Result};
use std::slice::Iter;

pub struct VirtualMachine<'a> {
    pub instructions: Iter<'a, Instruction>,
    pub constants: Vec<Object>,
    pub stack: Vec<Object>,
    pub stack_pointer: usize,
}

impl<'a> VirtualMachine<'a> {
    pub fn new(
        instructions: &'a [Instruction],
        constants: Vec<Object>,
    ) -> Self {
        Self {
            instructions: instructions.iter(),
            constants,
            stack: Vec::new(),
            stack_pointer: 0,
        }
    }

    pub fn execute_cycle(&mut self) -> Result<()> {
        if let Some(instruction) = self.instructions.next() {
            match instruction.opcode {
                Opcode::Null => { /* Nothing to do */ }
                Opcode::Constant => {
                    let value = instruction.operands[0];
                }
            }
        }
        Ok(())
    }

    pub fn stack_top(&mut self) -> Result<&Object> {
        self.stack
            .get(self.stack_pointer - 1)
            .context("Failed to get top of stack!")
    }
}

#[cfg(test)]
mod tests {
    use crate::{Bytecode, Compiler, Lexer, Object, Parser, VirtualMachine};
    use anyhow::Result;

    #[test]
    fn test_integer_object() -> Result<()> {
        let tests = [
            ("1", Object::Integer(1)),
            ("2", Object::Integer(2)),
            ("1 + 2", Object::Integer(2)), // FIXME
        ];

        for (input, expected_result) in tests {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            let mut compiler = Compiler::new(&program);

            let Bytecode {
                instructions,
                constants,
            } = compiler.compile()?;

            let mut virtual_machine =
                VirtualMachine::new(&instructions, constants);
            virtual_machine.execute_cycle()?;

            assert_eq!(virtual_machine.stack_top()?, &expected_result);
        }

        Ok(())
    }
}
