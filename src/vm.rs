#[repr(u8)]
pub enum Opcode {
    Constant,
}

pub type Instructions = Vec<Opcode>;

#[derive(Default)]
pub struct VirtualMachine {
    pub program_counter: u8,
    pub stack: Vec<u8>,
    pub stack_pointer: u8,
}

impl VirtualMachine {
    pub fn execute_cycle(&mut self) {}
}

#[cfg(test)]
mod tests {
    use crate::Opcode;
    use anyhow::Result;

    fn make(opcode: Opcode, operands: Vec<u16>) -> Vec<u8> {
        let operands: Vec<u8> =
            operands.iter().flat_map(|x| x.to_be_bytes()).collect();
        [vec![opcode as u8], operands].concat()
    }

    #[test]
    fn test_make() -> Result<()> {
        let tests = [(
            Opcode::Constant,
            vec![u16::MAX],
            vec![
                vec![Opcode::Constant as u8],
                u16::MAX.to_be_bytes().to_vec(),
            ]
            .concat(),
        )];

        for (opcode, operands, expected_result) in tests {
            let instruction = make(opcode, operands);
            assert_eq!(instruction, *expected_result);
        }

        Ok(())
    }
}
