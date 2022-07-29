use crate::Bytecode;

#[derive(Default)]
pub struct VirtualMachine {
    pub program_counter: u8,
    pub stack: Vec<u8>,
    pub stack_pointer: u8,
}

impl VirtualMachine {
    pub fn execute_cycle(&mut self, _bytecode: Bytecode) {}
}
