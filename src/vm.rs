use crate::{Closure, CompiledFunction, Instruction, Object, Opcode};
use anyhow::{bail, Context, Result};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

type BuiltinAction = Rc<RefCell<dyn Fn(Vec<Object>) -> Result<Object>>>;

const STACK_SIZE: usize = 2048;
const GLOBALS_SIZE: usize = 65536;
const MAX_FRAMES: usize = 1024;

#[derive(Clone)]
pub struct Frame {
    pub closure: Closure,
    pub ip: usize,
    pub base_pointer: usize,
}

impl Frame {
    pub fn new(closure: Closure, base_pointer: usize) -> Self {
        Self {
            closure,
            ip: 0,
            base_pointer,
        }
    }

    pub fn instructions(&self) -> &[Instruction] {
        &self.closure.function.instructions
    }
}

pub struct VirtualMachine {
    pub constants: Vec<Object>,
    pub stack: Vec<Object>,
    pub stack_pointer: usize,
    pub globals: Vec<Object>,
    pub frames: Vec<Frame>,
    pub frame_index: usize,
}

static BUILTINS: &[&str] = &["len", "first", "last", "rest", "push", "print"];

impl VirtualMachine {
    pub fn new(constants: Vec<Object>) -> Self {
        Self {
            constants,
            stack: Vec::with_capacity(STACK_SIZE),
            stack_pointer: 0,
            globals: vec![Object::Null; GLOBALS_SIZE],
            frames: Vec::with_capacity(MAX_FRAMES),
            frame_index: 0,
        }
    }

    fn current_frame(&self) -> &Frame {
        &self.frames[self.frame_index - 1]
    }

    fn current_frame_mut(&mut self) -> &mut Frame {
        &mut self.frames[self.frame_index - 1]
    }

    fn push_frame(&mut self, frame: Frame) {
        self.frames.push(frame);
        self.frame_index += 1;
    }

    fn pop_frame(&mut self) -> Frame {
        self.frame_index -= 1;
        self.frames.pop().unwrap()
    }

    pub fn run(&mut self, instructions: &[Instruction]) -> Result<()> {
        let main_fn = CompiledFunction {
            instructions: instructions.to_vec(),
            num_locals: 0,
            num_parameters: 0,
        };
        let main_closure = Closure {
            function: main_fn,
            free: vec![],
        };
        let main_frame = Frame::new(main_closure, 0);
        self.push_frame(main_frame);

        while self.frame_index > 0 {
            let frame = self.current_frame();
            if frame.ip >= frame.instructions().len() {
                break;
            }

            let instruction = frame.instructions()[frame.ip].clone();
            self.current_frame_mut().ip += 1;

            match instruction.opcode {
                Opcode::Constant => {
                    let constant_index = instruction.operands[0] as usize;
                    let constant = self.constants[constant_index].clone();
                    self.push(constant)?;
                }
                Opcode::Pop => {
                    self.pop()?;
                }
                Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div => {
                    self.execute_binary_operation(instruction.opcode)?;
                }
                Opcode::True => {
                    self.push(Object::Boolean(true))?;
                }
                Opcode::False => {
                    self.push(Object::Boolean(false))?;
                }
                Opcode::Equal | Opcode::NotEqual | Opcode::GreaterThan => {
                    self.execute_comparison(instruction.opcode)?;
                }
                Opcode::Minus => {
                    let operand = self.pop()?;
                    match operand {
                        Object::Integer(value) => self.push(Object::Integer(-value))?,
                        _ => bail!("unsupported type for negation"),
                    }
                }
                Opcode::Bang => {
                    let operand = self.pop()?;
                    let result = match operand {
                        Object::Boolean(value) => !value,
                        Object::Null => true,
                        _ => false,
                    };
                    self.push(Object::Boolean(result))?;
                }
                Opcode::Jump => {
                    let target = instruction.operands[0] as usize;
                    self.current_frame_mut().ip = target;
                }
                Opcode::JumpNotTruthy => {
                    let target = instruction.operands[0] as usize;
                    let condition = self.pop()?;
                    if !self.is_truthy(&condition) {
                        self.current_frame_mut().ip = target;
                    }
                }
                Opcode::Null => {
                    self.push(Object::Null)?;
                }
                Opcode::SetGlobal => {
                    let global_index = instruction.operands[0] as usize;
                    let value = self.pop()?;
                    self.globals[global_index] = value;
                }
                Opcode::GetGlobal => {
                    let global_index = instruction.operands[0] as usize;
                    let value = self.globals[global_index].clone();
                    self.push(value)?;
                }
                Opcode::Array => {
                    let num_elements = instruction.operands[0] as usize;
                    let mut elements = Vec::with_capacity(num_elements);
                    for index in (self.stack_pointer - num_elements)..self.stack_pointer {
                        elements.push(self.stack[index].clone());
                    }
                    self.stack_pointer -= num_elements;
                    self.push(Object::Array(elements))?;
                }
                Opcode::Hash => {
                    let num_elements = instruction.operands[0] as usize;
                    let mut hash = HashMap::new();
                    let start = self.stack_pointer - num_elements;
                    for index in (start..self.stack_pointer).step_by(2) {
                        let key = &self.stack[index];
                        let value = self.stack[index + 1].clone();
                        let hash_key = self.hash_key(key)?;
                        hash.insert(hash_key, value);
                    }
                    self.stack_pointer -= num_elements;
                    self.push(Object::HashMap(hash))?;
                }
                Opcode::Index => {
                    let index = self.pop()?;
                    let left = self.pop()?;
                    self.execute_index_expression(left, index)?;
                }
                Opcode::Call => {
                    let num_args = instruction.operands[0] as usize;
                    self.execute_call(num_args)?;
                }
                Opcode::ReturnValue => {
                    let return_value = self.pop()?;
                    let frame = self.pop_frame();
                    self.stack_pointer = frame.base_pointer - 1;
                    self.push(return_value)?;
                }
                Opcode::Return => {
                    let frame = self.pop_frame();
                    self.stack_pointer = frame.base_pointer - 1;
                    self.push(Object::Null)?;
                }
                Opcode::SetLocal => {
                    let local_index = instruction.operands[0] as usize;
                    let base_pointer = self.current_frame().base_pointer;
                    let value = self.pop()?;
                    let stack_index = base_pointer + local_index;
                    if stack_index >= self.stack.len() {
                        self.stack.resize(stack_index + 1, Object::Null);
                    }
                    self.stack[stack_index] = value;
                }
                Opcode::GetLocal => {
                    let local_index = instruction.operands[0] as usize;
                    let base_pointer = self.current_frame().base_pointer;
                    let value = self.stack[base_pointer + local_index].clone();
                    self.push(value)?;
                }
                Opcode::GetBuiltin => {
                    let builtin_index = instruction.operands[0] as usize;
                    let builtin = self.get_builtin(builtin_index)?;
                    self.push(builtin)?;
                }
                Opcode::Closure => {
                    let constant_index = instruction.operands[0] as usize;
                    let num_free = instruction.operands[1] as usize;

                    let constant = self.constants[constant_index].clone();
                    let function = match constant {
                        Object::CompiledFunction(f) => f,
                        _ => bail!("not a function"),
                    };

                    let mut free = Vec::with_capacity(num_free);
                    for index in 0..num_free {
                        free.push(self.stack[self.stack_pointer - num_free + index].clone());
                    }
                    self.stack_pointer -= num_free;

                    let closure = Closure { function, free };
                    self.push(Object::Closure(closure))?;
                }
                Opcode::GetFree => {
                    let free_index = instruction.operands[0] as usize;
                    let closure = &self.current_frame().closure;
                    let value = closure.free[free_index].clone();
                    self.push(value)?;
                }
                Opcode::CurrentClosure => {
                    let closure = self.current_frame().closure.clone();
                    self.push(Object::Closure(closure))?;
                }
            }
        }
        Ok(())
    }

    fn execute_call(&mut self, num_args: usize) -> Result<()> {
        let callee = self.stack[self.stack_pointer - 1 - num_args].clone();
        match callee {
            Object::Closure(closure) => self.call_closure(closure, num_args),
            Object::BuiltInFunction(builtin) => self.call_builtin(&builtin, num_args),
            _ => bail!("calling non-function"),
        }
    }

    fn call_closure(&mut self, closure: Closure, num_args: usize) -> Result<()> {
        if num_args != closure.function.num_parameters {
            bail!(
                "wrong number of arguments: want={}, got={}",
                closure.function.num_parameters,
                num_args
            );
        }

        let num_locals = closure.function.num_locals;
        let base_pointer = self.stack_pointer - num_args;
        let frame = Frame::new(closure, base_pointer);

        self.push_frame(frame);
        self.stack_pointer = base_pointer + num_locals;

        while self.stack.len() < self.stack_pointer {
            self.stack.push(Object::Null);
        }

        Ok(())
    }

    fn call_builtin(
        &mut self,
        builtin: &crate::BuiltInFunction,
        num_args: usize,
    ) -> Result<()> {
        let mut args = Vec::with_capacity(num_args);
        for index in (self.stack_pointer - num_args)..self.stack_pointer {
            args.push(self.stack[index].clone());
        }
        self.stack_pointer -= num_args + 1;

        let result = (builtin.action.borrow())(args)?;
        self.push(result)?;
        Ok(())
    }

    fn get_builtin(&self, index: usize) -> Result<Object> {
        let name = BUILTINS
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("unknown builtin index"))?;
        Ok(self.create_builtin(name))
    }

    fn create_builtin(&self, name: &str) -> Object {
        let action: BuiltinAction = match name {
            "len" => Rc::new(RefCell::new(|args: Vec<Object>| {
                if args.len() != 1 {
                    bail!("wrong number of arguments for len");
                }
                match &args[0] {
                    Object::String(s) => Ok(Object::Integer(s.len() as i64)),
                    Object::Array(a) => Ok(Object::Integer(a.len() as i64)),
                    _ => bail!("argument to len not supported"),
                }
            })),
            "first" => Rc::new(RefCell::new(|args: Vec<Object>| {
                if args.len() != 1 {
                    bail!("wrong number of arguments for first");
                }
                match &args[0] {
                    Object::Array(a) => Ok(a.first().cloned().unwrap_or(Object::Null)),
                    _ => bail!("argument to first must be array"),
                }
            })),
            "last" => Rc::new(RefCell::new(|args: Vec<Object>| {
                if args.len() != 1 {
                    bail!("wrong number of arguments for last");
                }
                match &args[0] {
                    Object::Array(a) => Ok(a.last().cloned().unwrap_or(Object::Null)),
                    _ => bail!("argument to last must be array"),
                }
            })),
            "rest" => Rc::new(RefCell::new(|args: Vec<Object>| {
                if args.len() != 1 {
                    bail!("wrong number of arguments for rest");
                }
                match &args[0] {
                    Object::Array(a) => {
                        if a.is_empty() {
                            Ok(Object::Array(vec![]))
                        } else {
                            Ok(Object::Array(a[1..].to_vec()))
                        }
                    }
                    _ => bail!("argument to rest must be array"),
                }
            })),
            "push" => Rc::new(RefCell::new(|args: Vec<Object>| {
                if args.len() != 2 {
                    bail!("wrong number of arguments for push");
                }
                match &args[0] {
                    Object::Array(a) => {
                        let mut new_array = a.clone();
                        new_array.push(args[1].clone());
                        Ok(Object::Array(new_array))
                    }
                    _ => bail!("first argument to push must be array"),
                }
            })),
            "print" => Rc::new(RefCell::new(|args: Vec<Object>| {
                for arg in args {
                    println!("{}", arg);
                }
                Ok(Object::Null)
            })),
            _ => Rc::new(RefCell::new(|_| bail!("unknown builtin"))),
        };

        Object::BuiltInFunction(crate::BuiltInFunction {
            name: name.to_string(),
            action,
        })
    }

    fn execute_binary_operation(&mut self, opcode: Opcode) -> Result<()> {
        let right = self.pop()?;
        let left = self.pop()?;

        match (&left, &right) {
            (Object::Integer(left_val), Object::Integer(right_val)) => {
                let result = match opcode {
                    Opcode::Add => left_val + right_val,
                    Opcode::Sub => left_val - right_val,
                    Opcode::Mul => left_val * right_val,
                    Opcode::Div => left_val / right_val,
                    _ => bail!("unknown integer operator"),
                };
                self.push(Object::Integer(result))
            }
            (Object::String(left_val), Object::String(right_val)) => {
                if opcode == Opcode::Add {
                    self.push(Object::String(format!("{}{}", left_val, right_val)))
                } else {
                    bail!("unknown string operator")
                }
            }
            _ => bail!("unsupported types for binary operation"),
        }
    }

    fn execute_comparison(&mut self, opcode: Opcode) -> Result<()> {
        let right = self.pop()?;
        let left = self.pop()?;

        match (&left, &right) {
            (Object::Integer(left_val), Object::Integer(right_val)) => {
                let result = match opcode {
                    Opcode::Equal => left_val == right_val,
                    Opcode::NotEqual => left_val != right_val,
                    Opcode::GreaterThan => left_val > right_val,
                    _ => bail!("unknown comparison operator"),
                };
                self.push(Object::Boolean(result))
            }
            (Object::Boolean(left_val), Object::Boolean(right_val)) => {
                let result = match opcode {
                    Opcode::Equal => left_val == right_val,
                    Opcode::NotEqual => left_val != right_val,
                    _ => bail!("unknown boolean comparison operator"),
                };
                self.push(Object::Boolean(result))
            }
            _ => bail!("unsupported types for comparison"),
        }
    }

    fn execute_index_expression(&mut self, left: Object, index: Object) -> Result<()> {
        match (&left, &index) {
            (Object::Array(elements), Object::Integer(idx)) => {
                if *idx < 0 || *idx as usize >= elements.len() {
                    self.push(Object::Null)
                } else {
                    self.push(elements[*idx as usize].clone())
                }
            }
            (Object::HashMap(hash), _) => {
                let hash_key = self.hash_key(&index)?;
                let value = hash.get(&hash_key).cloned().unwrap_or(Object::Null);
                self.push(value)
            }
            _ => bail!("index operator not supported"),
        }
    }

    fn hash_key(&self, key: &Object) -> Result<u64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        match key {
            Object::Integer(value) => value.hash(&mut hasher),
            Object::Boolean(value) => value.hash(&mut hasher),
            Object::String(value) => value.hash(&mut hasher),
            _ => bail!("unusable as hash key"),
        }
        Ok(hasher.finish())
    }

    fn is_truthy(&self, object: &Object) -> bool {
        match object {
            Object::Boolean(value) => *value,
            Object::Null => false,
            _ => true,
        }
    }

    fn push(&mut self, object: Object) -> Result<()> {
        if self.stack_pointer >= STACK_SIZE {
            bail!("stack overflow");
        }
        if self.stack_pointer >= self.stack.len() {
            self.stack.push(object);
        } else {
            self.stack[self.stack_pointer] = object;
        }
        self.stack_pointer += 1;
        Ok(())
    }

    fn pop(&mut self) -> Result<Object> {
        if self.stack_pointer == 0 {
            bail!("stack underflow");
        }
        self.stack_pointer -= 1;
        Ok(self.stack[self.stack_pointer].clone())
    }

    pub fn stack_top(&self) -> Result<&Object> {
        if self.stack_pointer == 0 {
            bail!("stack is empty");
        }
        self.stack
            .get(self.stack_pointer - 1)
            .context("Failed to get top of stack!")
    }

    pub fn last_popped(&self) -> Result<&Object> {
        self.stack
            .get(self.stack_pointer)
            .context("No popped element")
    }
}

#[cfg(test)]
mod tests {
    use crate::{Bytecode, Compiler, Lexer, Object, Parser, VirtualMachine};
    use anyhow::Result;

    fn run_vm_test(input: &str) -> Result<Object> {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;
        let mut compiler = Compiler::new(&program);
        let Bytecode {
            instructions,
            constants,
        } = compiler.compile()?;
        let mut vm = VirtualMachine::new(constants);
        vm.run(&instructions)?;
        Ok(vm.last_popped()?.clone())
    }

    #[test]
    fn test_integer_arithmetic() -> Result<()> {
        let tests = [
            ("1", Object::Integer(1)),
            ("2", Object::Integer(2)),
            ("1 + 2", Object::Integer(3)),
            ("1 - 2", Object::Integer(-1)),
            ("1 * 2", Object::Integer(2)),
            ("4 / 2", Object::Integer(2)),
            ("50 / 2 * 2 + 10 - 5", Object::Integer(55)),
            ("5 + 5 + 5 + 5 - 10", Object::Integer(10)),
            ("2 * 2 * 2 * 2 * 2", Object::Integer(32)),
            ("5 * 2 + 10", Object::Integer(20)),
            ("5 + 2 * 10", Object::Integer(25)),
            ("5 * (2 + 10)", Object::Integer(60)),
            ("-5", Object::Integer(-5)),
            ("-10", Object::Integer(-10)),
            ("-50 + 100 + -50", Object::Integer(0)),
            ("(5 + 10 * 2 + 15 / 3) * 2 + -10", Object::Integer(50)),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }

        Ok(())
    }

    #[test]
    fn test_boolean_expressions() -> Result<()> {
        let tests = [
            ("true", Object::Boolean(true)),
            ("false", Object::Boolean(false)),
            ("1 < 2", Object::Boolean(true)),
            ("1 > 2", Object::Boolean(false)),
            ("1 < 1", Object::Boolean(false)),
            ("1 > 1", Object::Boolean(false)),
            ("1 == 1", Object::Boolean(true)),
            ("1 != 1", Object::Boolean(false)),
            ("1 == 2", Object::Boolean(false)),
            ("1 != 2", Object::Boolean(true)),
            ("true == true", Object::Boolean(true)),
            ("false == false", Object::Boolean(true)),
            ("true == false", Object::Boolean(false)),
            ("true != false", Object::Boolean(true)),
            ("false != true", Object::Boolean(true)),
            ("(1 < 2) == true", Object::Boolean(true)),
            ("(1 < 2) == false", Object::Boolean(false)),
            ("(1 > 2) == true", Object::Boolean(false)),
            ("(1 > 2) == false", Object::Boolean(true)),
            ("!true", Object::Boolean(false)),
            ("!false", Object::Boolean(true)),
            ("!5", Object::Boolean(false)),
            ("!!true", Object::Boolean(true)),
            ("!!false", Object::Boolean(false)),
            ("!!5", Object::Boolean(true)),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }

        Ok(())
    }

    #[test]
    fn test_conditionals() -> Result<()> {
        let tests = [
            ("if (true) { 10 }", Object::Integer(10)),
            ("if (true) { 10 } else { 20 }", Object::Integer(10)),
            ("if (false) { 10 } else { 20 }", Object::Integer(20)),
            ("if (1) { 10 }", Object::Integer(10)),
            ("if (1 < 2) { 10 }", Object::Integer(10)),
            ("if (1 < 2) { 10 } else { 20 }", Object::Integer(10)),
            ("if (1 > 2) { 10 } else { 20 }", Object::Integer(20)),
            ("if (1 > 2) { 10 }", Object::Null),
            ("if (false) { 10 }", Object::Null),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }

        Ok(())
    }

    #[test]
    fn test_global_let_statements() -> Result<()> {
        let tests = [
            ("let one = 1; one", Object::Integer(1)),
            ("let one = 1; let two = 2; one + two", Object::Integer(3)),
            (
                "let one = 1; let two = one + one; one + two",
                Object::Integer(3),
            ),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }

        Ok(())
    }

    #[test]
    fn test_string_expressions() -> Result<()> {
        let tests = [
            (r#""monkey""#, Object::String("monkey".to_string())),
            (r#""mon" + "key""#, Object::String("monkey".to_string())),
            (
                r#""mon" + "key" + "banana""#,
                Object::String("monkeybanana".to_string()),
            ),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }

        Ok(())
    }

    #[test]
    fn test_array_literals() -> Result<()> {
        let tests = [
            ("[]", Object::Array(vec![])),
            (
                "[1, 2, 3]",
                Object::Array(vec![
                    Object::Integer(1),
                    Object::Integer(2),
                    Object::Integer(3),
                ]),
            ),
            (
                "[1 + 2, 3 * 4, 5 + 6]",
                Object::Array(vec![
                    Object::Integer(3),
                    Object::Integer(12),
                    Object::Integer(11),
                ]),
            ),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }

        Ok(())
    }

    #[test]
    fn test_index_expressions() -> Result<()> {
        let tests = [
            ("[1, 2, 3][1]", Object::Integer(2)),
            ("[1, 2, 3][0 + 2]", Object::Integer(3)),
            ("[[1, 1, 1]][0][0]", Object::Integer(1)),
            ("[][0]", Object::Null),
            ("[1, 2, 3][99]", Object::Null),
            ("[1][-1]", Object::Null),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }

        Ok(())
    }

    #[test]
    fn test_calling_functions_without_arguments() -> Result<()> {
        let tests = [
            ("let fivePlusTen = fn() { 5 + 10; }; fivePlusTen();", Object::Integer(15)),
            (
                "let one = fn() { 1; }; let two = fn() { 2; }; one() + two()",
                Object::Integer(3),
            ),
            (
                "let a = fn() { 1 }; let b = fn() { a() + 1 }; let c = fn() { b() + 1 }; c();",
                Object::Integer(3),
            ),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }

        Ok(())
    }

    #[test]
    fn test_functions_with_return_statement() -> Result<()> {
        let tests = [
            (
                "let earlyExit = fn() { return 99; 100; }; earlyExit();",
                Object::Integer(99),
            ),
            (
                "let earlyExit = fn() { return 99; return 100; }; earlyExit();",
                Object::Integer(99),
            ),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }

        Ok(())
    }

    #[test]
    fn test_functions_without_return_value() -> Result<()> {
        let tests = [
            ("let noReturn = fn() { }; noReturn();", Object::Null),
            (
                "let noReturn = fn() { }; let noReturnTwo = fn() { noReturn(); }; noReturn(); noReturnTwo();",
                Object::Null,
            ),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }

        Ok(())
    }

    #[test]
    fn test_calling_functions_with_bindings() -> Result<()> {
        let tests = [
            (
                "let one = fn() { let one = 1; one }; one();",
                Object::Integer(1),
            ),
            (
                "let oneAndTwo = fn() { let one = 1; let two = 2; one + two; }; oneAndTwo();",
                Object::Integer(3),
            ),
            (
                "let oneAndTwo = fn() { let one = 1; let two = 2; one + two; }; let threeAndFour = fn() { let three = 3; let four = 4; three + four; }; oneAndTwo() + threeAndFour();",
                Object::Integer(10),
            ),
            (
                "let firstFoobar = fn() { let foobar = 50; foobar; }; let secondFoobar = fn() { let foobar = 100; foobar; }; firstFoobar() + secondFoobar();",
                Object::Integer(150),
            ),
            (
                "let globalSeed = 50; let minusOne = fn() { let num = 1; globalSeed - num; }; let minusTwo = fn() { let num = 2; globalSeed - num; }; minusOne() + minusTwo();",
                Object::Integer(97),
            ),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }

        Ok(())
    }

    #[test]
    fn test_calling_functions_with_arguments_and_bindings() -> Result<()> {
        let tests = [
            (
                "let identity = fn(a) { a; }; identity(4);",
                Object::Integer(4),
            ),
            (
                "let sum = fn(a, b) { a + b; }; sum(1, 2);",
                Object::Integer(3),
            ),
            (
                "let sum = fn(a, b) { let c = a + b; c; }; sum(1, 2);",
                Object::Integer(3),
            ),
            (
                "let sum = fn(a, b) { let c = a + b; c; }; sum(1, 2) + sum(3, 4);",
                Object::Integer(10),
            ),
            (
                "let sum = fn(a, b) { let c = a + b; c; }; let outer = fn() { sum(1, 2) + sum(3, 4); }; outer();",
                Object::Integer(10),
            ),
            (
                "let globalNum = 10; let sum = fn(a, b) { let c = a + b; c + globalNum; }; let outer = fn() { sum(1, 2) + sum(3, 4) + globalNum; }; outer() + globalNum;",
                Object::Integer(50),
            ),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }

        Ok(())
    }

    #[test]
    fn test_builtin_functions() -> Result<()> {
        let tests = [
            (r#"len("")"#, Object::Integer(0)),
            (r#"len("four")"#, Object::Integer(4)),
            (r#"len("hello world")"#, Object::Integer(11)),
            ("len([1, 2, 3])", Object::Integer(3)),
            ("len([])", Object::Integer(0)),
            ("first([1, 2, 3])", Object::Integer(1)),
            ("first([])", Object::Null),
            ("last([1, 2, 3])", Object::Integer(3)),
            ("last([])", Object::Null),
            (
                "rest([1, 2, 3])",
                Object::Array(vec![Object::Integer(2), Object::Integer(3)]),
            ),
            ("rest([])", Object::Array(vec![])),
            (
                "push([], 1)",
                Object::Array(vec![Object::Integer(1)]),
            ),
            (
                "push([1, 2], 3)",
                Object::Array(vec![
                    Object::Integer(1),
                    Object::Integer(2),
                    Object::Integer(3),
                ]),
            ),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }

        Ok(())
    }

    #[test]
    fn test_closures() -> Result<()> {
        let tests = [
            (
                "let newClosure = fn(a) { fn() { a; }; }; let closure = newClosure(99); closure();",
                Object::Integer(99),
            ),
            (
                "let newAdder = fn(a, b) { fn(c) { a + b + c }; }; let adder = newAdder(1, 2); adder(8);",
                Object::Integer(11),
            ),
            (
                "let newAdder = fn(a, b) { let c = a + b; fn(d) { c + d }; }; let adder = newAdder(1, 2); adder(8);",
                Object::Integer(11),
            ),
            (
                "let newAdderOuter = fn(a, b) { let c = a + b; fn(d) { let e = d + c; fn(f) { e + f; }; }; }; let newAdderInner = newAdderOuter(1, 2); let adder = newAdderInner(3); adder(8);",
                Object::Integer(14),
            ),
            (
                "let a = 1; let newAdderOuter = fn(b) { fn(c) { fn(d) { a + b + c + d }; }; }; let newAdderInner = newAdderOuter(2); let adder = newAdderInner(3); adder(8);",
                Object::Integer(14),
            ),
            (
                "let newClosure = fn(a, b) { let one = fn() { a; }; let two = fn() { b; }; fn() { one() + two(); }; }; let closure = newClosure(9, 90); closure();",
                Object::Integer(99),
            ),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }

        Ok(())
    }

    #[test]
    fn test_recursive_functions() -> Result<()> {
        let tests = [
            (
                "let countDown = fn(x) { if (x == 0) { return 0; } else { countDown(x - 1); } }; countDown(1);",
                Object::Integer(0),
            ),
            (
                "let countDown = fn(x) { if (x == 0) { return 0; } else { countDown(x - 1); } }; let wrapper = fn() { countDown(1); }; wrapper();",
                Object::Integer(0),
            ),
            (
                "let wrapper = fn() { let countDown = fn(x) { if (x == 0) { return 0; } else { countDown(x - 1); } }; countDown(1); }; wrapper();",
                Object::Integer(0),
            ),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }

        Ok(())
    }

    #[test]
    fn test_fibonacci() -> Result<()> {
        let input = r#"
            let fibonacci = fn(x) {
                if (x == 0) {
                    return 0;
                } else {
                    if (x == 1) {
                        return 1;
                    } else {
                        fibonacci(x - 1) + fibonacci(x - 2);
                    }
                }
            };
            fibonacci(15);
        "#;

        let result = run_vm_test(input)?;
        assert_eq!(result, Object::Integer(610));

        Ok(())
    }
}
