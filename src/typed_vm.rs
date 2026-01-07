use crate::ffi::NativeRegistry;
use crate::value::{ArenaData, PoolData};
use crate::{
    CompiledFunction, HeapObject, Instruction, Opcode, TypedBuiltIn,
    TypedClosure, Value64,
};
use anyhow::{bail, Result};
use std::collections::HashMap;

const STACK_SIZE: usize = 2048;
const GLOBALS_SIZE: usize = 65536;
const MAX_FRAMES: usize = 1024;

#[derive(Debug, Clone, Default)]
pub struct RuntimeContext {
    pub allocator: Value64,
    pub temp_allocator: Value64,
    pub logger: Value64,
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub function_index: u32,
    pub ip: usize,
    pub base_pointer: usize,
    pub free_values: Vec<Value64>,
    pub context: RuntimeContext,
}

impl Frame {
    pub fn new(
        function_index: u32,
        base_pointer: usize,
        free_values: Vec<Value64>,
    ) -> Self {
        Self {
            function_index,
            ip: 0,
            base_pointer,
            free_values,
            context: RuntimeContext::default(),
        }
    }

    pub fn new_with_context(
        function_index: u32,
        base_pointer: usize,
        free_values: Vec<Value64>,
        context: RuntimeContext,
    ) -> Self {
        Self {
            function_index,
            ip: 0,
            base_pointer,
            free_values,
            context,
        }
    }
}

pub struct VirtualMachine {
    pub stack: Vec<Value64>,
    pub stack_pointer: usize,
    pub globals: Vec<Value64>,
    pub heap: Vec<HeapObject>,
    pub heap_free_list: Vec<u32>,
    pub constants: Vec<Value64>,
    pub functions: Vec<CompiledFunction>,
    pub frames: Vec<Frame>,
    pub frame_index: usize,
    pub native_registry: Option<NativeRegistry>,
    pub native_names: Vec<String>,
    pub context_stack: Vec<RuntimeContext>,
    freed_set: std::collections::HashSet<u32>,
}

static BUILTINS: &[&str] = &[
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

impl VirtualMachine {
    pub fn new(
        constants: Vec<Value64>,
        functions: Vec<CompiledFunction>,
        heap: Vec<HeapObject>,
    ) -> Self {
        Self {
            stack: Vec::with_capacity(STACK_SIZE),
            stack_pointer: 0,
            globals: vec![Value64::Null; GLOBALS_SIZE],
            heap,
            heap_free_list: Vec::new(),
            constants,
            functions,
            frames: Vec::with_capacity(MAX_FRAMES),
            frame_index: 0,
            native_registry: None,
            native_names: Vec::new(),
            context_stack: Vec::new(),
            freed_set: std::collections::HashSet::new(),
        }
    }

    pub fn new_with_natives(
        constants: Vec<Value64>,
        functions: Vec<CompiledFunction>,
        heap: Vec<HeapObject>,
        native_registry: NativeRegistry,
        native_names: Vec<String>,
    ) -> Self {
        Self {
            stack: Vec::with_capacity(STACK_SIZE),
            stack_pointer: 0,
            globals: vec![Value64::Null; GLOBALS_SIZE],
            heap,
            heap_free_list: Vec::new(),
            constants,
            functions,
            frames: Vec::with_capacity(MAX_FRAMES),
            frame_index: 0,
            native_registry: Some(native_registry),
            native_names,
            context_stack: Vec::new(),
            freed_set: std::collections::HashSet::new(),
        }
    }

    fn current_frame(&self) -> &Frame {
        &self.frames[self.frame_index - 1]
    }

    fn current_frame_mut(&mut self) -> &mut Frame {
        &mut self.frames[self.frame_index - 1]
    }

    fn current_instructions(&self) -> &[Instruction] {
        let frame = self.current_frame();
        &self.functions[frame.function_index as usize].instructions
    }

    fn push_frame(&mut self, frame: Frame) {
        self.frames.push(frame);
        self.frame_index += 1;
    }

    fn pop_frame(&mut self) -> Frame {
        self.frame_index -= 1;
        self.frames.pop().unwrap()
    }

    fn push(&mut self, value: Value64) -> Result<()> {
        if self.stack_pointer >= STACK_SIZE {
            bail!("stack overflow");
        }
        if self.stack_pointer >= self.stack.len() {
            self.stack.push(value);
        } else {
            self.stack[self.stack_pointer] = value;
        }
        self.stack_pointer += 1;
        Ok(())
    }

    fn pop(&mut self) -> Result<Value64> {
        if self.stack_pointer == 0 {
            bail!("stack underflow");
        }
        self.stack_pointer -= 1;
        Ok(self.stack[self.stack_pointer])
    }

    pub fn stack_top(&self) -> Result<Value64> {
        if self.stack_pointer == 0 {
            bail!("stack is empty");
        }
        Ok(self.stack[self.stack_pointer - 1])
    }

    pub fn last_popped(&self) -> Result<Value64> {
        Ok(self.stack[self.stack_pointer])
    }

    fn allocate_heap(&mut self, object: HeapObject) -> u32 {
        if let Some(index) = self.heap_free_list.pop() {
            self.freed_set.remove(&index);
            self.heap[index as usize] = object;
            index
        } else {
            let index = self.heap.len() as u32;
            self.heap.push(object);
            index
        }
    }

    fn free_heap(&mut self, index: u32) {
        if self.freed_set.contains(&index) {
            panic!("double free detected for HeapRef({})", index);
        }
        self.freed_set.insert(index);
        self.heap[index as usize] = HeapObject::Free;
        self.heap_free_list.push(index);
    }

    fn get_vec2(&self, value: Value64) -> Result<(f32, f32)> {
        match value {
            Value64::HeapRef(idx) => match &self.heap[idx as usize] {
                HeapObject::Vec2(x, y) => Ok((*x, *y)),
                _ => bail!("expected Vec2"),
            },
            _ => bail!("expected Vec2 heap reference"),
        }
    }

    fn get_vec3(&self, value: Value64) -> Result<(f32, f32, f32)> {
        match value {
            Value64::HeapRef(idx) => match &self.heap[idx as usize] {
                HeapObject::Vec3(x, y, z) => Ok((*x, *y, *z)),
                _ => bail!("expected Vec3"),
            },
            _ => bail!("expected Vec3 heap reference"),
        }
    }

    fn get_vec4(&self, value: Value64) -> Result<(f32, f32, f32, f32)> {
        match value {
            Value64::HeapRef(idx) => match &self.heap[idx as usize] {
                HeapObject::Vec4(x, y, z, w) => Ok((*x, *y, *z, *w)),
                _ => bail!("expected Vec4"),
            },
            _ => bail!("expected Vec4 heap reference"),
        }
    }

    fn get_quat(&self, value: Value64) -> Result<(f32, f32, f32, f32)> {
        match value {
            Value64::HeapRef(idx) => match &self.heap[idx as usize] {
                HeapObject::Quat(x, y, z, w) => Ok((*x, *y, *z, *w)),
                _ => bail!("expected Quat"),
            },
            _ => bail!("expected Quat heap reference"),
        }
    }

    fn drop_heap_value(&mut self, index: u32) {
        self.free_heap(index);
    }

    pub fn run(&mut self, main_instructions: &[Instruction]) -> Result<()> {
        let main_fn = CompiledFunction {
            instructions: main_instructions.to_vec(),
            num_locals: 0,
            num_parameters: 0,
        };

        let main_fn_index = self.functions.len() as u32;
        self.functions.push(main_fn);

        let default_allocator_idx = self.allocate_heap(HeapObject::Arena(ArenaData::new(65536)));
        let default_temp_allocator_idx = self.allocate_heap(HeapObject::Arena(ArenaData::new(65536)));
        let default_context = RuntimeContext {
            allocator: Value64::HeapRef(default_allocator_idx),
            temp_allocator: Value64::HeapRef(default_temp_allocator_idx),
            logger: Value64::Null,
        };

        let main_frame = Frame::new_with_context(main_fn_index, 0, vec![], default_context);
        self.push_frame(main_frame);

        while self.frame_index > 0 {
            let frame = self.current_frame();
            let instructions = self.current_instructions();
            if frame.ip >= instructions.len() {
                break;
            }

            let instruction = instructions[frame.ip].clone();
            self.current_frame_mut().ip += 1;

            match instruction.opcode {
                Opcode::Constant => {
                    let constant_index = instruction.operands[0] as usize;
                    let constant = self.constants[constant_index];
                    self.push(constant)?;
                }
                Opcode::Pop => {
                    self.pop()?;
                }
                Opcode::AddI64 => {
                    let right = self.pop()?.as_i64();
                    let left = self.pop()?.as_i64();
                    self.push(Value64::Integer(left + right))?;
                }
                Opcode::SubI64 => {
                    let right = self.pop()?.as_i64();
                    let left = self.pop()?.as_i64();
                    self.push(Value64::Integer(left - right))?;
                }
                Opcode::MulI64 => {
                    let right = self.pop()?.as_i64();
                    let left = self.pop()?.as_i64();
                    self.push(Value64::Integer(left * right))?;
                }
                Opcode::DivI64 => {
                    let right = self.pop()?.as_i64();
                    let left = self.pop()?.as_i64();
                    self.push(Value64::Integer(left / right))?;
                }
                Opcode::ModI64 => {
                    let right = self.pop()?.as_i64();
                    let left = self.pop()?.as_i64();
                    self.push(Value64::Integer(left % right))?;
                }
                Opcode::AddF64 => {
                    let right = self.pop()?.as_f64();
                    let left = self.pop()?.as_f64();
                    self.push(Value64::Float(left + right))?;
                }
                Opcode::SubF64 => {
                    let right = self.pop()?.as_f64();
                    let left = self.pop()?.as_f64();
                    self.push(Value64::Float(left - right))?;
                }
                Opcode::MulF64 => {
                    let right = self.pop()?.as_f64();
                    let left = self.pop()?.as_f64();
                    self.push(Value64::Float(left * right))?;
                }
                Opcode::DivF64 => {
                    let right = self.pop()?.as_f64();
                    let left = self.pop()?.as_f64();
                    self.push(Value64::Float(left / right))?;
                }
                Opcode::AddF32 => {
                    let right = self.pop()?.as_f32();
                    let left = self.pop()?.as_f32();
                    self.push(Value64::Float32(left + right))?;
                }
                Opcode::SubF32 => {
                    let right = self.pop()?.as_f32();
                    let left = self.pop()?.as_f32();
                    self.push(Value64::Float32(left - right))?;
                }
                Opcode::MulF32 => {
                    let right = self.pop()?.as_f32();
                    let left = self.pop()?.as_f32();
                    self.push(Value64::Float32(left * right))?;
                }
                Opcode::DivF32 => {
                    let right = self.pop()?.as_f32();
                    let left = self.pop()?.as_f32();
                    self.push(Value64::Float32(left / right))?;
                }
                Opcode::NegateF32 => {
                    let value = self.pop()?.as_f32();
                    self.push(Value64::Float32(-value))?;
                }
                Opcode::Add
                | Opcode::Sub
                | Opcode::Mul
                | Opcode::Div
                | Opcode::Mod => {
                    let right = self.pop()?;
                    let left = self.pop()?;
                    let result = match (left, right, instruction.opcode) {
                        (
                            Value64::Integer(l),
                            Value64::Integer(r),
                            Opcode::Add,
                        ) => Value64::Integer(l + r),
                        (
                            Value64::Integer(l),
                            Value64::Integer(r),
                            Opcode::Sub,
                        ) => Value64::Integer(l - r),
                        (
                            Value64::Integer(l),
                            Value64::Integer(r),
                            Opcode::Mul,
                        ) => Value64::Integer(l * r),
                        (
                            Value64::Integer(l),
                            Value64::Integer(r),
                            Opcode::Div,
                        ) => Value64::Integer(l / r),
                        (
                            Value64::Integer(l),
                            Value64::Integer(r),
                            Opcode::Mod,
                        ) => Value64::Integer(l % r),
                        (Value64::Float(l), Value64::Float(r), Opcode::Add) => {
                            Value64::Float(l + r)
                        }
                        (Value64::Float(l), Value64::Float(r), Opcode::Sub) => {
                            Value64::Float(l - r)
                        }
                        (Value64::Float(l), Value64::Float(r), Opcode::Mul) => {
                            Value64::Float(l * r)
                        }
                        (Value64::Float(l), Value64::Float(r), Opcode::Div) => {
                            Value64::Float(l / r)
                        }
                        (
                            Value64::HeapRef(l),
                            Value64::HeapRef(r),
                            Opcode::Add,
                        ) => {
                            match (
                                &self.heap[l as usize],
                                &self.heap[r as usize],
                            ) {
                                (
                                    HeapObject::String(ls),
                                    HeapObject::String(rs),
                                ) => {
                                    let combined = format!("{}{}", ls, rs);
                                    let index = self.allocate_heap(
                                        HeapObject::String(combined),
                                    );
                                    Value64::HeapRef(index)
                                }
                                _ => bail!("unsupported heap types for add"),
                            }
                        }
                        _ => bail!("unsupported types for binary operation"),
                    };
                    self.push(result)?;
                }
                Opcode::ShiftLeft => {
                    let right = self.pop()?.as_i64();
                    let left = self.pop()?.as_i64();
                    self.push(Value64::Integer(left << right))?;
                }
                Opcode::ShiftRight => {
                    let right = self.pop()?.as_i64();
                    let left = self.pop()?.as_i64();
                    self.push(Value64::Integer(left >> right))?;
                }
                Opcode::BitwiseAnd => {
                    let right = self.pop()?.as_i64();
                    let left = self.pop()?.as_i64();
                    self.push(Value64::Integer(left & right))?;
                }
                Opcode::BitwiseOr => {
                    let right = self.pop()?.as_i64();
                    let left = self.pop()?.as_i64();
                    self.push(Value64::Integer(left | right))?;
                }
                Opcode::True => {
                    self.push(Value64::Bool(true))?;
                }
                Opcode::False => {
                    self.push(Value64::Bool(false))?;
                }
                Opcode::EqualI64 => {
                    let right = self.pop()?.as_i64();
                    let left = self.pop()?.as_i64();
                    self.push(Value64::Bool(left == right))?;
                }
                Opcode::NotEqualI64 => {
                    let right = self.pop()?.as_i64();
                    let left = self.pop()?.as_i64();
                    self.push(Value64::Bool(left != right))?;
                }
                Opcode::LessThanI64 => {
                    let right = self.pop()?.as_i64();
                    let left = self.pop()?.as_i64();
                    self.push(Value64::Bool(left < right))?;
                }
                Opcode::GreaterThanI64 => {
                    let right = self.pop()?.as_i64();
                    let left = self.pop()?.as_i64();
                    self.push(Value64::Bool(left > right))?;
                }
                Opcode::GreaterThanOrEqualI64 => {
                    let right = self.pop()?.as_i64();
                    let left = self.pop()?.as_i64();
                    self.push(Value64::Bool(left >= right))?;
                }
                Opcode::LessThanOrEqualI64 => {
                    let right = self.pop()?.as_i64();
                    let left = self.pop()?.as_i64();
                    self.push(Value64::Bool(left <= right))?;
                }
                Opcode::EqualBool => {
                    let right = self.pop()?.as_bool();
                    let left = self.pop()?.as_bool();
                    self.push(Value64::Bool(left == right))?;
                }
                Opcode::NotEqualBool => {
                    let right = self.pop()?.as_bool();
                    let left = self.pop()?.as_bool();
                    self.push(Value64::Bool(left != right))?;
                }
                Opcode::Equal | Opcode::NotEqual | Opcode::GreaterThan => {
                    let right = self.pop()?;
                    let left = self.pop()?;
                    let result = match (left, right, instruction.opcode) {
                        (
                            Value64::Integer(l),
                            Value64::Integer(r),
                            Opcode::Equal,
                        ) => l == r,
                        (
                            Value64::Integer(l),
                            Value64::Integer(r),
                            Opcode::NotEqual,
                        ) => l != r,
                        (
                            Value64::Integer(l),
                            Value64::Integer(r),
                            Opcode::GreaterThan,
                        ) => l > r,
                        (
                            Value64::Float(l),
                            Value64::Float(r),
                            Opcode::Equal,
                        ) => l == r,
                        (
                            Value64::Float(l),
                            Value64::Float(r),
                            Opcode::NotEqual,
                        ) => l != r,
                        (
                            Value64::Float(l),
                            Value64::Float(r),
                            Opcode::GreaterThan,
                        ) => l > r,
                        (Value64::Bool(l), Value64::Bool(r), Opcode::Equal) => {
                            l == r
                        }
                        (
                            Value64::Bool(l),
                            Value64::Bool(r),
                            Opcode::NotEqual,
                        ) => l != r,
                        (Value64::HeapRef(l), Value64::HeapRef(r), op) => {
                            match (&self.heap[l as usize], &self.heap[r as usize]) {
                                (HeapObject::String(ls), HeapObject::String(rs)) => {
                                    match op {
                                        Opcode::Equal => ls == rs,
                                        Opcode::NotEqual => ls != rs,
                                        Opcode::GreaterThan => ls > rs,
                                        _ => unreachable!(),
                                    }
                                }
                                _ => bail!("unsupported heap comparison"),
                            }
                        }
                        (Value64::Integer(addr), Value64::HeapRef(r), op) if (addr as u64) & 0x80000000 != 0 => {
                            let stack_addr = (addr as u64 & !0x80000000) as usize;
                            if let Value64::HeapRef(l) = self.stack[stack_addr] {
                                match (&self.heap[l as usize], &self.heap[r as usize]) {
                                    (HeapObject::String(ls), HeapObject::String(rs)) => {
                                        match op {
                                            Opcode::Equal => ls == rs,
                                            Opcode::NotEqual => ls != rs,
                                            Opcode::GreaterThan => ls > rs,
                                            _ => unreachable!(),
                                        }
                                    }
                                    _ => bail!("unsupported heap comparison"),
                                }
                            } else {
                                bail!("expected string reference")
                            }
                        }
                        (Value64::HeapRef(l), Value64::Integer(addr), op) if (addr as u64) & 0x80000000 != 0 => {
                            let stack_addr = (addr as u64 & !0x80000000) as usize;
                            if let Value64::HeapRef(r) = self.stack[stack_addr] {
                                match (&self.heap[l as usize], &self.heap[r as usize]) {
                                    (HeapObject::String(ls), HeapObject::String(rs)) => {
                                        match op {
                                            Opcode::Equal => ls == rs,
                                            Opcode::NotEqual => ls != rs,
                                            Opcode::GreaterThan => ls > rs,
                                            _ => unreachable!(),
                                        }
                                    }
                                    _ => bail!("unsupported heap comparison"),
                                }
                            } else {
                                bail!("expected string reference")
                            }
                        }
                        _ => bail!("unsupported types for comparison"),
                    };
                    self.push(Value64::Bool(result))?;
                }
                Opcode::Minus | Opcode::NegateI64 => {
                    let operand = self.pop()?;
                    match operand {
                        Value64::Integer(v) => {
                            self.push(Value64::Integer(-v))?
                        }
                        Value64::Float(v) => self.push(Value64::Float(-v))?,
                        _ => bail!("unsupported type for negation"),
                    }
                }
                Opcode::NegateF64 => {
                    let v = self.pop()?.as_f64();
                    self.push(Value64::Float(-v))?;
                }
                Opcode::Bang => {
                    let operand = self.pop()?;
                    let result = !operand.is_truthy();
                    self.push(Value64::Bool(result))?;
                }
                Opcode::Jump => {
                    let target = instruction.operands[0] as usize;
                    self.current_frame_mut().ip = target;
                }
                Opcode::JumpNotTruthy => {
                    let target = instruction.operands[0] as usize;
                    let condition = self.pop()?;
                    if !condition.is_truthy() {
                        self.current_frame_mut().ip = target;
                    }
                }
                Opcode::Null => {
                    self.push(Value64::Null)?;
                }
                Opcode::SetGlobal => {
                    let global_index = instruction.operands[0] as usize;
                    let value = self.pop()?;
                    self.globals[global_index] = value;
                }
                Opcode::GetGlobal => {
                    let global_index = instruction.operands[0] as usize;
                    let value = self.globals[global_index];
                    self.push(value)?;
                }
                Opcode::SetLocal => {
                    let local_index = instruction.operands[0] as usize;
                    let base_pointer = self.current_frame().base_pointer;
                    let value = self.pop()?;
                    let stack_index = base_pointer + local_index;
                    if stack_index >= self.stack.len() {
                        self.stack.resize(stack_index + 1, Value64::Null);
                    }
                    self.stack[stack_index] = value;
                }
                Opcode::GetLocal => {
                    let local_index = instruction.operands[0] as usize;
                    let base_pointer = self.current_frame().base_pointer;
                    let value = self.stack[base_pointer + local_index];
                    self.push(value)?;
                }
                Opcode::Array => {
                    let num_elements = instruction.operands[0] as usize;
                    let mut elements = Vec::with_capacity(num_elements);
                    for index in
                        (self.stack_pointer - num_elements)..self.stack_pointer
                    {
                        elements.push(self.stack[index]);
                    }
                    self.stack_pointer -= num_elements;
                    let heap_index =
                        self.allocate_heap(HeapObject::Array(elements));
                    self.push(Value64::HeapRef(heap_index))?;
                }
                Opcode::Hash => {
                    let num_elements = instruction.operands[0] as usize;
                    let mut hash = HashMap::new();
                    let start = self.stack_pointer - num_elements;
                    for index in (start..self.stack_pointer).step_by(2) {
                        let key = self.stack[index];
                        let value = self.stack[index + 1];
                        let hash_key = self.hash_key(key)?;
                        hash.insert(hash_key, value);
                    }
                    self.stack_pointer -= num_elements;
                    let heap_index =
                        self.allocate_heap(HeapObject::HashMap(hash));
                    self.push(Value64::HeapRef(heap_index))?;
                }
                Opcode::Index => {
                    let index = self.pop()?;
                    let left = self.pop()?;
                    self.execute_index_expression(left, index)?;
                }
                Opcode::IndexSet => {
                    let value = self.pop()?;
                    let index = self.pop()?;
                    let array = self.pop()?;
                    self.execute_index_set(array, index, value)?;
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
                    self.push(Value64::Null)?;
                }
                Opcode::GetBuiltin => {
                    let builtin_index = instruction.operands[0] as usize;
                    let name = BUILTINS[builtin_index].to_string();
                    let builtin = TypedBuiltIn {
                        name,
                        index: builtin_index as u32,
                    };
                    let heap_index =
                        self.allocate_heap(HeapObject::BuiltIn(builtin));
                    self.push(Value64::HeapRef(heap_index))?;
                }
                Opcode::GetNative => {
                    let native_index = instruction.operands[0] as usize;
                    let name = self.native_names[native_index].clone();
                    let heap_index =
                        self.allocate_heap(HeapObject::NativeFunction(name));
                    self.push(Value64::HeapRef(heap_index))?;
                }
                Opcode::Closure => {
                    let function_index = instruction.operands[0] as u32;
                    let num_free = instruction.operands[1] as usize;

                    let mut free = Vec::with_capacity(num_free);
                    for index in 0..num_free {
                        free.push(
                            self.stack[self.stack_pointer - num_free + index],
                        );
                    }
                    self.stack_pointer -= num_free;

                    let closure = TypedClosure {
                        function_index,
                        free,
                    };
                    let heap_index =
                        self.allocate_heap(HeapObject::Closure(closure));
                    self.push(Value64::HeapRef(heap_index))?;
                }
                Opcode::GetFree => {
                    let free_index = instruction.operands[0] as usize;
                    let value = self.current_frame().free_values[free_index];
                    self.push(value)?;
                }
                Opcode::CurrentClosure => {
                    let frame = self.current_frame();
                    let closure = TypedClosure {
                        function_index: frame.function_index,
                        free: frame.free_values.clone(),
                    };
                    let heap_index =
                        self.allocate_heap(HeapObject::Closure(closure));
                    self.push(Value64::HeapRef(heap_index))?;
                }
                Opcode::LoadPtr => {
                    let pointer = self.pop()?;
                    match pointer {
                        Value64::Integer(address) => {
                            let addr = address as usize;
                            if addr & 0x80000000 != 0 {
                                let stack_addr = addr & 0x7FFFFFFF;
                                let value = self.stack[stack_addr];
                                self.push(value)?;
                            } else if addr & 0x40000000 != 0 {
                                let global_idx = addr & 0x3FFFFFFF;
                                let value = self.globals[global_idx];
                                self.push(value)?;
                            } else {
                                bail!("heap pointer access not supported in typed VM yet");
                            }
                        }
                        _ => bail!("LoadPtr requires an integer pointer"),
                    }
                }
                Opcode::StorePtr => {
                    let value = self.pop()?;
                    let pointer = self.pop()?;
                    match pointer {
                        Value64::Integer(address) => {
                            let addr = address as usize;
                            if addr & 0x80000000 != 0 {
                                let stack_addr = addr & 0x7FFFFFFF;
                                self.stack[stack_addr] = value;
                            } else if addr & 0x40000000 != 0 {
                                let global_idx = addr & 0x3FFFFFFF;
                                self.globals[global_idx] = value;
                            } else {
                                bail!("heap pointer store not supported in typed VM yet");
                            }
                        }
                        _ => bail!("StorePtr requires an integer pointer"),
                    }
                }
                Opcode::AddressOfLocal => {
                    let local_index = instruction.operands[0] as usize;
                    let base_pointer = self.current_frame().base_pointer;
                    let stack_address = base_pointer + local_index;
                    self.push(Value64::Integer(
                        (stack_address | 0x80000000) as i64,
                    ))?;
                }
                Opcode::AddressOfGlobal => {
                    let global_index = instruction.operands[0] as usize;
                    self.push(Value64::Integer(
                        (global_index | 0x40000000) as i64,
                    ))?;
                }
                Opcode::Alloc => {
                    let size = instruction.operands[0] as usize;
                    let elements = vec![Value64::Null; size];
                    let heap_index =
                        self.allocate_heap(HeapObject::Array(elements));
                    self.push(Value64::HeapRef(heap_index))?;
                }
                Opcode::Free => {
                    let pointer = self.pop()?;
                    match pointer {
                        Value64::HeapRef(idx) => {
                            self.free_heap(idx);
                        }
                        _ => bail!("Free requires a heap reference"),
                    }
                }
                Opcode::StructAlloc => {
                    let size = instruction.operands[0] as usize;
                    let fields = vec![Value64::Null; size];
                    let heap_index = self.allocate_heap(HeapObject::Struct(
                        "".to_string(),
                        fields,
                    ));
                    self.push(Value64::HeapRef(heap_index))?;
                }
                Opcode::StructGet => {
                    let offset = instruction.operands[0] as usize;
                    let struct_val = self.pop()?;
                    match struct_val {
                        Value64::HeapRef(idx) => {
                            if let HeapObject::Struct(_, fields) =
                                &self.heap[idx as usize]
                            {
                                let value = fields
                                    .get(offset)
                                    .copied()
                                    .unwrap_or(Value64::Null);
                                self.push(value)?;
                            } else {
                                bail!("StructGet requires a struct");
                            }
                        }
                        _ => bail!("StructGet requires a heap reference"),
                    }
                }
                Opcode::StructSet => {
                    let offset = instruction.operands[0] as usize;
                    let value = self.pop()?;
                    let struct_val = self.pop()?;
                    match struct_val {
                        Value64::HeapRef(idx) => {
                            if let HeapObject::Struct(_, fields) =
                                &mut self.heap[idx as usize]
                            {
                                if offset < fields.len() {
                                    fields[offset] = value;
                                }
                            }
                            self.push(struct_val)?;
                        }
                        _ => bail!("StructSet requires a heap reference"),
                    }
                }
                Opcode::TaggedUnionAlloc => {
                    let num_fields = instruction.operands[0] as usize;
                    let fields = vec![Value64::Null; num_fields];
                    let heap_index =
                        self.allocate_heap(HeapObject::TaggedUnion(0, fields));
                    self.push(Value64::HeapRef(heap_index))?;
                }
                Opcode::TaggedUnionSetTag => {
                    let tag = instruction.operands[0] as u32;
                    let union_val = self.stack[self.stack_pointer - 1];
                    if let Value64::HeapRef(idx) = union_val {
                        if let HeapObject::TaggedUnion(t, _) =
                            &mut self.heap[idx as usize]
                        {
                            *t = tag;
                        }
                    }
                }
                Opcode::TaggedUnionGetTag => {
                    let union_val = self.pop()?;
                    match union_val {
                        Value64::HeapRef(idx) => {
                            if let HeapObject::TaggedUnion(tag, _) =
                                &self.heap[idx as usize]
                            {
                                self.push(Value64::Integer(*tag as i64))?;
                            } else {
                                bail!("TaggedUnionGetTag requires a tagged union");
                            }
                        }
                        Value64::Integer(tag) => {
                            self.push(Value64::Integer(tag))?;
                        }
                        _ => bail!("TaggedUnionGetTag requires a heap reference or integer"),
                    }
                }
                Opcode::TaggedUnionGetField => {
                    let offset = instruction.operands[0] as usize;
                    let union_val = self.pop()?;
                    if let Value64::HeapRef(idx) = union_val {
                        if let HeapObject::TaggedUnion(_, fields) =
                            &self.heap[idx as usize]
                        {
                            let value = fields
                                .get(offset)
                                .copied()
                                .unwrap_or(Value64::Null);
                            self.push(value)?;
                        } else {
                            bail!(
                                "TaggedUnionGetField requires a tagged union"
                            );
                        }
                    } else {
                        bail!("TaggedUnionGetField requires a heap reference");
                    }
                }
                Opcode::TaggedUnionSetField => {
                    let offset = instruction.operands[0] as usize;
                    let value = self.pop()?;
                    let union_val = self.stack[self.stack_pointer - 1];
                    if let Value64::HeapRef(idx) = union_val {
                        if let HeapObject::TaggedUnion(_, fields) =
                            &mut self.heap[idx as usize]
                        {
                            if offset < fields.len() {
                                fields[offset] = value;
                            }
                        }
                    }
                }
                Opcode::TupleAlloc => {
                    let num_elements = instruction.operands[0] as usize;
                    let mut elements = Vec::with_capacity(num_elements);
                    for _ in 0..num_elements {
                        elements.push(self.pop()?);
                    }
                    elements.reverse();
                    let heap_index =
                        self.allocate_heap(HeapObject::Array(elements));
                    self.push(Value64::HeapRef(heap_index))?;
                }
                Opcode::TupleGet => {
                    let index = instruction.operands[0] as usize;
                    let tuple_val = self.pop()?;
                    if let Value64::HeapRef(idx) = tuple_val {
                        if let HeapObject::Array(elements) =
                            &self.heap[idx as usize]
                        {
                            let value = elements
                                .get(index)
                                .copied()
                                .unwrap_or(Value64::Null);
                            self.push(value)?;
                        } else {
                            bail!("TupleGet requires an array/tuple");
                        }
                    } else {
                        bail!("TupleGet requires a heap reference");
                    }
                }
                Opcode::Dup => {
                    let value = self.stack[self.stack_pointer - 1];
                    self.push(value)?;
                }
                Opcode::Drop => {
                    let value = self.pop()?;
                    if let Value64::HeapRef(idx) = value {
                        self.drop_heap_value(idx);
                    }
                }
                Opcode::GetContext => {
                    let context = &self.current_frame().context;
                    let fields = vec![
                        context.allocator,
                        context.temp_allocator,
                        context.logger,
                    ];
                    let heap_index = self.allocate_heap(HeapObject::Struct("Context".to_string(), fields));
                    self.push(Value64::HeapRef(heap_index))?;
                }
                Opcode::SetContext => {
                    let context_val = self.pop()?;
                    if let Value64::HeapRef(idx) = context_val {
                        if let HeapObject::Struct(_, fields) = &self.heap[idx as usize] {
                            let new_context = RuntimeContext {
                                allocator: fields.first().copied().unwrap_or(Value64::Null),
                                temp_allocator: fields.get(1).copied().unwrap_or(Value64::Null),
                                logger: fields.get(2).copied().unwrap_or(Value64::Null),
                            };
                            self.current_frame_mut().context = new_context;
                        }
                    }
                }
                Opcode::GetContextField => {
                    let field_index = instruction.operands[0] as usize;
                    let context = &self.current_frame().context;
                    let value = match field_index {
                        0 => context.allocator,
                        1 => context.temp_allocator,
                        2 => context.logger,
                        _ => Value64::Null,
                    };
                    self.push(value)?;
                }
                Opcode::SetContextField => {
                    let field_index = instruction.operands[0] as usize;
                    let value = self.pop()?;
                    match field_index {
                        0 => self.current_frame_mut().context.allocator = value,
                        1 => self.current_frame_mut().context.temp_allocator = value,
                        2 => self.current_frame_mut().context.logger = value,
                        _ => {}
                    }
                }
                Opcode::PushContextScope => {
                    let current_context = self.current_frame().context.clone();
                    self.context_stack.push(current_context);
                }
                Opcode::PopContextScope => {
                    if let Some(previous_context) = self.context_stack.pop() {
                        self.current_frame_mut().context = previous_context;
                    }
                }
            }
        }
        Ok(())
    }

    fn execute_call(&mut self, num_args: usize) -> Result<()> {
        let callee = self.stack[self.stack_pointer - 1 - num_args];
        match callee {
            Value64::HeapRef(idx) => match &self.heap[idx as usize] {
                HeapObject::Closure(closure) => {
                    let function_index = closure.function_index;
                    let free_values = closure.free.clone();
                    let num_parameters =
                        self.functions[function_index as usize].num_parameters;
                    let num_locals =
                        self.functions[function_index as usize].num_locals;

                    if num_args != num_parameters {
                        bail!(
                            "wrong number of arguments: want={}, got={}",
                            num_parameters,
                            num_args
                        );
                    }

                    let base_pointer = self.stack_pointer - num_args;
                    let caller_context = self.current_frame().context.clone();
                    let frame =
                        Frame::new_with_context(function_index, base_pointer, free_values, caller_context);
                    self.push_frame(frame);
                    self.stack_pointer = base_pointer + num_locals;

                    while self.stack.len() < self.stack_pointer {
                        self.stack.push(Value64::Null);
                    }
                }
                HeapObject::BuiltIn(builtin) => {
                    let name = builtin.name.clone();
                    self.call_builtin(&name, num_args)?;
                }
                HeapObject::NativeFunction(name) => {
                    let name = name.clone();
                    self.call_native(&name, num_args)?;
                }
                _ => bail!("calling non-function heap object"),
            },
            _ => bail!("calling non-function"),
        }
        Ok(())
    }

    fn call_builtin(&mut self, name: &str, num_args: usize) -> Result<()> {
        let mut args = Vec::with_capacity(num_args);
        for index in (self.stack_pointer - num_args)..self.stack_pointer {
            args.push(self.stack[index]);
        }
        self.stack_pointer -= num_args + 1;

        let result = match name {
            "len" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for len");
                }
                match args[0] {
                    Value64::HeapRef(idx) => match &self.heap[idx as usize] {
                        HeapObject::String(s) => {
                            Value64::Integer(s.len() as i64)
                        }
                        HeapObject::Array(a) => {
                            Value64::Integer(a.len() as i64)
                        }
                        _ => bail!("argument to len not supported"),
                    },
                    _ => bail!("argument to len must be a heap reference"),
                }
            }
            "first" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for first");
                }
                match args[0] {
                    Value64::HeapRef(idx) => {
                        if let HeapObject::Array(a) = &self.heap[idx as usize] {
                            a.first().copied().unwrap_or(Value64::Null)
                        } else {
                            bail!("argument to first must be array");
                        }
                    }
                    _ => bail!("argument to first must be a heap reference"),
                }
            }
            "last" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for last");
                }
                match args[0] {
                    Value64::HeapRef(idx) => {
                        if let HeapObject::Array(a) = &self.heap[idx as usize] {
                            a.last().copied().unwrap_or(Value64::Null)
                        } else {
                            bail!("argument to last must be array");
                        }
                    }
                    _ => bail!("argument to last must be a heap reference"),
                }
            }
            "rest" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for rest");
                }
                match args[0] {
                    Value64::HeapRef(idx) => {
                        if let HeapObject::Array(a) = &self.heap[idx as usize] {
                            let rest = if a.is_empty() {
                                vec![]
                            } else {
                                a[1..].to_vec()
                            };
                            let heap_idx =
                                self.allocate_heap(HeapObject::Array(rest));
                            Value64::HeapRef(heap_idx)
                        } else {
                            bail!("argument to rest must be array");
                        }
                    }
                    _ => bail!("argument to rest must be a heap reference"),
                }
            }
            "push" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for push");
                }
                match args[0] {
                    Value64::HeapRef(idx) => {
                        if let HeapObject::Array(a) = &self.heap[idx as usize] {
                            let mut new_array = a.clone();
                            new_array.push(args[1]);
                            let heap_idx = self
                                .allocate_heap(HeapObject::Array(new_array));
                            Value64::HeapRef(heap_idx)
                        } else {
                            bail!("first argument to push must be array");
                        }
                    }
                    _ => {
                        bail!("first argument to push must be a heap reference")
                    }
                }
            }
            "print" => {
                for arg in args {
                    match arg {
                        Value64::HeapRef(idx) => {
                            println!("{}", self.heap[idx as usize])
                        }
                        _ => println!("{}", arg),
                    }
                }
                Value64::Null
            }
            "abs" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for abs");
                }
                match args[0] {
                    Value64::Integer(n) => Value64::Integer(n.abs()),
                    Value64::Float(n) => Value64::Float(n.abs()),
                    _ => bail!("argument to abs must be a number"),
                }
            }
            "min" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for min");
                }
                match (args[0], args[1]) {
                    (Value64::Integer(a), Value64::Integer(b)) => {
                        Value64::Integer(a.min(b))
                    }
                    (Value64::Float(a), Value64::Float(b)) => {
                        Value64::Float(a.min(b))
                    }
                    _ => {
                        bail!("arguments to min must be the same numeric type")
                    }
                }
            }
            "max" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for max");
                }
                match (args[0], args[1]) {
                    (Value64::Integer(a), Value64::Integer(b)) => {
                        Value64::Integer(a.max(b))
                    }
                    (Value64::Float(a), Value64::Float(b)) => {
                        Value64::Float(a.max(b))
                    }
                    _ => {
                        bail!("arguments to max must be the same numeric type")
                    }
                }
            }
            "substr" => {
                if args.len() != 3 {
                    bail!("wrong number of arguments for substr");
                }
                match (args[0], args[1], args[2]) {
                    (
                        Value64::HeapRef(idx),
                        Value64::Integer(start),
                        Value64::Integer(length),
                    ) => {
                        if let HeapObject::String(s) = &self.heap[idx as usize]
                        {
                            let start = start as usize;
                            let length = length as usize;
                            let sub = if start >= s.len() {
                                String::new()
                            } else {
                                s.chars().skip(start).take(length).collect()
                            };
                            let heap_idx =
                                self.allocate_heap(HeapObject::String(sub));
                            Value64::HeapRef(heap_idx)
                        } else {
                            bail!("first argument to substr must be a string");
                        }
                    }
                    _ => bail!("substr takes (string, start, length)"),
                }
            }
            "contains" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for contains");
                }
                match (args[0], args[1]) {
                    (Value64::HeapRef(idx1), Value64::HeapRef(idx2)) => {
                        match (
                            &self.heap[idx1 as usize],
                            &self.heap[idx2 as usize],
                        ) {
                            (
                                HeapObject::String(haystack),
                                HeapObject::String(needle),
                            ) => Value64::Bool(
                                haystack.contains(needle.as_str()),
                            ),
                            _ => bail!("arguments to contains must be strings"),
                        }
                    }
                    _ => bail!("arguments to contains must be strings"),
                }
            }
            "to_string" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for to_string");
                }
                let s = match args[0] {
                    Value64::Integer(n) => n.to_string(),
                    Value64::Float(n) => n.to_string(),
                    Value64::Float32(n) => n.to_string(),
                    Value64::Bool(b) => b.to_string(),
                    Value64::Null => "null".to_string(),
                    Value64::HeapRef(idx) => match &self.heap[idx as usize] {
                        HeapObject::String(s) => s.clone(),
                        obj => format!("{}", obj),
                    },
                };
                let heap_idx = self.allocate_heap(HeapObject::String(s));
                Value64::HeapRef(heap_idx)
            }
            "parse_int" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for parse_int");
                }
                match args[0] {
                    Value64::HeapRef(idx) => {
                        if let HeapObject::String(s) = &self.heap[idx as usize]
                        {
                            match s.trim().parse::<i64>() {
                                Ok(n) => Value64::Integer(n),
                                Err(_) => Value64::Null,
                            }
                        } else {
                            bail!("argument to parse_int must be a string");
                        }
                    }
                    _ => bail!("argument to parse_int must be a string"),
                }
            }
            "floor" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for floor");
                }
                match args[0] {
                    Value64::Float(n) => Value64::Float(n.floor()),
                    Value64::Integer(n) => Value64::Integer(n),
                    _ => bail!("argument to floor must be a number"),
                }
            }
            "ceil" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for ceil");
                }
                match args[0] {
                    Value64::Float(n) => Value64::Float(n.ceil()),
                    Value64::Integer(n) => Value64::Integer(n),
                    _ => bail!("argument to ceil must be a number"),
                }
            }
            "sqrt" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for sqrt");
                }
                match args[0] {
                    Value64::Float(n) => Value64::Float(n.sqrt()),
                    Value64::Integer(n) => Value64::Float((n as f64).sqrt()),
                    _ => bail!("argument to sqrt must be a number"),
                }
            }
            "read_file" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for read_file");
                }
                match args[0] {
                    Value64::HeapRef(idx) => {
                        if let HeapObject::String(path) =
                            &self.heap[idx as usize]
                        {
                            match std::fs::read_to_string(path) {
                                Ok(contents) => {
                                    let heap_idx = self.allocate_heap(
                                        HeapObject::String(contents),
                                    );
                                    Value64::HeapRef(heap_idx)
                                }
                                Err(_) => Value64::Null,
                            }
                        } else {
                            bail!("argument to read_file must be a string");
                        }
                    }
                    _ => bail!("argument to read_file must be a string"),
                }
            }
            "write_file" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for write_file");
                }
                match (args[0], args[1]) {
                    (Value64::HeapRef(idx1), Value64::HeapRef(idx2)) => {
                        match (
                            &self.heap[idx1 as usize],
                            &self.heap[idx2 as usize],
                        ) {
                            (
                                HeapObject::String(path),
                                HeapObject::String(content),
                            ) => match std::fs::write(path, content) {
                                Ok(_) => Value64::Bool(true),
                                Err(_) => Value64::Bool(false),
                            },
                            _ => {
                                bail!("arguments to write_file must be strings")
                            }
                        }
                    }
                    _ => bail!("arguments to write_file must be strings"),
                }
            }
            "file_exists" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for file_exists");
                }
                match args[0] {
                    Value64::HeapRef(idx) => {
                        if let HeapObject::String(path) =
                            &self.heap[idx as usize]
                        {
                            Value64::Bool(std::path::Path::new(path).exists())
                        } else {
                            bail!("argument to file_exists must be a string");
                        }
                    }
                    _ => bail!("argument to file_exists must be a string"),
                }
            }
            "append_file" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for append_file");
                }
                match (args[0], args[1]) {
                    (Value64::HeapRef(idx1), Value64::HeapRef(idx2)) => {
                        match (
                            &self.heap[idx1 as usize],
                            &self.heap[idx2 as usize],
                        ) {
                            (
                                HeapObject::String(path),
                                HeapObject::String(content),
                            ) => {
                                use std::io::Write;
                                match std::fs::OpenOptions::new()
                                    .create(true)
                                    .append(true)
                                    .open(path)
                                {
                                    Ok(mut file) => {
                                        match file.write_all(content.as_bytes())
                                        {
                                            Ok(_) => Value64::Bool(true),
                                            Err(_) => Value64::Bool(false),
                                        }
                                    }
                                    Err(_) => Value64::Bool(false),
                                }
                            }
                            _ => bail!(
                                "arguments to append_file must be strings"
                            ),
                        }
                    }
                    _ => bail!("arguments to append_file must be strings"),
                }
            }
            "vec2" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec2");
                }
                let x = args[0].as_f32();
                let y = args[1].as_f32();
                let heap_idx = self.allocate_heap(HeapObject::Vec2(x, y));
                Value64::HeapRef(heap_idx)
            }
            "vec3" => {
                if args.len() != 3 {
                    bail!("wrong number of arguments for vec3");
                }
                let x = args[0].as_f32();
                let y = args[1].as_f32();
                let z = args[2].as_f32();
                let heap_idx = self.allocate_heap(HeapObject::Vec3(x, y, z));
                Value64::HeapRef(heap_idx)
            }
            "vec4" => {
                if args.len() != 4 {
                    bail!("wrong number of arguments for vec4");
                }
                let x = args[0].as_f32();
                let y = args[1].as_f32();
                let z = args[2].as_f32();
                let w = args[3].as_f32();
                let heap_idx = self.allocate_heap(HeapObject::Vec4(x, y, z, w));
                Value64::HeapRef(heap_idx)
            }
            "quat" => {
                if args.len() != 4 {
                    bail!("wrong number of arguments for quat");
                }
                let x = args[0].as_f32();
                let y = args[1].as_f32();
                let z = args[2].as_f32();
                let w = args[3].as_f32();
                let heap_idx = self.allocate_heap(HeapObject::Quat(x, y, z, w));
                Value64::HeapRef(heap_idx)
            }
            "quat_identity" => {
                if !args.is_empty() {
                    bail!("wrong number of arguments for quat_identity");
                }
                let heap_idx =
                    self.allocate_heap(HeapObject::Quat(0.0, 0.0, 0.0, 1.0));
                Value64::HeapRef(heap_idx)
            }
            "vec2_add" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec2_add");
                }
                let (x1, y1) = self.get_vec2(args[0])?;
                let (x2, y2) = self.get_vec2(args[1])?;
                let heap_idx =
                    self.allocate_heap(HeapObject::Vec2(x1 + x2, y1 + y2));
                Value64::HeapRef(heap_idx)
            }
            "vec2_sub" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec2_sub");
                }
                let (x1, y1) = self.get_vec2(args[0])?;
                let (x2, y2) = self.get_vec2(args[1])?;
                let heap_idx =
                    self.allocate_heap(HeapObject::Vec2(x1 - x2, y1 - y2));
                Value64::HeapRef(heap_idx)
            }
            "vec2_mul" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec2_mul");
                }
                let (x1, y1) = self.get_vec2(args[0])?;
                let scalar = args[1].as_f32();
                let heap_idx = self
                    .allocate_heap(HeapObject::Vec2(x1 * scalar, y1 * scalar));
                Value64::HeapRef(heap_idx)
            }
            "vec2_div" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec2_div");
                }
                let (x1, y1) = self.get_vec2(args[0])?;
                let scalar = args[1].as_f32();
                let heap_idx = self
                    .allocate_heap(HeapObject::Vec2(x1 / scalar, y1 / scalar));
                Value64::HeapRef(heap_idx)
            }
            "vec2_dot" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec2_dot");
                }
                let (x1, y1) = self.get_vec2(args[0])?;
                let (x2, y2) = self.get_vec2(args[1])?;
                Value64::Float32(x1 * x2 + y1 * y2)
            }
            "vec2_len" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for vec2_len");
                }
                let (x, y) = self.get_vec2(args[0])?;
                Value64::Float32((x * x + y * y).sqrt())
            }
            "vec2_normalize" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for vec2_normalize");
                }
                let (x, y) = self.get_vec2(args[0])?;
                let len = (x * x + y * y).sqrt();
                let heap_idx =
                    self.allocate_heap(HeapObject::Vec2(x / len, y / len));
                Value64::HeapRef(heap_idx)
            }
            "vec3_add" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec3_add");
                }
                let (x1, y1, z1) = self.get_vec3(args[0])?;
                let (x2, y2, z2) = self.get_vec3(args[1])?;
                let heap_idx = self.allocate_heap(HeapObject::Vec3(
                    x1 + x2,
                    y1 + y2,
                    z1 + z2,
                ));
                Value64::HeapRef(heap_idx)
            }
            "vec3_sub" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec3_sub");
                }
                let (x1, y1, z1) = self.get_vec3(args[0])?;
                let (x2, y2, z2) = self.get_vec3(args[1])?;
                let heap_idx = self.allocate_heap(HeapObject::Vec3(
                    x1 - x2,
                    y1 - y2,
                    z1 - z2,
                ));
                Value64::HeapRef(heap_idx)
            }
            "vec3_mul" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec3_mul");
                }
                let (x1, y1, z1) = self.get_vec3(args[0])?;
                let scalar = args[1].as_f32();
                let heap_idx = self.allocate_heap(HeapObject::Vec3(
                    x1 * scalar,
                    y1 * scalar,
                    z1 * scalar,
                ));
                Value64::HeapRef(heap_idx)
            }
            "vec3_div" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec3_div");
                }
                let (x1, y1, z1) = self.get_vec3(args[0])?;
                let scalar = args[1].as_f32();
                let heap_idx = self.allocate_heap(HeapObject::Vec3(
                    x1 / scalar,
                    y1 / scalar,
                    z1 / scalar,
                ));
                Value64::HeapRef(heap_idx)
            }
            "vec3_dot" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec3_dot");
                }
                let (x1, y1, z1) = self.get_vec3(args[0])?;
                let (x2, y2, z2) = self.get_vec3(args[1])?;
                Value64::Float32(x1 * x2 + y1 * y2 + z1 * z2)
            }
            "vec3_cross" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec3_cross");
                }
                let (x1, y1, z1) = self.get_vec3(args[0])?;
                let (x2, y2, z2) = self.get_vec3(args[1])?;
                let heap_idx = self.allocate_heap(HeapObject::Vec3(
                    y1 * z2 - z1 * y2,
                    z1 * x2 - x1 * z2,
                    x1 * y2 - y1 * x2,
                ));
                Value64::HeapRef(heap_idx)
            }
            "vec3_len" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for vec3_len");
                }
                let (x, y, z) = self.get_vec3(args[0])?;
                Value64::Float32((x * x + y * y + z * z).sqrt())
            }
            "vec3_normalize" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for vec3_normalize");
                }
                let (x, y, z) = self.get_vec3(args[0])?;
                let len = (x * x + y * y + z * z).sqrt();
                let heap_idx = self.allocate_heap(HeapObject::Vec3(
                    x / len,
                    y / len,
                    z / len,
                ));
                Value64::HeapRef(heap_idx)
            }
            "vec4_add" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec4_add");
                }
                let (x1, y1, z1, w1) = self.get_vec4(args[0])?;
                let (x2, y2, z2, w2) = self.get_vec4(args[1])?;
                let heap_idx = self.allocate_heap(HeapObject::Vec4(
                    x1 + x2,
                    y1 + y2,
                    z1 + z2,
                    w1 + w2,
                ));
                Value64::HeapRef(heap_idx)
            }
            "vec4_sub" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec4_sub");
                }
                let (x1, y1, z1, w1) = self.get_vec4(args[0])?;
                let (x2, y2, z2, w2) = self.get_vec4(args[1])?;
                let heap_idx = self.allocate_heap(HeapObject::Vec4(
                    x1 - x2,
                    y1 - y2,
                    z1 - z2,
                    w1 - w2,
                ));
                Value64::HeapRef(heap_idx)
            }
            "vec4_mul" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec4_mul");
                }
                let (x1, y1, z1, w1) = self.get_vec4(args[0])?;
                let scalar = args[1].as_f32();
                let heap_idx = self.allocate_heap(HeapObject::Vec4(
                    x1 * scalar,
                    y1 * scalar,
                    z1 * scalar,
                    w1 * scalar,
                ));
                Value64::HeapRef(heap_idx)
            }
            "vec4_div" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec4_div");
                }
                let (x1, y1, z1, w1) = self.get_vec4(args[0])?;
                let scalar = args[1].as_f32();
                let heap_idx = self.allocate_heap(HeapObject::Vec4(
                    x1 / scalar,
                    y1 / scalar,
                    z1 / scalar,
                    w1 / scalar,
                ));
                Value64::HeapRef(heap_idx)
            }
            "vec4_dot" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for vec4_dot");
                }
                let (x1, y1, z1, w1) = self.get_vec4(args[0])?;
                let (x2, y2, z2, w2) = self.get_vec4(args[1])?;
                Value64::Float32(x1 * x2 + y1 * y2 + z1 * z2 + w1 * w2)
            }
            "vec4_len" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for vec4_len");
                }
                let (x, y, z, w) = self.get_vec4(args[0])?;
                Value64::Float32((x * x + y * y + z * z + w * w).sqrt())
            }
            "vec4_normalize" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for vec4_normalize");
                }
                let (x, y, z, w) = self.get_vec4(args[0])?;
                let len = (x * x + y * y + z * z + w * w).sqrt();
                let heap_idx = self.allocate_heap(HeapObject::Vec4(
                    x / len,
                    y / len,
                    z / len,
                    w / len,
                ));
                Value64::HeapRef(heap_idx)
            }
            "quat_mul" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for quat_mul");
                }
                let (x1, y1, z1, w1) = self.get_quat(args[0])?;
                let (x2, y2, z2, w2) = self.get_quat(args[1])?;
                let heap_idx = self.allocate_heap(HeapObject::Quat(
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                ));
                Value64::HeapRef(heap_idx)
            }
            "quat_normalize" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for quat_normalize");
                }
                let (x, y, z, w) = self.get_quat(args[0])?;
                let len = (x * x + y * y + z * z + w * w).sqrt();
                let heap_idx = self.allocate_heap(HeapObject::Quat(
                    x / len,
                    y / len,
                    z / len,
                    w / len,
                ));
                Value64::HeapRef(heap_idx)
            }
            "quat_from_euler" => {
                if args.len() != 3 {
                    bail!("wrong number of arguments for quat_from_euler");
                }
                let pitch = args[0].as_f32();
                let yaw = args[1].as_f32();
                let roll = args[2].as_f32();
                let half_pitch = pitch * 0.5;
                let half_yaw = yaw * 0.5;
                let half_roll = roll * 0.5;
                let cp = half_pitch.cos();
                let sp = half_pitch.sin();
                let cy = half_yaw.cos();
                let sy = half_yaw.sin();
                let cr = half_roll.cos();
                let sr = half_roll.sin();
                let heap_idx = self.allocate_heap(HeapObject::Quat(
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy,
                    cr * cp * cy + sr * sp * sy,
                ));
                Value64::HeapRef(heap_idx)
            }
            "quat_rotate_vec3" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for quat_rotate_vec3");
                }
                let (qx, qy, qz, qw) = self.get_quat(args[0])?;
                let (vx, vy, vz) = self.get_vec3(args[1])?;
                let tx = 2.0 * (qy * vz - qz * vy);
                let ty = 2.0 * (qz * vx - qx * vz);
                let tz = 2.0 * (qx * vy - qy * vx);
                let heap_idx = self.allocate_heap(HeapObject::Vec3(
                    vx + qw * tx + qy * tz - qz * ty,
                    vy + qw * ty + qz * tx - qx * tz,
                    vz + qw * tz + qx * ty - qy * tx,
                ));
                Value64::HeapRef(heap_idx)
            }
            "arena_new" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for arena_new");
                }
                let capacity = args[0].as_i64() as usize;
                let arena = ArenaData::new(capacity);
                let heap_idx = self.allocate_heap(HeapObject::Arena(arena));
                Value64::HeapRef(heap_idx)
            }
            "arena_alloc" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for arena_alloc");
                }
                let arena_ref = args[0].as_heap_ref();
                let value = args[1];
                if let HeapObject::Arena(arena) = &mut self.heap[arena_ref as usize] {
                    match arena.alloc(value) {
                        Some(index) => Value64::Integer(index as i64),
                        None => Value64::Null,
                    }
                } else {
                    bail!("first argument to arena_alloc must be an Arena");
                }
            }
            "arena_reset" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for arena_reset");
                }
                let arena_ref = args[0].as_heap_ref();
                if let HeapObject::Arena(arena) = &mut self.heap[arena_ref as usize] {
                    arena.reset();
                    Value64::Null
                } else {
                    bail!("argument to arena_reset must be an Arena");
                }
            }
            "arena_get" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for arena_get");
                }
                let arena_ref = args[0].as_heap_ref();
                let index = args[1].as_i64() as usize;
                if let HeapObject::Arena(arena) = &self.heap[arena_ref as usize] {
                    if index < arena.next_index {
                        arena.storage[index]
                    } else {
                        Value64::Null
                    }
                } else {
                    bail!("first argument to arena_get must be an Arena");
                }
            }
            "pool_new" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for pool_new");
                }
                let capacity = args[0].as_i64() as usize;
                let pool = PoolData::new(capacity);
                let heap_idx = self.allocate_heap(HeapObject::Pool(pool));
                Value64::HeapRef(heap_idx)
            }
            "pool_alloc" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for pool_alloc");
                }
                let pool_ref = args[0].as_heap_ref();
                let value = args[1];
                if let HeapObject::Pool(pool) = &mut self.heap[pool_ref as usize] {
                    match pool.alloc(value) {
                        Some((index, generation)) => {
                            let handle = HeapObject::Handle(index, generation);
                            let heap_idx = self.allocate_heap(handle);
                            Value64::HeapRef(heap_idx)
                        }
                        None => Value64::Null,
                    }
                } else {
                    bail!("first argument to pool_alloc must be a Pool");
                }
            }
            "pool_get" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for pool_get");
                }
                let pool_ref = args[0].as_heap_ref();
                let handle_ref = args[1].as_heap_ref();
                let (index, generation) = if let HeapObject::Handle(i, g) = &self.heap[handle_ref as usize] {
                    (*i, *g)
                } else {
                    bail!("second argument to pool_get must be a Handle");
                };
                if let HeapObject::Pool(pool) = &self.heap[pool_ref as usize] {
                    match pool.get(index, generation) {
                        Some(value) => value,
                        None => Value64::Null,
                    }
                } else {
                    bail!("first argument to pool_get must be a Pool");
                }
            }
            "pool_free" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for pool_free");
                }
                let pool_ref = args[0].as_heap_ref();
                let handle_ref = args[1].as_heap_ref();
                let (index, generation) = if let HeapObject::Handle(i, g) = &self.heap[handle_ref as usize] {
                    (*i, *g)
                } else {
                    bail!("second argument to pool_free must be a Handle");
                };
                if let HeapObject::Pool(pool) = &mut self.heap[pool_ref as usize] {
                    Value64::Bool(pool.free(index, generation))
                } else {
                    bail!("first argument to pool_free must be a Pool");
                }
            }
            "handle_index" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for handle_index");
                }
                let handle_ref = args[0].as_heap_ref();
                if let HeapObject::Handle(index, _) = &self.heap[handle_ref as usize] {
                    Value64::Integer(*index as i64)
                } else {
                    bail!("argument to handle_index must be a Handle");
                }
            }
            "handle_generation" => {
                if args.len() != 1 {
                    bail!("wrong number of arguments for handle_generation");
                }
                let handle_ref = args[0].as_heap_ref();
                if let HeapObject::Handle(_, generation) = &self.heap[handle_ref as usize] {
                    Value64::Integer(*generation as i64)
                } else {
                    bail!("argument to handle_generation must be a Handle");
                }
            }
            "char_at" => {
                if args.len() != 2 {
                    bail!("wrong number of arguments for char_at");
                }
                match (args[0], args[1]) {
                    (Value64::HeapRef(idx), Value64::Integer(pos)) => {
                        if let HeapObject::String(s) = &self.heap[idx as usize] {
                            let pos = pos as usize;
                            if pos >= s.len() {
                                Value64::Integer(-1)
                            } else {
                                let c = s.chars().nth(pos).unwrap_or('\0');
                                Value64::Integer(c as i64)
                            }
                        } else {
                            bail!("first argument to char_at must be a string");
                        }
                    }
                    _ => bail!("char_at takes (string, index)"),
                }
            }
            "assert" => {
                if args.is_empty() || args.len() > 2 {
                    bail!("assert takes 1 or 2 arguments");
                }
                let condition = match args[0] {
                    Value64::Bool(b) => b,
                    Value64::Integer(n) => n != 0,
                    _ => bail!("assert condition must be a boolean or integer"),
                };
                if !condition {
                    let message = if args.len() == 2 {
                        if let Value64::HeapRef(idx) = args[1] {
                            if let HeapObject::String(s) = &self.heap[idx as usize] {
                                s.clone()
                            } else {
                                "assertion failed".to_string()
                            }
                        } else {
                            "assertion failed".to_string()
                        }
                    } else {
                        "assertion failed".to_string()
                    };
                    bail!("Assertion failed: {}", message);
                }
                Value64::Null
            }
            "alloc" => {
                if args.len() != 1 {
                    bail!("alloc takes 1 argument");
                }
                let allocator_ref = self.current_frame().context.allocator;
                if let Value64::HeapRef(arena_idx) = allocator_ref {
                    if let HeapObject::Arena(arena) = &mut self.heap[arena_idx as usize] {
                        match arena.alloc(args[0]) {
                            Some(index) => Value64::Integer(index as i64),
                            None => bail!("alloc failed: arena is full"),
                        }
                    } else {
                        bail!("context.allocator is not an arena");
                    }
                } else {
                    bail!("context.allocator is not set");
                }
            }
            "temp_alloc" => {
                if args.len() != 1 {
                    bail!("temp_alloc takes 1 argument");
                }
                let temp_allocator_ref = self.current_frame().context.temp_allocator;
                if let Value64::HeapRef(arena_idx) = temp_allocator_ref {
                    if let HeapObject::Arena(arena) = &mut self.heap[arena_idx as usize] {
                        match arena.alloc(args[0]) {
                            Some(index) => Value64::Integer(index as i64),
                            None => bail!("temp_alloc failed: arena is full"),
                        }
                    } else {
                        bail!("context.temp_allocator is not an arena");
                    }
                } else {
                    bail!("context.temp_allocator is not set");
                }
            }
            _ => bail!("unknown builtin: {}", name),
        };

        self.push(result)?;
        Ok(())
    }

    fn call_native(&mut self, name: &str, num_args: usize) -> Result<()> {
        let mut args = Vec::with_capacity(num_args);
        for index in (self.stack_pointer - num_args)..self.stack_pointer {
            args.push(self.stack[index]);
        }
        self.stack_pointer -= num_args + 1;

        let registry = self
            .native_registry
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("no native registry available"))?;

        let native_fn = registry.functions.get(name).ok_or_else(|| {
            anyhow::anyhow!("unknown native function: {}", name)
        })?;

        let result = (native_fn.function)(&args, &mut registry.handles)?;
        self.push(result)?;
        Ok(())
    }

    fn execute_index_expression(
        &mut self,
        left: Value64,
        index: Value64,
    ) -> Result<()> {
        match left {
            Value64::HeapRef(idx) => match &self.heap[idx as usize] {
                HeapObject::Array(elements) => {
                    let i = index.as_i64();
                    if i < 0 || i as usize >= elements.len() {
                        self.push(Value64::Null)?;
                    } else {
                        self.push(elements[i as usize])?;
                    }
                }
                HeapObject::HashMap(hash) => {
                    let hash_key = self.hash_key(index)?;
                    let value =
                        hash.get(&hash_key).copied().unwrap_or(Value64::Null);
                    self.push(value)?;
                }
                _ => bail!("index operator not supported for this heap type"),
            },
            _ => bail!("index operator not supported for this type"),
        }
        Ok(())
    }

    fn execute_index_set(
        &mut self,
        array: Value64,
        index: Value64,
        value: Value64,
    ) -> Result<()> {
        match array {
            Value64::HeapRef(idx) => {
                let i = index.as_i64();
                let hash_key = self.hash_key(index)?;
                match &mut self.heap[idx as usize] {
                    HeapObject::Array(elements) => {
                        if i < 0 || i as usize >= elements.len() {
                            bail!("array index out of bounds: {} (len: {})", i, elements.len());
                        }
                        elements[i as usize] = value;
                    }
                    HeapObject::HashMap(hash) => {
                        hash.insert(hash_key, value);
                    }
                    _ => bail!("index set not supported for this heap type"),
                }
                self.push(array)?;
            }
            _ => bail!("index set not supported for this type"),
        }
        Ok(())
    }

    fn hash_key(&self, key: Value64) -> Result<u64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        match key {
            Value64::Integer(v) => v.hash(&mut hasher),
            Value64::Bool(v) => v.hash(&mut hasher),
            Value64::HeapRef(idx) => {
                if let HeapObject::String(s) = &self.heap[idx as usize] {
                    s.hash(&mut hasher);
                } else {
                    bail!("unusable as hash key");
                }
            }
            _ => bail!("unusable as hash key"),
        }
        Ok(hasher.finish())
    }

    pub fn call_global(
        &mut self,
        global_index: usize,
        args: &[Value64],
    ) -> Result<Value64> {
        let callee = self.globals[global_index];

        for arg in args {
            self.push(*arg)?;
        }
        self.push(callee)?;

        let num_args = args.len();

        match callee {
            Value64::HeapRef(idx) => match &self.heap[idx as usize] {
                HeapObject::Closure(closure) => {
                    let function_index = closure.function_index;
                    let free_values = closure.free.clone();
                    let num_parameters =
                        self.functions[function_index as usize].num_parameters;
                    let num_locals =
                        self.functions[function_index as usize].num_locals;

                    if num_args != num_parameters {
                        bail!(
                            "wrong number of arguments: want={}, got={}",
                            num_parameters,
                            num_args
                        );
                    }

                    let base_pointer = self.stack_pointer - num_args - 1;
                    let frame =
                        Frame::new(function_index, base_pointer, free_values);
                    self.push_frame(frame);
                    self.stack_pointer = base_pointer + num_locals;

                    while self.stack.len() < self.stack_pointer {
                        self.stack.push(Value64::Null);
                    }

                    while self.frame_index > 0 {
                        let frame = self.current_frame();
                        let instructions = self.current_instructions();
                        if frame.ip >= instructions.len() {
                            break;
                        }

                        let instruction = instructions[frame.ip].clone();
                        self.current_frame_mut().ip += 1;
                        self.execute_instruction(instruction)?;
                    }

                    self.last_popped()
                }
                _ => bail!("global is not a callable closure"),
            },
            _ => bail!("global is not a function"),
        }
    }

    pub fn create_struct(&mut self, name: &str, fields: Vec<Value64>) -> Value64 {
        let heap_index = self.allocate_heap(HeapObject::Struct(name.to_string(), fields));
        Value64::HeapRef(heap_index)
    }

    pub fn get_struct_field(&self, struct_ref: Value64, field_index: usize) -> Result<Value64> {
        match struct_ref {
            Value64::HeapRef(idx) => {
                if let HeapObject::Struct(_, fields) = &self.heap[idx as usize] {
                    fields.get(field_index).copied().ok_or_else(|| {
                        anyhow::anyhow!("field index {} out of bounds", field_index)
                    })
                } else {
                    bail!("expected struct, got different heap object")
                }
            }
            _ => bail!("expected heap reference to struct"),
        }
    }

    fn execute_instruction(&mut self, instruction: Instruction) -> Result<()> {
        match instruction.opcode {
            Opcode::Constant => {
                let constant_index = instruction.operands[0] as usize;
                let constant = self.constants[constant_index];
                self.push(constant)?;
            }
            Opcode::Pop => {
                self.pop()?;
            }
            Opcode::AddI64 => {
                let right = self.pop()?.as_i64();
                let left = self.pop()?.as_i64();
                self.push(Value64::Integer(left + right))?;
            }
            Opcode::SubI64 => {
                let right = self.pop()?.as_i64();
                let left = self.pop()?.as_i64();
                self.push(Value64::Integer(left - right))?;
            }
            Opcode::MulI64 => {
                let right = self.pop()?.as_i64();
                let left = self.pop()?.as_i64();
                self.push(Value64::Integer(left * right))?;
            }
            Opcode::DivI64 => {
                let right = self.pop()?.as_i64();
                let left = self.pop()?.as_i64();
                self.push(Value64::Integer(left / right))?;
            }
            Opcode::ModI64 => {
                let right = self.pop()?.as_i64();
                let left = self.pop()?.as_i64();
                self.push(Value64::Integer(left % right))?;
            }
            Opcode::AddF64 => {
                let right = self.pop()?.as_f64();
                let left = self.pop()?.as_f64();
                self.push(Value64::Float(left + right))?;
            }
            Opcode::SubF64 => {
                let right = self.pop()?.as_f64();
                let left = self.pop()?.as_f64();
                self.push(Value64::Float(left - right))?;
            }
            Opcode::MulF64 => {
                let right = self.pop()?.as_f64();
                let left = self.pop()?.as_f64();
                self.push(Value64::Float(left * right))?;
            }
            Opcode::DivF64 => {
                let right = self.pop()?.as_f64();
                let left = self.pop()?.as_f64();
                self.push(Value64::Float(left / right))?;
            }
            Opcode::AddF32 => {
                let right = self.pop()?.as_f32();
                let left = self.pop()?.as_f32();
                self.push(Value64::Float32(left + right))?;
            }
            Opcode::SubF32 => {
                let right = self.pop()?.as_f32();
                let left = self.pop()?.as_f32();
                self.push(Value64::Float32(left - right))?;
            }
            Opcode::MulF32 => {
                let right = self.pop()?.as_f32();
                let left = self.pop()?.as_f32();
                self.push(Value64::Float32(left * right))?;
            }
            Opcode::DivF32 => {
                let right = self.pop()?.as_f32();
                let left = self.pop()?.as_f32();
                self.push(Value64::Float32(left / right))?;
            }
            Opcode::NegateF32 => {
                let value = self.pop()?.as_f32();
                self.push(Value64::Float32(-value))?;
            }
            Opcode::Add
            | Opcode::Sub
            | Opcode::Mul
            | Opcode::Div
            | Opcode::Mod => {
                let right = self.pop()?;
                let left = self.pop()?;
                let result = match (left, right, instruction.opcode) {
                    (Value64::Integer(l), Value64::Integer(r), Opcode::Add) => {
                        Value64::Integer(l + r)
                    }
                    (Value64::Integer(l), Value64::Integer(r), Opcode::Sub) => {
                        Value64::Integer(l - r)
                    }
                    (Value64::Integer(l), Value64::Integer(r), Opcode::Mul) => {
                        Value64::Integer(l * r)
                    }
                    (Value64::Integer(l), Value64::Integer(r), Opcode::Div) => {
                        Value64::Integer(l / r)
                    }
                    (Value64::Integer(l), Value64::Integer(r), Opcode::Mod) => {
                        Value64::Integer(l % r)
                    }
                    (Value64::Float(l), Value64::Float(r), Opcode::Add) => {
                        Value64::Float(l + r)
                    }
                    (Value64::Float(l), Value64::Float(r), Opcode::Sub) => {
                        Value64::Float(l - r)
                    }
                    (Value64::Float(l), Value64::Float(r), Opcode::Mul) => {
                        Value64::Float(l * r)
                    }
                    (Value64::Float(l), Value64::Float(r), Opcode::Div) => {
                        Value64::Float(l / r)
                    }
                    (Value64::Float32(l), Value64::Float32(r), Opcode::Add) => {
                        Value64::Float32(l + r)
                    }
                    (Value64::Float32(l), Value64::Float32(r), Opcode::Sub) => {
                        Value64::Float32(l - r)
                    }
                    (Value64::Float32(l), Value64::Float32(r), Opcode::Mul) => {
                        Value64::Float32(l * r)
                    }
                    (Value64::Float32(l), Value64::Float32(r), Opcode::Div) => {
                        Value64::Float32(l / r)
                    }
                    (Value64::HeapRef(l), Value64::HeapRef(r), Opcode::Add) => {
                        match (&self.heap[l as usize], &self.heap[r as usize]) {
                            (
                                HeapObject::String(ls),
                                HeapObject::String(rs),
                            ) => {
                                let combined = format!("{}{}", ls, rs);
                                let index = self.allocate_heap(
                                    HeapObject::String(combined),
                                );
                                Value64::HeapRef(index)
                            }
                            _ => bail!("unsupported heap operation"),
                        }
                    }
                    _ => bail!("unsupported operation"),
                };
                self.push(result)?;
            }
            Opcode::ShiftLeft => {
                let right = self.pop()?.as_i64();
                let left = self.pop()?.as_i64();
                self.push(Value64::Integer(left << right))?;
            }
            Opcode::ShiftRight => {
                let right = self.pop()?.as_i64();
                let left = self.pop()?.as_i64();
                self.push(Value64::Integer(left >> right))?;
            }
            Opcode::BitwiseAnd => {
                let right = self.pop()?.as_i64();
                let left = self.pop()?.as_i64();
                self.push(Value64::Integer(left & right))?;
            }
            Opcode::BitwiseOr => {
                let right = self.pop()?.as_i64();
                let left = self.pop()?.as_i64();
                self.push(Value64::Integer(left | right))?;
            }
            Opcode::True => self.push(Value64::Bool(true))?,
            Opcode::False => self.push(Value64::Bool(false))?,
            Opcode::Null => self.push(Value64::Null)?,
            Opcode::Equal | Opcode::NotEqual | Opcode::GreaterThan => {
                let right = self.pop()?;
                let left = self.pop()?;
                let result = match instruction.opcode {
                    Opcode::Equal => Value64::Bool(left == right),
                    Opcode::NotEqual => Value64::Bool(left != right),
                    Opcode::GreaterThan => match (left, right) {
                        (Value64::Integer(l), Value64::Integer(r)) => {
                            Value64::Bool(l > r)
                        }
                        (Value64::Float(l), Value64::Float(r)) => {
                            Value64::Bool(l > r)
                        }
                        _ => bail!("unsupported comparison"),
                    },
                    _ => unreachable!(),
                };
                self.push(result)?;
            }
            Opcode::EqualI64 => {
                let right = self.pop()?.as_i64();
                let left = self.pop()?.as_i64();
                self.push(Value64::Bool(left == right))?;
            }
            Opcode::NotEqualI64 => {
                let right = self.pop()?.as_i64();
                let left = self.pop()?.as_i64();
                self.push(Value64::Bool(left != right))?;
            }
            Opcode::LessThanI64 => {
                let right = self.pop()?.as_i64();
                let left = self.pop()?.as_i64();
                self.push(Value64::Bool(left < right))?;
            }
            Opcode::GreaterThanI64 => {
                let right = self.pop()?.as_i64();
                let left = self.pop()?.as_i64();
                self.push(Value64::Bool(left > right))?;
            }
            Opcode::GreaterThanOrEqualI64 => {
                let right = self.pop()?.as_i64();
                let left = self.pop()?.as_i64();
                self.push(Value64::Bool(left >= right))?;
            }
            Opcode::LessThanOrEqualI64 => {
                let right = self.pop()?.as_i64();
                let left = self.pop()?.as_i64();
                self.push(Value64::Bool(left <= right))?;
            }
            Opcode::EqualBool => {
                let right = self.pop()?.as_bool();
                let left = self.pop()?.as_bool();
                self.push(Value64::Bool(left == right))?;
            }
            Opcode::NotEqualBool => {
                let right = self.pop()?.as_bool();
                let left = self.pop()?.as_bool();
                self.push(Value64::Bool(left != right))?;
            }
            Opcode::Minus => {
                let value = self.pop()?;
                match value {
                    Value64::Integer(v) => self.push(Value64::Integer(-v))?,
                    Value64::Float(v) => self.push(Value64::Float(-v))?,
                    _ => bail!("unsupported negation"),
                }
            }
            Opcode::NegateI64 => {
                let value = self.pop()?.as_i64();
                self.push(Value64::Integer(-value))?;
            }
            Opcode::NegateF64 => {
                let value = self.pop()?.as_f64();
                self.push(Value64::Float(-value))?;
            }
            Opcode::Bang => {
                let value = self.pop()?;
                self.push(Value64::Bool(!value.is_truthy()))?;
            }
            Opcode::JumpNotTruthy => {
                let condition = self.pop()?;
                if !condition.is_truthy() {
                    let target = instruction.operands[0] as usize;
                    self.current_frame_mut().ip = target;
                }
            }
            Opcode::Jump => {
                let target = instruction.operands[0] as usize;
                self.current_frame_mut().ip = target;
            }
            Opcode::GetGlobal => {
                let global_index = instruction.operands[0] as usize;
                let value = self.globals[global_index];
                self.push(value)?;
            }
            Opcode::SetGlobal => {
                let global_index = instruction.operands[0] as usize;
                let value = self.stack_top()?;
                self.globals[global_index] = value;
            }
            Opcode::GetLocal => {
                let local_index = instruction.operands[0] as usize;
                let base_pointer = self.current_frame().base_pointer;
                let value = self.stack[base_pointer + local_index];
                self.push(value)?;
            }
            Opcode::SetLocal => {
                let local_index = instruction.operands[0] as usize;
                let base_pointer = self.current_frame().base_pointer;
                let value = self.stack_top()?;
                self.stack[base_pointer + local_index] = value;
            }
            Opcode::GetBuiltin => {
                let builtin_index = instruction.operands[0] as usize;
                let name = BUILTINS[builtin_index].to_string();
                let builtin = TypedBuiltIn {
                    name: name.clone(),
                    index: builtin_index as u32,
                };
                let heap_index =
                    self.allocate_heap(HeapObject::BuiltIn(builtin));
                self.push(Value64::HeapRef(heap_index))?;
            }
            Opcode::GetNative => {
                let native_index = instruction.operands[0] as usize;
                let name = self.native_names[native_index].clone();
                let heap_index =
                    self.allocate_heap(HeapObject::NativeFunction(name));
                self.push(Value64::HeapRef(heap_index))?;
            }
            Opcode::GetFree => {
                let free_index = instruction.operands[0] as usize;
                let value = self.current_frame().free_values[free_index];
                self.push(value)?;
            }
            Opcode::Closure => {
                let function_index = instruction.operands[0] as u32;
                let num_free = instruction.operands[1] as usize;
                let mut free_values = Vec::with_capacity(num_free);
                for index in (self.stack_pointer - num_free)..self.stack_pointer
                {
                    free_values.push(self.stack[index]);
                }
                self.stack_pointer -= num_free;
                let closure = TypedClosure {
                    function_index,
                    free: free_values,
                };
                let heap_index =
                    self.allocate_heap(HeapObject::Closure(closure));
                self.push(Value64::HeapRef(heap_index))?;
            }
            Opcode::CurrentClosure => {
                let frame = self.current_frame();
                let function_index = frame.function_index;
                let free_values = frame.free_values.clone();
                let closure = TypedClosure {
                    function_index,
                    free: free_values,
                };
                let heap_index =
                    self.allocate_heap(HeapObject::Closure(closure));
                self.push(Value64::HeapRef(heap_index))?;
            }
            Opcode::Array => {
                let num_elements = instruction.operands[0] as usize;
                let mut elements = Vec::with_capacity(num_elements);
                for index in
                    (self.stack_pointer - num_elements)..self.stack_pointer
                {
                    elements.push(self.stack[index]);
                }
                self.stack_pointer -= num_elements;
                let heap_index =
                    self.allocate_heap(HeapObject::Array(elements));
                self.push(Value64::HeapRef(heap_index))?;
            }
            Opcode::Hash => {
                let num_pairs = instruction.operands[0] as usize;
                let mut hash = HashMap::new();
                for index in (0..(num_pairs / 2)).rev() {
                    let value = self.stack[self.stack_pointer - 1 - index * 2];
                    let key = self.stack[self.stack_pointer - 2 - index * 2];
                    let hash_key = self.hash_key(key)?;
                    hash.insert(hash_key, value);
                }
                self.stack_pointer -= num_pairs;
                let heap_index = self.allocate_heap(HeapObject::HashMap(hash));
                self.push(Value64::HeapRef(heap_index))?;
            }
            Opcode::Index => {
                let index = self.pop()?;
                let left = self.pop()?;
                self.execute_index_expression(left, index)?;
            }
            Opcode::IndexSet => {
                let value = self.pop()?;
                let index = self.pop()?;
                let array = self.pop()?;
                self.execute_index_set(array, index, value)?;
            }
            Opcode::Call => {
                let num_args = instruction.operands[0] as usize;
                self.execute_call(num_args)?;
            }
            Opcode::ReturnValue => {
                let return_value = self.pop()?;
                let frame = self.pop_frame();
                self.stack_pointer = frame.base_pointer;
                self.push(return_value)?;
            }
            Opcode::Return => {
                let frame = self.pop_frame();
                self.stack_pointer = frame.base_pointer;
                self.push(Value64::Null)?;
            }
            Opcode::LoadPtr => {
                let addr = self.pop()?;
                match addr {
                    Value64::Integer(ptr) => {
                        let slot = ptr as usize;
                        if slot < self.stack.len() {
                            self.push(self.stack[slot])?;
                        } else if slot < GLOBALS_SIZE + self.stack.len() {
                            self.push(self.globals[slot - self.stack.len()])?;
                        } else {
                            bail!("invalid pointer");
                        }
                    }
                    _ => bail!("cannot dereference non-pointer"),
                }
            }
            Opcode::StorePtr => {
                let value = self.pop()?;
                let addr = self.pop()?;
                match addr {
                    Value64::Integer(ptr) => {
                        let slot = ptr as usize;
                        if slot < self.stack.len() {
                            self.stack[slot] = value;
                        } else if slot < GLOBALS_SIZE + self.stack.len() {
                            self.globals[slot - self.stack.len()] = value;
                        } else {
                            bail!("invalid pointer");
                        }
                    }
                    _ => bail!("cannot store through non-pointer"),
                }
            }
            Opcode::AddressOfLocal => {
                let local_index = instruction.operands[0] as usize;
                let base_pointer = self.current_frame().base_pointer;
                let address = base_pointer + local_index;
                self.push(Value64::Integer(address as i64))?;
            }
            Opcode::AddressOfGlobal => {
                let global_index = instruction.operands[0] as usize;
                let address = self.stack.len() + global_index;
                self.push(Value64::Integer(address as i64))?;
            }
            Opcode::Alloc => {
                let size = self.pop()?.as_i64() as usize;
                let elements = vec![Value64::Null; size];
                let heap_idx = self.allocate_heap(HeapObject::Array(elements));
                self.push(Value64::HeapRef(heap_idx))?;
            }
            Opcode::Free => {
                let ptr = self.pop()?;
                if let Value64::HeapRef(idx) = ptr {
                    self.free_heap(idx);
                }
            }
            Opcode::StructAlloc => {
                let size = instruction.operands[0] as usize;
                let fields = vec![Value64::Null; size];
                let struct_val = self
                    .allocate_heap(HeapObject::Struct("".to_string(), fields));
                self.push(Value64::HeapRef(struct_val))?;
            }
            Opcode::StructGet => {
                let offset = instruction.operands[0] as usize;
                let struct_ref = self.pop()?;
                if let Value64::HeapRef(idx) = struct_ref {
                    if let HeapObject::Struct(_, fields) =
                        &self.heap[idx as usize]
                    {
                        self.push(fields[offset])?;
                    } else {
                        bail!("expected struct");
                    }
                } else {
                    bail!("expected heap reference");
                }
            }
            Opcode::StructSet => {
                let offset = instruction.operands[0] as usize;
                let value = self.pop()?;
                let struct_ref = self.stack_top()?;
                if let Value64::HeapRef(idx) = struct_ref {
                    if let HeapObject::Struct(_, fields) =
                        &mut self.heap[idx as usize]
                    {
                        fields[offset] = value;
                    } else {
                        bail!("expected struct");
                    }
                } else {
                    bail!("expected heap reference");
                }
            }
            Opcode::TaggedUnionAlloc => {
                let num_fields = instruction.operands[0] as usize;
                let fields = vec![Value64::Null; num_fields];
                let heap_index =
                    self.allocate_heap(HeapObject::TaggedUnion(0, fields));
                self.push(Value64::HeapRef(heap_index))?;
            }
            Opcode::TaggedUnionSetTag => {
                let tag = instruction.operands[0] as u32;
                let union_val = self.stack[self.stack_pointer - 1];
                if let Value64::HeapRef(idx) = union_val {
                    if let HeapObject::TaggedUnion(t, _) =
                        &mut self.heap[idx as usize]
                    {
                        *t = tag;
                    }
                }
            }
            Opcode::TaggedUnionGetTag => {
                let union_val = self.pop()?;
                match union_val {
                    Value64::HeapRef(idx) => {
                        if let HeapObject::TaggedUnion(tag, _) =
                            &self.heap[idx as usize]
                        {
                            self.push(Value64::Integer(*tag as i64))?;
                        } else {
                            bail!("TaggedUnionGetTag requires a tagged union");
                        }
                    }
                    Value64::Integer(tag) => {
                        self.push(Value64::Integer(tag))?;
                    }
                    _ => bail!("TaggedUnionGetTag requires a heap reference or integer"),
                }
            }
            Opcode::TaggedUnionGetField => {
                let offset = instruction.operands[0] as usize;
                let union_val = self.pop()?;
                if let Value64::HeapRef(idx) = union_val {
                    if let HeapObject::TaggedUnion(_, fields) =
                        &self.heap[idx as usize]
                    {
                        let value = fields
                            .get(offset)
                            .copied()
                            .unwrap_or(Value64::Null);
                        self.push(value)?;
                    } else {
                        bail!("TaggedUnionGetField requires a tagged union");
                    }
                } else {
                    bail!("TaggedUnionGetField requires a heap reference");
                }
            }
            Opcode::TaggedUnionSetField => {
                let offset = instruction.operands[0] as usize;
                let value = self.pop()?;
                let union_val = self.stack[self.stack_pointer - 1];
                if let Value64::HeapRef(idx) = union_val {
                    if let HeapObject::TaggedUnion(_, fields) =
                        &mut self.heap[idx as usize]
                    {
                        if offset < fields.len() {
                            fields[offset] = value;
                        }
                    }
                }
            }
            Opcode::TupleAlloc => {
                let num_elements = instruction.operands[0] as usize;
                let mut elements = Vec::with_capacity(num_elements);
                for _ in 0..num_elements {
                    elements.push(self.pop()?);
                }
                elements.reverse();
                let heap_index =
                    self.allocate_heap(HeapObject::Array(elements));
                self.push(Value64::HeapRef(heap_index))?;
            }
            Opcode::TupleGet => {
                let index = instruction.operands[0] as usize;
                let tuple_val = self.pop()?;
                if let Value64::HeapRef(idx) = tuple_val {
                    if let HeapObject::Array(elements) =
                        &self.heap[idx as usize]
                    {
                        let value = elements
                            .get(index)
                            .copied()
                            .unwrap_or(Value64::Null);
                        self.push(value)?;
                    } else {
                        bail!("TupleGet requires an array/tuple");
                    }
                } else {
                    bail!("TupleGet requires a heap reference");
                }
            }
            Opcode::Dup => {
                let value = self.stack[self.stack_pointer - 1];
                self.push(value)?;
            }
            Opcode::Drop => {
                let value = self.pop()?;
                if let Value64::HeapRef(idx) = value {
                    self.drop_heap_value(idx);
                }
            }
            Opcode::GetContext => {
                let context = &self.current_frame().context;
                let fields = vec![
                    context.allocator,
                    context.temp_allocator,
                    context.logger,
                ];
                let heap_index = self.allocate_heap(HeapObject::Struct("Context".to_string(), fields));
                self.push(Value64::HeapRef(heap_index))?;
            }
            Opcode::SetContext => {
                let context_val = self.pop()?;
                if let Value64::HeapRef(idx) = context_val {
                    if let HeapObject::Struct(_, fields) = &self.heap[idx as usize] {
                        let new_context = RuntimeContext {
                            allocator: fields.first().copied().unwrap_or(Value64::Null),
                            temp_allocator: fields.get(1).copied().unwrap_or(Value64::Null),
                            logger: fields.get(2).copied().unwrap_or(Value64::Null),
                        };
                        self.current_frame_mut().context = new_context;
                    }
                }
            }
            Opcode::GetContextField => {
                let field_index = instruction.operands[0] as usize;
                let context = &self.current_frame().context;
                let value = match field_index {
                    0 => context.allocator,
                    1 => context.temp_allocator,
                    2 => context.logger,
                    _ => Value64::Null,
                };
                self.push(value)?;
            }
            Opcode::SetContextField => {
                let field_index = instruction.operands[0] as usize;
                let value = self.pop()?;
                match field_index {
                    0 => self.current_frame_mut().context.allocator = value,
                    1 => self.current_frame_mut().context.temp_allocator = value,
                    2 => self.current_frame_mut().context.logger = value,
                    _ => {}
                }
            }
            Opcode::PushContextScope => {
                let current_context = self.current_frame().context.clone();
                self.context_stack.push(current_context);
            }
            Opcode::PopContextScope => {
                if let Some(previous_context) = self.context_stack.pop() {
                    self.current_frame_mut().context = previous_context;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Compiler, Lexer, Parser};

    fn run_vm_test(input: &str) -> Result<Value64> {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;
        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        let mut vm = VirtualMachine::new(
            bytecode.constants,
            bytecode.functions,
            bytecode.heap,
        );
        vm.run(&bytecode.instructions)?;
        vm.last_popped()
    }

    fn run_vm_test_string(input: &str) -> Result<String> {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;
        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        let mut vm = VirtualMachine::new(
            bytecode.constants,
            bytecode.functions,
            bytecode.heap,
        );
        vm.run(&bytecode.instructions)?;
        let result = vm.last_popped()?;
        if let Value64::HeapRef(index) = result {
            if let HeapObject::String(s) = &vm.heap[index as usize] {
                return Ok(s.clone());
            }
        }
        bail!("Expected string result, got {:?}", result)
    }

    #[test]
    fn test_typed_vm_integer_arithmetic() -> Result<()> {
        let tests = [
            ("1", Value64::Integer(1)),
            ("2", Value64::Integer(2)),
            ("1 + 2", Value64::Integer(3)),
            ("1 - 2", Value64::Integer(-1)),
            ("1 * 2", Value64::Integer(2)),
            ("4 / 2", Value64::Integer(2)),
            ("50 / 2 * 2 + 10 - 5", Value64::Integer(55)),
            ("-5", Value64::Integer(-5)),
            ("-10", Value64::Integer(-10)),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_typed_vm_boolean_expressions() -> Result<()> {
        let tests = [
            ("true", Value64::Bool(true)),
            ("false", Value64::Bool(false)),
            ("1 < 2", Value64::Bool(true)),
            ("1 > 2", Value64::Bool(false)),
            ("1 == 1", Value64::Bool(true)),
            ("1 != 1", Value64::Bool(false)),
            ("true == true", Value64::Bool(true)),
            ("true != false", Value64::Bool(true)),
            ("!true", Value64::Bool(false)),
            ("!false", Value64::Bool(true)),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_typed_vm_conditionals() -> Result<()> {
        let tests = [
            ("if (true) { 10 }", Value64::Integer(10)),
            ("if (true) { 10 } else { 20 }", Value64::Integer(10)),
            ("if (false) { 10 } else { 20 }", Value64::Integer(20)),
            ("if (1 < 2) { 10 }", Value64::Integer(10)),
            ("if (1 > 2) { 10 }", Value64::Null),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_typed_vm_global_let_statements() -> Result<()> {
        let tests = [
            ("one := 1; one", Value64::Integer(1)),
            ("one := 1; two := 2; one + two", Value64::Integer(3)),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_typed_vm_float_arithmetic() -> Result<()> {
        let tests = [
            ("3.14", Value64::Float(3.14)),
            ("2.5 + 1.5", Value64::Float(4.0)),
            ("2.0 * 3.0", Value64::Float(6.0)),
            ("-3.5", Value64::Float(-3.5)),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_for_loop() -> Result<()> {
        let input = r#"
            mut sum := 0;
            for i in 0..5 {
                sum = sum + i;
            }
            sum
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(10));
        Ok(())
    }

    #[test]
    fn test_inclusive_range() -> Result<()> {
        let input = r#"
            mut sum := 0;
            for i in 0..=5 {
                sum = sum + i;
            }
            sum
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(15));
        Ok(())
    }

    #[test]
    fn test_if_let_binding() -> Result<()> {
        let input = r#"
            x := 42;
            if let y = x {
                y
            } else {
                0
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(42));
        Ok(())
    }

    #[test]
    fn test_if_let_enum_pattern() -> Result<()> {
        let input = r#"
            Option :: enum { Some { value: i64 }, None }
            x: Option = Option::Some { value = 42 }
            if let .Some { value } = x {
                value
            } else {
                0
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(42));
        Ok(())
    }

    #[test]
    fn test_if_let_enum_no_match() -> Result<()> {
        let input = r#"
            Option :: enum { Some { value: i64 }, None }
            x: Option = Option::None
            if let .Some { value } = x {
                value
            } else {
                -1
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(-1));
        Ok(())
    }

    #[test]
    fn test_while_loop() -> Result<()> {
        let input = r#"
            mut i := 0;
            mut sum := 0;
            while (i < 5) {
                sum = sum + i;
                i = i + 1;
            }
            sum
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(10));
        Ok(())
    }

    #[test]
    fn test_while_break() -> Result<()> {
        let input = r#"
            mut i := 0;
            while (i < 100) {
                if (i == 5) {
                    break;
                }
                i = i + 1;
            }
            i
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(5));
        Ok(())
    }

    #[test]
    fn test_while_continue() -> Result<()> {
        let input = r#"
            mut i := 0;
            mut sum := 0;
            while (i < 10) {
                i = i + 1;
                if (i % 2 == 0) {
                    continue;
                }
                sum = sum + i;
            }
            sum
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(25));
        Ok(())
    }

    #[test]
    fn test_functions_and_closures() -> Result<()> {
        let tests = [
            ("add := fn(a, b) { a + b }; add(2, 3)", Value64::Integer(5)),
            ("fac := fn(n) { if (n < 2) { 1 } else { n * fac(n - 1) } }; fac(5)", Value64::Integer(120)),
            ("make := fn(x) { fn(y) { x + y } }; add5 := make(5); add5(3)", Value64::Integer(8)),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_f32_arithmetic() -> Result<()> {
        let tests = [
            ("1.0f32 + 2.0f32", Value64::Float32(3.0)),
            ("5.0f32 - 2.0f32", Value64::Float32(3.0)),
            ("3.0f32 * 4.0f32", Value64::Float32(12.0)),
            ("10.0f32 / 2.0f32", Value64::Float32(5.0)),
            ("x : f32 = 3.5f32; x + 1.5f32", Value64::Float32(5.0)),
        ];
        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_vec3_operations() -> Result<()> {
        let input = r#"
            v1 := vec3(1.0f32, 2.0f32, 3.0f32);
            v2 := vec3(4.0f32, 5.0f32, 6.0f32);
            v3 := vec3_add(v1, v2);
            vec3_dot(v1, v2)
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Float32(32.0));
        Ok(())
    }

    #[test]
    fn test_vec3_cross() -> Result<()> {
        let input = r#"
            v1 := vec3(1.0f32, 0.0f32, 0.0f32);
            v2 := vec3(0.0f32, 1.0f32, 0.0f32);
            v3 := vec3_cross(v1, v2);
            vec3_len(v3)
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Float32(1.0));
        Ok(())
    }

    #[test]
    fn test_quat_operations() -> Result<()> {
        let input = r#"
            q := quat_identity();
            v := vec3(1.0f32, 0.0f32, 0.0f32);
            rotated := quat_rotate_vec3(q, v);
            vec3_len(rotated)
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Float32(1.0));
        Ok(())
    }

    #[test]
    fn test_native_function_binding() -> Result<()> {
        use crate::ffi::NativeRegistry;

        let input = r#"
            result := multiply_add(2, 3, 4);
            result
        "#;

        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        let native_names = ["multiply_add"];
        let mut compiler = Compiler::new_with_natives(&program, &native_names);
        let bytecode = compiler.compile()?;

        let mut registry = NativeRegistry::new();
        registry.register("multiply_add", 3, |args, _handles| {
            let a = args[0].as_i64();
            let b = args[1].as_i64();
            let c = args[2].as_i64();
            Ok(Value64::Integer(a * b + c))
        });

        let mut vm = VirtualMachine::new_with_natives(
            bytecode.constants,
            bytecode.functions,
            bytecode.heap,
            registry,
            compiler.native_names,
        );
        vm.run(&bytecode.instructions)?;

        let result = vm.last_popped()?;
        assert_eq!(result, Value64::Integer(10));
        Ok(())
    }

    #[test]
    fn test_import_module() -> Result<()> {
        use std::io::Write;

        let temp_dir = std::env::temp_dir();
        let module_path = temp_dir.join("frost_test_math.frost");

        let mut file = std::fs::File::create(&module_path)?;
        writeln!(file, "add :: fn(a: i64, b: i64) -> i64 {{ a + b }}")?;
        writeln!(file, "mul :: fn(a: i64, b: i64) -> i64 {{ a * b }}")?;
        drop(file);

        let input = format!(
            r#"
            import "{}"
            result := add(3, 4);
            result
            "#,
            module_path.display()
        );

        let result = run_vm_test(&input)?;
        std::fs::remove_file(&module_path).ok();
        assert_eq!(result, Value64::Integer(7));
        Ok(())
    }

    #[test]
    fn test_structs() -> Result<()> {
        let input = r#"
            Point :: struct { x: i64, y: i64 }
            p := Point { x = 10, y = 20 };
            p.x + p.y
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(30));
        Ok(())
    }

    #[test]
    fn test_struct_field_assignment() -> Result<()> {
        let input = r#"
            Point :: struct { x: i64, y: i64 }
            mut p := Point { x = 10, y = 20 };
            p.x = 100;
            p.x + p.y
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(120));
        Ok(())
    }

    #[test]
    fn test_struct_multiple_field_assignments() -> Result<()> {
        let input = r#"
            Point :: struct { x: i64, y: i64, z: i64 }
            mut p := Point { x = 1, y = 2, z = 3 };
            p.x = 10;
            p.y = 20;
            p.z = 30;
            p.x + p.y + p.z
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(60));
        Ok(())
    }

    #[test]
    fn test_hashmap_creation_and_access() -> Result<()> {
        let input = r#"
            m := { 1: 100, 2: 200, 3: 300 };
            m[2]
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(200));
        Ok(())
    }

    #[test]
    fn test_hashmap_assignment() -> Result<()> {
        let input = r#"
            mut m := { 1: 10 };
            m[1] = 999;
            m[1]
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(999));
        Ok(())
    }

    #[test]
    fn test_named_return_single() -> Result<()> {
        let input = r#"
            add :: fn(a: i64, b: i64) -> (result: i64) {
                result = a + b;
            };
            add(10, 20)
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(30));
        Ok(())
    }

    #[test]
    fn test_named_return_with_computation() -> Result<()> {
        let input = r#"
            factorial :: fn(n: i64) -> (result: i64) {
                result = 1;
                mut index := 1;
                while (index <= n) {
                    result = result * index;
                    index = index + 1;
                }
            };
            factorial(5)
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(120));
        Ok(())
    }

    #[test]
    fn test_named_return_early_exit() -> Result<()> {
        let input = r#"
            check :: fn(value: i64) -> (result: i64) {
                result = 0;
                if (value > 10) {
                    result = 100;
                    return result;
                };
                result = value * 2;
            };
            check(5) + check(15)
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(110));
        Ok(())
    }

    #[test]
    fn test_string_concatenation() -> Result<()> {
        let tests = [
            (r#""hello" + " world""#, "hello world"),
            (r#"a := "foo"; b := "bar"; a + b"#, "foobar"),
            (r#""a" + "b" + "c""#, "abc"),
        ];
        for (input, expected) in tests {
            let result = run_vm_test_string(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_arrays_and_builtins() -> Result<()> {
        let tests = [
            ("a := [1, 2, 3]; a[0]", Value64::Integer(1)),
            ("a := [1, 2, 3]; len(a)", Value64::Integer(3)),
            ("a := [1, 2, 3]; first(a)", Value64::Integer(1)),
            ("a := [1, 2, 3]; last(a)", Value64::Integer(3)),
            (
                "a := [1, 2, 3]; b := push(a, 4); last(b)",
                Value64::Integer(4),
            ),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_math_builtins() -> Result<()> {
        let tests = [
            ("abs(-5)", Value64::Integer(5)),
            ("abs(5)", Value64::Integer(5)),
            ("abs(-3.5)", Value64::Float(3.5)),
            ("min(3, 7)", Value64::Integer(3)),
            ("max(3, 7)", Value64::Integer(7)),
            ("min(3.5, 2.1)", Value64::Float(2.1)),
            ("max(3.5, 2.1)", Value64::Float(3.5)),
            ("floor(3.7)", Value64::Float(3.0)),
            ("ceil(3.2)", Value64::Float(4.0)),
            ("sqrt(16.0)", Value64::Float(4.0)),
            ("sqrt(9)", Value64::Float(3.0)),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_string_builtins() -> Result<()> {
        let result = run_vm_test_string(r#"substr("hello world", 0, 5)"#)?;
        assert_eq!(result, "hello");

        let result = run_vm_test_string(r#"substr("hello world", 6, 5)"#)?;
        assert_eq!(result, "world");

        let result = run_vm_test(r#"contains("hello world", "world")"#)?;
        assert_eq!(result, Value64::Bool(true));

        let result = run_vm_test(r#"contains("hello world", "xyz")"#)?;
        assert_eq!(result, Value64::Bool(false));

        let result = run_vm_test_string("to_string(42)")?;
        assert_eq!(result, "42");

        let result = run_vm_test_string("to_string(3.14)")?;
        assert_eq!(result, "3.14");

        let result = run_vm_test(r#"parse_int("123")"#)?;
        assert_eq!(result, Value64::Integer(123));

        let result = run_vm_test(r#"parse_int("-42")"#)?;
        assert_eq!(result, Value64::Integer(-42));

        Ok(())
    }

    #[test]
    fn test_file_io() -> Result<()> {
        let test_path = std::env::temp_dir().join("frost_test_file.txt");
        let test_path_str = test_path.to_string_lossy().replace('\\', "\\\\");

        let write_test =
            format!(r#"write_file("{}", "hello from frost")"#, test_path_str);
        let result = run_vm_test(&write_test)?;
        assert_eq!(result, Value64::Bool(true));

        let exists_test = format!(r#"file_exists("{}")"#, test_path_str);
        let result = run_vm_test(&exists_test)?;
        assert_eq!(result, Value64::Bool(true));

        let read_test = format!(r#"read_file("{}")"#, test_path_str);
        let result = run_vm_test_string(&read_test)?;
        assert_eq!(result, "hello from frost");

        let append_test =
            format!(r#"append_file("{}", " - appended")"#, test_path_str);
        let result = run_vm_test(&append_test)?;
        assert_eq!(result, Value64::Bool(true));

        let result = run_vm_test_string(&read_test)?;
        assert_eq!(result, "hello from frost - appended");

        std::fs::remove_file(&test_path).ok();

        let exists_test = format!(r#"file_exists("{}")"#, test_path_str);
        let result = run_vm_test(&exists_test)?;
        assert_eq!(result, Value64::Bool(false));

        Ok(())
    }

    #[test]
    fn test_declarations() -> Result<()> {
        let tests = [
            ("x := 5; x", Value64::Integer(5)),
            ("x := 10; y := 20; x + y", Value64::Integer(30)),
            ("x : i64 = 42; x", Value64::Integer(42)),
            ("add := fn(a, b) { a + b }; add(3, 4)", Value64::Integer(7)),
            ("PI :: 3.14; PI", Value64::Float(3.14)),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_mutability() -> Result<()> {
        let tests = [
            ("mut x := 5; x = 10; x", Value64::Integer(10)),
            ("mut x : i64 = 5; x = 20; x", Value64::Integer(20)),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_immutability_error() {
        let input = "x := 5; x = 10; x";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let program = parser.parse().unwrap();
        let mut compiler = Compiler::new(&program);
        let result = compiler.compile();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("immutable"));
    }

    #[test]
    fn test_immutability_typed_error() {
        let input = "x : i64 = 5; x = 10; x";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let program = parser.parse().unwrap();
        let mut compiler = Compiler::new(&program);
        let result = compiler.compile();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("immutable"));
    }

    #[test]
    fn test_constant_immutability_error() {
        let input = "PI :: 3.14; PI = 3.0; PI";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let program = parser.parse().unwrap();
        let mut compiler = Compiler::new(&program);
        let result = compiler.compile();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("immutable"));
    }

    #[test]
    fn test_mutable_multiple_reassignments() -> Result<()> {
        let input = "mut x := 1; x = 2; x = 3; x = 4; x";
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(4));
        Ok(())
    }

    #[test]
    fn test_mutable_local_in_function() -> Result<()> {
        let input = r#"
            counter := fn() {
                mut x := 0;
                x = x + 1;
                x = x + 1;
                x = x + 1;
                x
            };
            counter()
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(3));
        Ok(())
    }

    #[test]
    fn test_defer() -> Result<()> {
        let input = r#"
            mut result := 0;
            test := fn() {
                defer result = result + 1;
                defer result = result * 10;
                result = 5;
                return result;
            };
            test();
            result
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(51));
        Ok(())
    }

    #[test]
    fn test_pointer_global_address_and_load() -> Result<()> {
        let input = r#"
            x := 42;
            p := &x;
            p^
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(42));
        Ok(())
    }

    #[test]
    fn test_pointer_store() -> Result<()> {
        let input = r#"
            mut x := 10;
            p := &x;
            p^ = 20;
            x
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(20));
        Ok(())
    }

    #[test]
    fn test_modulo_operator() -> Result<()> {
        let tests = [
            ("10 % 3", Value64::Integer(1)),
            ("17 % 5", Value64::Integer(2)),
            ("100 % 10", Value64::Integer(0)),
            ("7 % 2", Value64::Integer(1)),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_comparison_operators() -> Result<()> {
        let tests = [
            ("5 <= 10", Value64::Bool(true)),
            ("5 <= 5", Value64::Bool(true)),
            ("10 <= 5", Value64::Bool(false)),
            ("10 >= 5", Value64::Bool(true)),
            ("5 >= 5", Value64::Bool(true)),
            ("5 >= 10", Value64::Bool(false)),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_match_expression() -> Result<()> {
        let tests = [
            ("match 1 { case 1: 100 case _: 0 }", Value64::Integer(100)),
            ("match 2 { case 1: 100 case _: 0 }", Value64::Integer(0)),
            (
                "match 3 { case 1: 10 case 2: 20 case 3: 30 case _: 0 }",
                Value64::Integer(30),
            ),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_tuple_expression() -> Result<()> {
        let result = run_vm_test("x := (1, 2, 3); x")?;
        assert!(matches!(result, Value64::HeapRef(_)));
        Ok(())
    }

    #[test]
    fn test_tuple_pattern_matching() -> Result<()> {
        let tests = [
            ("match (0, 0) { case (0, 0): 1 case (0, _): 2 case (_, 0): 3 case (_, _): 4 }", Value64::Integer(1)),
            ("match (0, 1) { case (0, 0): 1 case (0, _): 2 case (_, 0): 3 case (_, _): 4 }", Value64::Integer(2)),
            ("match (1, 0) { case (0, 0): 1 case (0, _): 2 case (_, 0): 3 case (_, _): 4 }", Value64::Integer(3)),
            ("match (1, 1) { case (0, 0): 1 case (0, _): 2 case (_, 0): 3 case (_, _): 4 }", Value64::Integer(4)),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_tagged_union_construction() -> Result<()> {
        let input = "Result :: enum { Ok { value: i64 }, Err { code: i64 } }; r := Result::Ok { value = 42 }; r";
        let result = run_vm_test(input)?;
        assert!(matches!(result, Value64::HeapRef(_)));
        Ok(())
    }

    #[test]
    fn test_tagged_union_switch() -> Result<()> {
        let input = r#"
            Result :: enum { Ok { value: i64 }, Err { code: i64 } };
            r := Result::Ok { value = 42 };
            match r {
                case .Ok { value }: value
                case .Err { code }: 0 - code
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(42));
        Ok(())
    }

    #[test]
    fn test_enum_unit_variants() -> Result<()> {
        let input = r#"
            Color :: enum { Red, Green, Blue };
            c := Color::Green;
            match c {
                case .Red: 1
                case .Green: 2
                case .Blue: 3
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(2));
        Ok(())
    }

    #[test]
    fn test_enum_unit_variant_red() -> Result<()> {
        let input = r#"
            Color :: enum { Red, Green, Blue };
            c := Color::Red;
            match c {
                case .Red: 1
                case .Green: 2
                case .Blue: 3
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(1));
        Ok(())
    }

    #[test]
    fn test_enum_unit_variant_blue() -> Result<()> {
        let input = r#"
            Color :: enum { Red, Green, Blue };
            c := Color::Blue;
            match c {
                case .Red: 1
                case .Green: 2
                case .Blue: 3
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(3));
        Ok(())
    }

    #[test]
    fn test_enum_two_variants_second() -> Result<()> {
        let input = r#"
            Bool :: enum { False, True };
            b := Bool::True;
            match b {
                case .False: 0
                case .True: 1
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(1));
        Ok(())
    }

    #[test]
    fn test_enum_with_type_annotation() -> Result<()> {
        let input = r#"
            Status :: enum { Active, Inactive, Pending };
            s : Status = Status::Active;
            match s {
                case .Active: 100
                case .Inactive: 0
                case .Pending: 50
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(100));
        Ok(())
    }

    #[test]
    fn test_enum_mixed_unit_and_data_variants() -> Result<()> {
        let input = r#"
            Option :: enum { None, Some { value: i64 } };
            x := Option::Some { value = 42 };
            match x {
                case .None: 0
                case .Some { value }: value
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(42));
        Ok(())
    }

    #[test]
    fn test_enum_mixed_none_case() -> Result<()> {
        let input = r#"
            Option :: enum { None, Some { value: i64 } };
            x := Option::None;
            match x {
                case .None: 0
                case .Some { value }: value
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(0));
        Ok(())
    }

    #[test]
    fn test_enum_variant_with_multiple_fields() -> Result<()> {
        let input = r#"
            Point :: enum { Origin, At { x: i64, y: i64 } };
            p := Point::At { x = 10, y = 20 };
            match p {
                case .Origin: 0
                case .At { x, y }: x + y
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(30));
        Ok(())
    }

    #[test]
    fn test_enum_switch_with_default() -> Result<()> {
        let input = r#"
            Status :: enum { A, B, C, D, E };
            s := Status::D;
            match s {
                case .A: 1
                case .B: 2
                case _: 99
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(99));
        Ok(())
    }

    #[test]
    fn test_enum_from_function_return() -> Result<()> {
        let input = r#"
            Result :: enum { Ok { value: i64 }, Err { code: i64 } };

            make_ok :: fn(v: i64) -> Result {
                Result::Ok { value = v }
            }

            r := make_ok(123);
            match r {
                case .Ok { value }: value
                case .Err { code }: 0 - code
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(123));
        Ok(())
    }

    #[test]
    fn test_enum_as_function_param() -> Result<()> {
        let input = r#"
            Result :: enum { Ok { value: i64 }, Err { code: i64 } };

            unwrap :: fn(r: Result) -> i64 {
                match r {
                    case .Ok { value }: value
                    case .Err { code }: 0 - code
                }
            }

            unwrap(Result::Ok { value = 55 })
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(55));
        Ok(())
    }

    #[test]
    fn test_enum_err_variant_matching() -> Result<()> {
        let input = r#"
            Result :: enum { Ok { value: i64 }, Err { code: i64 } };
            r := Result::Err { code = 404 };
            match r {
                case .Ok { value }: value
                case .Err { code }: 0 - code
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(-404));
        Ok(())
    }

    #[test]
    fn test_enum_in_loop() -> Result<()> {
        let input = r#"
            Status :: enum { Active, Inactive };
            mut count := 0;
            for i in 0..5 {
                s := Status::Active;
                result := match s {
                    case .Active: 1
                    case .Inactive: 0
                };
                count = count + result;
            }
            count
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(5));
        Ok(())
    }

    #[test]
    fn test_enum_fully_qualified_pattern() -> Result<()> {
        let input = r#"
            Color :: enum { Red, Green, Blue };
            c := Color::Blue;
            match c {
                case Color::Red: 1
                case Color::Green: 2
                case Color::Blue: 3
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(3));
        Ok(())
    }

    #[test]
    fn test_switch_integer_with_binding() -> Result<()> {
        let input = r#"
            x := 42;
            match x {
                case 0: 0
                case n: n * 2
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(84));
        Ok(())
    }

    #[test]
    fn test_switch_bool_patterns() -> Result<()> {
        let tests = [
            ("match true { case true: 1 case false: 0 }", 1),
            ("match false { case true: 1 case false: 0 }", 0),
        ];

        for (input, expected) in tests {
            let result = run_vm_test(input)?;
            assert_eq!(result, Value64::Integer(expected), "Failed for input: {}", input);
        }
        Ok(())
    }

    #[test]
    fn test_enum_three_data_variants() -> Result<()> {
        let input = r#"
            Shape :: enum {
                Circle { radius: i64 },
                Rectangle { width: i64, height: i64 },
                Triangle { base: i64, height: i64 }
            };

            area :: fn(s: Shape) -> i64 {
                match s {
                    case .Circle { radius }: radius * radius * 3
                    case .Rectangle { width, height }: width * height
                    case .Triangle { base, height }: base * height / 2
                }
            }

            area(Shape::Rectangle { width = 10, height = 5 })
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(50));
        Ok(())
    }

    #[test]
    fn test_enum_nested_switch() -> Result<()> {
        let input = r#"
            Outer :: enum { A, B };
            Inner :: enum { X, Y };

            o := Outer::A;
            i := Inner::Y;

            match o {
                case .A: match i {
                    case .X: 1
                    case .Y: 2
                }
                case .B: 3
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(2));
        Ok(())
    }

    #[test]
    fn test_use_after_move_error() {
        let input = r#"
            Point :: struct { x: i64, y: i64 }
            take_point :: fn(p: Point) { p.x }
            p := Point { x = 10, y = 20 };
            take_point(p);
            p.x
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let program = parser.parse().unwrap();
        let mut compiler = Compiler::new(&program);
        let result = compiler.compile();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("moved"));
    }

    #[test]
    fn test_copy_types_not_moved() -> Result<()> {
        let input = r#"
            x := 42;
            y := x;
            x + y
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(84));
        Ok(())
    }

    #[test]
    fn test_immutable_borrow() -> Result<()> {
        let input = r#"
            x := 42;
            p := &x;
            p^
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(42));
        Ok(())
    }

    #[test]
    fn test_mutable_borrow() -> Result<()> {
        let input = r#"
            mut x := 10;
            p := &mut x;
            p^ = 20;
            x
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(20));
        Ok(())
    }

    #[test]
    fn test_cannot_mut_borrow_immutable() {
        let input = r#"
            x := 42;
            p := &mut x;
            p^
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let program = parser.parse().unwrap();
        let mut compiler = Compiler::new(&program);
        let result = compiler.compile();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("immutable"));
    }

    #[test]
    fn test_exclusive_mutable_borrow() {
        let input = r#"
            mut x := 42;
            p1 := &mut x;
            p2 := &mut x;
            p1^
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let program = parser.parse().unwrap();
        let mut compiler = Compiler::new(&program);
        let result = compiler.compile();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("borrow"));
    }

    #[test]
    fn test_multiple_immutable_borrows() -> Result<()> {
        let input = r#"
            x := 42;
            p1 := &x;
            p2 := &x;
            p1^ + p2^
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(84));
        Ok(())
    }

    #[test]
    fn test_struct_field_access_no_move() -> Result<()> {
        let input = r#"
            Point :: struct { x: i64, y: i64 }
            p := Point { x = 10, y = 20 };
            p.x + p.y
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(30));
        Ok(())
    }

    #[test]
    fn test_struct_field_access_in_function() -> Result<()> {
        let input = r#"
            Person :: struct { name: str, age: i64 }
            get_age :: fn(p: Person) -> i64 { p.age }
            alice := Person { name = "Alice", age = 30 };
            get_age(alice)
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(30));
        Ok(())
    }

    #[test]
    fn test_struct_field_comparison_in_function() -> Result<()> {
        let input = r#"
            Person :: struct { name: str, age: i64 }
            is_adult :: fn(p: Person) -> bool { p.age >= 18 }
            alice := Person { name = "Alice", age = 30 };
            is_adult(alice)
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Bool(true));
        Ok(())
    }

    #[test]
    fn test_struct_return_from_function() -> Result<()> {
        let input = r#"
            Point :: struct { x: i64, y: i64 }
            make_point :: fn(a: i64, b: i64) -> Point {
                Point { x = a, y = b }
            }
            p := make_point(10, 20);
            p.x + p.y
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(30));
        Ok(())
    }

    #[test]
    fn test_struct_return_three_fields() -> Result<()> {
        let input = r#"
            Vec3 :: struct { x: i64, y: i64, z: i64 }
            make_vec :: fn(a: i64, b: i64, c: i64) -> Vec3 {
                Vec3 { x = a, y = b, z = c }
            }
            v := make_vec(100, 200, 300);
            v.x + v.y + v.z
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(600));
        Ok(())
    }

    #[test]
    fn test_struct_return_with_string_field() -> Result<()> {
        let input = r#"
            Person :: struct { name: str, age: i64 }
            make_person :: fn(n: str, a: i64) -> Person {
                Person { name = n, age = a }
            }
            p := make_person("Alice", 30);
            p.age
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(30));
        Ok(())
    }

    #[test]
    fn test_struct_return_second_field_access() -> Result<()> {
        let input = r#"
            Pair :: struct { first: i64, second: i64 }
            make_pair :: fn(a: i64, b: i64) -> Pair {
                Pair { first = a, second = b }
            }
            p := make_pair(111, 222);
            p.second
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(222));
        Ok(())
    }

    #[test]
    fn test_struct_return_nested_calls() -> Result<()> {
        let input = r#"
            Point :: struct { x: i64, y: i64 }
            make_point :: fn(a: i64, b: i64) -> Point {
                Point { x = a, y = b }
            }
            double_point :: fn(p: Point) -> Point {
                Point { x = p.x * 2, y = p.y * 2 }
            }
            p1 := make_point(5, 10);
            p2 := double_point(p1);
            p2.x + p2.y
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(30));
        Ok(())
    }

    #[test]
    fn test_enum_return_from_function() -> Result<()> {
        let input = r#"
            Color :: enum { Red, Green, Blue }
            get_color :: fn() -> Color {
                Color::Blue
            }
            c := get_color();
            match c {
                case .Red: 1
                case .Green: 2
                case .Blue: 3
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(3));
        Ok(())
    }

    #[test]
    fn test_enum_return_with_fields() -> Result<()> {
        let input = r#"
            Token :: enum { Number { value: i64 }, Plus, End }
            get_token :: fn() -> Token {
                Token::Number { value = 42 }
            }
            t := get_token();
            match t {
                case .Number { value }: value
                case .Plus: 0
                case .End: 0
            }
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(42));
        Ok(())
    }

    #[test]
    fn test_enum_return_multiple_variants() -> Result<()> {
        let input = r#"
            Result :: enum { Ok { value: i64 }, Err { code: i64 } }
            make_ok :: fn(v: i64) -> Result {
                Result::Ok { value = v }
            }
            make_err :: fn(c: i64) -> Result {
                Result::Err { code = c }
            }
            r1 := make_ok(100);
            r2 := make_err(404);
            v1 := match r1 {
                case .Ok { value }: value
                case .Err { code }: code
            };
            v2 := match r2 {
                case .Ok { value }: value
                case .Err { code }: code
            };
            v1 + v2
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(504));
        Ok(())
    }

    #[test]
    fn test_comptime_for_typename() -> Result<()> {
        let input = r#"
            Position :: struct { x: f64, y: f64 }
            comptime for T in [Position] {
                print(typename(T))
            }
            42
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(42));
        Ok(())
    }

    #[test]
    fn test_shift_operators() -> Result<()> {
        let input = "1 << 3";
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(8));
        Ok(())
    }

    #[test]
    fn test_shift_right_operator() -> Result<()> {
        let input = "16 >> 2";
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(4));
        Ok(())
    }

    #[test]
    fn test_bitwise_or_operator() -> Result<()> {
        let input = "1 | 2 | 4";
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(7));
        Ok(())
    }

    #[test]
    fn test_bitwise_and_operator() -> Result<()> {
        let input = "7 & 3";
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(3));
        Ok(())
    }

    #[test]
    fn test_bitwise_and_masking() -> Result<()> {
        let input = "255 & 15";
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(15));
        Ok(())
    }

    #[test]
    fn test_comptime_bitmask_generation() -> Result<()> {
        let input = r#"
            Position :: struct { x: f64, y: f64 }
            Velocity :: struct { dx: f64, dy: f64 }
            Health :: struct { value: i64 }
            comptime for index, T in [Position, Velocity, Health] {
                BIT_#T :: 1 << index
            }
            BIT_Position | BIT_Velocity | BIT_Health
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(7));
        Ok(())
    }

    #[test]
    fn test_array_index_assignment_in_loop_with_conditional() -> Result<()> {
        let input = r#"
            Point :: struct { x: i64, y: i64 }

            update :: fn(mut masks: []i64, mut points: []Point, count: i64) -> i64 {
                for i in 0..count {
                    mask := masks[i]
                    if (mask != 0) {
                        point := points[i]
                        points[i] = Point { x = point.x + 1, y = point.y + 2 }
                    }
                }
                p0 := points[0]
                p1 := points[1]
                p2 := points[2]
                p0.x + p0.y + p1.x + p1.y + p2.x + p2.y
            }

            mut masks := [1, 1, 0]
            mut points := [
                Point { x = 10, y = 20 },
                Point { x = 30, y = 40 },
                Point { x = 50, y = 60 }
            ]
            update(masks, points, 3)
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(33 + 73 + 110));
        Ok(())
    }

    #[test]
    fn test_array_index_assignment_global() -> Result<()> {
        let input = r#"
            Point :: struct { x: i64, y: i64 }

            mut points := [
                Point { x = 10, y = 20 },
                Point { x = 30, y = 40 }
            ]

            for i in 0..2 {
                point := points[i]
                points[i] = Point { x = point.x + 5, y = point.y + 10 }
            }

            p0 := points[0]
            p1 := points[1]
            p0.x + p0.y + p1.x + p1.y
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(45 + 85));
        Ok(())
    }

    #[test]
    fn test_embedding_api() -> Result<()> {
        let input = r#"
            Input :: struct { x: f64, y: f64, dt: f64 }
            Output :: struct { x: f64, y: f64 }

            update :: fn(input: Input) -> Output {
                Output {
                    x = input.x + 10.0 * input.dt,
                    y = input.y + 5.0 * input.dt
                }
            }
        "#;

        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;
        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        assert!(bytecode.global_symbols.contains_key("update"));
        let update_idx = *bytecode.global_symbols.get("update").unwrap();

        let mut vm = VirtualMachine::new(
            bytecode.constants,
            bytecode.functions,
            bytecode.heap,
        );
        vm.run(&bytecode.instructions)?;

        let input_struct = vm.create_struct("Input", vec![
            Value64::Float(100.0),
            Value64::Float(200.0),
            Value64::Float(0.016),
        ]);

        let output = vm.call_global(update_idx, &[input_struct])?;

        let new_x = vm.get_struct_field(output, 0)?;
        let new_y = vm.get_struct_field(output, 1)?;

        assert_eq!(new_x.as_f64(), 100.16);
        assert_eq!(new_y.as_f64(), 200.08);

        Ok(())
    }

    #[test]
    fn test_embedding_api_nonexistent_global() -> Result<()> {
        let input = r#"
            foo :: 42
        "#;

        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;
        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        assert!(bytecode.global_symbols.get("foo").is_some());
        assert!(bytecode.global_symbols.get("bar").is_none());

        Ok(())
    }

    #[test]
    fn test_embedding_api_field_index_out_of_bounds() -> Result<()> {
        let input = r#"
            Point :: struct { x: f64, y: f64 }
        "#;

        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;
        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        let mut vm = VirtualMachine::new(
            bytecode.constants,
            bytecode.functions,
            bytecode.heap,
        );
        vm.run(&bytecode.instructions)?;

        let point = vm.create_struct("Point", vec![1.0.into(), 2.0.into()]);

        assert!(vm.get_struct_field(point, 0).is_ok());
        assert!(vm.get_struct_field(point, 1).is_ok());
        assert!(vm.get_struct_field(point, 2).is_err());
        assert!(vm.get_struct_field(point, 100).is_err());

        Ok(())
    }

    #[test]
    fn test_embedding_api_get_field_on_non_struct() -> Result<()> {
        let input = r#"
            x := 42
        "#;

        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;
        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        let mut vm = VirtualMachine::new(
            bytecode.constants,
            bytecode.functions,
            bytecode.heap,
        );
        vm.run(&bytecode.instructions)?;

        let integer = Value64::Integer(42);
        assert!(vm.get_struct_field(integer, 0).is_err());

        let float = Value64::Float(3.14);
        assert!(vm.get_struct_field(float, 0).is_err());

        let boolean = Value64::Bool(true);
        assert!(vm.get_struct_field(boolean, 0).is_err());

        Ok(())
    }

    #[test]
    fn test_arena_new_and_alloc() -> Result<()> {
        let input = r#"
            arena := arena_new(100);
            idx1 := arena_alloc(arena, 42);
            idx2 := arena_alloc(arena, 99);
            val1 := arena_get(arena, idx1);
            val2 := arena_get(arena, idx2);
            val1 + val2
        "#;

        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;
        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        let mut vm = VirtualMachine::new(
            bytecode.constants,
            bytecode.functions,
            bytecode.heap,
        );
        vm.run(&bytecode.instructions)?;

        assert_eq!(vm.last_popped()?, Value64::Integer(141));
        Ok(())
    }

    #[test]
    fn test_arena_reset() -> Result<()> {
        let input = r#"
            arena := arena_new(10);
            arena_alloc(arena, 1);
            arena_alloc(arena, 2);
            arena_alloc(arena, 3);
            arena_reset(arena);
            idx := arena_alloc(arena, 100);
            arena_get(arena, idx)
        "#;

        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;
        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        let mut vm = VirtualMachine::new(
            bytecode.constants,
            bytecode.functions,
            bytecode.heap,
        );
        vm.run(&bytecode.instructions)?;

        assert_eq!(vm.last_popped()?, Value64::Integer(100));
        Ok(())
    }

    #[test]
    fn test_pool_alloc_and_get() -> Result<()> {
        let input = r#"
            pool := pool_new(100);
            h1 := pool_alloc(pool, 42);
            h2 := pool_alloc(pool, 99);
            val1 := pool_get(pool, h1);
            val2 := pool_get(pool, h2);
            val1 + val2
        "#;

        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;
        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        let mut vm = VirtualMachine::new(
            bytecode.constants,
            bytecode.functions,
            bytecode.heap,
        );
        vm.run(&bytecode.instructions)?;

        assert_eq!(vm.last_popped()?, Value64::Integer(141));
        Ok(())
    }

    #[test]
    fn test_pool_free_invalidates_handle() -> Result<()> {
        let input = r#"
            pool := pool_new(100);
            h := pool_alloc(pool, 42);
            pool_free(pool, h);
            pool_get(pool, h)
        "#;

        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;
        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        let mut vm = VirtualMachine::new(
            bytecode.constants,
            bytecode.functions,
            bytecode.heap,
        );
        vm.run(&bytecode.instructions)?;

        assert_eq!(vm.last_popped()?, Value64::Null);
        Ok(())
    }

    #[test]
    fn test_handle_index_and_generation() -> Result<()> {
        let input = r#"
            pool := pool_new(100);
            h := pool_alloc(pool, 42);
            handle_index(h) + handle_generation(h)
        "#;

        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;
        let mut compiler = Compiler::new(&program);
        let bytecode = compiler.compile()?;

        let mut vm = VirtualMachine::new(
            bytecode.constants,
            bytecode.functions,
            bytecode.heap,
        );
        vm.run(&bytecode.instructions)?;

        let result = vm.last_popped()?.as_i64();
        assert!(result >= 0);
        Ok(())
    }

    #[test]
    fn test_context_access_allocator() -> Result<()> {
        let input = r#"
            a := context.allocator;
            a
        "#;
        let result = run_vm_test(input)?;
        assert!(matches!(result, Value64::HeapRef(_)));
        Ok(())
    }

    #[test]
    fn test_context_alloc_builtin() -> Result<()> {
        let input = r#"
            idx := alloc(42);
            idx
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(0));
        Ok(())
    }

    #[test]
    fn test_context_temp_alloc_builtin() -> Result<()> {
        let input = r#"
            idx := temp_alloc(99);
            idx
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(0));
        Ok(())
    }

    #[test]
    fn test_context_alloc_multiple() -> Result<()> {
        let input = r#"
            idx1 := alloc(10);
            idx2 := alloc(20);
            idx3 := alloc(30);
            idx1 + idx2 + idx3
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(0 + 1 + 2));
        Ok(())
    }

    #[test]
    fn test_context_propagates_to_functions() -> Result<()> {
        let input = r#"
            do_alloc :: fn() -> i64 {
                alloc(100)
            };
            idx1 := alloc(50);
            idx2 := do_alloc();
            idx1 + idx2
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(0 + 1));
        Ok(())
    }

    #[test]
    fn test_push_allocator_scope() -> Result<()> {
        let input = r#"
            custom_arena := arena_new(100);
            idx_before := alloc(1);
            push_allocator(custom_arena) {
                idx_inside := alloc(2);
            }
            idx_after := alloc(3);
            idx_before + idx_after
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(0 + 1));
        Ok(())
    }

    #[test]
    fn test_generic_identity_function() -> Result<()> {
        let input = r#"
            identity :: fn(x: $T) -> T { x }
            identity(42)
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(42));
        Ok(())
    }

    #[test]
    fn test_generic_identity_with_bool() -> Result<()> {
        let input = r#"
            identity :: fn(x: $T) -> T { x }
            identity(true)
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Bool(true));
        Ok(())
    }

    #[test]
    fn test_generic_first_of_two() -> Result<()> {
        let input = r#"
            first :: fn(a: $T, b: T) -> T { a }
            first(10, 20)
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(10));
        Ok(())
    }

    #[test]
    fn test_generic_multiple_type_params() -> Result<()> {
        let input = r#"
            second :: fn(a: $T, b: $U) -> U { b }
            second(1, 99)
        "#;
        let result = run_vm_test(input)?;
        assert_eq!(result, Value64::Integer(99));
        Ok(())
    }
}
