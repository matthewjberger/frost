use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Value64 {
    Integer(i64),
    Float(f64),
    Float32(f32),
    Bool(bool),
    #[default]
    Null,
    HeapRef(u32),
}

impl Value64 {
    pub fn as_i64(self) -> i64 {
        match self {
            Value64::Integer(v) => v,
            _ => panic!("expected integer, got {:?}", self),
        }
    }

    pub fn as_f64(self) -> f64 {
        match self {
            Value64::Float(v) => v,
            Value64::Float32(v) => v as f64,
            _ => panic!("expected float, got {:?}", self),
        }
    }

    pub fn as_f32(self) -> f32 {
        match self {
            Value64::Float32(v) => v,
            Value64::Float(v) => v as f32,
            _ => panic!("expected f32, got {:?}", self),
        }
    }

    pub fn as_bool(self) -> bool {
        match self {
            Value64::Bool(v) => v,
            _ => panic!("expected bool, got {:?}", self),
        }
    }

    pub fn as_heap_ref(self) -> u32 {
        match self {
            Value64::HeapRef(v) => v,
            _ => panic!("expected heap ref, got {:?}", self),
        }
    }

    pub fn is_truthy(self) -> bool {
        match self {
            Value64::Bool(v) => v,
            Value64::Null => false,
            Value64::Integer(0) => false,
            _ => true,
        }
    }
}

impl Display for Value64 {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Value64::Integer(v) => write!(f, "{}", v),
            Value64::Float(v) => write!(f, "{}", v),
            Value64::Float32(v) => write!(f, "{}f32", v),
            Value64::Bool(v) => write!(f, "{}", v),
            Value64::Null => write!(f, "null"),
            Value64::HeapRef(v) => write!(f, "heap@{}", v),
        }
    }
}

impl From<i64> for Value64 {
    fn from(value: i64) -> Self {
        Value64::Integer(value)
    }
}

impl From<i32> for Value64 {
    fn from(value: i32) -> Self {
        Value64::Integer(value as i64)
    }
}

impl From<f64> for Value64 {
    fn from(value: f64) -> Self {
        Value64::Float(value)
    }
}

impl From<f32> for Value64 {
    fn from(value: f32) -> Self {
        Value64::Float32(value)
    }
}

impl From<bool> for Value64 {
    fn from(value: bool) -> Self {
        Value64::Bool(value)
    }
}

#[derive(Debug, Clone)]
pub struct TypedClosure {
    pub function_index: u32,
    pub free: Vec<Value64>,
}

#[derive(Debug, Clone)]
pub struct TypedBuiltIn {
    pub name: String,
    pub index: u32,
}

#[derive(Debug, Clone)]
pub struct ArenaData {
    pub storage: Vec<Value64>,
    pub next_index: usize,
    pub capacity: usize,
}

impl ArenaData {
    pub fn new(capacity: usize) -> Self {
        Self {
            storage: vec![Value64::Null; capacity],
            next_index: 0,
            capacity,
        }
    }

    pub fn alloc(&mut self, value: Value64) -> Option<usize> {
        if self.next_index >= self.capacity {
            return None;
        }
        let index = self.next_index;
        self.storage[index] = value;
        self.next_index += 1;
        Some(index)
    }

    pub fn reset(&mut self) {
        self.next_index = 0;
    }
}

#[derive(Debug, Clone)]
pub struct PoolData {
    pub storage: Vec<Value64>,
    pub generations: Vec<u32>,
    pub free_list: Vec<u32>,
    pub capacity: usize,
}

impl PoolData {
    pub fn new(capacity: usize) -> Self {
        Self {
            storage: vec![Value64::Null; capacity],
            generations: vec![0; capacity],
            free_list: (0..capacity as u32).rev().collect(),
            capacity,
        }
    }

    pub fn alloc(&mut self, value: Value64) -> Option<(u32, u32)> {
        let index = self.free_list.pop()?;
        self.storage[index as usize] = value;
        let generation = self.generations[index as usize];
        Some((index, generation))
    }

    pub fn get(&self, index: u32, generation: u32) -> Option<Value64> {
        if index as usize >= self.capacity {
            return None;
        }
        if self.generations[index as usize] != generation {
            return None;
        }
        Some(self.storage[index as usize])
    }

    pub fn free(&mut self, index: u32, generation: u32) -> bool {
        if index as usize >= self.capacity {
            return false;
        }
        if self.generations[index as usize] != generation {
            return false;
        }
        self.generations[index as usize] = self.generations[index as usize].wrapping_add(1);
        self.free_list.push(index);
        true
    }
}

#[derive(Debug, Clone)]
pub enum HeapObject {
    String(String),
    Array(Vec<Value64>),
    HashMap(HashMap<u64, Value64>),
    Closure(TypedClosure),
    Struct(String, Vec<Value64>),
    BuiltIn(TypedBuiltIn),
    NativeFunction(String),
    NativeHandle(u32),
    TaggedUnion(u32, Vec<Value64>),
    Vec2(f32, f32),
    Vec3(f32, f32, f32),
    Vec4(f32, f32, f32, f32),
    Quat(f32, f32, f32, f32),
    Mat4([f32; 16]),
    Arena(ArenaData),
    Pool(PoolData),
    Handle(u32, u32),
    Free,
}

impl Display for HeapObject {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            HeapObject::String(s) => write!(f, "\"{}\"", s),
            HeapObject::Array(elements) => {
                let strs: Vec<String> =
                    elements.iter().map(|e| e.to_string()).collect();
                write!(f, "[{}]", strs.join(", "))
            }
            HeapObject::HashMap(map) => write!(f, "{:?}", map),
            HeapObject::Closure(closure) => {
                write!(
                    f,
                    "Closure[fn={}, free={}]",
                    closure.function_index,
                    closure.free.len()
                )
            }
            HeapObject::Struct(name, fields) => {
                let strs: Vec<String> =
                    fields.iter().map(|e| e.to_string()).collect();
                write!(f, "{} {{ {} }}", name, strs.join(", "))
            }
            HeapObject::BuiltIn(builtin) => {
                write!(f, "builtin:{}", builtin.name)
            }
            HeapObject::NativeFunction(name) => write!(f, "native:{}", name),
            HeapObject::NativeHandle(index) => write!(f, "handle@{}", index),
            HeapObject::TaggedUnion(tag, fields) => {
                let strs: Vec<String> =
                    fields.iter().map(|e| e.to_string()).collect();
                write!(f, "TaggedUnion(tag={}, {{ {} }})", tag, strs.join(", "))
            }
            HeapObject::Vec2(x, y) => write!(f, "Vec2({}, {})", x, y),
            HeapObject::Vec3(x, y, z) => write!(f, "Vec3({}, {}, {})", x, y, z),
            HeapObject::Vec4(x, y, z, w) => {
                write!(f, "Vec4({}, {}, {}, {})", x, y, z, w)
            }
            HeapObject::Quat(x, y, z, w) => {
                write!(f, "Quat({}, {}, {}, {})", x, y, z, w)
            }
            HeapObject::Mat4(m) => write!(f, "Mat4({:?})", m),
            HeapObject::Arena(arena) => {
                write!(f, "Arena(used={}/{})", arena.next_index, arena.capacity)
            }
            HeapObject::Pool(pool) => {
                write!(
                    f,
                    "Pool(allocated={}/{})",
                    pool.capacity - pool.free_list.len(),
                    pool.capacity
                )
            }
            HeapObject::Handle(index, generation) => write!(f, "Handle({}, {})", index, generation),
            HeapObject::Free => write!(f, "<freed>"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value64_integer() {
        let v = Value64::Integer(42);
        assert_eq!(v.as_i64(), 42);
        assert!(v.is_truthy());
    }

    #[test]
    fn test_value64_float() {
        let v = Value64::Float(3.14);
        assert_eq!(v.as_f64(), 3.14);
        assert!(v.is_truthy());
    }

    #[test]
    fn test_value64_bool() {
        assert!(Value64::Bool(true).as_bool());
        assert!(!Value64::Bool(false).as_bool());
        assert!(Value64::Bool(true).is_truthy());
        assert!(!Value64::Bool(false).is_truthy());
    }

    #[test]
    fn test_value64_null() {
        assert!(!Value64::Null.is_truthy());
    }

    #[test]
    fn test_value64_heap_ref() {
        let v = Value64::HeapRef(123);
        assert_eq!(v.as_heap_ref(), 123);
        assert!(v.is_truthy());
    }

    #[test]
    fn test_value64_copy() {
        let v1 = Value64::Integer(42);
        let v2 = v1;
        assert_eq!(v1.as_i64(), 42);
        assert_eq!(v2.as_i64(), 42);
    }

    #[test]
    fn test_heap_object_string() {
        let obj = HeapObject::String("hello".to_string());
        assert_eq!(format!("{}", obj), "\"hello\"");
    }

    #[test]
    fn test_heap_object_array() {
        let obj =
            HeapObject::Array(vec![Value64::Integer(1), Value64::Integer(2)]);
        assert_eq!(format!("{}", obj), "[1, 2]");
    }

    #[test]
    fn test_value64_from_i64() {
        let v: Value64 = 42i64.into();
        assert_eq!(v.as_i64(), 42);
    }

    #[test]
    fn test_value64_from_i32() {
        let v: Value64 = 42i32.into();
        assert_eq!(v.as_i64(), 42);
    }

    #[test]
    fn test_value64_from_f64() {
        let v: Value64 = 3.14f64.into();
        assert_eq!(v.as_f64(), 3.14);
    }

    #[test]
    fn test_value64_from_f32() {
        let v: Value64 = 3.14f32.into();
        assert_eq!(v.as_f32(), 3.14);
    }

    #[test]
    fn test_value64_from_bool() {
        let v: Value64 = true.into();
        assert!(v.as_bool());
    }
}
