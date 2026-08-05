use std::fmt::Display;

/// How long an array is, before the size parameters written in it are known.
///
/// A number, a name that stands for one, or arithmetic over those. It is
/// arithmetic and nothing else: no call, no comparison, no name that is not a
/// size. So working one out is bounded by how it was written, which is the same
/// bound every other compile-time construct in the language has.
#[derive(
    serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq, Hash,
)]
pub enum SizeExpr {
    Number(i64),
    Named(String),
    Binary(Box<SizeExpr>, SizeOp, Box<SizeExpr>),
}

#[derive(
    serde::Serialize,
    serde::Deserialize,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
)]
pub enum SizeOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
}

impl SizeExpr {
    /// The number this works out to, given what the names in it stand for, or
    /// nothing when a name is still unbound or the arithmetic has no answer.
    pub fn evaluate(
        &self,
        bound: &dyn Fn(&str) -> Option<i64>,
    ) -> Option<i64> {
        match self {
            SizeExpr::Number(value) => Some(*value),
            SizeExpr::Named(name) => bound(name),
            SizeExpr::Binary(left, op, right) => {
                let left = left.evaluate(bound)?;
                let right = right.evaluate(bound)?;
                match op {
                    SizeOp::Add => left.checked_add(right),
                    SizeOp::Subtract => left.checked_sub(right),
                    SizeOp::Multiply => left.checked_mul(right),
                    SizeOp::Divide => left.checked_div(right),
                    SizeOp::Modulo => left.checked_rem(right),
                }
            }
        }
    }

    /// The size parameters written in this, so a walk that collects what a
    /// generic body depends on finds them.
    pub fn names(&self, out: &mut Vec<String>) {
        match self {
            SizeExpr::Number(_) => {}
            SizeExpr::Named(name) => out.push(name.clone()),
            SizeExpr::Binary(left, _, right) => {
                left.names(out);
                right.names(out);
            }
        }
    }
}

impl Display for SizeExpr {
    /// Written back with every operation bracketed, so two spellings that mean
    /// the same length read the same and the name an instance is mangled under
    /// does not depend on how it was written.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SizeExpr::Number(value) => write!(f, "{value}"),
            SizeExpr::Named(name) => write!(f, "{name}"),
            SizeExpr::Binary(left, op, right) => {
                let op = match op {
                    SizeOp::Add => "+",
                    SizeOp::Subtract => "-",
                    SizeOp::Multiply => "*",
                    SizeOp::Divide => "/",
                    SizeOp::Modulo => "%",
                };
                write!(f, "({left} {op} {right})")
            }
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub enum Type {
    I8,
    I16,
    I32,
    I64,
    Isize,
    U8,
    U16,
    U32,
    U64,
    Usize,
    F32,
    F64,
    Bool,
    Str,
    Void,
    Ptr(Box<Type>),
    Ref(Box<Type>),
    RefMut(Box<Type>),
    Array(Box<Type>, usize),
    ArrayGeneric(Box<Type>, SizeExpr),
    ConstUsize(usize),
    // A function named at a call as a compile-time argument, so the body it is
    // substituted into calls it directly rather than through a pointer.
    ConstFn(String),
    // A constant named at a call as a compile-time argument. This is what a
    // capability bundle is passed as: the body names the constant wherever it
    // named the parameter, so a call through one of its fields is a call to
    // the function that field was given.
    ConstValue(String),
    Slice(Box<Type>),
    Proc(Vec<Type>, Box<Type>),
    Struct(String),
    Enum(String),
    // `Meters :: distinct i64`: the representation of the inner type under a
    // name of its own, so a Meters is not an i64 and not a Feet. The name is
    // what makes it nominal. Without it two distinct types over the same
    // representation would compare equal.
    Distinct(String, Box<Type>),
    Handle(Box<Type>),
    TypeParam(String),
    Unknown,
}

impl Type {
    pub fn size_of(&self) -> usize {
        match self {
            Type::I8 | Type::U8 | Type::Bool => 1,
            Type::I16 | Type::U16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64 | Type::U64 | Type::Isize | Type::Usize | Type::F64 => 8,
            Type::Ptr(_) | Type::Ref(_) | Type::RefMut(_) => 8,
            Type::Str => 16,
            Type::Void => 0,
            Type::Array(inner, count) => inner.size_of() * count,
            Type::ArrayGeneric(..)
            | Type::ConstUsize(_)
            | Type::ConstFn(_)
            | Type::ConstValue(_) => 0,
            Type::Slice(_) => 16,
            Type::Proc(_, _) => 8,
            Type::Struct(_) => 0,
            Type::Enum(_) => 4,
            Type::Distinct(_, inner) => inner.size_of(),
            Type::Handle(_) => 8,
            Type::TypeParam(_) => 0,
            Type::Unknown => 0,
        }
    }

    pub fn align_of(&self) -> usize {
        match self {
            Type::I8 | Type::U8 | Type::Bool => 1,
            Type::I16 | Type::U16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64
            | Type::U64
            | Type::Isize
            | Type::Usize
            | Type::F64
            | Type::Ptr(_)
            | Type::Ref(_)
            | Type::RefMut(_) => 8,
            Type::Str | Type::Slice(_) => 8,
            Type::Void => 1,
            Type::Array(inner, _) => inner.align_of(),
            Type::ArrayGeneric(..)
            | Type::ConstUsize(_)
            | Type::ConstFn(_)
            | Type::ConstValue(_) => 1,
            Type::Proc(_, _) => 8,
            Type::Struct(_) => 8,
            Type::Enum(_) => 4,
            Type::Distinct(_, inner) => inner.align_of(),
            Type::Handle(_) => 4,
            Type::TypeParam(_) => 1,
            Type::Unknown => 1,
        }
    }

    /// The name a generic instance was stamped from: `Vec<i64>` is `Vec`. A
    /// `linear` is written once, on the template, so that is where the answer
    /// lives.
    pub fn template_of(name: &str) -> &str {
        match name.find('<') {
            Some(at) => &name[..at],
            None => name,
        }
    }

    /// Whether a value of this type has to be consumed exactly once, given the
    /// set of types declared `linear` and the ones that hold such a value.
    /// Both the AST-level ownership pass and the IR builder ask this, so they
    /// cannot drift apart on what a resource is.
    pub fn is_linear_with(
        &self,
        linear: &std::collections::HashSet<String>,
    ) -> bool {
        match self {
            // The instantiation first, then the template it came from. A
            // generic declared `linear` is a resource whatever it is bound to,
            // which is what the template answers for. An ordinary generic is a
            // resource only where what it was bound to is one, and that is a
            // question about `Pool<File>` rather than about `Pool`, so asking
            // only the template could never tell the two apart.
            Type::Struct(name) | Type::Enum(name) => {
                linear.contains(name.as_str())
                    || linear.contains(Type::template_of(name))
            }
            Type::Distinct(_, inner) => inner.is_linear_with(linear),
            // A run of resources is a resource: freeing the run is not freeing
            // what is in it, and a fixed array holds its elements by value.
            Type::Array(inner, _) => inner.is_linear_with(linear),
            _ => false,
        }
    }

    pub fn is_copy(&self) -> bool {
        match self {
            Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::Isize => true,
            Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::Usize => true,
            Type::F32 | Type::F64 | Type::Bool => true,
            Type::Ref(_) | Type::RefMut(_) | Type::Ptr(_) => true,
            Type::Proc(_, _) | Type::Void => true,
            Type::Array(_, _) => true,
            Type::ArrayGeneric(..)
            | Type::ConstUsize(_)
            | Type::ConstFn(_)
            | Type::ConstValue(_) => false,
            Type::Str | Type::Slice(_) => true,
            Type::Struct(_) | Type::Enum(_) => false,
            Type::Distinct(_, inner) => inner.is_copy(),
            Type::Handle(_) => true,
            Type::TypeParam(_) => false,
            Type::Unknown => false,
        }
    }

    pub fn needs_drop(&self) -> bool {
        match self {
            Type::Str | Type::Slice(_) => false,
            Type::Struct(_) | Type::Enum(_) => true,
            Type::Array(inner, _) => inner.needs_drop(),
            Type::Distinct(_, inner) => inner.needs_drop(),
            _ => false,
        }
    }

    pub fn is_reference(&self) -> bool {
        matches!(self, Type::Ref(_) | Type::RefMut(_))
    }

    // A distinct type computes, is passed and is stored as what it is
    // represented by, so anything asking which kind of number it is has to look
    // through the name.
    pub fn is_float(&self) -> bool {
        match self {
            Type::F32 | Type::F64 => true,
            Type::Distinct(_, inner) => inner.is_float(),
            _ => false,
        }
    }

    // Which kind of number this is, looked at through any name it carries, for
    // the same reason `is_float` looks through one.
    pub fn is_integer(&self) -> bool {
        match self {
            Type::I8
            | Type::I16
            | Type::I32
            | Type::I64
            | Type::Isize
            | Type::U8
            | Type::U16
            | Type::U32
            | Type::U64
            | Type::Usize => true,
            Type::Distinct(_, inner) => inner.is_integer(),
            _ => false,
        }
    }

    pub fn contains_reference(&self) -> bool {
        match self {
            Type::Ref(_) | Type::RefMut(_) => true,
            Type::Array(inner, _) => inner.contains_reference(),
            Type::Slice(inner) => inner.contains_reference(),
            Type::Ptr(inner) => inner.contains_reference(),
            Type::Distinct(_, inner) => inner.contains_reference(),
            Type::Handle(inner) => inner.contains_reference(),
            _ => false,
        }
    }

    pub fn is_second_class(&self) -> bool {
        self.is_reference()
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::I8 => write!(f, "i8"),
            Type::I16 => write!(f, "i16"),
            Type::I32 => write!(f, "i32"),
            Type::I64 => write!(f, "i64"),
            Type::Isize => write!(f, "isize"),
            Type::U8 => write!(f, "u8"),
            Type::U16 => write!(f, "u16"),
            Type::U32 => write!(f, "u32"),
            Type::U64 => write!(f, "u64"),
            Type::Usize => write!(f, "usize"),
            Type::F32 => write!(f, "f32"),
            Type::F64 => write!(f, "f64"),
            Type::Bool => write!(f, "bool"),
            Type::Str => write!(f, "str"),
            Type::Void => write!(f, "void"),
            Type::Ptr(inner) => write!(f, "^{}", inner),
            Type::Ref(inner) => write!(f, "&{}", inner),
            Type::RefMut(inner) => write!(f, "&mut {}", inner),
            Type::Array(inner, size) => write!(f, "[{}]{}", size, inner),
            Type::ArrayGeneric(inner, size) => {
                write!(f, "[{}]{}", size, inner)
            }
            Type::ConstUsize(value) => write!(f, "{}", value),
            Type::ConstFn(name) | Type::ConstValue(name) => {
                write!(f, "{}", name)
            }
            Type::Slice(inner) => write!(f, "[]{}", inner),
            Type::Proc(params, ret) => {
                let param_strs: Vec<String> =
                    params.iter().map(|p| p.to_string()).collect();
                write!(f, "proc({}) -> {}", param_strs.join(", "), ret)
            }
            Type::Struct(name) => write!(f, "{}", name),
            Type::Enum(name) => write!(f, "{}", name),
            Type::Distinct(name, _) => write!(f, "{}", name),
            Type::Handle(inner) => write!(f, "Handle<{}>", inner),
            Type::TypeParam(name) => write!(f, "${}", name),
            Type::Unknown => write!(f, "?"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_types_display() {
        assert_eq!(Type::I64.to_string(), "i64");
        assert_eq!(Type::Bool.to_string(), "bool");
        assert_eq!(Type::Str.to_string(), "str");
        assert_eq!(Type::Void.to_string(), "void");
    }

    #[test]
    fn pointer_type_display() {
        let ptr_i64 = Type::Ptr(Box::new(Type::I64));
        assert_eq!(ptr_i64.to_string(), "^i64");

        let ptr_ptr_i64 = Type::Ptr(Box::new(Type::Ptr(Box::new(Type::I64))));
        assert_eq!(ptr_ptr_i64.to_string(), "^^i64");
    }

    #[test]
    fn array_type_display() {
        let arr = Type::Array(Box::new(Type::I64), 10);
        assert_eq!(arr.to_string(), "[10]i64");
    }

    #[test]
    fn slice_type_display() {
        let slice = Type::Slice(Box::new(Type::F32));
        assert_eq!(slice.to_string(), "[]f32");
    }

    #[test]
    fn proc_type_display() {
        let proc_type =
            Type::Proc(vec![Type::I64, Type::I64], Box::new(Type::I64));
        assert_eq!(proc_type.to_string(), "proc(i64, i64) -> i64");

        let proc_void = Type::Proc(vec![], Box::new(Type::Void));
        assert_eq!(proc_void.to_string(), "proc() -> void");
    }

    // A distinct type prints as the name it was declared under, which is what
    // a diagnostic about it should say. `distinct i64` would name the thing it
    // is deliberately not.
    #[test]
    fn distinct_type_display() {
        let distinct =
            Type::Distinct("Meters".to_string(), Box::new(Type::I64));
        assert_eq!(distinct.to_string(), "Meters");
    }

    #[test]
    fn struct_type_display() {
        let struct_type = Type::Struct("Vec3".to_string());
        assert_eq!(struct_type.to_string(), "Vec3");
    }

    #[test]
    fn sizeof_primitives() {
        assert_eq!(Type::I8.size_of(), 1);
        assert_eq!(Type::I16.size_of(), 2);
        assert_eq!(Type::I32.size_of(), 4);
        assert_eq!(Type::I64.size_of(), 8);
        assert_eq!(Type::U8.size_of(), 1);
        assert_eq!(Type::U16.size_of(), 2);
        assert_eq!(Type::U32.size_of(), 4);
        assert_eq!(Type::U64.size_of(), 8);
        assert_eq!(Type::F32.size_of(), 4);
        assert_eq!(Type::F64.size_of(), 8);
        assert_eq!(Type::Bool.size_of(), 1);
        assert_eq!(Type::Void.size_of(), 0);
    }

    #[test]
    fn sizeof_compound() {
        assert_eq!(Type::Ptr(Box::new(Type::I64)).size_of(), 8);
        assert_eq!(Type::Array(Box::new(Type::I64), 10).size_of(), 80);
        assert_eq!(Type::Slice(Box::new(Type::I64)).size_of(), 16);
        assert_eq!(Type::Str.size_of(), 16);
    }

    #[test]
    fn alignof_primitives() {
        assert_eq!(Type::I8.align_of(), 1);
        assert_eq!(Type::I16.align_of(), 2);
        assert_eq!(Type::I32.align_of(), 4);
        assert_eq!(Type::I64.align_of(), 8);
        assert_eq!(Type::Ptr(Box::new(Type::I64)).align_of(), 8);
    }

    #[test]
    fn is_reference() {
        assert!(Type::Ref(Box::new(Type::I64)).is_reference());
        assert!(Type::RefMut(Box::new(Type::I64)).is_reference());
        assert!(!Type::Ptr(Box::new(Type::I64)).is_reference());
        assert!(!Type::I64.is_reference());
    }

    #[test]
    fn contains_reference() {
        assert!(Type::Ref(Box::new(Type::I64)).contains_reference());
        assert!(Type::RefMut(Box::new(Type::I64)).contains_reference());
        assert!(
            Type::Array(Box::new(Type::Ref(Box::new(Type::I64))), 10)
                .contains_reference()
        );
        assert!(!Type::Array(Box::new(Type::I64), 10).contains_reference());
        assert!(!Type::Ptr(Box::new(Type::I64)).contains_reference());
        assert!(!Type::I64.contains_reference());
    }

    #[test]
    fn is_second_class() {
        assert!(Type::Ref(Box::new(Type::I64)).is_second_class());
        assert!(Type::RefMut(Box::new(Type::I64)).is_second_class());
        assert!(!Type::Ptr(Box::new(Type::I64)).is_second_class());
        assert!(!Type::I64.is_second_class());
    }

    #[test]
    fn handle_is_a_copyable_value() {
        let handle = Type::Handle(Box::new(Type::Struct("Entity".to_string())));
        assert_eq!(handle.size_of(), 8);
        assert!(handle.is_copy());
        assert_eq!(handle.to_string(), "Handle<Entity>");
    }

    #[test]
    fn str_is_a_copyable_view() {
        assert_eq!(Type::Str.size_of(), 16);
        assert_eq!(Type::Str.align_of(), 8);
        assert!(Type::Str.is_copy());
        assert!(!Type::Str.needs_drop());
        assert!(!Type::Str.contains_reference());
    }

    #[test]
    fn type_param_display() {
        assert_eq!(Type::TypeParam("T".to_string()).to_string(), "$T");
        assert_eq!(Type::TypeParam("U".to_string()).to_string(), "$U");
    }
}
