use std::fmt::Display;

#[derive(Debug, Clone, PartialEq)]
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
    Slice(Box<Type>),
    Proc(Vec<Type>, Box<Type>),
    Struct(String),
    Enum(String),
    Distinct(Box<Type>),
    Arena,
    Context,
    Handle(Box<Type>),
    Optional(Box<Type>),
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
            Type::Slice(_) => 16,
            Type::Proc(_, _) => 8,
            Type::Struct(_) => 0,
            Type::Enum(_) => 4,
            Type::Distinct(inner) => inner.size_of(),
            Type::Arena => 24,
            Type::Context => 24,
            Type::Handle(_) => 8,
            Type::Optional(inner) => 1 + inner.size_of(),
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
            Type::Proc(_, _) => 8,
            Type::Struct(_) => 8,
            Type::Enum(_) => 4,
            Type::Distinct(inner) => inner.align_of(),
            Type::Arena => 8,
            Type::Context => 8,
            Type::Handle(_) => 4,
            Type::Optional(inner) => inner.align_of(),
            Type::TypeParam(_) => 1,
            Type::Unknown => 1,
        }
    }

    pub fn is_copy(&self) -> bool {
        match self {
            Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::Isize => true,
            Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::Usize => true,
            Type::F32 | Type::F64 | Type::Bool => true,
            Type::Ref(_) | Type::RefMut(_) | Type::Ptr(_) => true,
            Type::Proc(_, _) | Type::Void => true,
            Type::Slice(_) | Type::Array(_, _) => true,
            Type::Str | Type::Struct(_) | Type::Enum(_) => false,
            Type::Distinct(inner) => inner.is_copy(),
            Type::Arena => false,
            Type::Context => false,
            Type::Handle(_) => true,
            Type::Optional(inner) => inner.is_copy(),
            Type::TypeParam(_) => false,
            Type::Unknown => false,
        }
    }

    pub fn needs_drop(&self) -> bool {
        match self {
            Type::Str | Type::Slice(_) | Type::Struct(_) | Type::Enum(_) => true,
            Type::Array(inner, _) => inner.needs_drop(),
            Type::Distinct(inner) => inner.needs_drop(),
            Type::Arena => true,
            Type::Context => false,
            Type::Optional(inner) => inner.needs_drop(),
            _ => false,
        }
    }

    pub fn is_reference(&self) -> bool {
        matches!(self, Type::Ref(_) | Type::RefMut(_))
    }

    pub fn contains_reference(&self) -> bool {
        match self {
            Type::Ref(_) | Type::RefMut(_) => true,
            Type::Array(inner, _) => inner.contains_reference(),
            Type::Slice(inner) => inner.contains_reference(),
            Type::Ptr(inner) => inner.contains_reference(),
            Type::Distinct(inner) => inner.contains_reference(),
            Type::Handle(inner) => inner.contains_reference(),
            Type::Optional(inner) => inner.contains_reference(),
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
            Type::Slice(inner) => write!(f, "[]{}", inner),
            Type::Proc(params, ret) => {
                let param_strs: Vec<String> =
                    params.iter().map(|p| p.to_string()).collect();
                write!(f, "proc({}) -> {}", param_strs.join(", "), ret)
            }
            Type::Struct(name) => write!(f, "{}", name),
            Type::Enum(name) => write!(f, "{}", name),
            Type::Distinct(inner) => write!(f, "distinct {}", inner),
            Type::Arena => write!(f, "Arena"),
            Type::Context => write!(f, "Context"),
            Type::Handle(inner) => write!(f, "Handle<{}>", inner),
            Type::Optional(inner) => write!(f, "?{}", inner),
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

    #[test]
    fn distinct_type_display() {
        let distinct = Type::Distinct(Box::new(Type::I64));
        assert_eq!(distinct.to_string(), "distinct i64");
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
        assert!(Type::Array(Box::new(Type::Ref(Box::new(Type::I64))), 10).contains_reference());
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
    fn type_param_display() {
        assert_eq!(Type::TypeParam("T".to_string()).to_string(), "$T");
        assert_eq!(Type::TypeParam("U".to_string()).to_string(), "$U");
    }
}
