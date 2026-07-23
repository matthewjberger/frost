use std::fmt::{self, Display, Formatter};

use crate::lexer::Position;
use crate::types::Type;

pub type LocalId = usize;
pub type BlockId = usize;

#[derive(Debug, Clone)]
pub struct IrModule {
    pub functions: Vec<IrFunction>,
    pub externs: Vec<IrExtern>,
    // Functions this part calls but does not contain, because another module
    // does. They are declared so a call can be emitted and left for the linker
    // to resolve, and they are carried as whole `IrFunction`s rather than as
    // externs so the signature is built by exactly the same code that built the
    // definition. An extern's ABI is not a Frost function's ABI: aggregate
    // returns go through a hidden out-pointer, and describing one as an extern
    // would silently disagree with the object that defines it.
    pub imported: Vec<IrFunction>,
}

impl IrModule {
    // One `IrModule` per source module, each holding that module's own
    // functions and the specializations it instantiated. Every extern goes in
    // every part, since an extern is a declaration and an object that does not
    // reference one simply does not emit it.
    //
    // This is what makes each module a compilation unit. A specialization two
    // modules both instantiate appears in both parts, which is the duplication
    // the design accepts in exchange for a module's work depending only on the
    // module; see docs/separate-compilation.md. `FROST_MODULE_REPORT` measures
    // how much of it there is.
    pub fn split_by_module(self) -> Vec<IrModule> {
        let mut order: Vec<u32> = Vec::new();
        for function in &self.functions {
            if !order.contains(&function.module) {
                order.push(function.module);
            }
        }
        let externs = self.externs;
        // Functions another object already defines, declared for every part
        // because any of them may call one. These come from a module the build
        // cache answered for, whose object is linked rather than rebuilt.
        let declared = self.imported;
        // Every part may call any module's exported function, so each declares
        // the ones it does not define. A local function belongs to exactly one
        // part and is never reachable from another.
        let shared: Vec<IrFunction> = self
            .functions
            .iter()
            .filter(|function| !function.local)
            .cloned()
            .collect();
        let mut functions = self.functions;
        let mut parts = Vec::with_capacity(order.len());
        for module in order {
            let (mine, rest): (Vec<_>, Vec<_>) = functions
                .into_iter()
                .partition(|function| function.module == module);
            functions = rest;
            let mut imported: Vec<IrFunction> = shared
                .iter()
                .filter(|function| function.module != module)
                .cloned()
                .collect();
            imported.extend(declared.iter().cloned());
            parts.push(IrModule {
                functions: mine,
                externs: externs.clone(),
                imported,
            });
        }
        parts
    }
}

#[derive(Debug, Clone)]
pub struct StructLayout {
    pub size: usize,
    pub align: usize,
    pub fields: Vec<FieldLayout>,
}

impl StructLayout {
    pub fn field(&self, name: &str) -> Option<&FieldLayout> {
        self.fields.iter().find(|field| field.name == name)
    }
}

#[derive(Debug, Clone)]
pub struct FieldLayout {
    pub name: String,
    pub ty: Type,
    pub offset: usize,
}

#[derive(Debug, Clone)]
pub struct EnumLayout {
    pub size: usize,
    pub align: usize,
    pub variants: Vec<EnumVariantLayout>,
}

impl EnumLayout {
    pub fn variant(&self, name: &str) -> Option<&EnumVariantLayout> {
        self.variants.iter().find(|variant| variant.name == name)
    }
}

#[derive(Debug, Clone)]
pub struct EnumVariantLayout {
    pub name: String,
    pub tag: u32,
    pub fields: Vec<FieldLayout>,
}

#[derive(Debug, Clone)]
pub struct IrExtern {
    pub name: String,
    pub params: Vec<Type>,
    pub return_type: Type,
    // Present when the return type is an aggregate, which C returns by a rule
    // of its own that the backends have to follow rather than reuse Frost's.
    // See src/c_abi.rs and item 4 of docs/roadmap.md.
    pub return_layout: Option<crate::c_abi::CLayout>,
}

#[derive(Debug, Clone)]
pub struct IrFunction {
    pub name: String,
    pub param_count: usize,
    pub return_type: Type,
    pub locals: Vec<IrLocal>,
    pub blocks: Vec<IrBlock>,
    pub entry: BlockId,
    // The module this function belongs to, as a source map file id. For a
    // specialization it is the module that instantiated it, which is the module
    // that would emit it once modules are compilation units. See
    // docs/separate-compilation.md.
    pub module: u32,
    // Whether this symbol is private to its object file. True for
    // specializations and anonymous function literals, which are only ever
    // called from the module that produced them, and which therefore must not
    // be exported: two modules that instantiate the same generic each emit
    // their own copy, and two objects exporting one name is a duplicate symbol.
    pub local: bool,
}

impl IrFunction {
    pub fn local_type(&self, local: LocalId) -> &Type {
        &self.locals[local].ty
    }
}

#[derive(Debug, Clone)]
pub struct IrLocal {
    pub ty: Type,
    pub name: Option<String>,
    pub in_memory: bool,
    pub size: usize,
    pub linear: bool,
    pub position: Position,
}

#[derive(Debug, Clone)]
pub struct IrBlock {
    pub statements: Vec<IrStatement>,
    pub terminator: IrTerminator,
}

#[derive(Debug, Clone)]
pub enum IrStatement {
    Assign(LocalId, IrRvalue),
    Store {
        address: IrOperand,
        value: IrOperand,
    },
    Copy {
        destination: IrOperand,
        source: IrOperand,
        size: usize,
    },
    Own(LocalId),
    Consume(LocalId),
}

#[derive(Debug, Clone)]
pub enum IrRvalue {
    Use(IrOperand),
    Binary(IrBinOp, IrOperand, IrOperand),
    Unary(IrUnOp, IrOperand),
    Cast(IrOperand, Type),
    AddressOf {
        local: LocalId,
        offset: usize,
    },
    FieldAddress {
        base: IrOperand,
        offset: usize,
    },
    ElementAddress {
        base: IrOperand,
        index: IrOperand,
        element_size: usize,
    },
    Load {
        address: IrOperand,
        ty: Type,
    },
    Call {
        function: String,
        arguments: Vec<IrOperand>,
    },
    FunctionAddress(String),
    CallIndirect {
        callee: IrOperand,
        arguments: Vec<IrOperand>,
        parameter_types: Vec<Type>,
        return_type: Type,
    },
}

#[derive(Debug, Clone)]
pub enum IrOperand {
    Constant(IrConstant),
    Local(LocalId),
}

#[derive(Debug, Clone)]
pub enum IrConstant {
    Integer(i64, Type),
    Float(f64, Type),
    Bool(bool),
    CString(String),
    Unit,
}

impl IrConstant {
    pub fn constant_type(&self) -> Type {
        match self {
            IrConstant::Integer(_, ty) => ty.clone(),
            IrConstant::Float(_, ty) => ty.clone(),
            IrConstant::Bool(_) => Type::Bool,
            IrConstant::CString(_) => Type::Ptr(Box::new(Type::I8)),
            IrConstant::Unit => Type::Void,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrBinOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    BitwiseAnd,
    BitwiseOr,
    ShiftLeft,
    ShiftRight,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

impl IrBinOp {
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            IrBinOp::Equal
                | IrBinOp::NotEqual
                | IrBinOp::LessThan
                | IrBinOp::LessThanOrEqual
                | IrBinOp::GreaterThan
                | IrBinOp::GreaterThanOrEqual
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrUnOp {
    Negate,
    Not,
}

#[derive(Debug, Clone)]
pub enum IrTerminator {
    Return(Option<IrOperand>),
    Jump(BlockId),
    Branch {
        condition: IrOperand,
        then_block: BlockId,
        else_block: BlockId,
    },
    Unreachable,
}

impl Display for IrModule {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        for external in &self.externs {
            let params: Vec<String> =
                external.params.iter().map(|ty| ty.to_string()).collect();
            writeln!(
                f,
                "extern fn {}({}) -> {}",
                external.name,
                params.join(", "),
                external.return_type
            )?;
        }
        for function in &self.functions {
            write!(f, "{function}")?;
        }
        Ok(())
    }
}

impl Display for IrFunction {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(
            f,
            "fn {}(params: {}) -> {} {{",
            self.name, self.param_count, self.return_type
        )?;
        for (index, local) in self.locals.iter().enumerate() {
            let name = local.name.as_deref().unwrap_or("");
            writeln!(f, "  local _{index}: {} {}", local.ty, name)?;
        }
        for (index, block) in self.blocks.iter().enumerate() {
            writeln!(f, " block{index}:")?;
            for statement in &block.statements {
                writeln!(f, "    {statement}")?;
            }
            writeln!(f, "    {}", block.terminator)?;
        }
        writeln!(f, "}}")
    }
}

impl Display for IrStatement {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            IrStatement::Assign(local, rvalue) => {
                write!(f, "_{local} = {rvalue}")
            }
            IrStatement::Store { address, value } => {
                write!(f, "store {value} -> [{address}]")
            }
            IrStatement::Copy {
                destination,
                source,
                size,
            } => {
                write!(f, "copy {size} bytes [{source}] -> [{destination}]")
            }
            IrStatement::Own(local) => write!(f, "own _{local}"),
            IrStatement::Consume(local) => write!(f, "consume _{local}"),
        }
    }
}

impl Display for IrRvalue {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            IrRvalue::Use(operand) => write!(f, "{operand}"),
            IrRvalue::Binary(op, left, right) => {
                write!(f, "{op:?}({left}, {right})")
            }
            IrRvalue::Unary(op, operand) => write!(f, "{op:?}({operand})"),
            IrRvalue::Cast(operand, ty) => write!(f, "{operand} as {ty}"),
            IrRvalue::AddressOf { local, offset } => {
                write!(f, "&_{local}+{offset}")
            }
            IrRvalue::FieldAddress { base, offset } => {
                write!(f, "{base}+{offset}")
            }
            IrRvalue::ElementAddress {
                base,
                index,
                element_size,
            } => {
                write!(f, "{base}+{index}*{element_size}")
            }
            IrRvalue::Load { address, ty } => {
                write!(f, "load {ty} [{address}]")
            }
            IrRvalue::Call {
                function,
                arguments,
            } => {
                let args: Vec<String> =
                    arguments.iter().map(|arg| arg.to_string()).collect();
                write!(f, "{function}({})", args.join(", "))
            }
            IrRvalue::FunctionAddress(name) => write!(f, "&fn {name}"),
            IrRvalue::CallIndirect {
                callee, arguments, ..
            } => {
                let args: Vec<String> =
                    arguments.iter().map(|arg| arg.to_string()).collect();
                write!(f, "(*{callee})({})", args.join(", "))
            }
        }
    }
}

impl Display for IrOperand {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            IrOperand::Constant(constant) => write!(f, "{constant}"),
            IrOperand::Local(local) => write!(f, "_{local}"),
        }
    }
}

impl Display for IrConstant {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            IrConstant::Integer(value, ty) => write!(f, "{value}_{ty}"),
            IrConstant::Float(value, ty) => write!(f, "{value}_{ty}"),
            IrConstant::Bool(value) => write!(f, "{value}"),
            IrConstant::CString(value) => write!(f, "{value:?}"),
            IrConstant::Unit => write!(f, "()"),
        }
    }
}

impl Display for IrTerminator {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            IrTerminator::Return(None) => write!(f, "return"),
            IrTerminator::Return(Some(operand)) => {
                write!(f, "return {operand}")
            }
            IrTerminator::Jump(block) => write!(f, "jump block{block}"),
            IrTerminator::Branch {
                condition,
                then_block,
                else_block,
            } => {
                write!(
                    f,
                    "branch {condition} ? block{then_block} : block{else_block}"
                )
            }
            IrTerminator::Unreachable => write!(f, "unreachable"),
        }
    }
}
