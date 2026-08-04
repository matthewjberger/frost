// The AST as flat arenas. A node names its children by index into the arena
// that holds them, a name is an interned symbol, and a node's place in the
// source is a pair of token indices into the position table the lexer built.
// The tree shape lives in the ids, so walking is index arithmetic over
// contiguous arrays rather than pointer chasing, which is what keeps the front
// end on goal 9's curve as programs grow. The self-hosted compiler has always
// had this shape (core.frost's Node arena); this is the bootstrap holding to
// the same design.

use std::collections::HashMap;

use crate::lexer::Position;

/// A float written as its bits. An interface has to mean the same thing
/// written down as it does in memory, and a decimal is only lossless where
/// both ends round correctly. The bits carry no such question.
mod double_bits {
    pub fn serialize<S: serde::Serializer>(
        value: &f64,
        writer: S,
    ) -> Result<S::Ok, S::Error> {
        writer.serialize_u64(value.to_bits())
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        reader: D,
    ) -> Result<f64, D::Error> {
        let bits: u64 = serde::Deserialize::deserialize(reader)?;
        Ok(f64::from_bits(bits))
    }
}

mod single_bits {
    pub fn serialize<S: serde::Serializer>(
        value: &f32,
        writer: S,
    ) -> Result<S::Ok, S::Error> {
        writer.serialize_u32(value.to_bits())
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        reader: D,
    ) -> Result<f32, D::Error> {
        let bits: u32 = serde::Deserialize::deserialize(reader)?;
        Ok(f32::from_bits(bits))
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq, Clone)]
pub enum Literal {
    Integer(i64),
    #[serde(with = "double_bits")]
    Float(f64),
    #[serde(with = "single_bits")]
    Float32(f32),
    Boolean(bool),
    String(String),
    Array(Range32),
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
    PartialOrd,
    Ord,
)]
pub struct Symbol(pub u32);

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
pub struct ExprId(pub u32);

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
pub struct StmtId(pub u32);

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
pub struct PatternId(pub u32);

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
pub struct SignatureId(pub u32);

// A run of consecutive entries in one of the side arrays. Which array is a
// property of the field holding the range, the way an arena id's arena is.
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
pub struct Range32 {
    pub start: u32,
    pub len: u32,
}

impl Range32 {
    pub const EMPTY: Range32 = Range32 { start: 0, len: 0 };

    pub fn indices(self) -> std::ops::Range<usize> {
        self.start as usize..(self.start + self.len) as usize
    }

    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    pub fn len(self) -> usize {
        self.len as usize
    }
}

// The span of a node, as token indices into the owning arena's position
// table. A synthesized node carries the span of what it was made from, so a
// diagnostic about it lands on the line the reader wrote.
#[derive(
    serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq,
)]
pub struct TokenSpan {
    pub first: u32,
    pub last: u32,
}

impl TokenSpan {
    // The span of a node no source wrote. `position_of` answers the default
    // position for it, the way a synthesized node used to carry
    // `Position::default()`.
    pub const NONE: TokenSpan = TokenSpan {
        first: u32::MAX,
        last: u32::MAX,
    };
}

// Interned names. The strings are stored once, in first-use order, so the ids
// an interface arena hands out are deterministic given the order its
// declarations are copied in, which is what lets a fingerprint hash the
// serialized form directly. The lookup map is rebuilt on deserialization
// rather than stored.
#[derive(Debug, Default, Clone)]
pub struct SymbolTable {
    names: Vec<String>,
    lookup: HashMap<String, Symbol>,
}

impl PartialEq for SymbolTable {
    fn eq(&self, other: &Self) -> bool {
        self.names == other.names
    }
}

impl SymbolTable {
    pub fn intern(&mut self, name: &str) -> Symbol {
        if let Some(found) = self.lookup.get(name) {
            return *found;
        }
        let symbol = Symbol(self.names.len() as u32);
        self.names.push(name.to_string());
        self.lookup.insert(name.to_string(), symbol);
        symbol
    }

    pub fn get(&self, name: &str) -> Option<Symbol> {
        self.lookup.get(name).copied()
    }

    pub fn name(&self, symbol: Symbol) -> &str {
        &self.names[symbol.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

impl serde::Serialize for SymbolTable {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        self.names.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for SymbolTable {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        let names = Vec::<String>::deserialize(deserializer)?;
        let lookup = names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), Symbol(index as u32)))
            .collect();
        Ok(SymbolTable { names, lookup })
    }
}

// The arenas and every side array a node's ranges index into. One per module
// out of the parser; import resolution splices modules into one by copying
// nodes across, which is also where private names are rewritten, since a
// symbol translated during the copy is a rename that costs nothing extra.
#[derive(
    serde::Serialize, serde::Deserialize, Debug, Default, Clone, PartialEq,
)]
pub struct Ast {
    pub expressions: Vec<Expression>,
    pub statements: Vec<Statement>,
    pub expr_spans: Vec<TokenSpan>,
    pub stmt_spans: Vec<TokenSpan>,
    pub expr_list: Vec<ExprId>,
    pub stmt_list: Vec<StmtId>,
    pub named_exprs: Vec<NamedExpr>,
    pub parameters: Vec<Parameter>,
    pub bindings: Vec<MultiBinding>,
    pub struct_fields: Vec<StructField>,
    pub enum_variants: Vec<EnumVariant>,
    pub flag_bits: Vec<FlagBit>,
    pub renames: Vec<ImportRename>,
    pub cases: Vec<SwitchCase>,
    pub patterns: Vec<Pattern>,
    pub pattern_list: Vec<PatternId>,
    pub pattern_bindings: Vec<PatternBinding>,
    pub signatures: Vec<ReturnSignature>,
    pub return_values: Vec<ReturnValue>,
    pub symbol_list: Vec<Symbol>,
    pub symbols: SymbolTable,
    // One Position per token, the lexer's table carried through. Splicing
    // concatenates them, and a node's span indexes the table of the Ast that
    // holds it.
    pub token_positions: Vec<Position>,
}

// A field of a struct literal or an enum-variant literal: the name and what
// it is set to.
#[derive(
    serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq,
)]
pub struct NamedExpr {
    pub name: Symbol,
    pub value: ExprId,
}

// One `name` binding a payload field of a matched enum variant: the field,
// and the name the arm reads it by.
#[derive(
    serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq,
)]
pub struct PatternBinding {
    pub field: Symbol,
    pub binding: Symbol,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: Symbol,
    pub type_annotation: Option<crate::types::Type>,
    pub mutable: bool,
    pub mode: crate::parser::ParamMode,
    pub compile_time_signature: Option<crate::types::Type>,
    pub pack: bool,
}

#[derive(
    serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq,
)]
pub struct MultiBinding {
    pub name: Symbol,
    pub mutable: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct StructField {
    pub name: Symbol,
    pub field_type: crate::types::Type,
}

// A variant's fields are a run in `struct_fields`; a unit variant records
// that it has none rather than an empty run, since `A` and `A {}` differ.
#[derive(
    serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq,
)]
pub struct EnumVariant {
    pub name: Symbol,
    pub fields: Option<Range32>,
}

#[derive(
    serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq,
)]
pub struct FlagBit {
    pub name: Symbol,
    pub value: i64,
}

#[derive(
    serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq,
)]
pub struct ImportRename {
    pub exported: Symbol,
    pub local: Symbol,
}

#[derive(
    serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq,
)]
pub struct SwitchCase {
    pub pattern: PatternId,
    pub body: Range32,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub enum Pattern {
    Wildcard,
    Literal(Literal),
    Identifier(Symbol),
    EnumVariant {
        enum_name: Option<Symbol>,
        variant_name: Symbol,
        bindings: Range32,
    },
    Tuple(Range32),
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct ReturnValue {
    pub name: Option<Symbol>,
    pub value_type: crate::types::Type,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub enum ReturnKind {
    None,
    Single(crate::types::Type),
    Multiple(Range32),
    Fallible(crate::types::Type, crate::types::Type),
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct ReturnSignature {
    pub kind: ReturnKind,
    pub uses: Vec<crate::types::Type>,
    pub bound: Option<ExprId>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub enum Expression {
    Identifier(Symbol),
    Literal(Literal),
    Boolean(bool),
    Prefix(crate::parser::Operator, ExprId),
    Infix(ExprId, crate::parser::Operator, ExprId),
    If(ExprId, Range32, Option<Range32>),
    Function(Range32, SignatureId, Range32),
    Proc(Range32, SignatureId, Range32),
    Call(ExprId, Range32),
    Index(ExprId, ExprId),
    FieldAccess(ExprId, Symbol),
    AddressOf(ExprId),
    Borrow(ExprId),
    BorrowMut(ExprId),
    Dereference(ExprId),
    StructInit(Symbol, Range32),
    PackMap(ExprId, Symbol, Symbol),
    Range(ExprId, ExprId, bool),
    Switch(ExprId, Range32),
    Tuple(Range32),
    EnumVariantInit(Symbol, Symbol, Range32),
    TypeValue(crate::types::Type),
    Unsafe(Range32),
    UnsafeFn(ExprId),
    Try(ExprId),
    ArrayRepeat(ExprId, Symbol),
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub enum Statement {
    Let {
        name: Symbol,
        type_annotation: Option<crate::types::Type>,
        value: ExprId,
        mutable: bool,
    },
    LetMultiple(Range32, ExprId),
    Constant(Symbol, ExprId),
    Return(ExprId),
    Expression(ExprId),
    Print(ExprId, Range32),
    Struct(Symbol, Range32, Range32),
    Enum(Symbol, Range32, Range32),
    Flags(Symbol, crate::types::Type, Range32),
    TypeAlias(Symbol, crate::types::Type),
    Defer(StmtId),
    Assignment(ExprId, ExprId),
    For(Symbol, Option<Symbol>, ExprId, Range32),
    While(ExprId, Range32),
    With(Symbol, Range32),
    Break,
    Continue,
    Import(String, Range32),
    Extern {
        name: Symbol,
        params: Range32,
        return_type: Option<crate::types::Type>,
        safe: bool,
    },
    Declared {
        name: Symbol,
        params: Range32,
        return_sig: SignatureId,
    },
}

// A range must be one contiguous run, and children are built while their
// parent is still being read, so a builder collects ids locally and flushes
// them as a run once the parent closes. These append-and-answer methods are
// that flush.
impl Ast {
    pub fn intern(&mut self, name: &str) -> Symbol {
        self.symbols.intern(name)
    }

    pub fn name(&self, symbol: Symbol) -> &str {
        self.symbols.name(symbol)
    }

    pub fn push_expr(
        &mut self,
        expression: Expression,
        span: TokenSpan,
    ) -> ExprId {
        let id = ExprId(self.expressions.len() as u32);
        self.expressions.push(expression);
        self.expr_spans.push(span);
        id
    }

    pub fn push_stmt(
        &mut self,
        statement: Statement,
        span: TokenSpan,
    ) -> StmtId {
        let id = StmtId(self.statements.len() as u32);
        self.statements.push(statement);
        self.stmt_spans.push(span);
        id
    }

    pub fn expr(&self, id: ExprId) -> &Expression {
        &self.expressions[id.0 as usize]
    }

    pub fn stmt(&self, id: StmtId) -> &Statement {
        &self.statements[id.0 as usize]
    }

    pub fn expr_span(&self, id: ExprId) -> TokenSpan {
        self.expr_spans[id.0 as usize]
    }

    pub fn stmt_span(&self, id: StmtId) -> TokenSpan {
        self.stmt_spans[id.0 as usize]
    }

    pub fn expr_position(&self, id: ExprId) -> Position {
        self.position_of(self.expr_span(id))
    }

    pub fn stmt_position(&self, id: StmtId) -> Position {
        self.position_of(self.stmt_span(id))
    }

    pub fn position_of(&self, span: TokenSpan) -> Position {
        self.token_positions
            .get(span.first as usize)
            .copied()
            .unwrap_or_default()
    }

    pub fn end_position_of(&self, span: TokenSpan) -> Position {
        self.token_positions
            .get(span.last as usize)
            .copied()
            .unwrap_or_default()
    }

    pub fn add_expr_list(&mut self, ids: &[ExprId]) -> Range32 {
        let start = self.expr_list.len() as u32;
        self.expr_list.extend_from_slice(ids);
        Range32 {
            start,
            len: ids.len() as u32,
        }
    }

    pub fn add_stmt_list(&mut self, ids: &[StmtId]) -> Range32 {
        let start = self.stmt_list.len() as u32;
        self.stmt_list.extend_from_slice(ids);
        Range32 {
            start,
            len: ids.len() as u32,
        }
    }

    pub fn add_named_exprs(&mut self, entries: &[NamedExpr]) -> Range32 {
        let start = self.named_exprs.len() as u32;
        self.named_exprs.extend_from_slice(entries);
        Range32 {
            start,
            len: entries.len() as u32,
        }
    }

    pub fn add_parameters(&mut self, entries: Vec<Parameter>) -> Range32 {
        let start = self.parameters.len() as u32;
        let len = entries.len() as u32;
        self.parameters.extend(entries);
        Range32 { start, len }
    }

    pub fn add_bindings(&mut self, entries: &[MultiBinding]) -> Range32 {
        let start = self.bindings.len() as u32;
        self.bindings.extend_from_slice(entries);
        Range32 {
            start,
            len: entries.len() as u32,
        }
    }

    pub fn add_struct_fields(&mut self, entries: Vec<StructField>) -> Range32 {
        let start = self.struct_fields.len() as u32;
        let len = entries.len() as u32;
        self.struct_fields.extend(entries);
        Range32 { start, len }
    }

    pub fn add_enum_variants(&mut self, entries: &[EnumVariant]) -> Range32 {
        let start = self.enum_variants.len() as u32;
        self.enum_variants.extend_from_slice(entries);
        Range32 {
            start,
            len: entries.len() as u32,
        }
    }

    pub fn add_flag_bits(&mut self, entries: &[FlagBit]) -> Range32 {
        let start = self.flag_bits.len() as u32;
        self.flag_bits.extend_from_slice(entries);
        Range32 {
            start,
            len: entries.len() as u32,
        }
    }

    pub fn add_renames(&mut self, entries: &[ImportRename]) -> Range32 {
        let start = self.renames.len() as u32;
        self.renames.extend_from_slice(entries);
        Range32 {
            start,
            len: entries.len() as u32,
        }
    }

    pub fn add_cases(&mut self, entries: &[SwitchCase]) -> Range32 {
        let start = self.cases.len() as u32;
        self.cases.extend_from_slice(entries);
        Range32 {
            start,
            len: entries.len() as u32,
        }
    }

    pub fn push_pattern(&mut self, pattern: Pattern) -> PatternId {
        let id = PatternId(self.patterns.len() as u32);
        self.patterns.push(pattern);
        id
    }

    pub fn pattern(&self, id: PatternId) -> &Pattern {
        &self.patterns[id.0 as usize]
    }

    pub fn add_pattern_list(&mut self, ids: &[PatternId]) -> Range32 {
        let start = self.pattern_list.len() as u32;
        self.pattern_list.extend_from_slice(ids);
        Range32 {
            start,
            len: ids.len() as u32,
        }
    }

    pub fn add_pattern_bindings(
        &mut self,
        entries: &[PatternBinding],
    ) -> Range32 {
        let start = self.pattern_bindings.len() as u32;
        self.pattern_bindings.extend_from_slice(entries);
        Range32 {
            start,
            len: entries.len() as u32,
        }
    }

    pub fn push_signature(
        &mut self,
        signature: ReturnSignature,
    ) -> SignatureId {
        let id = SignatureId(self.signatures.len() as u32);
        self.signatures.push(signature);
        id
    }

    pub fn signature(&self, id: SignatureId) -> &ReturnSignature {
        &self.signatures[id.0 as usize]
    }

    pub fn add_return_values(&mut self, entries: Vec<ReturnValue>) -> Range32 {
        let start = self.return_values.len() as u32;
        let len = entries.len() as u32;
        self.return_values.extend(entries);
        Range32 { start, len }
    }

    pub fn add_symbol_list(&mut self, entries: &[Symbol]) -> Range32 {
        let start = self.symbol_list.len() as u32;
        self.symbol_list.extend_from_slice(entries);
        Range32 {
            start,
            len: entries.len() as u32,
        }
    }

    pub fn exprs_in(&self, range: Range32) -> &[ExprId] {
        &self.expr_list[range.indices()]
    }

    pub fn stmts_in(&self, range: Range32) -> &[StmtId] {
        &self.stmt_list[range.indices()]
    }

    pub fn named_in(&self, range: Range32) -> &[NamedExpr] {
        &self.named_exprs[range.indices()]
    }

    pub fn params_in(&self, range: Range32) -> &[Parameter] {
        &self.parameters[range.indices()]
    }

    pub fn bindings_in(&self, range: Range32) -> &[MultiBinding] {
        &self.bindings[range.indices()]
    }

    pub fn fields_in(&self, range: Range32) -> &[StructField] {
        &self.struct_fields[range.indices()]
    }

    pub fn variants_in(&self, range: Range32) -> &[EnumVariant] {
        &self.enum_variants[range.indices()]
    }

    pub fn flag_bits_in(&self, range: Range32) -> &[FlagBit] {
        &self.flag_bits[range.indices()]
    }

    pub fn renames_in(&self, range: Range32) -> &[ImportRename] {
        &self.renames[range.indices()]
    }

    pub fn cases_in(&self, range: Range32) -> &[SwitchCase] {
        &self.cases[range.indices()]
    }

    pub fn patterns_in(&self, range: Range32) -> &[PatternId] {
        &self.pattern_list[range.indices()]
    }

    pub fn pattern_bindings_in(&self, range: Range32) -> &[PatternBinding] {
        &self.pattern_bindings[range.indices()]
    }

    pub fn return_values_in(&self, range: Range32) -> &[ReturnValue] {
        &self.return_values[range.indices()]
    }

    pub fn symbols_in(&self, range: Range32) -> &[Symbol] {
        &self.symbol_list[range.indices()]
    }
}

impl ReturnSignature {
    pub fn plain(kind: ReturnKind) -> Self {
        Self {
            kind,
            uses: Vec::new(),
            bound: None,
        }
    }

    // The error enum of a fallible return, if any.
    pub fn failure_type(&self) -> Option<&crate::types::Type> {
        match &self.kind {
            ReturnKind::Fallible(_, error) => Some(error),
            _ => None,
        }
    }

    pub fn is_multiple(&self) -> bool {
        matches!(self.kind, ReturnKind::Multiple(_))
    }
}

// The signature questions that need the arena, because a return type list's
// values live in a side array the kind only names.
impl Ast {
    pub fn signature_to_type(
        &self,
        signature: &ReturnSignature,
    ) -> Option<crate::types::Type> {
        match &signature.kind {
            ReturnKind::None => None,
            ReturnKind::Single(t) => Some(t.clone()),
            ReturnKind::Multiple(values) => Some(crate::types::Type::Struct(
                self.multi_return_struct_name(*values),
            )),
            ReturnKind::Fallible(value, _) => Some(value.clone()),
        }
    }

    // One rendered value of a return type list, the shape the old node's
    // Display printed, since the struct name mangled from it has to stay the
    // same name across a rebuild.
    fn render_return_value(&self, value: &ReturnValue) -> String {
        match value.name {
            Some(name) => {
                format!("{}: {}", self.name(name), value.value_type)
            }
            None => value.value_type.to_string(),
        }
    }

    // The struct a return type list becomes. One per distinct list, named
    // after what it holds, so two functions returning the same list under the
    // same names share it and two that name their values differently do not.
    pub fn multi_return_struct_name(&self, values: Range32) -> String {
        let mut name = String::from("__multi");
        for held in self.return_values_in(values) {
            name.push('_');
            for character in self.render_return_value(held).chars() {
                if character.is_ascii_alphanumeric() {
                    name.push(character);
                } else {
                    name.push('_');
                }
            }
        }
        name
    }

    // The field a return type list's nth value lives in: the name the
    // signature gave it, or its position when the signature gave none.
    pub fn multi_return_field_name(
        &self,
        values: Range32,
        index: usize,
    ) -> String {
        match self
            .return_values_in(values)
            .get(index)
            .and_then(|held| held.name)
        {
            Some(name) => self.name(name).to_string(),
            None => format!("value{index}"),
        }
    }

    pub fn signature_has_second_class(
        &self,
        signature: &ReturnSignature,
    ) -> Option<crate::types::Type> {
        match &signature.kind {
            ReturnKind::None => None,
            ReturnKind::Single(t) => {
                if t.is_second_class() {
                    Some(t.clone())
                } else {
                    None
                }
            }
            ReturnKind::Multiple(values) => self
                .return_values_in(*values)
                .iter()
                .map(|held| &held.value_type)
                .find(|held| held.is_second_class())
                .cloned(),
            ReturnKind::Fallible(value, _) => {
                if value.is_second_class() {
                    Some(value.clone())
                } else {
                    None
                }
            }
        }
    }

    pub fn signature_contains_reference(
        &self,
        signature: &ReturnSignature,
    ) -> Option<crate::types::Type> {
        match &signature.kind {
            ReturnKind::None => None,
            ReturnKind::Single(t) => {
                if t.contains_reference() {
                    Some(t.clone())
                } else {
                    None
                }
            }
            ReturnKind::Multiple(values) => self
                .return_values_in(*values)
                .iter()
                .map(|held| &held.value_type)
                .find(|held| held.contains_reference())
                .cloned(),
            ReturnKind::Fallible(value, _) => {
                if value.contains_reference() {
                    Some(value.clone())
                } else {
                    None
                }
            }
        }
    }
}

// A module as parsed: its arena and the top-level statements in source order.
#[derive(
    serde::Serialize, serde::Deserialize, Debug, Default, Clone, PartialEq,
)]
pub struct Module {
    pub ast: Ast,
    pub roots: Vec<StmtId>,
}

// Copies nodes from one arena into another. Import resolution splices every
// module into one program with it, renaming a module's private names as the
// symbols cross, and interface construction builds its small deterministic
// arena the same way with the identity rename. Token positions are appended
// once per source and every copied span shifts by that offset, so a spliced
// node still answers the file and line it was written on.
pub fn splice_positions(dest: &mut Ast, source: &Ast) -> u32 {
    let offset = dest.token_positions.len() as u32;
    dest.token_positions
        .extend_from_slice(&source.token_positions);
    offset
}

fn shift(span: TokenSpan, offset: u32) -> TokenSpan {
    if span.first == u32::MAX {
        return span;
    }
    TokenSpan {
        first: span.first + offset,
        last: span.last + offset,
    }
}

pub struct Splicer<'a> {
    pub source: &'a Ast,
    pub offset: u32,
}

impl<'a> Splicer<'a> {
    pub fn new(source: &'a Ast, offset: u32) -> Self {
        Self { source, offset }
    }

    pub fn shifted(&self, span: TokenSpan) -> TokenSpan {
        shift(span, self.offset)
    }

    pub fn copy_parameters(&self, dest: &mut Ast, list: Range32) -> Range32 {
        self.parameters(dest, list, &mut |name| name.to_string())
    }

    pub fn copy_signature(
        &self,
        dest: &mut Ast,
        id: SignatureId,
    ) -> SignatureId {
        self.signature(dest, id, &mut |name| name.to_string())
    }

    fn symbol(
        &self,
        dest: &mut Ast,
        symbol: Symbol,
        rename: &mut impl FnMut(&str) -> String,
    ) -> Symbol {
        let renamed = rename(self.source.name(symbol));
        dest.intern(&renamed)
    }

    pub fn statement(
        &self,
        dest: &mut Ast,
        id: StmtId,
        rename: &mut impl FnMut(&str) -> String,
    ) -> StmtId {
        let span = shift(self.source.stmt_span(id), self.offset);
        let node = match self.source.stmt(id).clone() {
            Statement::Let {
                name,
                type_annotation,
                value,
                mutable,
            } => Statement::Let {
                name: self.symbol(dest, name, rename),
                type_annotation,
                value: self.expression(dest, value, rename),
                mutable,
            },
            Statement::LetMultiple(bindings, value) => {
                let copied: Vec<MultiBinding> = self
                    .source
                    .bindings_in(bindings)
                    .to_vec()
                    .into_iter()
                    .map(|binding| MultiBinding {
                        name: self.symbol(dest, binding.name, rename),
                        mutable: binding.mutable,
                    })
                    .collect();
                let value = self.expression(dest, value, rename);
                Statement::LetMultiple(dest.add_bindings(&copied), value)
            }
            Statement::Constant(name, value) => Statement::Constant(
                self.symbol(dest, name, rename),
                self.expression(dest, value, rename),
            ),
            Statement::Return(value) => {
                Statement::Return(self.expression(dest, value, rename))
            }
            Statement::Expression(value) => {
                Statement::Expression(self.expression(dest, value, rename))
            }
            Statement::Print(value, arguments) => {
                let value = self.expression(dest, value, rename);
                let arguments = self.expressions(dest, arguments, rename);
                Statement::Print(value, arguments)
            }
            Statement::Struct(name, type_params, fields) => Statement::Struct(
                self.symbol(dest, name, rename),
                self.symbols(dest, type_params, rename),
                self.fields(dest, fields, rename),
            ),
            Statement::Enum(name, type_params, variants) => {
                let name = self.symbol(dest, name, rename);
                let type_params = self.symbols(dest, type_params, rename);
                let copied: Vec<EnumVariant> = self
                    .source
                    .variants_in(variants)
                    .to_vec()
                    .into_iter()
                    .map(|variant| EnumVariant {
                        name: self.symbol(dest, variant.name, rename),
                        fields: variant
                            .fields
                            .map(|held| self.fields(dest, held, rename)),
                    })
                    .collect();
                Statement::Enum(
                    name,
                    type_params,
                    dest.add_enum_variants(&copied),
                )
            }
            Statement::Flags(name, repr, bits) => {
                let name = self.symbol(dest, name, rename);
                let copied: Vec<FlagBit> = self
                    .source
                    .flag_bits_in(bits)
                    .to_vec()
                    .into_iter()
                    .map(|bit| FlagBit {
                        name: self.symbol(dest, bit.name, rename),
                        value: bit.value,
                    })
                    .collect();
                Statement::Flags(name, repr, dest.add_flag_bits(&copied))
            }
            Statement::TypeAlias(name, ty) => {
                Statement::TypeAlias(self.symbol(dest, name, rename), ty)
            }
            Statement::Defer(inner) => {
                Statement::Defer(self.statement(dest, inner, rename))
            }
            Statement::Assignment(place, value) => Statement::Assignment(
                self.expression(dest, place, rename),
                self.expression(dest, value, rename),
            ),
            Statement::For(iterator, second, sequence, body) => Statement::For(
                self.symbol(dest, iterator, rename),
                second.map(|held| self.symbol(dest, held, rename)),
                self.expression(dest, sequence, rename),
                self.block(dest, body, rename),
            ),
            Statement::While(condition, body) => Statement::While(
                self.expression(dest, condition, rename),
                self.block(dest, body, rename),
            ),
            Statement::With(capability, body) => Statement::With(
                self.symbol(dest, capability, rename),
                self.block(dest, body, rename),
            ),
            Statement::Break => Statement::Break,
            Statement::Continue => Statement::Continue,
            Statement::Import(path, renames) => {
                let copied: Vec<ImportRename> = self
                    .source
                    .renames_in(renames)
                    .to_vec()
                    .into_iter()
                    .map(|held| ImportRename {
                        exported: self.symbol(dest, held.exported, rename),
                        local: self.symbol(dest, held.local, rename),
                    })
                    .collect();
                Statement::Import(path, dest.add_renames(&copied))
            }
            Statement::Extern {
                name,
                params,
                return_type,
                safe,
            } => Statement::Extern {
                name: self.symbol(dest, name, rename),
                params: self.parameters(dest, params, rename),
                return_type,
                safe,
            },
            Statement::Declared {
                name,
                params,
                return_sig,
            } => Statement::Declared {
                name: self.symbol(dest, name, rename),
                params: self.parameters(dest, params, rename),
                return_sig: self.signature(dest, return_sig, rename),
            },
        };
        dest.push_stmt(node, span)
    }

    fn block(
        &self,
        dest: &mut Ast,
        block: Range32,
        rename: &mut impl FnMut(&str) -> String,
    ) -> Range32 {
        let copied: Vec<StmtId> = self
            .source
            .stmts_in(block)
            .to_vec()
            .into_iter()
            .map(|held| self.statement(dest, held, rename))
            .collect();
        dest.add_stmt_list(&copied)
    }

    fn expressions(
        &self,
        dest: &mut Ast,
        list: Range32,
        rename: &mut impl FnMut(&str) -> String,
    ) -> Range32 {
        let copied: Vec<ExprId> = self
            .source
            .exprs_in(list)
            .to_vec()
            .into_iter()
            .map(|held| self.expression(dest, held, rename))
            .collect();
        dest.add_expr_list(&copied)
    }

    fn named(
        &self,
        dest: &mut Ast,
        list: Range32,
        rename: &mut impl FnMut(&str) -> String,
    ) -> Range32 {
        let copied: Vec<NamedExpr> = self
            .source
            .named_in(list)
            .to_vec()
            .into_iter()
            .map(|held| NamedExpr {
                name: self.symbol(dest, held.name, rename),
                value: self.expression(dest, held.value, rename),
            })
            .collect();
        dest.add_named_exprs(&copied)
    }

    fn symbols(
        &self,
        dest: &mut Ast,
        list: Range32,
        rename: &mut impl FnMut(&str) -> String,
    ) -> Range32 {
        let copied: Vec<Symbol> = self
            .source
            .symbols_in(list)
            .to_vec()
            .into_iter()
            .map(|held| self.symbol(dest, held, rename))
            .collect();
        dest.add_symbol_list(&copied)
    }

    fn fields(
        &self,
        dest: &mut Ast,
        list: Range32,
        rename: &mut impl FnMut(&str) -> String,
    ) -> Range32 {
        let copied: Vec<StructField> = self
            .source
            .fields_in(list)
            .to_vec()
            .into_iter()
            .map(|field| StructField {
                name: self.symbol(dest, field.name, rename),
                field_type: field.field_type,
            })
            .collect();
        dest.add_struct_fields(copied)
    }

    fn parameters(
        &self,
        dest: &mut Ast,
        list: Range32,
        rename: &mut impl FnMut(&str) -> String,
    ) -> Range32 {
        let copied: Vec<Parameter> = self
            .source
            .params_in(list)
            .to_vec()
            .into_iter()
            .map(|parameter| Parameter {
                name: self.symbol(dest, parameter.name, rename),
                ..parameter
            })
            .collect();
        dest.add_parameters(copied)
    }

    fn signature(
        &self,
        dest: &mut Ast,
        id: SignatureId,
        rename: &mut impl FnMut(&str) -> String,
    ) -> SignatureId {
        let held = self.source.signature(id).clone();
        let kind = match held.kind {
            ReturnKind::Multiple(values) => {
                let copied: Vec<ReturnValue> = self
                    .source
                    .return_values_in(values)
                    .to_vec()
                    .into_iter()
                    .map(|value| ReturnValue {
                        name: value
                            .name
                            .map(|held| self.symbol(dest, held, rename)),
                        value_type: value.value_type,
                    })
                    .collect();
                ReturnKind::Multiple(dest.add_return_values(copied))
            }
            other => other,
        };
        let bound = held.bound.map(|held| self.expression(dest, held, rename));
        dest.push_signature(ReturnSignature {
            kind,
            uses: held.uses,
            bound,
        })
    }

    fn pattern(
        &self,
        dest: &mut Ast,
        id: PatternId,
        rename: &mut impl FnMut(&str) -> String,
    ) -> PatternId {
        let node = match self.source.pattern(id).clone() {
            Pattern::Wildcard => Pattern::Wildcard,
            Pattern::Literal(literal) => {
                Pattern::Literal(self.literal(dest, literal, rename))
            }
            Pattern::Identifier(name) => {
                Pattern::Identifier(self.symbol(dest, name, rename))
            }
            Pattern::EnumVariant {
                enum_name,
                variant_name,
                bindings,
            } => {
                let copied: Vec<PatternBinding> = self
                    .source
                    .pattern_bindings_in(bindings)
                    .to_vec()
                    .into_iter()
                    .map(|held| PatternBinding {
                        field: self.symbol(dest, held.field, rename),
                        binding: self.symbol(dest, held.binding, rename),
                    })
                    .collect();
                Pattern::EnumVariant {
                    enum_name: enum_name
                        .map(|held| self.symbol(dest, held, rename)),
                    variant_name: self.symbol(dest, variant_name, rename),
                    bindings: dest.add_pattern_bindings(&copied),
                }
            }
            Pattern::Tuple(patterns) => {
                let copied: Vec<PatternId> = self
                    .source
                    .patterns_in(patterns)
                    .to_vec()
                    .into_iter()
                    .map(|held| self.pattern(dest, held, rename))
                    .collect();
                Pattern::Tuple(dest.add_pattern_list(&copied))
            }
        };
        dest.push_pattern(node)
    }

    fn literal(
        &self,
        dest: &mut Ast,
        literal: Literal,
        rename: &mut impl FnMut(&str) -> String,
    ) -> Literal {
        match literal {
            Literal::Array(elements) => {
                Literal::Array(self.expressions(dest, elements, rename))
            }
            other => other,
        }
    }

    pub fn expression(
        &self,
        dest: &mut Ast,
        id: ExprId,
        rename: &mut impl FnMut(&str) -> String,
    ) -> ExprId {
        let span = shift(self.source.expr_span(id), self.offset);
        let node = match self.source.expr(id).clone() {
            Expression::Identifier(name) => {
                Expression::Identifier(self.symbol(dest, name, rename))
            }
            Expression::Literal(literal) => {
                Expression::Literal(self.literal(dest, literal, rename))
            }
            Expression::Boolean(value) => Expression::Boolean(value),
            Expression::Prefix(operator, inner) => Expression::Prefix(
                operator,
                self.expression(dest, inner, rename),
            ),
            Expression::Infix(left, operator, right) => Expression::Infix(
                self.expression(dest, left, rename),
                operator,
                self.expression(dest, right, rename),
            ),
            Expression::If(condition, consequence, alternative) => {
                Expression::If(
                    self.expression(dest, condition, rename),
                    self.block(dest, consequence, rename),
                    alternative.map(|held| self.block(dest, held, rename)),
                )
            }
            Expression::Function(params, signature, body) => {
                Expression::Function(
                    self.parameters(dest, params, rename),
                    self.signature(dest, signature, rename),
                    self.block(dest, body, rename),
                )
            }
            Expression::Proc(params, signature, body) => Expression::Proc(
                self.parameters(dest, params, rename),
                self.signature(dest, signature, rename),
                self.block(dest, body, rename),
            ),
            Expression::Call(callee, arguments) => Expression::Call(
                self.expression(dest, callee, rename),
                self.expressions(dest, arguments, rename),
            ),
            Expression::Index(base, index) => Expression::Index(
                self.expression(dest, base, rename),
                self.expression(dest, index, rename),
            ),
            Expression::FieldAccess(base, field) => Expression::FieldAccess(
                self.expression(dest, base, rename),
                self.symbol(dest, field, rename),
            ),
            Expression::AddressOf(inner) => {
                Expression::AddressOf(self.expression(dest, inner, rename))
            }
            Expression::Borrow(inner) => {
                Expression::Borrow(self.expression(dest, inner, rename))
            }
            Expression::BorrowMut(inner) => {
                Expression::BorrowMut(self.expression(dest, inner, rename))
            }
            Expression::Dereference(inner) => {
                Expression::Dereference(self.expression(dest, inner, rename))
            }
            Expression::StructInit(name, fields) => Expression::StructInit(
                self.symbol(dest, name, rename),
                self.named(dest, fields, rename),
            ),
            Expression::PackMap(inner, variable, list) => Expression::PackMap(
                self.expression(dest, inner, rename),
                self.symbol(dest, variable, rename),
                self.symbol(dest, list, rename),
            ),
            Expression::Range(start, end, inclusive) => Expression::Range(
                self.expression(dest, start, rename),
                self.expression(dest, end, rename),
                inclusive,
            ),
            Expression::Switch(scrutinee, cases) => {
                let scrutinee = self.expression(dest, scrutinee, rename);
                let copied: Vec<SwitchCase> = self
                    .source
                    .cases_in(cases)
                    .to_vec()
                    .into_iter()
                    .map(|case| SwitchCase {
                        pattern: self.pattern(dest, case.pattern, rename),
                        body: self.block(dest, case.body, rename),
                    })
                    .collect();
                Expression::Switch(scrutinee, dest.add_cases(&copied))
            }
            Expression::Tuple(elements) => {
                Expression::Tuple(self.expressions(dest, elements, rename))
            }
            Expression::EnumVariantInit(enum_name, variant, fields) => {
                Expression::EnumVariantInit(
                    self.symbol(dest, enum_name, rename),
                    self.symbol(dest, variant, rename),
                    self.named(dest, fields, rename),
                )
            }
            Expression::TypeValue(ty) => Expression::TypeValue(ty),
            Expression::Unsafe(body) => {
                Expression::Unsafe(self.block(dest, body, rename))
            }
            Expression::UnsafeFn(inner) => {
                Expression::UnsafeFn(self.expression(dest, inner, rename))
            }
            Expression::Try(inner) => {
                Expression::Try(self.expression(dest, inner, rename))
            }
            Expression::ArrayRepeat(value, count) => Expression::ArrayRepeat(
                self.expression(dest, value, rename),
                self.symbol(dest, count, rename),
            ),
        };
        dest.push_expr(node, span)
    }
}
