use crate::{flatten, hash, lexer::Token, types::Type};
use anyhow::{bail, Result};
use std::{
    fmt::{Display, Formatter, Result as FmtResult},
    matches,
    slice::Iter,
};

pub type Identifier = String;

pub type Block = Vec<Statement>;

#[derive(Debug, PartialEq, Clone)]
pub struct Parameter {
    pub name: Identifier,
    pub type_annotation: Option<Type>,
}

impl Display for Parameter {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        match &self.type_annotation {
            Some(typ) => write!(f, "{}: {}", self.name, typ),
            None => write!(f, "{}", self.name),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct StructField {
    pub name: Identifier,
    pub field_type: Type,
}

#[derive(Debug, PartialEq, Clone)]
pub struct EnumVariant {
    pub name: Identifier,
    pub fields: Option<Vec<StructField>>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct SwitchCase {
    pub pattern: Pattern,
    pub body: Block,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Pattern {
    Wildcard,
    Literal(Literal),
    Identifier(Identifier),
    EnumVariant {
        enum_name: Option<Identifier>,
        variant_name: Identifier,
        bindings: Vec<(Identifier, Identifier)>,
    },
    Tuple(Vec<Pattern>),
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Operator {
    Add,
    And,
    Divide,
    Multiply,
    Modulo,
    Not,
    Negate,
    Or,
    Subtract,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
    NotEqual,
}

impl Operator {
    pub fn from_token(token: &Token, is_prefix: bool) -> Result<Self> {
        Ok(match token {
            Token::Plus => Self::Add,
            Token::Slash => Self::Divide,
            Token::Asterisk => Self::Multiply,
            Token::Percent => Self::Modulo,
            Token::Bang => Self::Not,
            Token::Minus if is_prefix => Self::Negate,
            Token::Minus if !is_prefix => Self::Subtract,
            Token::LessThan => Self::LessThan,
            Token::LessThanOrEqual => Self::LessThanOrEqual,
            Token::GreaterThan => Self::GreaterThan,
            Token::GreaterThanOrEqual => Self::GreaterThanOrEqual,
            Token::Equal => Self::Equal,
            Token::NotEqual => Self::NotEqual,
            Token::And => Self::And,
            Token::Or => Self::Or,
            _ => bail!("Token is not an operator: {}", token),
        })
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let statement = match self {
            Self::Add => "+",
            Self::And => "&&",
            Self::Subtract | Self::Negate => "-",
            Self::Divide => "/",
            Self::Multiply => "*",
            Self::Modulo => "%",
            Self::Not => "!",
            Self::Or => "||",
            Self::LessThan => "<",
            Self::LessThanOrEqual => "<=",
            Self::GreaterThan => ">",
            Self::GreaterThanOrEqual => ">=",
            Self::Equal => "==",
            Self::NotEqual => "!=",
        };
        write!(f, "{}", statement)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    Let {
        name: Identifier,
        type_annotation: Option<Type>,
        value: Expression,
        mutable: bool,
    },
    Constant(Identifier, Expression),
    Return(Expression),
    Expression(Expression),
    Struct(Identifier, Vec<StructField>),
    Enum(Identifier, Vec<EnumVariant>),
    TypeAlias(Identifier, Type),
    Defer(Box<Statement>),
    Assignment(Expression, Expression),
    For(Identifier, Expression, Block),
    While(Expression, Block),
    Break,
    Continue,
    Import(String),
}

impl Display for Statement {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let statement = match self {
            Self::Let {
                name,
                type_annotation,
                value,
                mutable,
            } => {
                let mut_str = if *mutable { "mut " } else { "" };
                match type_annotation {
                    Some(typ) => {
                        format!("{}{} : {} = {};", mut_str, name, typ, value)
                    }
                    None => format!("{}{} := {};", mut_str, name, value),
                }
            }
            Self::Constant(identifier, expression) => {
                format!("{} :: {};", identifier, expression)
            }
            Self::Return(expression) => format!("return {};", expression),
            Self::Expression(expression) => expression.to_string(),
            Self::Struct(name, fields) => {
                let field_strs: Vec<String> = fields
                    .iter()
                    .map(|field| {
                        format!("{}: {}", field.name, field.field_type)
                    })
                    .collect();
                format!("{} :: struct {{ {} }}", name, field_strs.join(", "))
            }
            Self::Enum(name, variants) => {
                let variant_strs: Vec<String> = variants
                    .iter()
                    .map(|v| match &v.fields {
                        Some(fields) => {
                            let field_strs: Vec<String> = fields
                                .iter()
                                .map(|f| {
                                    format!("{}: {}", f.name, f.field_type)
                                })
                                .collect();
                            format!(
                                "{} {{ {} }}",
                                v.name,
                                field_strs.join(", ")
                            )
                        }
                        None => v.name.clone(),
                    })
                    .collect();
                format!("{} :: enum {{ {} }}", name, variant_strs.join(", "))
            }
            Self::TypeAlias(name, typ) => {
                format!("{} :: {};", name, typ)
            }
            Self::Defer(statement) => format!("defer {}", statement),
            Self::Assignment(lhs, rhs) => format!("{} = {}", lhs, rhs),
            Self::For(iterator, range, body) => {
                let body_str: Vec<String> =
                    body.iter().map(|s| s.to_string()).collect();
                format!(
                    "for {} in {} {{ {} }}",
                    iterator,
                    range,
                    body_str.join("; ")
                )
            }
            Self::While(condition, body) => {
                let body_str: Vec<String> =
                    body.iter().map(|s| s.to_string()).collect();
                format!("while ({}) {{ {} }}", condition, body_str.join("; "))
            }
            Self::Break => "break".to_string(),
            Self::Continue => "continue".to_string(),
            Self::Import(path) => format!("import \"{}\"", path),
        };
        write!(f, "{}", statement)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    Identifier(Identifier),
    Literal(Literal),
    Boolean(bool),
    Prefix(Operator, Box<Expression>),
    Infix(Box<Expression>, Operator, Box<Expression>),
    If(Box<Expression>, Block, Option<Block>),
    Function(Vec<Parameter>, Option<Type>, Block),
    Proc(Vec<Parameter>, Option<Type>, Block),
    Call(Box<Expression>, Vec<Expression>),
    Index(Box<Expression>, Box<Expression>),
    FieldAccess(Box<Expression>, Identifier),
    AddressOf(Box<Expression>),
    Borrow(Box<Expression>),
    BorrowMut(Box<Expression>),
    Dereference(Box<Expression>),
    StructInit(Identifier, Vec<(Identifier, Expression)>),
    Sizeof(Type),
    Range(Box<Expression>, Box<Expression>),
    Switch(Box<Expression>, Vec<SwitchCase>),
    Tuple(Vec<Expression>),
    EnumVariantInit(Identifier, Identifier, Vec<(Identifier, Expression)>),
}

impl Expression {
    pub fn hash(&self) -> u64 {
        hash(&format!("{:?}", self))
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let expression = match self {
            Self::Identifier(identifier) => identifier.to_string(),
            Self::Literal(literal) => literal.to_string(),
            Self::Boolean(boolean) => boolean.to_string(),
            Self::Prefix(operator, expression) => {
                format!("({}{})", operator, expression)
            }
            Self::Infix(left_expression, operator, right_expression) => {
                format!(
                    "({} {} {})",
                    left_expression, operator, right_expression
                )
            }
            Self::If(condition, consequence, alternative) => {
                let statement = format!(
                    "if ({}) {{ {} }}",
                    condition,
                    flatten(consequence, "\n"),
                );

                let mut result = String::new();
                result.push_str(statement.as_str());

                if let Some(alternative) = alternative {
                    let else_statement =
                        format!("else {{ {} }}", flatten(alternative, "\n"));
                    result.push_str(&else_statement);
                }

                result
            }
            Self::Function(parameters, return_type, body) => {
                let params_str = parameters
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                match return_type {
                    Some(typ) => format!(
                        "fn({}) -> {} {{ {} }}",
                        params_str,
                        typ,
                        flatten(body, "\n")
                    ),
                    None => format!(
                        "fn({}) {{ {} }}",
                        params_str,
                        flatten(body, "\n")
                    ),
                }
            }
            Self::Proc(parameters, return_type, body) => {
                let params_str = parameters
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                match return_type {
                    Some(typ) => format!(
                        "proc({}) -> {} {{ {} }}",
                        params_str,
                        typ,
                        flatten(body, "\n")
                    ),
                    None => format!(
                        "proc({}) {{ {} }}",
                        params_str,
                        flatten(body, "\n")
                    ),
                }
            }
            Self::Call(expression, arguments) => {
                format!("{}({})", expression, flatten(arguments, ", "),)
            }
            Self::Index(left_expression, index_expression) => {
                format!("({}[{}])", left_expression, index_expression,)
            }
            Self::FieldAccess(expression, field) => {
                format!("{}.{}", expression, field)
            }
            Self::AddressOf(expression) => {
                format!("(&{})", expression)
            }
            Self::Borrow(expression) => {
                format!("(&{})", expression)
            }
            Self::BorrowMut(expression) => {
                format!("(&mut {})", expression)
            }
            Self::Dereference(expression) => {
                format!("({}^)", expression)
            }
            Self::StructInit(name, fields) => {
                let field_strs: Vec<String> = fields
                    .iter()
                    .map(|(n, v)| format!("{} = {}", n, v))
                    .collect();
                format!("{} {{ {} }}", name, field_strs.join(", "))
            }
            Self::Sizeof(typ) => {
                format!("sizeof({})", typ)
            }
            Self::Range(start, end) => {
                format!("{}..{}", start, end)
            }
            Self::Switch(scrutinee, cases) => {
                let case_strs: Vec<String> = cases
                    .iter()
                    .map(|c| {
                        let pattern_str = match &c.pattern {
                            Pattern::Wildcard => "_".to_string(),
                            Pattern::Literal(lit) => lit.to_string(),
                            Pattern::Identifier(id) => id.clone(),
                            Pattern::EnumVariant {
                                enum_name,
                                variant_name,
                                bindings,
                            } => {
                                let prefix = match enum_name {
                                    Some(e) => format!("{}::", e),
                                    None => ".".to_string(),
                                };
                                if bindings.is_empty() {
                                    format!("{}{}", prefix, variant_name)
                                } else {
                                    let binding_strs: Vec<String> = bindings
                                        .iter()
                                        .map(|(f, b)| {
                                            if f == b {
                                                f.clone()
                                            } else {
                                                format!("{} = {}", f, b)
                                            }
                                        })
                                        .collect();
                                    format!(
                                        "{}{} {{ {} }}",
                                        prefix,
                                        variant_name,
                                        binding_strs.join(", ")
                                    )
                                }
                            }
                            Pattern::Tuple(patterns) => {
                                let pat_strs: Vec<String> = patterns
                                    .iter()
                                    .map(|p| match p {
                                        Pattern::Wildcard => "_".to_string(),
                                        Pattern::Literal(lit) => {
                                            lit.to_string()
                                        }
                                        Pattern::Identifier(id) => id.clone(),
                                        _ => format!("{:?}", p),
                                    })
                                    .collect();
                                format!("({})", pat_strs.join(", "))
                            }
                        };
                        let body_str: Vec<String> =
                            c.body.iter().map(|s| s.to_string()).collect();
                        format!(
                            "case {}: {{ {} }}",
                            pattern_str,
                            body_str.join("; ")
                        )
                    })
                    .collect();
                format!("switch {} {{ {} }}", scrutinee, case_strs.join(" "))
            }
            Self::Tuple(elements) => {
                let elem_strs: Vec<String> =
                    elements.iter().map(|e| e.to_string()).collect();
                format!("({})", elem_strs.join(", "))
            }
            Self::EnumVariantInit(enum_name, variant_name, fields) => {
                let field_strs: Vec<String> = fields
                    .iter()
                    .map(|(n, v)| format!("{} = {}", n, v))
                    .collect();
                format!(
                    "{}::{} {{ {} }}",
                    enum_name,
                    variant_name,
                    field_strs.join(", ")
                )
            }
        };
        write!(f, "{}", expression)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    Float32(f32),
    String(String),
    Array(Vec<Expression>),
    HashMap(Vec<(Expression, Expression)>),
}

impl Display for Literal {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let literal = match self {
            Self::Integer(x) => x.to_string(),
            Self::Float(x) => x.to_string(),
            Self::Float32(x) => format!("{}f32", x),
            Self::String(x) => x.to_string(),
            Self::Array(array) => {
                let expressions =
                    array.iter().map(|e| e.to_string()).collect::<Vec<_>>();
                format!("[{}]", expressions.join(", "))
            }
            Self::HashMap(key_value_pairs) => {
                let pairs: Vec<String> = key_value_pairs
                    .iter()
                    .map(|(key, value)| format!("{}: {}", key, value))
                    .collect();
                format!("{{ {} }}", pairs.join(", "))
            }
        };
        write!(f, "{}", literal)
    }
}

#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
pub enum Precedence {
    Lowest,
    Range,
    LogicalOr,
    LogicalAnd,
    Equals,
    LessThanGreaterThan,
    Sum,
    Product,
    Prefix,
    Call,
    Index,
    FieldAccess,
}

impl From<&Token> for Precedence {
    fn from(token: &Token) -> Self {
        match token {
            Token::Or => Self::LogicalOr,
            Token::And => Self::LogicalAnd,
            Token::DotDot => Self::Range,
            Token::Equal => Self::Equals,
            Token::NotEqual => Self::Equals,
            Token::LessThan => Self::LessThanGreaterThan,
            Token::LessThanOrEqual => Self::LessThanGreaterThan,
            Token::GreaterThan => Self::LessThanGreaterThan,
            Token::GreaterThanOrEqual => Self::LessThanGreaterThan,
            Token::Plus => Self::Sum,
            Token::Minus => Self::Sum,
            Token::Slash => Self::Product,
            Token::Asterisk => Self::Product,
            Token::Percent => Self::Product,
            Token::LeftParentheses => Self::Call,
            Token::LeftBracket => Self::Index,
            Token::Dot => Self::FieldAccess,
            Token::Caret => Self::FieldAccess,
            Token::LeftBrace => Self::Range,
            Token::DoubleColon => Self::FieldAccess,
            _ => Self::Lowest,
        }
    }
}

pub type Program = Vec<Statement>;

pub struct Parser<'a> {
    pub tokens: Iter<'a, Token>,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token]) -> Self {
        Self {
            tokens: tokens.iter(),
        }
    }

    pub fn parse(&mut self) -> Result<Program> {
        let mut program = Program::new();
        while let Some(statement) = self.parse_statement()? {
            program.push(statement);
        }
        Ok(program)
    }

    pub fn parse_statement(&mut self) -> Result<Option<Statement>> {
        Ok(match self.peek_nth(0) {
            Token::EndOfFile => None,
            Token::Return => Some(self.parse_return_statement()?),
            Token::Defer => Some(self.parse_defer_statement()?),
            Token::For => Some(self.parse_for_statement()?),
            Token::While => Some(self.parse_while_statement()?),
            Token::Break => {
                self.read_token();
                if matches!(self.peek_nth(0), Token::Semicolon) {
                    self.read_token();
                }
                Some(Statement::Break)
            }
            Token::Continue => {
                self.read_token();
                if matches!(self.peek_nth(0), Token::Semicolon) {
                    self.read_token();
                }
                Some(Statement::Continue)
            }
            Token::Import => Some(self.parse_import_statement()?),
            Token::Mut => Some(self.parse_mutable_declaration()?),
            Token::Identifier(_)
                if matches!(self.peek_nth(1), Token::ColonAssign) =>
            {
                Some(self.parse_declaration(false)?)
            }
            Token::Identifier(_)
                if matches!(self.peek_nth(1), Token::Colon)
                    && !matches!(self.peek_nth(2), Token::Colon) =>
            {
                Some(self.parse_typed_declaration(false)?)
            }
            Token::Identifier(_)
                if matches!(self.peek_nth(1), Token::DoubleColon)
                    && matches!(
                        self.peek_nth(2),
                        Token::Struct
                            | Token::Enum
                            | Token::Distinct
                            | Token::Integer(_)
                            | Token::Float(_)
                            | Token::StringLiteral(_)
                            | Token::True
                            | Token::False
                            | Token::Function
                            | Token::Proc
                            | Token::LeftBracket
                            | Token::LeftBrace
                            | Token::Minus
                            | Token::Bang
                            | Token::LeftParentheses
                    ) =>
            {
                Some(self.parse_constant_or_struct_statement()?)
            }
            _ => Some(self.parse_expression_statement()?),
        })
    }

    fn parse_import_statement(&mut self) -> Result<Statement> {
        self.read_token();
        let path = match self.read_token() {
            Token::StringLiteral(path) => path.clone(),
            _ => bail!("Expected string literal after 'import'"),
        };
        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }
        Ok(Statement::Import(path))
    }

    fn parse_defer_statement(&mut self) -> Result<Statement> {
        self.read_token();
        let statement = self
            .parse_statement()?
            .ok_or_else(|| anyhow::anyhow!("Expected statement after defer"))?;
        Ok(Statement::Defer(Box::new(statement)))
    }

    fn parse_for_statement(&mut self) -> Result<Statement> {
        self.read_token();

        let iterator = match self.read_token() {
            Token::Identifier(name) => name.to_string(),
            _ => bail!("Expected identifier after 'for'"),
        };

        if !matches!(self.read_token(), Token::In) {
            bail!("Expected 'in' after for loop iterator");
        }

        let range = self.parse_expression(Precedence::Lowest)?;

        let body = self.parse_block()?;

        Ok(Statement::For(iterator, range, body))
    }

    fn parse_while_statement(&mut self) -> Result<Statement> {
        self.read_token();

        if !matches!(self.peek_nth(0), Token::LeftParentheses) {
            bail!("Expected '(' after 'while'");
        }
        self.read_token();

        let condition = self.parse_expression(Precedence::Lowest)?;

        if !matches!(self.peek_nth(0), Token::RightParentheses) {
            bail!("Expected ')' after while condition");
        }
        self.read_token();

        let body = self.parse_block()?;

        Ok(Statement::While(condition, body))
    }

    fn parse_mutable_declaration(&mut self) -> Result<Statement> {
        self.read_token();

        if matches!(self.peek_nth(1), Token::ColonAssign) {
            self.parse_declaration(true)
        } else if matches!(self.peek_nth(1), Token::Colon) {
            self.parse_typed_declaration(true)
        } else {
            bail!("Expected ':=' or ': type =' after 'mut identifier'")
        }
    }

    fn parse_declaration(&mut self, mutable: bool) -> Result<Statement> {
        let name = match self.read_token() {
            Token::Identifier(name) => name.to_string(),
            _ => bail!("Expected identifier"),
        };

        if !matches!(self.read_token(), Token::ColonAssign) {
            bail!("Expected ':='");
        }

        let value = self.parse_expression(Precedence::Lowest)?;

        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }

        Ok(Statement::Let {
            name,
            type_annotation: None,
            value,
            mutable,
        })
    }

    fn parse_typed_declaration(&mut self, mutable: bool) -> Result<Statement> {
        let name = match self.read_token() {
            Token::Identifier(name) => name.to_string(),
            _ => bail!("Expected identifier"),
        };

        if !matches!(self.read_token(), Token::Colon) {
            bail!("Expected ':'");
        }

        let type_annotation = Some(self.parse_type()?);

        if !matches!(self.read_token(), Token::Assign) {
            bail!("Expected '=' after type annotation");
        }

        let value = self.parse_expression(Precedence::Lowest)?;

        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }

        Ok(Statement::Let {
            name,
            type_annotation,
            value,
            mutable,
        })
    }

    fn parse_constant_or_struct_statement(&mut self) -> Result<Statement> {
        let identifier = match self.read_token() {
            Token::Identifier(name) => name.to_string(),
            _ => bail!("Expected identifier"),
        };

        if !matches!(self.read_token(), Token::DoubleColon) {
            bail!("Expected '::'");
        }

        if matches!(self.peek_nth(0), Token::Struct) {
            self.read_token();
            if !matches!(self.read_token(), Token::LeftBrace) {
                bail!("Expected '{{' after struct");
            }
            let mut fields = Vec::new();
            while self.peek_nth(0) != &Token::RightBrace {
                let field_name = match self.read_token() {
                    Token::Identifier(name) => name.to_string(),
                    _ => bail!("Expected field name"),
                };
                if !matches!(self.read_token(), Token::Colon) {
                    bail!("Expected ':' after field name");
                }
                let field_type = self.parse_type()?;
                fields.push(StructField {
                    name: field_name,
                    field_type,
                });
                if matches!(self.peek_nth(0), Token::Comma) {
                    self.read_token();
                }
            }
            self.read_token();
            if matches!(self.peek_nth(0), Token::Semicolon) {
                self.read_token();
            }
            Ok(Statement::Struct(identifier, fields))
        } else if matches!(self.peek_nth(0), Token::Enum) {
            self.read_token();
            if !matches!(self.read_token(), Token::LeftBrace) {
                bail!("Expected '{{' after enum");
            }
            let mut variants = Vec::new();
            while self.peek_nth(0) != &Token::RightBrace {
                let variant_name = match self.read_token() {
                    Token::Identifier(name) => name.to_string(),
                    _ => bail!("Expected variant name"),
                };
                let fields = if matches!(self.peek_nth(0), Token::LeftBrace) {
                    self.read_token();
                    let mut variant_fields = Vec::new();
                    while self.peek_nth(0) != &Token::RightBrace {
                        let field_name = match self.read_token() {
                            Token::Identifier(name) => name.to_string(),
                            _ => bail!("Expected field name in enum variant"),
                        };
                        if !matches!(self.read_token(), Token::Colon) {
                            bail!(
                                "Expected ':' after field name in enum variant"
                            );
                        }
                        let field_type = self.parse_type()?;
                        variant_fields.push(StructField {
                            name: field_name,
                            field_type,
                        });
                        if matches!(self.peek_nth(0), Token::Comma) {
                            self.read_token();
                        }
                    }
                    self.read_token();
                    Some(variant_fields)
                } else {
                    None
                };
                variants.push(EnumVariant {
                    name: variant_name,
                    fields,
                });
                if matches!(self.peek_nth(0), Token::Comma) {
                    self.read_token();
                }
            }
            self.read_token();
            if matches!(self.peek_nth(0), Token::Semicolon) {
                self.read_token();
            }
            Ok(Statement::Enum(identifier, variants))
        } else if matches!(self.peek_nth(0), Token::Distinct) {
            let typ = self.parse_type()?;
            if matches!(self.peek_nth(0), Token::Semicolon) {
                self.read_token();
            }
            Ok(Statement::TypeAlias(identifier, typ))
        } else {
            let expression = self.parse_expression(Precedence::Lowest)?;
            if matches!(self.peek_nth(0), Token::Semicolon) {
                self.read_token();
            }
            Ok(Statement::Constant(identifier, expression))
        }
    }

    fn parse_return_statement(&mut self) -> Result<Statement> {
        if !matches!(self.read_token(), Token::Return) {
            bail!("Expected 'Return' token!");
        }

        let expression = self.parse_expression(Precedence::Lowest)?;

        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }

        Ok(Statement::Return(expression))
    }

    fn parse_expression_statement(&mut self) -> Result<Statement> {
        let expression = self.parse_expression(Precedence::Lowest)?;
        let statement = if matches!(self.peek_nth(0), Token::Assign) {
            self.read_token();
            let rhs = self.parse_expression(Precedence::Lowest)?;
            Statement::Assignment(expression, rhs)
        } else {
            Statement::Expression(expression)
        };
        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }
        Ok(statement)
    }

    fn parse_expression(
        &mut self,
        precedence: Precedence,
    ) -> Result<Expression> {
        let mut advance = true;
        let mut expression = match self.peek_nth(0) {
            Token::Identifier(identifier) => {
                Expression::Identifier(identifier.to_string())
            }
            Token::StringLiteral(string) => {
                Expression::Literal(Literal::String(string.to_string()))
            }
            Token::Integer(value) => {
                Expression::Literal(Literal::Integer(*value))
            }
            Token::Float(value) => Expression::Literal(Literal::Float(*value)),
            Token::Float32(value) => Expression::Literal(Literal::Float32(*value)),
            Token::Bang | Token::Minus => {
                advance = false;
                self.parse_prefix_expression()?
            }
            Token::Ampersand => {
                advance = false;
                self.parse_address_of()?
            }
            Token::Sizeof => {
                advance = false;
                self.parse_sizeof()?
            }
            Token::True => Expression::Boolean(true),
            Token::False => Expression::Boolean(false),
            Token::LeftBrace => {
                advance = false;
                self.parse_hashmap_literal()?
            }
            Token::LeftBracket => {
                advance = false;
                self.parse_array_literal()?
            }
            Token::LeftParentheses => {
                advance = false;
                self.parse_grouped_expressions()?
            }
            Token::If => {
                advance = false;
                self.parse_if_expression()?
            }
            Token::Function => {
                advance = false;
                self.parse_function_literal()?
            }
            Token::Proc => {
                advance = false;
                self.parse_proc_literal()?
            }
            Token::Switch => {
                advance = false;
                self.parse_switch_expression()?
            }
            Token::EndOfFile => {
                bail!("Unexpected end of file")
            }
            token => bail!("Token not valid for an expression: {:?}", token),
        };

        if advance {
            self.read_token();
        }

        while self.peek_nth(0) != &Token::Semicolon
            && precedence < Precedence::from(self.peek_nth(0))
        {
            match self.peek_nth(0) {
                Token::Plus
                | Token::Minus
                | Token::Slash
                | Token::Asterisk
                | Token::Percent
                | Token::Equal
                | Token::NotEqual
                | Token::LessThan
                | Token::LessThanOrEqual
                | Token::GreaterThan
                | Token::GreaterThanOrEqual
                | Token::And
                | Token::Or => {
                    expression =
                        self.parse_infix_expression(expression.clone())?;
                }
                Token::DotDot => {
                    expression =
                        self.parse_range_expression(expression.clone())?;
                }
                Token::LeftBracket => {
                    expression =
                        self.parse_index_expression(expression.clone())?;
                }
                Token::LeftParentheses => {
                    expression =
                        self.parse_call_expression(expression.clone())?;
                }
                Token::Dot => {
                    expression = self.parse_field_access(expression.clone())?;
                }
                Token::Caret => {
                    expression = self.parse_dereference(expression.clone())?;
                }
                Token::LeftBrace => {
                    if self.peek_nth(1) == &Token::Case {
                        return Ok(expression);
                    }
                    if let Expression::Identifier(name) = &expression {
                        expression = self.parse_struct_init(name.clone())?;
                    } else {
                        return Ok(expression);
                    }
                }
                Token::DoubleColon => {
                    if let Expression::Identifier(enum_name) = &expression {
                        self.read_token();
                        let variant_name = match self.read_token() {
                            Token::Identifier(v) => v.to_string(),
                            _ => bail!("Expected identifier after '::'"),
                        };
                        if matches!(self.peek_nth(0), Token::LeftBrace) {
                            self.read_token();
                            let mut fields = Vec::new();
                            while self.peek_nth(0) != &Token::RightBrace {
                                let field_name = match self.read_token() {
                                    Token::Identifier(name) => name.to_string(),
                                    token => bail!("Expected field name in enum variant init, found {:?}", token),
                                };
                                if !matches!(self.read_token(), Token::Assign) {
                                    bail!("Expected '=' after field name in enum variant init");
                                }
                                let value =
                                    self.parse_expression(Precedence::Lowest)?;
                                fields.push((field_name, value));
                                if matches!(self.peek_nth(0), Token::Comma) {
                                    self.read_token();
                                }
                            }
                            self.read_token();
                            expression = Expression::EnumVariantInit(
                                enum_name.clone(),
                                variant_name,
                                fields,
                            );
                        } else {
                            expression = Expression::Identifier(format!(
                                "{}::{}",
                                enum_name, variant_name
                            ));
                        }
                    } else {
                        return Ok(expression);
                    }
                }
                _ => return Ok(expression),
            };
        }

        Ok(expression)
    }

    fn parse_prefix_expression(&mut self) -> Result<Expression> {
        let operator = Operator::from_token(self.peek_nth(0), true)?;
        self.read_token();
        Ok(Expression::Prefix(
            operator,
            Box::new(self.parse_expression(Precedence::Prefix)?),
        ))
    }

    fn parse_infix_expression(
        &mut self,
        left_expression: Expression,
    ) -> Result<Expression> {
        let operator = Operator::from_token(self.peek_nth(0), false)?;
        let precedence = Precedence::from(self.peek_nth(0));
        self.read_token();
        Ok(Expression::Infix(
            Box::new(left_expression),
            operator,
            Box::new(self.parse_expression(precedence)?),
        ))
    }

    fn parse_range_expression(
        &mut self,
        left_expression: Expression,
    ) -> Result<Expression> {
        self.read_token();
        let right_expression = self.parse_expression(Precedence::Range)?;
        Ok(Expression::Range(
            Box::new(left_expression),
            Box::new(right_expression),
        ))
    }

    fn parse_array_literal(&mut self) -> Result<Expression> {
        let elements = self.parse_expression_list(&Token::RightBracket)?;
        Ok(Expression::Literal(Literal::Array(elements)))
    }

    fn parse_call_expression(
        &mut self,
        expression: Expression,
    ) -> Result<Expression> {
        let elements = self.parse_expression_list(&Token::RightParentheses)?;
        Ok(Expression::Call(Box::new(expression), elements))
    }

    fn parse_index_expression(
        &mut self,
        expression: Expression,
    ) -> Result<Expression> {
        self.read_token();
        let index_expression = self.parse_expression(Precedence::Lowest)?;
        self.read_token();
        Ok(Expression::Index(
            Box::new(expression),
            Box::new(index_expression),
        ))
    }

    fn parse_address_of(&mut self) -> Result<Expression> {
        self.read_token();
        if matches!(self.peek_nth(0), Token::Mut) {
            self.read_token();
            let expression = self.parse_expression(Precedence::Prefix)?;
            Ok(Expression::BorrowMut(Box::new(expression)))
        } else {
            let expression = self.parse_expression(Precedence::Prefix)?;
            Ok(Expression::Borrow(Box::new(expression)))
        }
    }

    fn parse_sizeof(&mut self) -> Result<Expression> {
        self.read_token();
        if !matches!(self.read_token(), Token::LeftParentheses) {
            bail!("Expected '(' after sizeof");
        }
        let typ = self.parse_type()?;
        if !matches!(self.read_token(), Token::RightParentheses) {
            bail!("Expected ')' after type in sizeof");
        }
        Ok(Expression::Sizeof(typ))
    }

    fn parse_field_access(
        &mut self,
        expression: Expression,
    ) -> Result<Expression> {
        self.read_token();
        let field_name = match self.read_token() {
            Token::Identifier(name) => name.to_string(),
            token => bail!("Expected field name after '.', found {:?}", token),
        };
        Ok(Expression::FieldAccess(Box::new(expression), field_name))
    }

    fn parse_dereference(
        &mut self,
        expression: Expression,
    ) -> Result<Expression> {
        self.read_token();
        Ok(Expression::Dereference(Box::new(expression)))
    }

    fn parse_struct_init(&mut self, struct_name: String) -> Result<Expression> {
        self.read_token();
        let mut fields = Vec::new();
        while self.peek_nth(0) != &Token::RightBrace {
            let field_name = match self.read_token() {
                Token::Identifier(name) => name.to_string(),
                token => bail!(
                    "Expected field name in struct init, found {:?}",
                    token
                ),
            };
            if !matches!(self.read_token(), Token::Assign) {
                bail!("Expected '=' after field name in struct init");
            }
            let value = self.parse_expression(Precedence::Lowest)?;
            fields.push((field_name, value));
            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            }
        }
        self.read_token();
        Ok(Expression::StructInit(struct_name, fields))
    }

    fn parse_hashmap_literal(&mut self) -> Result<Expression> {
        let mut pairs = Vec::new();
        self.read_token(); // {
        while self.peek_nth(0) != &Token::RightBrace {
            let key = self.parse_expression(Precedence::Lowest)?;
            self.read_token(); // :
            let value = self.parse_expression(Precedence::Lowest)?;
            if matches!(self.peek_nth(0), &Token::Comma) {
                self.read_token();
            }
            pairs.push((key, value));
        }
        self.read_token(); // }
        Ok(Expression::Literal(Literal::HashMap(pairs)))
    }

    fn parse_expression_list(
        &mut self,
        end_token: &Token,
    ) -> Result<Vec<Expression>> {
        self.read_token();
        let mut elements = Vec::new();
        while self.peek_nth(0) != end_token {
            elements.push(self.parse_expression(Precedence::Lowest)?);

            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            }
        }
        self.read_token();
        Ok(elements)
    }

    fn parse_grouped_expressions(&mut self) -> Result<Expression> {
        self.read_token();
        let first_expression = self.parse_expression(Precedence::Lowest)?;
        if matches!(self.peek_nth(0), Token::Comma) {
            self.read_token();
            let mut elements = vec![first_expression];
            while self.peek_nth(0) != &Token::RightParentheses {
                elements.push(self.parse_expression(Precedence::Lowest)?);
                if matches!(self.peek_nth(0), Token::Comma) {
                    self.read_token();
                }
            }
            self.read_token();
            Ok(Expression::Tuple(elements))
        } else {
            if matches!(self.peek_nth(0), Token::RightParentheses) {
                self.read_token();
            }
            Ok(first_expression)
        }
    }

    fn parse_switch_expression(&mut self) -> Result<Expression> {
        self.read_token();
        let scrutinee = self.parse_expression(Precedence::Lowest)?;
        if !matches!(self.read_token(), Token::LeftBrace) {
            bail!("Expected '{{' after switch expression");
        }
        let mut cases = Vec::new();
        while self.peek_nth(0) != &Token::RightBrace {
            if !matches!(self.read_token(), Token::Case) {
                bail!("Expected 'case' in switch");
            }
            let pattern = self.parse_pattern()?;
            if !matches!(self.read_token(), Token::Colon) {
                bail!("Expected ':' after pattern in switch case");
            }
            let body = if matches!(self.peek_nth(0), Token::LeftBrace) {
                self.parse_block()?
            } else {
                let expr = self.parse_expression(Precedence::Lowest)?;
                vec![Statement::Expression(expr)]
            };
            cases.push(SwitchCase { pattern, body });
        }
        self.read_token();
        Ok(Expression::Switch(Box::new(scrutinee), cases))
    }

    fn parse_pattern(&mut self) -> Result<Pattern> {
        match self.peek_nth(0) {
            Token::Underscore => {
                self.read_token();
                Ok(Pattern::Wildcard)
            }
            Token::Integer(value) => {
                let value = *value;
                self.read_token();
                Ok(Pattern::Literal(Literal::Integer(value)))
            }
            Token::Float(value) => {
                let value = *value;
                self.read_token();
                Ok(Pattern::Literal(Literal::Float(value)))
            }
            Token::Float32(value) => {
                let value = *value;
                self.read_token();
                Ok(Pattern::Literal(Literal::Float32(value)))
            }
            Token::StringLiteral(s) => {
                let s = s.clone();
                self.read_token();
                Ok(Pattern::Literal(Literal::String(s)))
            }
            Token::True => {
                self.read_token();
                Ok(Pattern::Literal(Literal::Integer(1)))
            }
            Token::False => {
                self.read_token();
                Ok(Pattern::Literal(Literal::Integer(0)))
            }
            Token::Dot => {
                self.read_token();
                let variant_name = match self.read_token() {
                    Token::Identifier(name) => name.to_string(),
                    _ => bail!("Expected variant name after '.'"),
                };
                let bindings = if matches!(self.peek_nth(0), Token::LeftBrace) {
                    self.read_token();
                    let mut bindings = Vec::new();
                    while self.peek_nth(0) != &Token::RightBrace {
                        let field_name = match self.read_token() {
                            Token::Identifier(name) => name.to_string(),
                            _ => bail!("Expected binding name in pattern"),
                        };
                        bindings.push((field_name.clone(), field_name));
                        if matches!(self.peek_nth(0), Token::Comma) {
                            self.read_token();
                        }
                    }
                    self.read_token();
                    bindings
                } else {
                    Vec::new()
                };
                Ok(Pattern::EnumVariant {
                    enum_name: None,
                    variant_name,
                    bindings,
                })
            }
            Token::LeftParentheses => {
                self.read_token();
                let mut patterns = Vec::new();
                while self.peek_nth(0) != &Token::RightParentheses {
                    patterns.push(self.parse_pattern()?);
                    if matches!(self.peek_nth(0), Token::Comma) {
                        self.read_token();
                    }
                }
                self.read_token();
                Ok(Pattern::Tuple(patterns))
            }
            Token::Identifier(name) => {
                let name = name.clone();
                self.read_token();
                if matches!(self.peek_nth(0), Token::DoubleColon) {
                    self.read_token();
                    let variant_name = match self.read_token() {
                        Token::Identifier(v) => v.to_string(),
                        _ => bail!("Expected variant name after '::'"),
                    };
                    let bindings =
                        if matches!(self.peek_nth(0), Token::LeftBrace) {
                            self.read_token();
                            let mut bindings = Vec::new();
                            while self.peek_nth(0) != &Token::RightBrace {
                                let field_name = match self.read_token() {
                                    Token::Identifier(n) => n.to_string(),
                                    _ => bail!(
                                        "Expected binding name in pattern"
                                    ),
                                };
                                bindings.push((field_name.clone(), field_name));
                                if matches!(self.peek_nth(0), Token::Comma) {
                                    self.read_token();
                                }
                            }
                            self.read_token();
                            bindings
                        } else {
                            Vec::new()
                        };
                    Ok(Pattern::EnumVariant {
                        enum_name: Some(name),
                        variant_name,
                        bindings,
                    })
                } else {
                    Ok(Pattern::Identifier(name))
                }
            }
            token => bail!("Unexpected token in pattern: {:?}", token),
        }
    }

    fn parse_if_expression(&mut self) -> Result<Expression> {
        self.read_token();

        if !matches!(self.peek_nth(0), Token::LeftParentheses) {
            bail!("Expected a left parentheses in if expression!");
        }
        self.read_token();

        let condition = self.parse_expression(Precedence::Lowest)?;

        if !matches!(self.peek_nth(0), Token::RightParentheses) {
            bail!("Expected a right parentheses in if expression!");
        }
        self.read_token();

        let consequence = self.parse_block()?;

        let mut alternative = None;
        if matches!(self.peek_nth(0), Token::Else) {
            self.read_token();
            if matches!(self.peek_nth(0), Token::If) {
                let else_if = self.parse_if_expression()?;
                alternative = Some(vec![Statement::Expression(else_if)]);
            } else {
                alternative = Some(self.parse_block()?);
            }
        }

        Ok(Expression::If(
            Box::new(condition),
            consequence,
            alternative,
        ))
    }

    fn parse_function_literal(&mut self) -> Result<Expression> {
        self.read_token();
        let parameters = self.parse_function_parameters()?;
        let return_type = if matches!(self.peek_nth(0), Token::Arrow) {
            self.read_token();
            Some(self.parse_type()?)
        } else {
            None
        };
        let block = self.parse_block()?;
        Ok(Expression::Function(parameters, return_type, block))
    }

    fn parse_proc_literal(&mut self) -> Result<Expression> {
        self.read_token();
        let parameters = self.parse_function_parameters()?;
        let return_type = if matches!(self.peek_nth(0), Token::Arrow) {
            self.read_token();
            Some(self.parse_type()?)
        } else {
            None
        };
        let block = self.parse_block()?;
        Ok(Expression::Proc(parameters, return_type, block))
    }

    fn parse_function_parameters(&mut self) -> Result<Vec<Parameter>> {
        if !matches!(self.peek_nth(0), Token::LeftParentheses) {
            bail!("Expected a left parentheses in parameter list!");
        }
        self.read_token();

        let mut parameters = Vec::new();
        while self.peek_nth(0) != &Token::RightParentheses {
            if let Token::Identifier(name) = self.peek_nth(0) {
                let name = name.to_string();
                self.read_token();

                let type_annotation =
                    if matches!(self.peek_nth(0), Token::Colon) {
                        self.read_token();
                        Some(self.parse_type()?)
                    } else {
                        None
                    };

                parameters.push(Parameter {
                    name,
                    type_annotation,
                });
            }

            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            }
        }

        if !matches!(self.peek_nth(0), Token::RightParentheses) {
            bail!("Expected a right parentheses in parameter list!");
        }
        self.read_token();

        Ok(parameters)
    }

    fn parse_type(&mut self) -> Result<Type> {
        let base_type = match self.peek_nth(0) {
            Token::TypeI8 => {
                self.read_token();
                Type::I8
            }
            Token::TypeI16 => {
                self.read_token();
                Type::I16
            }
            Token::TypeI32 => {
                self.read_token();
                Type::I32
            }
            Token::TypeI64 => {
                self.read_token();
                Type::I64
            }
            Token::TypeU8 => {
                self.read_token();
                Type::U8
            }
            Token::TypeU16 => {
                self.read_token();
                Type::U16
            }
            Token::TypeU32 => {
                self.read_token();
                Type::U32
            }
            Token::TypeU64 => {
                self.read_token();
                Type::U64
            }
            Token::TypeF32 => {
                self.read_token();
                Type::F32
            }
            Token::TypeF64 => {
                self.read_token();
                Type::F64
            }
            Token::TypeBool => {
                self.read_token();
                Type::Bool
            }
            Token::TypeStr => {
                self.read_token();
                Type::Str
            }
            Token::TypeVoid => {
                self.read_token();
                Type::Void
            }
            Token::Caret => {
                self.read_token();
                Type::Ptr(Box::new(self.parse_type()?))
            }
            Token::Ampersand => {
                self.read_token();
                if matches!(self.peek_nth(0), Token::Mut) {
                    self.read_token();
                    Type::RefMut(Box::new(self.parse_type()?))
                } else {
                    Type::Ref(Box::new(self.parse_type()?))
                }
            }
            Token::LeftBracket => {
                self.read_token();
                if matches!(self.peek_nth(0), Token::RightBracket) {
                    self.read_token();
                    Type::Slice(Box::new(self.parse_type()?))
                } else if let Token::Integer(size) = self.peek_nth(0) {
                    let size = *size as usize;
                    self.read_token();
                    if !matches!(self.read_token(), Token::RightBracket) {
                        bail!("Expected ']' after array size");
                    }
                    Type::Array(Box::new(self.parse_type()?), size)
                } else {
                    bail!("Expected array size or ']' for slice");
                }
            }
            Token::Proc => {
                self.read_token();
                if !matches!(self.peek_nth(0), Token::LeftParentheses) {
                    bail!("Expected '(' after 'proc' in type");
                }
                self.read_token();
                let mut param_types = Vec::new();
                while self.peek_nth(0) != &Token::RightParentheses {
                    param_types.push(self.parse_type()?);
                    if matches!(self.peek_nth(0), Token::Comma) {
                        self.read_token();
                    }
                }
                self.read_token();
                let return_type = if matches!(self.peek_nth(0), Token::Arrow) {
                    self.read_token();
                    self.parse_type()?
                } else {
                    Type::Void
                };
                Type::Proc(param_types, Box::new(return_type))
            }
            Token::Distinct => {
                self.read_token();
                Type::Distinct(Box::new(self.parse_type()?))
            }
            Token::Identifier(name) => {
                let name = name.to_string();
                self.read_token();
                Type::Struct(name)
            }
            token => bail!("Expected type, found {:?}", token),
        };
        Ok(base_type)
    }

    fn parse_block(&mut self) -> Result<Block> {
        if !matches!(self.peek_nth(0), Token::LeftBrace) {
            bail!("Expected a left brace in block!");
        }
        self.read_token();

        let mut statements = Vec::new();

        while self.peek_nth(0) != &Token::RightBrace
            && self.peek_nth(0) != &Token::EndOfFile
        {
            if let Some(statement) = self.parse_statement()? {
                statements.push(statement);
            }
        }

        if !matches!(self.peek_nth(0), Token::RightBrace) {
            bail!("Expected a right brace in block!");
        }
        self.read_token();

        Ok(statements)
    }

    fn read_token(&mut self) -> &Token {
        self.tokens.next().unwrap_or(&Token::EndOfFile)
    }

    fn peek_nth(&self, n: usize) -> &Token {
        self.tokens.clone().nth(n).unwrap_or(&Token::EndOfFile)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Expression, Literal, Parameter, Parser, Statement, StructField,
    };
    use crate::{lexer::Lexer, types::Type, Operator};
    use anyhow::{bail, Result};

    #[test]
    fn test_let_statements() -> Result<()> {
        let tests = [
            (
                "x := 5;",
                "x".to_string(),
                Expression::Literal(Literal::Integer(5)),
            ),
            (
                "y := 10;",
                "y".to_string(),
                Expression::Literal(Literal::Integer(10)),
            ),
            (
                "foobar := 838383;",
                "foobar".to_string(),
                Expression::Literal(Literal::Integer(838383)),
            ),
            (
                "foobar := y;",
                "foobar".to_string(),
                Expression::Identifier("y".to_string()),
            ),
        ];

        for (input, expected_identifier, expected_expression) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            for statement in program.into_iter() {
                match statement {
                    Statement::Let { name, value, .. } => {
                        assert_eq!(name, *expected_identifier);
                        assert_eq!(value, *expected_expression);
                    }
                    _ => bail!("Expected a let statement!"),
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_return_statements() -> Result<()> {
        let tests = [
            ("return 5;", Expression::Literal(Literal::Integer(5))),
            ("return 10;", Expression::Literal(Literal::Integer(10))),
            (
                "return 993322;",
                Expression::Literal(Literal::Integer(993322)),
            ),
            ("return y;", Expression::Identifier("y".to_string())),
        ];

        for (input, expected_expression) in tests.iter() {
            parse_statement(input, expected_expression)?;
        }

        Ok(())
    }

    #[test]
    fn ast() -> Result<()> {
        let output = "myVar := anotherVar;";
        let ast = Statement::Let {
            name: "myVar".to_string(),
            type_annotation: None,
            value: Expression::Identifier("anotherVar".to_string()),
            mutable: false,
        };
        assert_eq!(ast.to_string(), output.to_string());
        Ok(())
    }

    #[test]
    fn identifier_expressions() -> Result<()> {
        parse_statement(
            "foobar;",
            &Expression::Identifier("foobar".to_string()),
        )
    }

    #[test]
    fn integer_expressions() -> Result<()> {
        parse_statement("5;", &Expression::Literal(Literal::Integer(5)))
    }

    #[test]
    fn boolean_expressions() -> Result<()> {
        let tests = [("true;", true), ("false;", false)];

        for (input, expected_value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            if let Some(Statement::Expression(Expression::Boolean(value))) =
                program.into_iter().next()
            {
                assert_eq!(value, *expected_value)
            } else {
                bail!("Expected a boolean expression statement!");
            }
        }

        Ok(())
    }

    #[test]
    fn prefix_expressions() -> Result<()> {
        let tests = [("!5;", Operator::Not, 5), ("-15;", Operator::Negate, 15)];

        for (input, operator, value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            if let Some(statement) = program.into_iter().next() {
                match statement {
                    Statement::Expression(expression) => {
                        assert_eq!(
                            expression,
                            Expression::Prefix(
                                *operator,
                                Box::new(Expression::Literal(
                                    Literal::Integer(*value)
                                )),
                            )
                        )
                    }
                    _ => bail!("Expected an expression statement!"),
                }
            }
        }

        Ok(())
    }

    #[test]
    fn prefix_boolean_expressions() -> Result<()> {
        let tests = [
            ("!true;", Operator::Not, true),
            ("!false;", Operator::Not, false),
        ];

        for (input, operator, value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            if let Some(statement) = program.into_iter().next() {
                match statement {
                    Statement::Expression(expression) => {
                        assert_eq!(
                            expression,
                            Expression::Prefix(
                                *operator,
                                Box::new(Expression::Boolean(*value)),
                            )
                        )
                    }
                    _ => bail!("Expected an expression statement!"),
                }
            }
        }

        Ok(())
    }

    #[test]
    fn infix_expressions() -> Result<()> {
        let tests = [
            ("5 + 5;", 5, Operator::Add, 5),
            ("5 - 5;", 5, Operator::Subtract, 5),
            ("5 * 5;", 5, Operator::Multiply, 5),
            ("5 / 5;", 5, Operator::Divide, 5),
            ("5 % 3;", 5, Operator::Modulo, 3),
            ("5 > 5;", 5, Operator::GreaterThan, 5),
            ("5 < 5;", 5, Operator::LessThan, 5),
            ("5 >= 5;", 5, Operator::GreaterThanOrEqual, 5),
            ("5 <= 5;", 5, Operator::LessThanOrEqual, 5),
            ("5 == 5;", 5, Operator::Equal, 5),
            ("5 != 5;", 5, Operator::NotEqual, 5),
        ];

        for (input, left_value, operator, right_value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            if let Some(statement) = program.into_iter().next() {
                match statement {
                    Statement::Expression(expression) => {
                        assert_eq!(
                            expression,
                            Expression::Infix(
                                Box::new(Expression::Literal(
                                    Literal::Integer(*left_value)
                                )),
                                *operator,
                                Box::new(Expression::Literal(
                                    Literal::Integer(*right_value,)
                                )),
                            )
                        )
                    }
                    _ => bail!("Expected an expression statement!"),
                }
            }
        }

        Ok(())
    }

    #[test]
    fn infix_boolean_expressions() -> Result<()> {
        let tests = [
            ("true == true", true, Operator::Equal, true),
            ("true != false", true, Operator::NotEqual, false),
            ("false == false", false, Operator::Equal, false),
        ];

        for (input, left_value, operator, right_value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            if let Some(statement) = program.into_iter().next() {
                match statement {
                    Statement::Expression(expression) => {
                        assert_eq!(
                            expression,
                            Expression::Infix(
                                Box::new(Expression::Boolean(*left_value)),
                                *operator,
                                Box::new(Expression::Boolean(*right_value)),
                            )
                        )
                    }
                    _ => bail!("Expected an expression statement!"),
                }
            }
        }

        Ok(())
    }

    #[test]
    fn operator_precedence() -> Result<()> {
        let tests = [
            ("-a * b", "((-a) * b)"),
            ("!-a", "(!(-a))"),
            ("a + b + c", "((a + b) + c)"),
            ("a + b - c", "((a + b) - c)"),
            ("a * b * c", "((a * b) * c)"),
            ("a * b / c", "((a * b) / c)"),
            ("a + b / c", "(a + (b / c))"),
            ("a + b * c + d / e - f", "(((a + (b * c)) + (d / e)) - f)"),
            ("3 + 4; -5 * 5", "(3 + 4)((-5) * 5)"),
            ("5 > 4 == 3 < 4", "((5 > 4) == (3 < 4))"),
            ("5 < 4 != 3 > 4", "((5 < 4) != (3 > 4))"),
            (
                "3 + 4 * 5 == 3 * 1 + 4 * 5",
                "((3 + (4 * 5)) == ((3 * 1) + (4 * 5)))",
            ),
            ("true", "true"),
            ("false", "false"),
            ("3 > 5 == false", "((3 > 5) == false)"),
            ("3 < 5 == true", "((3 < 5) == true)"),
            ("1 + (2 + 3) + 4", "((1 + (2 + 3)) + 4)"),
            ("(5 + 5) * 2", "((5 + 5) * 2)"),
            ("2 / (5 + 5)", "(2 / (5 + 5))"),
            ("-(5 + 5)", "(-(5 + 5))"),
            ("!(true == true)", "(!(true == true))"),
            ("a + add(b * c) + d", "((a + add((b * c))) + d)"),
            (
                "add(a, b, 1, 2 * 3, 4 + 5, add(6, 7 * 8))",
                "add(a, b, 1, (2 * 3), (4 + 5), add(6, (7 * 8)))",
            ),
            (
                "add(a + b + c * d / f + g)",
                "add((((a + b) + ((c * d) / f)) + g))",
            ),
            (
                "a * [1, 2, 3, 4][b * c] * d",
                "((a * ([1, 2, 3, 4][(b * c)])) * d)",
            ),
            (
                "add(a * b[2], b[1], 2 * [1, 2][1])",
                "add((a * (b[2])), (b[1]), (2 * ([1, 2][1])))",
            ),
        ];

        for (input, expected) in tests.iter() {
            let mut lexer = Lexer::new(input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;
            let program_string = program
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("");

            assert_eq!(program_string, expected.to_string());
        }

        Ok(())
    }

    #[test]
    fn if_expressions() -> Result<()> {
        let expression = Expression::If(
            Box::new(Expression::Infix(
                Box::new(Expression::Identifier("x".to_string())),
                Operator::LessThan,
                Box::new(Expression::Identifier("y".to_string())),
            )),
            vec![Statement::Expression(Expression::Identifier(
                "x".to_string(),
            ))],
            None,
        );

        parse_statement("if (x < y) { x }", &expression)
    }

    #[test]
    fn if_else_expressions() -> Result<()> {
        let expression = Expression::If(
            Box::new(Expression::Infix(
                Box::new(Expression::Identifier("x".to_string())),
                Operator::LessThan,
                Box::new(Expression::Identifier("y".to_string())),
            )),
            vec![Statement::Expression(Expression::Identifier(
                "x".to_string(),
            ))],
            Some(vec![Statement::Expression(Expression::Identifier(
                "y".to_string(),
            ))]),
        );

        parse_statement("if (x < y) { x } else { y }", &expression)
    }

    #[test]
    fn function_expressions() -> Result<()> {
        let expression = Expression::Function(
            vec![
                Parameter {
                    name: "x".to_string(),
                    type_annotation: None,
                },
                Parameter {
                    name: "y".to_string(),
                    type_annotation: None,
                },
            ],
            None,
            vec![Statement::Expression(Expression::Infix(
                Box::new(Expression::Identifier("x".to_string())),
                Operator::Add,
                Box::new(Expression::Identifier("y".to_string())),
            ))],
        );
        parse_statement("fn(x, y) { x + y; }", &expression)
    }

    #[test]
    fn function_parameter_parsing() -> Result<()> {
        let tests: Vec<(&str, Vec<Parameter>)> = vec![
            ("fn() {};", vec![]),
            (
                "fn(x) {};",
                vec![Parameter {
                    name: "x".to_string(),
                    type_annotation: None,
                }],
            ),
            (
                "fn(x, y, z) {};",
                vec![
                    Parameter {
                        name: "x".to_string(),
                        type_annotation: None,
                    },
                    Parameter {
                        name: "y".to_string(),
                        type_annotation: None,
                    },
                    Parameter {
                        name: "z".to_string(),
                        type_annotation: None,
                    },
                ],
            ),
        ];

        for (input, expected_parameters) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            if let Some(statement) = program.into_iter().next() {
                match statement {
                    Statement::Expression(expression) => {
                        assert_eq!(
                            expression,
                            Expression::Function(
                                expected_parameters.to_vec(),
                                None,
                                Vec::new()
                            )
                        )
                    }
                    _ => bail!("Expected an expression statement!"),
                }
            }
        }

        Ok(())
    }

    #[test]
    fn call_expressions() -> Result<()> {
        let expression = Expression::Call(
            Box::new(Expression::Identifier("add".to_string())),
            vec![
                Expression::Literal(Literal::Integer(1)),
                Expression::Infix(
                    Box::new(Expression::Literal(Literal::Integer(2))),
                    Operator::Multiply,
                    Box::new(Expression::Literal(Literal::Integer(3))),
                ),
                Expression::Infix(
                    Box::new(Expression::Literal(Literal::Integer(4))),
                    Operator::Add,
                    Box::new(Expression::Literal(Literal::Integer(5))),
                ),
            ],
        );

        parse_statement("add(1, 2 * 3, 4 + 5);", &expression)
    }

    #[test]
    fn string_literal_expression() -> Result<()> {
        parse_statement(
            "\"hello world\"",
            &Expression::Literal(Literal::String("hello world".to_string())),
        )
    }

    #[test]
    fn array_literal_expression() -> Result<()> {
        parse_statement(
            "[1, 2 * 2, 3 + 3]",
            &Expression::Literal(Literal::Array(vec![
                Expression::Literal(Literal::Integer(1)),
                Expression::Infix(
                    Box::new(Expression::Literal(Literal::Integer(2))),
                    Operator::Multiply,
                    Box::new(Expression::Literal(Literal::Integer(2))),
                ),
                Expression::Infix(
                    Box::new(Expression::Literal(Literal::Integer(3))),
                    Operator::Add,
                    Box::new(Expression::Literal(Literal::Integer(3))),
                ),
            ])),
        )
    }

    #[test]
    fn index_expression() -> Result<()> {
        parse_statement(
            "myArray[1 + 1]",
            &Expression::Index(
                Box::new(Expression::Identifier("myArray".to_string())),
                Box::new(Expression::Infix(
                    Box::new(Expression::Literal(Literal::Integer(1))),
                    Operator::Add,
                    Box::new(Expression::Literal(Literal::Integer(1))),
                )),
            ),
        )
    }

    #[test]
    fn hashmap_literal() -> Result<()> {
        parse_statement(
            r#"{"one": 1, "two": 2, "three": 3}"#,
            &Expression::Literal(Literal::HashMap(vec![
                (
                    Expression::Literal(Literal::String("one".to_string())),
                    Expression::Literal(Literal::Integer(1)),
                ),
                (
                    Expression::Literal(Literal::String("two".to_string())),
                    Expression::Literal(Literal::Integer(2)),
                ),
                (
                    Expression::Literal(Literal::String("three".to_string())),
                    Expression::Literal(Literal::Integer(3)),
                ),
            ])),
        )
    }

    #[test]
    fn empty_hashmap_literal() -> Result<()> {
        parse_statement("{}", &Expression::Literal(Literal::HashMap(vec![])))
    }

    #[test]
    fn hashmap_literal_with_expressions() -> Result<()> {
        parse_statement(
            "{\"one\": 0 + 1, \"two\": 10 - 8, \"three\": 15 / 5}",
            &Expression::Literal(Literal::HashMap(vec![
                (
                    Expression::Literal(Literal::String("one".to_string())),
                    Expression::Infix(
                        Box::new(Expression::Literal(Literal::Integer(0))),
                        Operator::Add,
                        Box::new(Expression::Literal(Literal::Integer(1))),
                    ),
                ),
                (
                    Expression::Literal(Literal::String("two".to_string())),
                    Expression::Infix(
                        Box::new(Expression::Literal(Literal::Integer(10))),
                        Operator::Subtract,
                        Box::new(Expression::Literal(Literal::Integer(8))),
                    ),
                ),
                (
                    Expression::Literal(Literal::String("three".to_string())),
                    Expression::Infix(
                        Box::new(Expression::Literal(Literal::Integer(15))),
                        Operator::Divide,
                        Box::new(Expression::Literal(Literal::Integer(5))),
                    ),
                ),
            ])),
        )
    }

    #[test]
    fn let_with_type_annotation() -> Result<()> {
        let input = "x : i64 = 5;";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        match &program[0] {
            Statement::Let {
                name,
                type_annotation,
                value,
                ..
            } => {
                assert_eq!(name, "x");
                assert_eq!(type_annotation, &Some(Type::I64));
                assert_eq!(value, &Expression::Literal(Literal::Integer(5)));
            }
            _ => bail!("Expected let statement"),
        }
        Ok(())
    }

    #[test]
    fn function_with_typed_parameters() -> Result<()> {
        let input = "fn(a: i64, b: i32) -> bool { true }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Expression(Expression::Function(
            params,
            return_type,
            body,
        )) = &program[0]
        {
            assert_eq!(params.len(), 2);
            assert_eq!(params[0].name, "a");
            assert_eq!(params[0].type_annotation, Some(Type::I64));
            assert_eq!(params[1].name, "b");
            assert_eq!(params[1].type_annotation, Some(Type::I32));
            assert_eq!(return_type, &Some(Type::Bool));
            assert_eq!(body.len(), 1);
        } else {
            bail!("Expected function expression");
        }
        Ok(())
    }

    #[test]
    fn proc_literal() -> Result<()> {
        let input = "proc(x: i64) -> i64 { x }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Expression(Expression::Proc(
            params,
            return_type,
            body,
        )) = &program[0]
        {
            assert_eq!(params.len(), 1);
            assert_eq!(params[0].name, "x");
            assert_eq!(params[0].type_annotation, Some(Type::I64));
            assert_eq!(return_type, &Some(Type::I64));
            assert_eq!(body.len(), 1);
        } else {
            bail!("Expected proc expression");
        }
        Ok(())
    }

    #[test]
    fn struct_declaration() -> Result<()> {
        let input = "Vec3 :: struct { x: f32, y: f32, z: f32 }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Struct(name, fields) = &program[0] {
            assert_eq!(name, "Vec3");
            assert_eq!(fields.len(), 3);
            assert_eq!(
                fields[0],
                StructField {
                    name: "x".to_string(),
                    field_type: Type::F32
                }
            );
            assert_eq!(
                fields[1],
                StructField {
                    name: "y".to_string(),
                    field_type: Type::F32
                }
            );
            assert_eq!(
                fields[2],
                StructField {
                    name: "z".to_string(),
                    field_type: Type::F32
                }
            );
        } else {
            bail!("Expected struct declaration");
        }
        Ok(())
    }

    #[test]
    fn constant_declaration() -> Result<()> {
        let input = "PI :: 3;";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Constant(name, expr) = &program[0] {
            assert_eq!(name, "PI");
            assert_eq!(expr, &Expression::Literal(Literal::Integer(3)));
        } else {
            bail!("Expected constant declaration");
        }
        Ok(())
    }

    #[test]
    fn defer_statement() -> Result<()> {
        let input = "defer return 5;";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Defer(inner) = &program[0] {
            if let Statement::Return(expr) = inner.as_ref() {
                assert_eq!(expr, &Expression::Literal(Literal::Integer(5)));
            } else {
                bail!("Expected return inside defer");
            }
        } else {
            bail!("Expected defer statement");
        }
        Ok(())
    }

    #[test]
    fn for_statement() -> Result<()> {
        let input = "for i in 0..10 { i }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::For(iterator, range, body) = &program[0] {
            assert_eq!(iterator, "i");
            if let Expression::Range(start, end) = range {
                assert_eq!(
                    start.as_ref(),
                    &Expression::Literal(Literal::Integer(0))
                );
                assert_eq!(
                    end.as_ref(),
                    &Expression::Literal(Literal::Integer(10))
                );
            } else {
                bail!("Expected range expression");
            }
            assert_eq!(body.len(), 1);
        } else {
            bail!("Expected for statement");
        }
        Ok(())
    }

    #[test]
    fn field_access() -> Result<()> {
        let input = "point.x";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Expression(Expression::FieldAccess(expr, field)) =
            &program[0]
        {
            assert_eq!(
                expr.as_ref(),
                &Expression::Identifier("point".to_string())
            );
            assert_eq!(field, "x");
        } else {
            bail!("Expected field access expression");
        }
        Ok(())
    }

    #[test]
    fn scoped_identifier() -> Result<()> {
        let input = "Color::Green";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Expression(Expression::Identifier(name)) = &program[0]
        {
            assert_eq!(name, "Color::Green");
        } else {
            bail!("Expected scoped identifier, got {:?}", program[0]);
        }
        Ok(())
    }

    #[test]
    fn borrow_expression() -> Result<()> {
        let input = "&x";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Expression(Expression::Borrow(expr)) = &program[0] {
            assert_eq!(expr.as_ref(), &Expression::Identifier("x".to_string()));
        } else {
            bail!("Expected borrow expression");
        }
        Ok(())
    }

    #[test]
    fn borrow_mut_expression() -> Result<()> {
        let input = "&mut x";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Expression(Expression::BorrowMut(expr)) = &program[0]
        {
            assert_eq!(expr.as_ref(), &Expression::Identifier("x".to_string()));
        } else {
            bail!("Expected borrow mut expression");
        }
        Ok(())
    }

    #[test]
    fn ref_type_annotation() -> Result<()> {
        let input = "r : &i64 = x;";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Let {
            name,
            type_annotation,
            ..
        } = &program[0]
        {
            assert_eq!(name, "r");
            assert_eq!(type_annotation, &Some(Type::Ref(Box::new(Type::I64))));
        } else {
            bail!("Expected let statement with ref type");
        }
        Ok(())
    }

    #[test]
    fn ref_mut_type_annotation() -> Result<()> {
        let input = "r : &mut i64 = x;";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Let {
            name,
            type_annotation,
            ..
        } = &program[0]
        {
            assert_eq!(name, "r");
            assert_eq!(
                type_annotation,
                &Some(Type::RefMut(Box::new(Type::I64)))
            );
        } else {
            bail!("Expected let statement with ref mut type");
        }
        Ok(())
    }

    #[test]
    fn dereference_expression() -> Result<()> {
        let input = "p^";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Expression(Expression::Dereference(expr)) =
            &program[0]
        {
            assert_eq!(expr.as_ref(), &Expression::Identifier("p".to_string()));
        } else {
            bail!("Expected dereference expression");
        }
        Ok(())
    }

    #[test]
    fn pointer_type_annotation() -> Result<()> {
        let input = "p : ^i64 = x;";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Let {
            name,
            type_annotation,
            ..
        } = &program[0]
        {
            assert_eq!(name, "p");
            assert_eq!(type_annotation, &Some(Type::Ptr(Box::new(Type::I64))));
        } else {
            bail!("Expected let statement with pointer type");
        }
        Ok(())
    }

    #[test]
    fn array_type_annotation() -> Result<()> {
        let input = "arr : [10]i64 = x;";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Let {
            name,
            type_annotation,
            ..
        } = &program[0]
        {
            assert_eq!(name, "arr");
            assert_eq!(
                type_annotation,
                &Some(Type::Array(Box::new(Type::I64), 10))
            );
        } else {
            bail!("Expected let statement with array type");
        }
        Ok(())
    }

    #[test]
    fn slice_type_annotation() -> Result<()> {
        let input = "slice : []f32 = x;";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Let {
            name,
            type_annotation,
            ..
        } = &program[0]
        {
            assert_eq!(name, "slice");
            assert_eq!(
                type_annotation,
                &Some(Type::Slice(Box::new(Type::F32)))
            );
        } else {
            bail!("Expected let statement with slice type");
        }
        Ok(())
    }

    #[test]
    fn proc_type_annotation() -> Result<()> {
        let input = "callback : proc(i64, i64) -> i64 = x;";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Let {
            name,
            type_annotation,
            ..
        } = &program[0]
        {
            assert_eq!(name, "callback");
            assert_eq!(
                type_annotation,
                &Some(Type::Proc(
                    vec![Type::I64, Type::I64],
                    Box::new(Type::I64)
                ))
            );
        } else {
            bail!("Expected let statement with proc type");
        }
        Ok(())
    }

    #[test]
    fn struct_init() -> Result<()> {
        let input = "Point { x = 1, y = 2 }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Expression(Expression::StructInit(name, fields)) =
            &program[0]
        {
            assert_eq!(name, "Point");
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].0, "x");
            assert_eq!(fields[1].0, "y");
        } else {
            bail!("Expected struct init expression");
        }
        Ok(())
    }

    #[test]
    fn pointer_store() -> Result<()> {
        let input = "p^ = 42";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Assignment(lhs, rhs) = &program[0] {
            if let Expression::Dereference(ptr) = lhs {
                assert_eq!(
                    ptr.as_ref(),
                    &Expression::Identifier("p".to_string())
                );
            } else {
                bail!("Expected dereference on lhs");
            }
            assert_eq!(rhs, &Expression::Literal(Literal::Integer(42)));
        } else {
            bail!("Expected assignment statement");
        }
        Ok(())
    }

    #[test]
    fn field_assignment() -> Result<()> {
        let input = "p.x = 42";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Assignment(lhs, rhs) = &program[0] {
            if let Expression::FieldAccess(expr, field) = lhs {
                assert_eq!(
                    expr.as_ref(),
                    &Expression::Identifier("p".to_string())
                );
                assert_eq!(field, "x");
            } else {
                bail!("Expected field access on lhs");
            }
            assert_eq!(rhs, &Expression::Literal(Literal::Integer(42)));
        } else {
            bail!("Expected assignment statement");
        }
        Ok(())
    }

    #[test]
    fn sizeof_expression() -> Result<()> {
        let input = "sizeof(i64)";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Expression(Expression::Sizeof(typ)) = &program[0] {
            assert_eq!(*typ, Type::I64);
        } else {
            bail!("Expected sizeof expression");
        }
        Ok(())
    }

    #[test]
    fn sizeof_pointer_expression() -> Result<()> {
        let input = "sizeof(^i64)";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Expression(Expression::Sizeof(typ)) = &program[0] {
            assert_eq!(*typ, Type::Ptr(Box::new(Type::I64)));
        } else {
            bail!("Expected sizeof expression");
        }
        Ok(())
    }

    fn parse_statement(
        input: &str,
        expected_expression: &Expression,
    ) -> Result<()> {
        let mut lexer = Lexer::new(&input);
        let tokens = lexer.tokenize()?;

        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        if let Some(statement) = program.into_iter().next() {
            match statement {
                Statement::Expression(expression)
                | Statement::Return(expression) => {
                    assert_eq!(expression, *expected_expression)
                }
                _ => bail!("Expected an expression statement!"),
            }
        }

        Ok(())
    }

    #[test]
    fn colon_assign_declaration() -> Result<()> {
        let input = "x := 5";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        assert_eq!(
            program[0],
            Statement::Let {
                name: "x".to_string(),
                type_annotation: None,
                value: Expression::Literal(Literal::Integer(5)),
                mutable: false,
            }
        );
        Ok(())
    }

    #[test]
    fn typed_declaration() -> Result<()> {
        let input = "x : i64 = 42";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        assert_eq!(
            program[0],
            Statement::Let {
                name: "x".to_string(),
                type_annotation: Some(Type::I64),
                value: Expression::Literal(Literal::Integer(42)),
                mutable: false,
            }
        );
        Ok(())
    }

    #[test]
    fn odin_style_function() -> Result<()> {
        let input = "add := fn(a, b) { a + b }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Let {
            name,
            type_annotation: None,
            value: Expression::Function(params, _, _),
            ..
        } = &program[0]
        {
            assert_eq!(name, "add");
            assert_eq!(params.len(), 2);
        } else {
            bail!("Expected let statement with function");
        }
        Ok(())
    }

    #[test]
    fn mutable_declaration() -> Result<()> {
        let input = "mut x := 5";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        assert_eq!(
            program[0],
            Statement::Let {
                name: "x".to_string(),
                type_annotation: None,
                value: Expression::Literal(Literal::Integer(5)),
                mutable: true,
            }
        );
        Ok(())
    }

    #[test]
    fn mutable_typed_declaration() -> Result<()> {
        let input = "mut x : i64 = 42";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        assert_eq!(
            program[0],
            Statement::Let {
                name: "x".to_string(),
                type_annotation: Some(Type::I64),
                value: Expression::Literal(Literal::Integer(42)),
                mutable: true,
            }
        );
        Ok(())
    }

    #[test]
    fn immutable_declaration_default() -> Result<()> {
        let input = "x := 5";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);
        if let Statement::Let { mutable, .. } = &program[0] {
            assert!(!mutable, "Declaration without mut should be immutable");
        } else {
            bail!("Expected let statement");
        }
        Ok(())
    }

    #[test]
    fn mutable_ast_display() -> Result<()> {
        let output = "mut myVar := anotherVar;";
        let ast = Statement::Let {
            name: "myVar".to_string(),
            type_annotation: None,
            value: Expression::Identifier("anotherVar".to_string()),
            mutable: true,
        };
        assert_eq!(ast.to_string(), output.to_string());
        Ok(())
    }

    #[test]
    fn mutable_typed_ast_display() -> Result<()> {
        let output = "mut x : i64 = 5;";
        let ast = Statement::Let {
            name: "x".to_string(),
            type_annotation: Some(Type::I64),
            value: Expression::Literal(Literal::Integer(5)),
            mutable: true,
        };
        assert_eq!(ast.to_string(), output.to_string());
        Ok(())
    }
}
