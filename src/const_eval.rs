use crate::ast::{Ast, ExprId, Expression, Literal, Range32, Statement};
use crate::lexer::Token;
use crate::parser::Operator;
use std::collections::HashMap;
use std::rc::Rc;

/// How far one compile-time evaluation may run. A body may loop, so how long
/// it takes is not read off the text, and this is what says a compile finishes.
const STEP_LIMIT: u64 = 1_000_000;

/// What a compile-time call is told when a fraction turns up in one.
const FRACTION: &str =
    "a compile-time value is a whole number or a yes or no, and this is a number with a fraction";

/// How deep calls may nest. A function that reaches itself is refused by name,
/// so this is what catches a long chain of distinct functions.
const DEPTH_LIMIT: usize = 32;

/// What a compile-time expression works out to.
///
/// A whole number or a yes or no, and nothing else. What a compile-time value
/// is for is a length, a capacity, a count or a branch, and folding a fraction
/// would mean two decimal-to-double readings having to agree bit for bit,
/// which is a guarantee neither compiler makes anywhere else.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Integer(i64),
    Boolean(bool),
}

impl Value {
    pub fn integer(self) -> Option<i64> {
        match self {
            Value::Integer(held) => Some(held),
            _ => None,
        }
    }

    fn describe(self) -> &'static str {
        match self {
            Value::Integer(_) => "a whole number",
            Value::Boolean(_) => "a yes or no",
        }
    }
}

/// A function body, parsed once and read for every call.
struct Body {
    ast: Ast,
    parameters: Vec<String>,
    statements: Range32,
}

/// The two sides of an `if`, and what decides between them.
struct Arms {
    condition: ExprId,
    then: Range32,
    otherwise: Option<Range32>,
}

/// What a statement did: ran, or left the function with a value.
enum Flow {
    Ran,
    /// An expression statement, which is a value where it is the last one of a
    /// block and nothing where it is not.
    Value(Value),
    Left(Value),
    Broke,
    Continued,
}

/// Works out what a call written where a compile-time value is read answers.
///
/// The functions it may call are the ones the file declares, parsed from their
/// tokens. That is what lets a constant call a function written below it, and
/// what keeps this out of the way of the parse it feeds.
pub struct Folder<'a> {
    tokens: &'a [Token],
    bodies: HashMap<String, (usize, usize)>,
    // What a file this one imports exports, as the run of tokens each body
    // occupies. Those come from another token stream, so they are held rather
    // than pointed at.
    imported: HashMap<String, Rc<Vec<Token>>>,
    parsed: HashMap<String, Option<Rc<Body>>>,
    steps: u64,
}

impl<'a> Folder<'a> {
    pub fn new(
        tokens: &'a [Token],
        bodies: HashMap<String, (usize, usize)>,
        imported: HashMap<String, Rc<Vec<Token>>>,
    ) -> Self {
        Self {
            tokens,
            bodies,
            imported,
            parsed: HashMap::new(),
            steps: 0,
        }
    }

    /// Whether this name has a body to read, which is what says a call written
    /// in a compile-time position was asking for one.
    pub fn declares(&self, name: &str) -> bool {
        self.bodies.contains_key(name) || self.imported.contains_key(name)
    }

    /// The value of an expression already parsed, over the constants known so
    /// far. A call is worked out here; anything else answers as it always did.
    pub fn expression(
        &mut self,
        ast: &Ast,
        expression: ExprId,
        known: &HashMap<String, Value>,
    ) -> Result<Value, String> {
        self.steps = 0;
        self.value(ast, expression, known, &mut Vec::new())
    }

    fn body_of(&mut self, name: &str) -> Option<Rc<Body>> {
        if let Some(held) = self.parsed.get(name) {
            return held.clone();
        }
        // A function this file declares is read out of the tokens the parser
        // is already holding. One it imports was lexed separately, and the
        // file's own declaration wins where both have the name.
        let held = self.imported.get(name).cloned();
        let made = self
            .bodies
            .get(name)
            .map(|(start, end)| &self.tokens[*start..*end])
            .or(held.as_deref().map(Vec::as_slice))
            .and_then(crate::parser::parse_function_value)
            .and_then(|(ast, expression)| {
                let (Expression::Function(parameters, _, statements)
                | Expression::Proc(parameters, _, statements)) =
                    ast.expr(expression)
                else {
                    return None;
                };
                let named = ast
                    .params_in(*parameters)
                    .iter()
                    .map(|parameter| ast.name(parameter.name).to_string())
                    .collect();
                let statements = *statements;
                Some(Rc::new(Body {
                    ast,
                    parameters: named,
                    statements,
                }))
            });
        self.parsed.insert(name.to_string(), made.clone());
        made
    }

    fn spend(&mut self) -> Result<(), String> {
        self.steps += 1;
        if self.steps > STEP_LIMIT {
            return Err(format!(
                "working this out at compile time took more than {STEP_LIMIT} steps, which is as far as it goes"
            ));
        }
        Ok(())
    }

    fn call(
        &mut self,
        name: &str,
        arguments: Vec<Value>,
        stack: &mut Vec<String>,
    ) -> Result<Value, String> {
        if stack.iter().any(|held| held == name) {
            return Err(format!(
                "'{name}' reaches itself, and a compile-time call is worked out by running it, which never ends"
            ));
        }
        if stack.len() >= DEPTH_LIMIT {
            return Err(format!(
                "a compile-time call may nest {DEPTH_LIMIT} deep and '{name}' is deeper"
            ));
        }
        let Some(body) = self.body_of(name) else {
            return Err(format!(
                "'{name}' is not a function this program declares, so there is nothing to work out here"
            ));
        };
        if body.parameters.len() != arguments.len() {
            return Err(format!(
                "'{name}' takes {} arguments and {} were written",
                body.parameters.len(),
                arguments.len()
            ));
        }
        let mut locals: HashMap<String, Value> = HashMap::new();
        for (parameter, argument) in body.parameters.iter().zip(arguments) {
            locals.insert(parameter.clone(), argument);
        }
        stack.push(name.to_string());
        let answered =
            self.block(&body.ast, body.statements, &mut locals, stack);
        stack.pop();
        match answered? {
            Flow::Left(value) => Ok(value),
            _ => Err(format!(
                "'{name}' reached its end without a value, so there is nothing for the constant to be"
            )),
        }
    }

    fn block(
        &mut self,
        ast: &Ast,
        statements: Range32,
        locals: &mut HashMap<String, Value>,
        stack: &mut Vec<String>,
    ) -> Result<Flow, String> {
        let held = ast.stmts_in(statements).to_vec();
        let mut answered = Flow::Ran;
        for statement in &held {
            self.spend()?;
            answered = match self.statement(ast, *statement, locals, stack)? {
                Flow::Ran => Flow::Ran,
                Flow::Value(value) => Flow::Value(value),
                left => return Ok(left),
            };
        }
        // A block answers with its trailing expression, so the last statement
        // of a body leaves with a value where an earlier one only ran.
        match answered {
            Flow::Value(value) => Ok(Flow::Left(value)),
            _ => Ok(Flow::Ran),
        }
    }

    fn statement(
        &mut self,
        ast: &Ast,
        statement: crate::ast::StmtId,
        locals: &mut HashMap<String, Value>,
        stack: &mut Vec<String>,
    ) -> Result<Flow, String> {
        match ast.stmt(statement) {
            Statement::Let { name, value, .. } => {
                let held = self.value(ast, *value, locals, stack)?;
                locals.insert(ast.name(*name).to_string(), held);
                Ok(Flow::Ran)
            }
            Statement::Assignment(place, value) => {
                let Expression::Identifier(name) = ast.expr(*place) else {
                    return Err(
                        "a compile-time call writes to a name and nothing else"
                            .to_string(),
                    );
                };
                let held = self.value(ast, *value, locals, stack)?;
                let name = ast.name(*name).to_string();
                if !locals.contains_key(&name) {
                    return Err(format!(
                        "'{name}' is not a name this call knows, so there is nothing to write to"
                    ));
                }
                locals.insert(name, held);
                Ok(Flow::Ran)
            }
            Statement::Return(value) => {
                let held = self.value(ast, *value, locals, stack)?;
                Ok(Flow::Left(held))
            }
            Statement::Expression(value) => {
                // A bare `if` is a statement here, and each arm may leave.
                if let Expression::If(condition, then, otherwise) =
                    ast.expr(*value)
                {
                    let arms = Arms {
                        condition: *condition,
                        then: *then,
                        otherwise: *otherwise,
                    };
                    return self.branch(ast, arms, locals, stack);
                }
                let held = self.value(ast, *value, locals, stack)?;
                Ok(Flow::Value(held))
            }
            Statement::While(condition, body) => {
                loop {
                    self.spend()?;
                    let Value::Boolean(holds) =
                        self.value(ast, *condition, locals, stack)?
                    else {
                        return Err(
                            "a `while` is run at compile time by asking its condition, which has to be a yes or no"
                                .to_string(),
                        );
                    };
                    if !holds {
                        return Ok(Flow::Ran);
                    }
                    match self.block(ast, *body, locals, stack)? {
                        Flow::Ran | Flow::Value(_) | Flow::Continued => {}
                        Flow::Broke => return Ok(Flow::Ran),
                        Flow::Left(value) => return Ok(Flow::Left(value)),
                    }
                }
            }
            Statement::Break => Ok(Flow::Broke),
            Statement::Continue => Ok(Flow::Continued),
            held => Err(format!(
                "{} is not something a compile-time call may do",
                describe_statement(held)
            )),
        }
    }

    fn branch(
        &mut self,
        ast: &Ast,
        arms: Arms,
        locals: &mut HashMap<String, Value>,
        stack: &mut Vec<String>,
    ) -> Result<Flow, String> {
        let Arms {
            condition,
            then,
            otherwise,
        } = arms;
        let Value::Boolean(holds) = self.value(ast, condition, locals, stack)?
        else {
            return Err(
                "an `if` is decided at compile time by asking its condition, which has to be a yes or no"
                    .to_string(),
            );
        };
        if holds {
            return self.block(ast, then, locals, stack);
        }
        match otherwise {
            Some(arm) => self.block(ast, arm, locals, stack),
            None => Ok(Flow::Ran),
        }
    }

    fn value(
        &mut self,
        ast: &Ast,
        expression: ExprId,
        locals: &HashMap<String, Value>,
        stack: &mut Vec<String>,
    ) -> Result<Value, String> {
        self.spend()?;
        match ast.expr(expression) {
            Expression::Literal(Literal::Integer(held)) => {
                Ok(Value::Integer(*held))
            }
            Expression::Literal(Literal::Float(_) | Literal::Float32(_)) => {
                Err(FRACTION.to_string())
            }
            Expression::Boolean(held) => Ok(Value::Boolean(*held)),
            Expression::Literal(Literal::Boolean(held)) => {
                Ok(Value::Boolean(*held))
            }
            Expression::Identifier(name) => {
                let name = ast.name(*name);
                match name {
                    "true" => return Ok(Value::Boolean(true)),
                    "false" => return Ok(Value::Boolean(false)),
                    _ => {}
                }
                locals.get(name).copied().ok_or_else(|| {
                    format!(
                        "'{name}' has no value at compile time, so this cannot be worked out before the program runs"
                    )
                })
            }
            Expression::Prefix(Operator::Negate, inner) => {
                match self.value(ast, *inner, locals, stack)? {
                    Value::Integer(held) => held
                        .checked_neg()
                        .map(Value::Integer)
                        .ok_or_else(|| "negating this overflows".to_string()),
                    held => Err(format!("'-' has no meaning for {}", held.describe())),
                }
            }
            Expression::Prefix(Operator::Not, inner) => {
                match self.value(ast, *inner, locals, stack)? {
                    Value::Boolean(held) => Ok(Value::Boolean(!held)),
                    held => Err(format!("'!' has no meaning for {}", held.describe())),
                }
            }
            Expression::Infix(left, operator, right) => {
                // `&&` and `||` answer without asking the right side when the
                // left one settles it, the way they do where the program runs.
                if matches!(operator, Operator::And | Operator::Or) {
                    let Value::Boolean(first) =
                        self.value(ast, *left, locals, stack)?
                    else {
                        return Err(
                            "'&&' and '||' join two yes-or-no answers"
                                .to_string(),
                        );
                    };
                    if matches!(operator, Operator::And) && !first {
                        return Ok(Value::Boolean(false));
                    }
                    if matches!(operator, Operator::Or) && first {
                        return Ok(Value::Boolean(true));
                    }
                    let Value::Boolean(second) =
                        self.value(ast, *right, locals, stack)?
                    else {
                        return Err(
                            "'&&' and '||' join two yes-or-no answers"
                                .to_string(),
                        );
                    };
                    return Ok(Value::Boolean(second));
                }
                let left = self.value(ast, *left, locals, stack)?;
                let right = self.value(ast, *right, locals, stack)?;
                combine(left, *operator, right)
            }
            Expression::If(condition, then, otherwise) => {
                let mut held = locals.clone();
                let arms = Arms {
                    condition: *condition,
                    then: *then,
                    otherwise: *otherwise,
                };
                match self.branch(ast, arms, &mut held, stack)? {
                    Flow::Left(value) => Ok(value),
                    _ => Err(
                        "this `if` answers with nothing, so there is no value here"
                            .to_string(),
                    ),
                }
            }
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(name) = ast.expr(*callee) else {
                    return Err(
                        "a compile-time call names the function it calls"
                            .to_string(),
                    );
                };
                let name = ast.name(*name).to_string();
                let mut held = Vec::new();
                for argument in ast.exprs_in(*arguments).to_vec() {
                    held.push(self.value(ast, argument, locals, stack)?);
                }
                self.call(&name, held, stack)
            }
            held => Err(format!(
                "{} is not something a compile-time call may do",
                describe_expression(held)
            )),
        }
    }
}

fn combine(
    left: Value,
    operator: Operator,
    right: Value,
) -> Result<Value, String> {
    match (left, right) {
        (Value::Integer(left), Value::Integer(right)) => {
            integers(left, operator, right)
        }
        (Value::Boolean(left), Value::Boolean(right)) => {
            match operator {
                Operator::Equal => Ok(Value::Boolean(left == right)),
                Operator::NotEqual => Ok(Value::Boolean(left != right)),
                _ => Err(
                    "a yes or no is compared and nothing else".to_string()
                ),
            }
        }
        (left, right) => Err(format!(
            "this joins {} to {}, and a compile-time value is worked out over one kind at a time",
            left.describe(),
            right.describe()
        )),
    }
}

fn integers(
    left: i64,
    operator: Operator,
    right: i64,
) -> Result<Value, String> {
    let overflowed = || "this overflows an i64".to_string();
    Ok(match operator {
        Operator::Add => {
            Value::Integer(left.checked_add(right).ok_or_else(overflowed)?)
        }
        Operator::Subtract => {
            Value::Integer(left.checked_sub(right).ok_or_else(overflowed)?)
        }
        Operator::Multiply => {
            Value::Integer(left.checked_mul(right).ok_or_else(overflowed)?)
        }
        Operator::Divide => Value::Integer(
            left.checked_div(right)
                .ok_or_else(|| "this divides by zero".to_string())?,
        ),
        Operator::Modulo => Value::Integer(
            left.checked_rem(right)
                .ok_or_else(|| "this divides by zero".to_string())?,
        ),
        Operator::ShiftLeft => Value::Integer(
            u32::try_from(right)
                .ok()
                .and_then(|by| left.checked_shl(by))
                .ok_or_else(overflowed)?,
        ),
        Operator::ShiftRight => Value::Integer(
            u32::try_from(right)
                .ok()
                .and_then(|by| left.checked_shr(by))
                .ok_or_else(overflowed)?,
        ),
        Operator::BitwiseAnd => Value::Integer(left & right),
        Operator::BitwiseOr => Value::Integer(left | right),
        Operator::LessThan => Value::Boolean(left < right),
        Operator::LessThanOrEqual => Value::Boolean(left <= right),
        Operator::GreaterThan => Value::Boolean(left > right),
        Operator::GreaterThanOrEqual => Value::Boolean(left >= right),
        Operator::Equal => Value::Boolean(left == right),
        Operator::NotEqual => Value::Boolean(left != right),
        _ => {
            return Err(
                "this operator has no compile-time answer for two whole numbers"
                    .to_string(),
            );
        }
    })
}

fn describe_statement(statement: &Statement) -> &'static str {
    match statement {
        Statement::For(..) => "a `for`",
        Statement::With(..) => "a `with`",
        Statement::Defer(_) => "a `defer`",
        Statement::ErrDefer(_) => "an `errdefer`",
        Statement::LetMultiple(..) => "a binding list",
        Statement::Import(..) => "an `import`",
        _ => "this",
    }
}

fn describe_expression(expression: &Expression) -> &'static str {
    match expression {
        Expression::Literal(Literal::String(_)) => "a string",
        Expression::Literal(Literal::Array(_)) => "an array",
        Expression::Index(..) => "an index",
        Expression::FieldAccess(..) => "a field",
        Expression::StructInit(..) => "a struct value",
        Expression::EnumVariantInit(..) => "an enum value",
        Expression::AddressOf(_) | Expression::Dereference(_) => "a pointer",
        Expression::Unsafe(_) | Expression::UnsafeFn(_) => "an `unsafe` block",
        Expression::Switch(..) => "a `match`",
        Expression::Try(_) => "a `?`",
        Expression::ArrayRepeat(..) => "an array",
        Expression::Function(..) | Expression::Proc(..) => "a function value",
        _ => "this",
    }
}
