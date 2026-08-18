use crate::ast::{
    Ast, ExprId, Expression, Literal, Pattern, PatternId, Range32, Statement,
};
use crate::lexer::Token;
use crate::parser::Operator;
use std::collections::HashMap;
use std::rc::Rc;

/// How far one compile-time evaluation may run. A body may loop, so how long
/// it takes is not read off the text, and this is what says a compile finishes.
const STEP_LIMIT: u64 = 1_000_000;

/// What a compile-time call is told when a fraction turns up in one.
const FRACTION: &str = "a compile-time value is a whole number or a yes or no, and this is a number with a fraction";

/// How deep calls may nest. A call may reach itself, so this is the whole of
/// what bounds a chain: a step that calls itself and a long chain of distinct
/// functions are the same shape once one of them is counted.
const DEPTH_LIMIT: usize = 32;

/// What a type answers once it has been laid out. Each is a call where a
/// program runs and each is answered by the layout pass, which reads the types
/// after this has settled every constant.
pub const LAYOUT_ANSWERS: &[&str] = &[
    "sizeof",
    "alignof",
    "name_of",
    "offset_of",
    "field_count",
    "typename",
    "type_id",
];

/// What a compile-time value is told when it asks a type for one of those. The
/// two answer sites cannot see each other, so this names the order rather than
/// the site, and it names the word that was written: the six answer different
/// questions and a reader who asked for one is told about that one.
pub fn layout_message(named: &str) -> String {
    format!(
        "'{named}' is answered once the types are read, and a compile-time value is worked out before that"
    )
}

/// What a type measures. These are what a constant may ask for, since a
/// measurement is what the layout pass works out and a constant is worked out
/// again after it. `type_id` is settled while the program is emitted and
/// `offset_of` names a field rather than a type, so neither is one of these.
pub const MEASUREMENTS: &[&str] =
    &["sizeof", "alignof", "field_count", "typename"];

/// Whether a value stopped because it asked a type what it measures. One that
/// did is set aside rather than refused: a constant is worked out again once the
/// types have been read, and there the question has an answer. A length is read
/// while the types are, so nothing sets that one aside.
pub fn asks_a_measurement(reason: &str) -> bool {
    MEASUREMENTS
        .iter()
        .any(|named| reason == layout_message(named))
}

/// What a compile-time expression works out to.
///
/// A whole number, a yes or no, and the three things built out of those: a run
/// of them, a set of named ones, and a run of bytes. No fraction: folding one
/// would mean two decimal-to-double readings having to agree bit for bit, which
/// is a guarantee neither compiler makes anywhere else.
///
/// An aggregate is *held* rather than read back out of the tokens each time.
/// An element may itself be a call, so re-reading would re-run it once per
/// index; and a value has to outlive the names that built it, which a position
/// in the token stream does not.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Integer(i64),
    Boolean(bool),
    Array(Rc<Vec<Value>>),
    /// The type it was written as, and its fields. The name rides along
    /// because a set of named values is written back into the program as a
    /// literal of that type, and a literal says which type it is.
    Record(Rc<String>, Rc<Vec<(String, Value)>>),
    Text(Rc<String>),
}

impl Value {
    pub fn integer(&self) -> Option<i64> {
        match self {
            Value::Integer(held) => Some(*held),
            _ => None,
        }
    }

    fn describe(&self) -> &'static str {
        match self {
            Value::Integer(_) => "a whole number",
            Value::Boolean(_) => "a yes or no",
            Value::Array(_) => "a run of values",
            Value::Record(..) => "a set of named values",
            Value::Text(_) => "a run of bytes",
        }
    }
}

/// A function body, parsed once and read for every call.
///
/// One read out of the tokens carries its own parse. One found in a tree that is
/// already built carries none: its statements belong to the tree the call is
/// being worked out in, which is the tree that is handed down.
struct Body {
    ast: Option<Ast>,
    parameters: Vec<String>,
    statements: Range32,
}

/// What a `match` reads, and the arms it chooses between.
struct Cases {
    subject: ExprId,
    arms: Range32,
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
///
/// A call may reach itself. What bounds the work is the step count and the
/// nesting depth, and a step that calls itself is counted by both, so nothing
/// here has to read the shape of the call graph to know a compile finishes.
pub struct Folder<'a> {
    tokens: &'a [Token],
    bodies: HashMap<String, (usize, usize)>,
    // What a file this one imports exports, as the run of tokens each body
    // occupies. Those come from another token stream, so they are held rather
    // than pointed at.
    imported: HashMap<String, Rc<Vec<Token>>>,
    parsed: HashMap<String, Option<Rc<Body>>>,
    // The constants settled so far. A body may name one, and a call binds its
    // parameters rather than carrying the caller's names, so what a body can
    // read is held here rather than passed down with the locals.
    constants: HashMap<String, Value>,
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
            constants: HashMap::new(),
            steps: 0,
        }
    }

    /// A folder over a tree that is already built, for the constants left to
    /// settle once the types have been laid out. It reads no tokens: every body
    /// it can run is one the tree already holds, which is what lets a constant
    /// asking a type for its layout be worked out after the layouts exist and
    /// still be the same evaluation as the one before the parse.
    pub fn over_tree(ast: &Ast, roots: &[crate::ast::StmtId]) -> Self {
        let mut parsed = HashMap::new();
        for statement in roots {
            let Statement::Constant(name, value) = ast.stmt(*statement) else {
                continue;
            };
            let (Expression::Function(parameters, _, statements)
            | Expression::Proc(parameters, _, statements)) = ast.expr(*value)
            else {
                continue;
            };
            let named = ast
                .params_in(*parameters)
                .iter()
                .map(|parameter| ast.name(parameter.name).to_string())
                .collect();
            parsed.insert(
                ast.name(*name).to_string(),
                Some(Rc::new(Body {
                    ast: None,
                    parameters: named,
                    statements: *statements,
                })),
            );
        }
        Self {
            tokens: &[],
            bodies: HashMap::new(),
            imported: HashMap::new(),
            parsed,
            constants: HashMap::new(),
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
        self.constants = known.clone();
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
                    ast: Some(ast),
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
        ast: &Ast,
        name: &str,
        arguments: Vec<Value>,
        stack: &mut Vec<String>,
    ) -> Result<Value, String> {
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
        let held = body.ast.as_ref().unwrap_or(ast);
        let answered = self.block(held, body.statements, &mut locals, stack);
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
                // A bare `match` is a statement here the same way, so an arm
                // may leave the call rather than answer with a value.
                if let Expression::Switch(subject, arms) = ast.expr(*value) {
                    let cases = Cases {
                        subject: *subject,
                        arms: *arms,
                    };
                    return self.arm(ast, cases, locals, stack);
                }
                let held = self.value(ast, *value, locals, stack)?;
                Ok(Flow::Value(held))
            }
            Statement::While(condition, body) => loop {
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
            },
            // A `for` over a span of whole numbers, or over a run this has
            // already worked out. Both have a length before the body is read,
            // so the walk is as bounded as the `while` above it is by its step
            // count. `for index, item in run` names the position as well.
            Statement::For(name, second, iterable, body) => {
                let items = self.items(ast, *iterable, locals, stack)?;
                let name = ast.name(*name).to_string();
                let second =
                    second.map(|held| ast.name(held).to_string());
                for (index, item) in items.into_iter().enumerate() {
                    self.spend()?;
                    match &second {
                        Some(element) => {
                            locals.insert(
                                name.clone(),
                                Value::Integer(index as i64),
                            );
                            locals.insert(element.clone(), item);
                        }
                        None => {
                            locals.insert(name.clone(), item);
                        }
                    }
                    match self.block(ast, *body, locals, stack)? {
                        Flow::Ran | Flow::Value(_) | Flow::Continued => {}
                        Flow::Broke => break,
                        Flow::Left(value) => return Ok(Flow::Left(value)),
                    }
                }
                Ok(Flow::Ran)
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
        let Value::Boolean(holds) =
            self.value(ast, condition, locals, stack)?
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

    // What a `for` walks: a span of whole numbers, or a run this has already
    // worked out. Both have a length before the body is read, so the walk is
    // bounded the way the step count bounds a `while`.
    fn items(
        &mut self,
        ast: &Ast,
        iterable: ExprId,
        locals: &HashMap<String, Value>,
        stack: &mut Vec<String>,
    ) -> Result<Vec<Value>, String> {
        if let Expression::Range(start, end, inclusive) = ast.expr(iterable) {
            let inclusive = *inclusive;
            let span = "a `for` over a span walks whole numbers";
            let Value::Integer(from) =
                self.value(ast, *start, locals, stack)?
            else {
                return Err(span.to_string());
            };
            let Value::Integer(to) = self.value(ast, *end, locals, stack)?
            else {
                return Err(span.to_string());
            };
            let last = match inclusive {
                true => to,
                false => match to.checked_sub(1) {
                    Some(last) => last,
                    None => return Ok(Vec::new()),
                },
            };
            let mut held = Vec::new();
            let mut at = from;
            while at <= last {
                self.spend()?;
                held.push(Value::Integer(at));
                let Some(next) = at.checked_add(1) else { break };
                at = next;
            }
            return Ok(held);
        }
        match self.value(ast, iterable, locals, stack)? {
            Value::Array(items) => Ok(items.to_vec()),
            Value::Text(held) => Ok(held
                .as_bytes()
                .iter()
                .map(|byte| Value::Integer(*byte as i64))
                .collect()),
            held => Err(format!(
                "a `for` walks a span or a run of values, and {} is neither",
                held.describe()
            )),
        }
    }

    // The arm a `match` takes. The subject is worked out here, so which arm it
    // is settled before any of them is read, and the arms that were not taken
    // are never worked out at all.
    fn arm(
        &mut self,
        ast: &Ast,
        cases: Cases,
        locals: &mut HashMap<String, Value>,
        stack: &mut Vec<String>,
    ) -> Result<Flow, String> {
        let held = self.value(ast, cases.subject, locals, stack)?;
        for case in ast.cases_in(cases.arms).to_vec() {
            self.spend()?;
            if covers(ast, case.pattern, &held)? {
                return self.block(ast, case.body, locals, stack);
            }
        }
        // No arm taken, so the `match` answered nothing and the body it stands
        // in reaches its end without a value. A `match` over a whole number
        // covers every one of them or the exhaustiveness rule has already
        // refused it, so this is only reached for a program that is refused
        // anyway, and what a reader is told is that the constant has nothing
        // to be.
        Ok(Flow::Ran)
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
            Expression::Literal(Literal::String(held)) => {
                Ok(Value::Text(Rc::new(held.clone())))
            }
            // A run of values written out. Each element is worked out once,
            // here, and what comes out is held: an element may itself be a
            // call, and reading the run again would run it again.
            Expression::Literal(Literal::Array(elements)) => {
                let mut held = Vec::new();
                for element in ast.exprs_in(*elements).to_vec() {
                    held.push(self.value(ast, element, locals, stack)?);
                }
                Ok(Value::Array(Rc::new(held)))
            }
            // `[value; n]`, the same value n times. The count is written as a
            // number or as a constant standing for one, which is how the
            // parser reads it everywhere else.
            Expression::ArrayRepeat(inner, count) => {
                let held = self.value(ast, *inner, locals, stack)?;
                let written = ast.name(*count);
                let Some(count) = written
                    .parse::<i64>()
                    .ok()
                    .or_else(|| locals.get(written)?.integer())
                else {
                    return Err(format!(
                        "'{written}' is not a count this can work out"
                    ));
                };
                let Ok(count) = usize::try_from(count) else {
                    return Err(format!(
                        "a run is written {count} long, and a length is not negative"
                    ));
                };
                for _ in 0..count {
                    self.spend()?;
                }
                Ok(Value::Array(Rc::new(vec![held; count])))
            }
            // A set of named values. Every field is named at the literal, so
            // what a field reads is decided here without a layout.
            Expression::StructInit(name, initializers) => {
                // A literal that named no type takes it from what the context
                // expects, and a value written back where a constant is named
                // carries its own type or none at all. Folded anyway it became
                // a set of named values belonging to nothing, and the reader was
                // told about the use rather than about the literal.
                if ast.name(*name).is_empty() {
                    return Err(
                        "this is not something a compile-time call may do"
                            .to_string(),
                    );
                }
                let mut held = Vec::new();
                for named in ast.named_in(*initializers).to_vec() {
                    let value = self.value(ast, named.value, locals, stack)?;
                    held.push((ast.name(named.name).to_string(), value));
                }
                Ok(Value::Record(
                    Rc::new(ast.name(*name).to_string()),
                    Rc::new(held),
                ))
            }
            Expression::Index(base, index) => {
                let base = self.value(ast, *base, locals, stack)?;
                let Value::Integer(index) =
                    self.value(ast, *index, locals, stack)?
                else {
                    return Err(
                        "what is read out of a run is named by a whole number"
                            .to_string(),
                    );
                };
                match &base {
                    Value::Array(items) => {
                        let at = in_range(index, items.len())?;
                        Ok(items[at].clone())
                    }
                    Value::Text(held) => {
                        let bytes = held.as_bytes();
                        let at = in_range(index, bytes.len())?;
                        Ok(Value::Integer(bytes[at] as i64))
                    }
                    held => Err(format!(
                        "{} is not something an index reads",
                        held.describe()
                    )),
                }
            }
            Expression::FieldAccess(base, field) => {
                let base = self.value(ast, *base, locals, stack)?;
                let name = ast.name(*field);
                let Value::Record(_, fields) = &base else {
                    return Err(format!(
                        "{} has no field to read",
                        base.describe()
                    ));
                };
                fields
                    .iter()
                    .find(|(held, _)| held == name)
                    .map(|(_, value)| value.clone())
                    .ok_or_else(|| format!("this has no field called '{name}'"))
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
                // A name this call bound first, then a constant settled
                // before this one. A parameter of the same name is the one in
                // hand, which is what a name means anywhere else.
                locals
                    .get(name)
                    .or_else(|| self.constants.get(name))
                    .cloned()
                    .ok_or_else(|| {
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
                    held => Err(format!(
                        "'-' has no meaning for {}",
                        held.describe()
                    )),
                }
            }
            Expression::Prefix(Operator::Not, inner) => {
                match self.value(ast, *inner, locals, stack)? {
                    Value::Boolean(held) => Ok(Value::Boolean(!held)),
                    held => Err(format!(
                        "'!' has no meaning for {}",
                        held.describe()
                    )),
                }
            }
            Expression::Infix(left, operator, right) => {
                // `&&` and `||` answer without asking the right side when the
                // left one settles it, the way they do where the program runs.
                if matches!(operator, Operator::And | Operator::Or) {
                    let Value::Boolean(first) =
                        self.value(ast, *left, locals, stack)?
                    else {
                        return Err("'&&' and '||' join two yes-or-no answers"
                            .to_string());
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
                        return Err("'&&' and '||' join two yes-or-no answers"
                            .to_string());
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
            Expression::Switch(subject, arms) => {
                let mut held = locals.clone();
                let cases = Cases {
                    subject: *subject,
                    arms: *arms,
                };
                match self.arm(ast, cases, &mut held, stack)? {
                    Flow::Left(value) => Ok(value),
                    _ => Err(
                        "this `match` answers with nothing, so there is no value here"
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
                if LAYOUT_ANSWERS.contains(&name.as_str()) {
                    return Err(layout_message(&name));
                }
                let mut held = Vec::new();
                for argument in ast.exprs_in(*arguments).to_vec() {
                    held.push(self.value(ast, argument, locals, stack)?);
                }
                // The three that keep the low bits. They exist so a value may
                // leave its range on purpose, and a hash worked out before the
                // program runs has to come out the same as one worked out
                // while it does.
                if let Some(answered) = builtin(&name, &held)? {
                    return Ok(answered);
                }
                self.call(ast, &name, held, stack)
            }
            held => Err(format!(
                "{} is not something a compile-time call may do",
                describe_expression(held)
            )),
        }
    }
}

// `wrap_add`, `wrap_sub` and `wrap_mul`, which are builtins the parser reads at
// a call rather than functions a program declares. Answers nothing for any
// other name, so an ordinary call goes on being one.
fn builtin(name: &str, given: &[Value]) -> Result<Option<Value>, String> {
    if matches!(name, "str_len" | "slice_len") {
        return match given {
            [Value::Text(held)] => Ok(Some(Value::Integer(held.len() as i64))),
            [Value::Array(items)] => {
                Ok(Some(Value::Integer(items.len() as i64)))
            }
            _ => Err(format!(
                "'{name}' reads the length of a run, and this is not one"
            )),
        };
    }
    if !matches!(name, "wrap_add" | "wrap_sub" | "wrap_mul") {
        return Ok(None);
    }
    let [Value::Integer(left), Value::Integer(right)] = given else {
        return Err(format!(
            "'{name}' keeps the low bits of two whole numbers, and this is not two of them"
        ));
    };
    Ok(Some(Value::Integer(match name {
        "wrap_add" => left.wrapping_add(*right),
        "wrap_sub" => left.wrapping_sub(*right),
        _ => left.wrapping_mul(*right),
    })))
}

// Whether an arm's pattern covers the value the subject worked out to. A
// pattern is a whole number, a yes or no, a span of them, or `_`, which is what
// a `case` may name anywhere: text belongs in an `if` and a variant is not a
// compile-time value.
fn covers(
    ast: &Ast,
    pattern: PatternId,
    held: &Value,
) -> Result<bool, String> {
    match ast.pattern(pattern) {
        Pattern::Wildcard => Ok(true),
        Pattern::Or(alternatives) => {
            for alternative in ast.patterns_in(*alternatives) {
                if covers(ast, *alternative, held)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Pattern::Range {
            low,
            high,
            inclusive,
        } => match held {
            Value::Integer(value) => Ok(value >= low
                && (value < high || (*inclusive && value == high))),
            held => Err(format!(
                "a span of whole numbers covers whole numbers, and this is {}",
                held.describe()
            )),
        },
        Pattern::Literal(Literal::Integer(written)) => {
            Ok(matches!(held, Value::Integer(value) if value == written))
        }
        Pattern::Literal(Literal::Boolean(written)) => {
            Ok(matches!(held, Value::Boolean(value) if value == written))
        }
        Pattern::Literal(Literal::Float(_) | Literal::Float32(_)) => {
            Err(FRACTION.to_string())
        }
        _ => Ok(false),
    }
}

// Where an index lands, or what is wrong with it. Bounds are decided here,
// where the index is written, so reading past the end is a compile error rather
// than an abort the program was going to reach.
fn in_range(index: i64, count: usize) -> Result<usize, String> {
    if count == 0 {
        return Err(format!(
            "this reads item {index} of a run of nothing, which has no items"
        ));
    }
    let last = count - 1;
    match usize::try_from(index) {
        Ok(at) if at < count => Ok(at),
        _ => Err(format!(
            "this reads item {index} of a run of {count}, whose items are numbered 0 to {last}"
        )),
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
        (Value::Boolean(left), Value::Boolean(right)) => match operator {
            Operator::Equal => Ok(Value::Boolean(left == right)),
            Operator::NotEqual => Ok(Value::Boolean(left != right)),
            _ => Err("a yes or no is compared and nothing else".to_string()),
        },
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
        // The remainder of the smallest by minus one is nothing, and that is
        // what the machine answers. Only the quotient leaves the range, so
        // refusing the remainder here would refuse arithmetic the program
        // would have carried out.
        Operator::Modulo if right == -1 => Value::Integer(0),
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
        Expression::Try(_) => "a `?`",
        Expression::ArrayRepeat(..) => "an array",
        Expression::Function(..) | Expression::Proc(..) => "a function value",
        _ => "this",
    }
}
