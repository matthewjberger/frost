use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::lexer::Position;
use crate::parser::{
    Block, Expression, Literal, Parameter, Spanned, Statement,
};
use crate::types::Type;

type Signatures = HashMap<String, Type>;

fn locate<T>(result: Result<T>, position: Position) -> Result<T> {
    result.map_err(|error| {
        let text = error.to_string();
        if position == Position::default() || text.starts_with("at line ") {
            error
        } else {
            anyhow::anyhow!(
                "at line {}, column {}: {text}",
                position.line,
                position.column
            )
        }
    })
}

pub fn check_ownership(
    statements: &[Spanned<Statement>],
    linear: &HashSet<String>,
) -> Result<()> {
    let signatures = collect_signatures(statements);
    for statement in statements {
        locate(
            check_statement(&statement.node, linear, &signatures),
            statement.position,
        )?;
    }
    Ok(())
}

fn collect_signatures(statements: &[Spanned<Statement>]) -> Signatures {
    let mut signatures = HashMap::new();
    for statement in statements {
        match &statement.node {
            Statement::Constant(
                name,
                Expression::Function(_, return_sig, _)
                | Expression::Proc(_, return_sig, _),
            ) => {
                signatures.insert(
                    name.clone(),
                    return_sig.to_type().unwrap_or(Type::Void),
                );
            }
            Statement::Extern {
                name, return_type, ..
            } => {
                signatures.insert(
                    name.clone(),
                    return_type.clone().unwrap_or(Type::Void),
                );
            }
            _ => {}
        }
    }
    signatures
}

fn check_statement(
    statement: &Statement,
    linear: &HashSet<String>,
    signatures: &Signatures,
) -> Result<()> {
    match statement {
        Statement::Struct(name, _, fields) => {
            for field in fields {
                if field.field_type.contains_reference() {
                    bail!(
                        "ownership: cannot store a reference in struct '{name}' (field '{}'); references are second-class",
                        field.name
                    );
                }
            }
        }
        Statement::Enum(name, variants) => {
            for variant in variants {
                let Some(fields) = &variant.fields else {
                    continue;
                };
                for field in fields {
                    if field.field_type.contains_reference() {
                        bail!(
                            "ownership: cannot store a reference in enum '{name}' (variant '{}', field '{}'); references are second-class",
                            variant.name,
                            field.name
                        );
                    }
                }
            }
        }
        Statement::Constant(
            name,
            Expression::Function(params, return_sig, body),
        )
        | Statement::Constant(
            name,
            Expression::Proc(params, return_sig, body),
        ) => {
            if let Some(reference) = return_sig.contains_reference() {
                bail!(
                    "ownership: function '{name}' cannot return the reference type '{reference}'; references are second-class"
                );
            }
            for inner in body {
                check_statement(inner, linear, signatures)?;
            }
            check_function_moves(params, body, linear, signatures)?;
        }
        Statement::Extern {
            name, return_type, ..
        } => {
            if let Some(return_type) = return_type
                && return_type.contains_reference()
            {
                bail!(
                    "ownership: extern function '{name}' cannot return a reference"
                );
            }
        }
        _ => {}
    }
    Ok(())
}

fn check_function_moves(
    params: &[Parameter],
    body: &Block,
    linear: &HashSet<String>,
    signatures: &Signatures,
) -> Result<()> {
    let mut checker = MoveChecker {
        types: HashMap::new(),
        states: HashMap::new(),
        linear,
        signatures,
        in_defer: false,
    };
    for parameter in params {
        if let Some(ty) = &parameter.type_annotation {
            checker.note_binding(&parameter.name, Some(ty.clone()));
        }
    }
    checker.check_function_body(body)?;
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MoveState {
    Live,
    Moved,
    MaybeMoved,
    Deferred,
}

fn join_state(left: MoveState, right: MoveState) -> MoveState {
    if left == right {
        left
    } else {
        MoveState::MaybeMoved
    }
}

struct MoveChecker<'a> {
    types: HashMap<String, Type>,
    states: HashMap<String, MoveState>,
    linear: &'a HashSet<String>,
    signatures: &'a Signatures,
    in_defer: bool,
}

impl MoveChecker<'_> {
    fn note_binding(&mut self, name: &str, ty: Option<Type>) {
        self.states.insert(name.to_string(), MoveState::Live);
        match ty {
            Some(ty) => {
                self.types.insert(name.to_string(), ty);
            }
            None => {
                self.types.remove(name);
            }
        }
    }

    fn state_of(&self, name: &str) -> MoveState {
        self.states.get(name).copied().unwrap_or(MoveState::Live)
    }

    fn check_block(&mut self, block: &Block) -> Result<bool> {
        let mut diverges = false;
        for statement in block {
            diverges = locate(
                self.check_statement(&statement.node),
                statement.position,
            )?;
            if diverges {
                break;
            }
        }
        Ok(diverges)
    }

    fn check_function_body(&mut self, block: &Block) -> Result<bool> {
        let mut diverges = false;
        for (index, statement) in block.iter().enumerate() {
            let is_last = index + 1 == block.len();
            let position = statement.position;
            if is_last
                && let Statement::Expression(expression) = &statement.node
            {
                if matches!(
                    expression,
                    Expression::If(..) | Expression::Switch(..)
                ) {
                    diverges =
                        locate(self.check_conditional(expression), position)?;
                } else {
                    locate(self.visit(expression, true), position)?;
                }
            } else {
                diverges =
                    locate(self.check_statement(&statement.node), position)?;
                if diverges {
                    break;
                }
            }
        }
        Ok(diverges)
    }

    fn check_statement(&mut self, statement: &Statement) -> Result<bool> {
        match statement {
            Statement::Let {
                name,
                type_annotation,
                value,
                ..
            } => {
                self.visit(value, true)?;
                let inferred = infer_type(
                    type_annotation.as_ref(),
                    value,
                    &self.types,
                    self.signatures,
                );
                self.note_binding(name, inferred);
                Ok(false)
            }
            Statement::Constant(
                _,
                Expression::Function(..) | Expression::Proc(..),
            ) => Ok(false),
            Statement::Constant(name, value) => {
                self.visit(value, true)?;
                let inferred =
                    infer_type(None, value, &self.types, self.signatures);
                self.note_binding(name, inferred);
                Ok(false)
            }
            Statement::Assignment(target, value) => {
                self.visit(value, true)?;
                if let Expression::Identifier(name) = target {
                    self.states.insert(name.clone(), MoveState::Live);
                } else {
                    self.visit(target, false)?;
                }
                Ok(false)
            }
            Statement::Return(expression) => {
                self.visit(expression, true)?;
                Ok(true)
            }
            Statement::Expression(expression) => {
                if matches!(
                    expression,
                    Expression::If(..) | Expression::Switch(..)
                ) {
                    self.check_conditional(expression)
                } else {
                    self.visit(expression, false)?;
                    Ok(false)
                }
            }
            Statement::While(condition, body) => {
                self.visit(condition, false)?;
                self.check_loop_body(body)?;
                Ok(false)
            }
            Statement::For(variable, range, body) => {
                self.visit(range, false)?;
                self.note_binding(variable, Some(Type::I64));
                self.check_loop_body(body)?;
                Ok(false)
            }
            Statement::Defer(inner) => {
                let was_in_defer = self.in_defer;
                self.in_defer = true;
                let result = self.check_statement(inner);
                self.in_defer = was_in_defer;
                result?;
                Ok(false)
            }
            Statement::Break | Statement::Continue => Ok(true),
            _ => Ok(false),
        }
    }

    fn check_loop_body(&mut self, body: &Block) -> Result<()> {
        let before = self.states.clone();
        self.check_block(body)?;
        for name in before.keys() {
            let previous = before.get(name).copied().unwrap_or(MoveState::Live);
            if previous == MoveState::Live
                && self.state_of(name) != MoveState::Live
                && self.is_move_variable(name)
            {
                if self.is_linear_variable(name) {
                    bail!(
                        "ownership: linear value '{name}' is consumed inside a loop; a linear resource must be consumed exactly once, not once per iteration"
                    );
                }
                bail!(
                    "ownership: value '{name}' is moved inside a loop; it would be used after move on a later iteration"
                );
            }
        }
        self.states = before;
        Ok(())
    }

    fn check_conditional(&mut self, expression: &Expression) -> Result<bool> {
        match expression {
            Expression::If(condition, consequence, alternative) => {
                self.check_if(condition, consequence, alternative.as_ref())
            }
            Expression::Switch(scrutinee, cases) => {
                self.check_switch(scrutinee, cases)
            }
            _ => {
                self.visit(expression, false)?;
                Ok(false)
            }
        }
    }

    fn check_arm(
        &mut self,
        block: &Block,
    ) -> Result<(HashMap<String, MoveState>, bool)> {
        let diverges = self.check_block(block)?;
        let states = self.states.clone();
        Ok((states, diverges))
    }

    fn check_if(
        &mut self,
        condition: &Expression,
        consequence: &Block,
        alternative: Option<&Block>,
    ) -> Result<bool> {
        self.visit(condition, false)?;
        let before = self.states.clone();

        let (then_states, then_diverges) = self.check_arm(consequence)?;

        self.states = before.clone();
        let (else_states, else_diverges) = match alternative {
            Some(block) => self.check_arm(block)?,
            None => (before.clone(), false),
        };

        self.states = self.merge_arms(
            &before,
            &[(then_states, then_diverges), (else_states, else_diverges)],
        );
        Ok(then_diverges && else_diverges)
    }

    fn check_switch(
        &mut self,
        scrutinee: &Expression,
        cases: &[crate::parser::SwitchCase],
    ) -> Result<bool> {
        self.visit(scrutinee, false)?;
        if let Expression::Identifier(name) = scrutinee
            && self.is_linear_variable(name)
        {
            self.states.insert(name.clone(), MoveState::Moved);
        }
        let before = self.states.clone();
        let mut arms = Vec::new();
        for case in cases {
            self.states = before.clone();
            arms.push(self.check_arm(&case.body)?);
        }
        let all_diverge =
            !arms.is_empty() && arms.iter().all(|(_, diverges)| *diverges);
        self.states = self.merge_arms(&before, &arms);
        Ok(all_diverge)
    }

    fn merge_arms(
        &self,
        before: &HashMap<String, MoveState>,
        arms: &[(HashMap<String, MoveState>, bool)],
    ) -> HashMap<String, MoveState> {
        let live: Vec<&HashMap<String, MoveState>> = arms
            .iter()
            .filter(|(_, diverges)| !diverges)
            .map(|(states, _)| states)
            .collect();
        if live.is_empty() {
            return before.clone();
        }
        let mut result = before.clone();
        for name in before.keys() {
            let mut merged: Option<MoveState> = None;
            for states in &live {
                let state =
                    states.get(name).copied().unwrap_or(MoveState::Live);
                merged = Some(match merged {
                    Some(previous) => join_state(previous, state),
                    None => state,
                });
            }
            if let Some(state) = merged {
                result.insert(name.clone(), state);
            }
        }
        result
    }

    fn visit(&mut self, expression: &Expression, moving: bool) -> Result<()> {
        match expression {
            Expression::Identifier(name) => {
                match self.state_of(name) {
                    MoveState::Live => {
                        if moving && self.is_move_variable(name) {
                            let consumed = if self.in_defer {
                                MoveState::Deferred
                            } else {
                                MoveState::Moved
                            };
                            self.states.insert(name.clone(), consumed);
                        }
                    }
                    MoveState::Deferred => {
                        if moving {
                            bail!(
                                "ownership: value '{name}' is already scheduled for consumption by a later defer; it cannot be moved again"
                            );
                        }
                    }
                    MoveState::Moved | MoveState::MaybeMoved => {
                        bail!("ownership: use of moved value '{name}'");
                    }
                }
                Ok(())
            }
            Expression::Borrow(inner)
            | Expression::BorrowMut(inner)
            | Expression::AddressOf(inner)
            | Expression::Dereference(inner) => self.visit(inner, false),
            Expression::FieldAccess(base, _) => self.visit(base, false),
            Expression::Index(base, index) => {
                self.visit(base, false)?;
                self.visit(index, false)
            }
            Expression::Prefix(_, operand) => self.visit(operand, false),
            Expression::Infix(left, _, right) => {
                self.visit(left, false)?;
                self.visit(right, false)
            }
            Expression::Call(callee, arguments) => {
                self.visit(callee, false)?;
                check_borrow_exclusivity(arguments)?;
                if let Expression::Identifier(name) = callee.as_ref()
                    && let Some(borrows) = builtin_borrows_first_argument(name)
                {
                    for (index, argument) in arguments.iter().enumerate() {
                        self.visit(argument, !(borrows && index == 0))?;
                    }
                    return Ok(());
                }
                for argument in arguments {
                    self.visit(argument, true)?;
                }
                Ok(())
            }
            Expression::StructInit(_, fields) => {
                for (_, value) in fields {
                    self.visit(value, true)?;
                }
                Ok(())
            }
            Expression::EnumVariantInit(_, _, fields) => {
                for (_, value) in fields {
                    self.visit(value, true)?;
                }
                Ok(())
            }
            Expression::Literal(Literal::Array(elements)) => {
                for element in elements {
                    self.visit(element, true)?;
                }
                Ok(())
            }
            Expression::If(..) | Expression::Switch(..) => {
                self.check_conditional(expression)?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn is_linear_variable(&self, name: &str) -> bool {
        self.types
            .get(name)
            .map(|ty| is_linear_type(ty, self.linear))
            .unwrap_or(false)
    }

    fn is_move_variable(&self, name: &str) -> bool {
        self.types
            .get(name)
            .map(|ty| !ty.is_copy())
            .unwrap_or(false)
    }
}

fn check_borrow_exclusivity(arguments: &[Expression]) -> Result<()> {
    let mut mutable: Vec<&str> = Vec::new();
    let mut shared: Vec<&str> = Vec::new();
    for argument in arguments {
        match argument {
            Expression::BorrowMut(inner) => {
                if let Expression::Identifier(name) = &**inner {
                    mutable.push(name);
                }
            }
            Expression::Borrow(inner) => {
                if let Expression::Identifier(name) = &**inner {
                    shared.push(name);
                }
            }
            _ => {}
        }
    }
    for (index, name) in mutable.iter().enumerate() {
        if mutable.iter().skip(index + 1).any(|other| other == name) {
            bail!(
                "ownership: '{name}' is borrowed as mutable more than once in a single call; mutable borrows are exclusive"
            );
        }
        if shared.iter().any(|other| other == name) {
            bail!(
                "ownership: '{name}' is borrowed as both shared and mutable in a single call; mutable borrows are exclusive"
            );
        }
    }
    Ok(())
}

/// `ptr_to` takes the address of its argument, which borrows rather than moves
/// it. Returns `Some(true)` for that borrow, and `None` for ordinary calls whose
/// arguments move normally.
fn builtin_borrows_first_argument(name: &str) -> Option<bool> {
    match name {
        "ptr_to" => Some(true),
        _ => None,
    }
}

fn is_linear_type(ty: &Type, linear: &HashSet<String>) -> bool {
    match ty {
        Type::Struct(name) | Type::Enum(name) => linear.contains(name),
        _ => false,
    }
}

fn infer_type(
    annotation: Option<&Type>,
    value: &Expression,
    types: &HashMap<String, Type>,
    signatures: &Signatures,
) -> Option<Type> {
    if let Some(ty) = annotation {
        return Some(ty.clone());
    }
    match value {
        Expression::StructInit(name, _) => Some(Type::Struct(name.clone())),
        Expression::EnumVariantInit(name, _, _) => {
            Some(Type::Enum(name.clone()))
        }
        Expression::Literal(Literal::String(_)) => Some(Type::Str),
        Expression::Literal(Literal::Integer(_)) => Some(Type::I64),
        Expression::Literal(Literal::Float(_)) => Some(Type::F64),
        Expression::Literal(Literal::Float32(_)) => Some(Type::F32),
        Expression::Literal(Literal::Boolean(_)) | Expression::Boolean(_) => {
            Some(Type::Bool)
        }
        Expression::Identifier(name) => types.get(name).cloned(),
        Expression::Call(callee, _) => {
            if let Expression::Identifier(name) = &**callee {
                signatures.get(name).cloned()
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Lexer, Parser};

    fn check(source: &str) -> Result<()> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let statements = parser.parse()?;
        let linear = parser.linear_types().clone();
        check_ownership(&statements, &linear)
    }

    #[test]
    fn reference_in_struct_is_rejected() {
        let source = "Bad :: struct { r: &i64 }";
        assert!(check(source).is_err());
    }

    #[test]
    fn reference_in_enum_is_rejected() {
        let source = "Bad :: enum { Holder { r: &mut i64 } }";
        assert!(check(source).is_err());
    }

    #[test]
    fn returning_a_reference_is_rejected() {
        let source = "bad :: fn(x: &i64) -> &i64 { x }";
        assert!(check(source).is_err());
    }

    #[test]
    fn reference_parameters_are_allowed() {
        let source = "read :: fn(x: &i64) -> i64 { x^ }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn owned_struct_is_allowed() {
        let source = "Point :: struct { x: i64, y: i64 }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn handles_can_be_stored_and_returned() {
        let source = "Store :: struct { h: Handle<i64> }\nget :: fn() -> Handle<i64> { make() }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn use_after_move_of_struct_is_rejected() {
        let source = "\
            Point :: struct { x: i64, y: i64 }\n\
            take :: fn(p: Point) -> i64 { p.x }\n\
            run :: fn() -> i64 {\n\
                p := Point { x = 1, y = 2 }\n\
                a := take(p)\n\
                b := take(p)\n\
                a + b\n\
            }";
        assert!(check(source).is_err());
    }

    #[test]
    fn copy_values_can_be_reused() {
        let source = "\
            add :: fn(a: i64, b: i64) -> i64 { a + b }\n\
            run :: fn() -> i64 {\n\
                x : i64 = 5\n\
                a := add(x, x)\n\
                b := add(x, x)\n\
                a + b\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn borrowing_does_not_move() {
        let source = "\
            Point :: struct { x: i64, y: i64 }\n\
            read :: fn(p: &Point) -> i64 { p.x }\n\
            run :: fn() -> i64 {\n\
                p := Point { x = 1, y = 2 }\n\
                a := read(&p)\n\
                b := read(&p)\n\
                a + b\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn consumed_linear_resource_is_accepted() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            open :: fn() -> File { File { handle = 1 } }\n\
            close :: extern fn(f: File)\n\
            run :: fn() {\n\
                f := open()\n\
                close(f)\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn linear_resource_used_twice_is_rejected() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            open :: fn() -> File { File { handle = 1 } }\n\
            close :: extern fn(f: File)\n\
            run :: fn() {\n\
                f := open()\n\
                close(f)\n\
                close(f)\n\
            }";
        assert!(check(source).is_err());
    }

    #[test]
    fn aliased_mutable_borrows_are_rejected() {
        let source = "\
            add :: fn(a: &mut i64, b: &mut i64) { }\n\
            run :: fn() {\n\
                mut x : i64 = 0\n\
                add(&mut x, &mut x)\n\
            }";
        assert!(check(source).is_err());
    }

    #[test]
    fn shared_and_mutable_borrow_of_same_is_rejected() {
        let source = "\
            mix :: fn(a: &i64, b: &mut i64) { }\n\
            run :: fn() {\n\
                mut x : i64 = 0\n\
                mix(&x, &mut x)\n\
            }";
        assert!(check(source).is_err());
    }

    #[test]
    fn distinct_mutable_borrows_are_allowed() {
        let source = "\
            add :: fn(a: &mut i64, b: &mut i64) { }\n\
            run :: fn() {\n\
                mut x : i64 = 0\n\
                mut y : i64 = 0\n\
                add(&mut x, &mut y)\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn multiple_shared_borrows_are_allowed() {
        let source = "\
            sum :: fn(a: &i64, b: &i64) -> i64 { a^ + b^ }\n\
            run :: fn() -> i64 {\n\
                x : i64 = 7\n\
                sum(&x, &x)\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn matching_a_linear_error_enum_consumes_it() {
        let source = "\
            Outcome :: linear enum { Ok { value: i64 }, Err { code: i64 } }\n\
            run_step :: fn() -> Outcome { Outcome::Ok { value = 1 } }\n\
            caller :: fn() -> i64 {\n\
                result := run_step()\n\
                match result {\n\
                    case .Ok { value }: value\n\
                    case .Err { code }: 0 - code\n\
                }\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn a_linear_destructor_may_be_written_in_frost() {
        // The destructor takes the linear value by value and unpacks it; the
        // parameter is not passed on, and that is allowed.
        let source = "\
            Arena :: linear struct { data: i64 }\n\
            free :: extern fn(handle: i64)\n\
            make :: fn() -> Arena { Arena { data = 1 } }\n\
            destroy :: fn(a: Arena) { free(a.data) }\n\
            run :: fn() {\n\
                a := make()\n\
                destroy(a)\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn returning_a_linear_resource_consumes_it() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            open :: fn() -> File { File { handle = 1 } }\n\
            forward :: fn() -> File {\n\
                f := open()\n\
                f\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn linear_consumed_on_both_if_branches_is_accepted() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            open :: fn() -> File { File { handle = 1 } }\n\
            close :: extern fn(f: File)\n\
            run :: fn() {\n\
                f := open()\n\
                if (1 == 1) { close(f) } else { close(f) }\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn linear_consumed_on_every_match_arm_is_accepted() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            Flag :: enum { A, B }\n\
            open :: fn() -> File { File { handle = 1 } }\n\
            close :: extern fn(f: File)\n\
            run :: fn() {\n\
                f := open()\n\
                flag := Flag::A\n\
                match flag { case .A: close(f)  case .B: close(f) }\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn linear_consumed_inside_a_loop_is_rejected() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            open :: fn() -> File { File { handle = 1 } }\n\
            close :: extern fn(f: File)\n\
            run :: fn() {\n\
                f := open()\n\
                mut i : i64 = 0\n\
                while (i < 3) { close(f)  i = i + 1 }\n\
            }";
        assert!(check(source).is_err());
    }

    #[test]
    fn reading_a_deferred_linear_value_is_accepted() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            open :: fn() -> File { File { handle = 1 } }\n\
            close :: extern fn(f: File)\n\
            run :: fn() -> i64 {\n\
                f := open()\n\
                defer close(f)\n\
                f.handle\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn moving_a_deferred_value_again_is_rejected() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            open :: fn() -> File { File { handle = 1 } }\n\
            close :: extern fn(f: File)\n\
            run :: fn() {\n\
                f := open()\n\
                defer close(f)\n\
                close(f)\n\
            }";
        assert!(check(source).is_err());
    }

    #[test]
    fn moving_an_owned_value_inside_a_loop_is_rejected() {
        let source = "\
            Point :: struct { x: i64, y: i64 }\n\
            take :: fn(p: Point) -> i64 { p.x }\n\
            run :: fn() {\n\
                p := Point { x = 1, y = 2 }\n\
                mut i : i64 = 0\n\
                while (i < 3) { take(p)  i = i + 1 }\n\
            }";
        assert!(check(source).is_err());
    }

    #[test]
    fn moving_a_value_declared_inside_a_loop_is_accepted() {
        let source = "\
            Point :: struct { x: i64, y: i64 }\n\
            take :: fn(p: Point) -> i64 { p.x }\n\
            make :: fn() -> Point { Point { x = 5, y = 6 } }\n\
            run :: fn() {\n\
                mut i : i64 = 0\n\
                while (i < 3) { p := make()  take(p)  i = i + 1 }\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn a_linear_consumed_by_defer_survives_a_nested_return() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            open :: fn() -> File { File { handle = 1 } }\n\
            close :: extern fn(f: File)\n\
            run :: fn(flag: i64) -> i64 {\n\
                f := open()\n\
                defer close(f)\n\
                if (flag == 0) { return 5 }\n\
                7\n\
            }";
        assert!(check(source).is_ok());
    }
}
