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
        moved: HashSet::new(),
        linear,
        signatures,
        linear_declared: Vec::new(),
    };
    for parameter in params {
        if let Some(ty) = &parameter.type_annotation {
            checker.note_binding(&parameter.name, Some(ty.clone()));
        }
    }
    checker.check_function_body(body)?;
    for name in &checker.linear_declared {
        if !checker.moved.contains(name) {
            bail!(
                "ownership: linear value '{name}' is never consumed; a linear resource must be moved exactly once"
            );
        }
    }
    Ok(())
}

struct MoveChecker<'a> {
    types: HashMap<String, Type>,
    moved: HashSet<String>,
    linear: &'a HashSet<String>,
    signatures: &'a Signatures,
    linear_declared: Vec<String>,
}

impl MoveChecker<'_> {
    fn note_binding(&mut self, name: &str, ty: Option<Type>) {
        self.moved.remove(name);
        match ty {
            Some(ty) => {
                if is_linear_type(&ty, self.linear) {
                    self.linear_declared.push(name.to_string());
                }
                self.types.insert(name.to_string(), ty);
            }
            None => {
                self.types.remove(name);
            }
        }
    }

    fn check_block(&mut self, block: &Block) -> Result<()> {
        for statement in block {
            locate(self.check_statement(&statement.node), statement.position)?;
        }
        Ok(())
    }

    fn check_function_body(&mut self, block: &Block) -> Result<()> {
        for (index, statement) in block.iter().enumerate() {
            let is_last = index + 1 == block.len();
            let position = statement.position;
            if is_last
                && let Statement::Expression(expression) = &statement.node
            {
                locate(self.visit(expression, true), position)?;
            } else {
                locate(self.check_statement(&statement.node), position)?;
            }
        }
        Ok(())
    }

    fn check_statement(&mut self, statement: &Statement) -> Result<()> {
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
                Ok(())
            }
            Statement::Constant(
                _,
                Expression::Function(..) | Expression::Proc(..),
            ) => Ok(()),
            Statement::Constant(name, value) => {
                self.visit(value, true)?;
                let inferred =
                    infer_type(None, value, &self.types, self.signatures);
                self.note_binding(name, inferred);
                Ok(())
            }
            Statement::Assignment(target, value) => {
                self.visit(value, true)?;
                if let Expression::Identifier(name) = target {
                    self.moved.remove(name);
                } else {
                    self.visit(target, false)?;
                }
                Ok(())
            }
            Statement::Return(expression) => self.visit(expression, true),
            Statement::Expression(expression) => self.visit(expression, false),
            Statement::While(condition, body) => {
                self.visit(condition, false)?;
                self.check_block(body)
            }
            Statement::For(variable, range, body) => {
                self.visit(range, false)?;
                self.note_binding(variable, Some(Type::I64));
                self.check_block(body)
            }
            Statement::Defer(inner) => self.check_statement(inner),
            _ => Ok(()),
        }
    }

    fn visit(&mut self, expression: &Expression, moving: bool) -> Result<()> {
        match expression {
            Expression::Identifier(name) => {
                if self.moved.contains(name) {
                    bail!("ownership: use of moved value '{name}'");
                }
                if moving && self.is_move_variable(name) {
                    self.moved.insert(name.clone());
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
                    && let Some(pool_borrows) = pool_operation_borrows(name)
                {
                    for (index, argument) in arguments.iter().enumerate() {
                        self.visit(argument, !(pool_borrows && index == 0))?;
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
            Expression::If(condition, consequence, alternative) => {
                self.visit(condition, false)?;
                self.check_block(consequence)?;
                if let Some(alternative) = alternative {
                    self.check_block(alternative)?;
                }
                Ok(())
            }
            Expression::Switch(scrutinee, cases) => {
                self.visit(scrutinee, false)?;
                if let Expression::Identifier(name) = scrutinee.as_ref()
                    && self.is_linear_variable(name)
                {
                    self.moved.insert(name.clone());
                }
                for case in cases {
                    self.check_block(&case.body)?;
                }
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

/// A pool operation's first argument is the pool. `pool_destroy` consumes it;
/// every other operation borrows it (the pool stays usable). Returns `None` for
/// names that are not pool operations, so their arguments move normally.
fn pool_operation_borrows(name: &str) -> Option<bool> {
    match name {
        "pool_alloc" | "pool_contains" | "pool_free" | "pool_get" => Some(true),
        "pool_destroy" => Some(false),
        _ => None,
    }
}

fn is_linear_type(ty: &Type, linear: &HashSet<String>) -> bool {
    match ty {
        Type::Struct(name) | Type::Enum(name) => linear.contains(name),
        Type::Pool(_) => true,
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
    fn unconsumed_linear_resource_is_rejected() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            open :: fn() -> File { File { handle = 1 } }\n\
            run :: fn() {\n\
                f := open()\n\
            }";
        assert!(check(source).is_err());
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
    fn ignored_linear_error_enum_is_rejected() {
        let source = "\
            Outcome :: linear enum { Ok { value: i64 }, Err { code: i64 } }\n\
            run_step :: fn() -> Outcome { Outcome::Ok { value = 1 } }\n\
            caller :: fn() -> i64 {\n\
                result := run_step()\n\
                7\n\
            }";
        assert!(check(source).is_err());
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
    fn undestroyed_pool_is_rejected() {
        let source = "\
            Entity :: struct { hp: i64 }\n\
            run :: fn() {\n\
                world : Pool<Entity> = pool_new($Entity, 4)\n\
            }";
        assert!(check(source).is_err());
    }

    #[test]
    fn destroyed_pool_is_accepted() {
        let source = "\
            Entity :: struct { hp: i64 }\n\
            run :: fn() {\n\
                world : Pool<Entity> = pool_new($Entity, 4)\n\
                pool_destroy(world)\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn pool_can_be_used_repeatedly_before_destroy() {
        let source = "\
            Entity :: struct { hp: i64 }\n\
            run :: fn() {\n\
                world : Pool<Entity> = pool_new($Entity, 4)\n\
                a := pool_alloc(world, Entity { hp = 1 })\n\
                b := pool_alloc(world, Entity { hp = 2 })\n\
                pool_destroy(world)\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn pool_use_after_destroy_is_rejected() {
        let source = "\
            Entity :: struct { hp: i64 }\n\
            run :: fn() {\n\
                world : Pool<Entity> = pool_new($Entity, 4)\n\
                pool_destroy(world)\n\
                a := pool_alloc(world, Entity { hp = 1 })\n\
            }";
        assert!(check(source).is_err());
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
}
