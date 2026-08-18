use std::collections::{HashMap, HashSet};

use anyhow::Result;

use crate::ast::{
    Ast, ExprId, Expression, Literal, Range32, ReturnKind, Statement, StmtId,
};
use crate::lexer::Position;
use crate::parser::{Diagnostic, ParamMode};
use crate::types::{SizeExpr, Type};

// The region check. A region is the scope in which an arena is live: the body of
// a `with arena { ... }` block, and the body of a `uses A` function (where the
// arena is the implicit capability). A raw pointer into the arena is region-bound
// and must not outlive its region.
//
// No lifetimes and no region types on pointers. Frost has no global arenas and no
// closures, so a `^T` can only point into an arena a function was handed directly
// (a parameter, a value derived from one, or a `uses` capability). That makes
// provenance a plain flow question, and the escape rule a plain scope question:
//   - inside a `with` block, an arena pointer may not be returned or stored in a
//     binding that lives past the block;
//   - inside a `uses` function, it may be returned (that hands it to the caller's
//     region, checked where the `with` block is) but not stored into a parameter.
// A pointer confined to a binding declared in the region is fine. That binding
// dies with the region.

struct Signatures {
    returns_view: HashMap<String, bool>,
    uses_arena: HashSet<String>,
    // The declared fields of every struct and what each function answers with,
    // so a read out of a value built in the region can be weighed against the
    // type it reads.
    fields: FieldTypes,
    returns: HashMap<String, Type>,
}

pub fn check_regions(ast: &Ast, roots: &[StmtId]) -> Result<()> {
    let diagnostics = check_regions_recovering(ast, roots);
    if diagnostics.is_empty() {
        return Ok(());
    }
    Err(anyhow::anyhow!(crate::flatten(&diagnostics, "\n")))
}

/// Check every region in the program, reporting each escape rather than
/// stopping at the first. Regions are independent of one another, and within
/// one an escape does not change what the rest of the block means.
pub fn check_regions_recovering(
    ast: &Ast,
    roots: &[StmtId],
) -> Vec<Diagnostic> {
    let fields = collect_field_types(ast, roots);
    let mut returns_view = HashMap::new();
    let mut uses_arena = HashSet::new();
    for statement in roots {
        if let Statement::Constant(name, value) = ast.stmt(*statement)
            && let Expression::Function(_, sig, _) | Expression::Proc(_, sig, _) =
                ast.expr(*value)
        {
            let sig = ast.signature(*sig);
            // Every view, not only a raw pointer, and a value holding one as
            // much as one written bare. A `[]T` or a `str` carved out of an
            // arena names the arena's storage exactly as a `^T` does, and a
            // container answering with itself carries the run inside it, which
            // is how a `Vec` built in a `with` block leaves the block.
            if matches!(
                &sig.kind,
                ReturnKind::Single(ty) | ReturnKind::Fallible(ty, _)
                    if holds_view(ty, &fields, &mut HashSet::new())
            ) {
                returns_view.insert(ast.name(*name).to_string(), true);
            }
            if !sig.uses.is_empty() {
                uses_arena.insert(ast.name(*name).to_string());
            }
        }
    }
    let signatures = Signatures {
        returns_view,
        uses_arena,
        returns: collect_return_types(ast, roots),
        fields,
    };

    let mut diagnostics = Vec::new();
    for statement in roots {
        if let Statement::Constant(_, value) = ast.stmt(*statement)
            && let Expression::Function(_, sig, body)
            | Expression::Proc(_, sig, body) = ast.expr(*value)
        {
            // A `uses` function's whole body is a region whose arena is the
            // implicit capability. It may return arena pointers but not leak
            // them into its parameters.
            if let Some(capability) = ast.signature(*sig).uses.first() {
                let mut region = Region::new(
                    ast,
                    capability_binding(capability),
                    &signatures,
                    true,
                );
                region.check(*body, true);
                diagnostics.append(&mut region.diagnostics);
            }
            find_regions(ast, *body, &signatures, &mut diagnostics);
        }
    }
    diagnostics
}

// Walk a block looking for `with` regions to check. An ordinary block imposes no
// region rule of its own.
fn find_regions(
    ast: &Ast,
    block: Range32,
    signatures: &Signatures,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for statement in ast.stmts_in(block) {
        match ast.stmt(*statement) {
            Statement::With(arena, body) => {
                let mut region = Region::new(
                    ast,
                    ast.name(*arena).to_string(),
                    signatures,
                    false,
                );
                region.check(*body, true);
                diagnostics.append(&mut region.diagnostics);
                find_regions(ast, *body, signatures, diagnostics);
            }
            Statement::While(_, body) | Statement::For(_, _, _, body) => {
                find_regions(ast, *body, signatures, diagnostics);
            }
            Statement::Defer(inner) | Statement::ErrDefer(inner) => {
                if let Statement::With(arena, body) = ast.stmt(*inner) {
                    let mut region = Region::new(
                        ast,
                        ast.name(*arena).to_string(),
                        signatures,
                        false,
                    );
                    region.check(*body, true);
                    diagnostics.append(&mut region.diagnostics);
                }
            }
            _ => {}
        }
    }
}

// The capability binding name for an arena type: its base name with the first
// letter lowercased, so `Arena<256>` binds `arena` (matching the allocation
// sources lowering).
pub(crate) fn capability_binding(capability: &Type) -> String {
    let name = match capability {
        Type::Struct(name) | Type::Enum(name) => name.clone(),
        other => other.to_string(),
    };
    let base = name.split('<').next().unwrap_or(&name);
    let mut characters = base.chars();
    match characters.next() {
        Some(first) => {
            first.to_lowercase().collect::<String>() + characters.as_str()
        }
        None => base.to_string(),
    }
}

/// Record an escape unless the same one is already recorded.
///
/// A block reachable both as a nested block and as the thing its enclosing
/// block answers with is walked down both roads, and the reader wants the
/// escape named once rather than once per road.
fn record_once(
    diagnostics: &mut Vec<Diagnostic>,
    at: Position,
    message: String,
) {
    let already = diagnostics
        .iter()
        .any(|held| held.position == at && held.message == message);
    if !already {
        diagnostics.push(Diagnostic {
            position: at,
            message,
            related: Vec::new(),
        });
    }
}

/// The expression an `unsafe` block answers with.
///
/// `ptr_to` and `slice_from` are refused outside such a block, so every pointer
/// a program can write leaves its frame or its region wrapped in one.
/// A check that does not look through the block does not see the pointer at all.
fn block_value(ast: &Ast, block: Range32) -> Option<ExprId> {
    match ast.stmt(*ast.stmts_in(block).last()?) {
        Statement::Expression(value) => Some(*value),
        _ => None,
    }
}

// The root variable a place is rooted at, so `s.field` and `xs[i]` are rooted at
// `s` and `xs`.
fn root_identifier(ast: &Ast, place: ExprId) -> Option<&str> {
    match ast.expr(place) {
        Expression::Identifier(name) => Some(ast.name(*name)),
        Expression::FieldAccess(base, _)
        | Expression::Dereference(base)
        | Expression::Index(base, _) => root_identifier(ast, *base),
        _ => None,
    }
}

struct Region<'a> {
    ast: &'a Ast,
    arena: String,
    signatures: &'a Signatures,
    // Whether a returned arena pointer is allowed (true in a `uses` body, false
    // in a `with` block).
    allow_return: bool,
    // Bindings declared inside the region. They die with it, so they may hold a
    // region pointer.
    inner: HashSet<String>,
    // Bindings that hold, or hold somewhere inside them, a region pointer. A
    // whole binding rather than the field carrying it, since a struct travels
    // as one value; which of its fields carries the storage is a question the
    // read asks of the field's type.
    bound: HashSet<String>,
    // Bindings holding the address of one of those, so reading back through one
    // hands the region pointer out again. This is what tells `pp^` from `p^`
    // without types: `pp` was taken from something already bound, `p` was not.
    via_pointer: HashSet<String>,
    // What each binding declared in the region holds, so a read of a field or an
    // element can be weighed against the type it reads.
    types: HashMap<String, Type>,
    diagnostics: Vec<Diagnostic>,
}

impl<'a> Region<'a> {
    fn new(
        ast: &'a Ast,
        arena: String,
        signatures: &'a Signatures,
        allow_return: bool,
    ) -> Self {
        Region {
            ast,
            arena,
            signatures,
            allow_return,
            inner: HashSet::new(),
            bound: HashSet::new(),
            via_pointer: HashSet::new(),
            types: HashMap::new(),
            diagnostics: Vec::new(),
        }
    }

    // Whether reading this place hands the region pointer back out. The root
    // has to hold one, and what is read has to be able to carry it: a number
    // read out of a struct that also holds arena storage is a number, and
    // reading it is how a count leaves a `with` block. A read whose type the
    // walk cannot name is refused, since it could be the storage itself.
    fn reads_out(&self, place: ExprId) -> bool {
        let Some(root) = root_identifier(self.ast, place) else {
            return false;
        };
        if !self.bound.contains(root) {
            return false;
        }
        match self.place_type(place) {
            Some(read) => {
                holds_view(&read, &self.signatures.fields, &mut HashSet::new())
            }
            None => true,
        }
    }

    fn place_type(&self, place: ExprId) -> Option<Type> {
        place_type(self.ast, &self.types, &self.signatures.fields, place)
    }

    // What a binding declared in the region holds, so a later read of one of
    // its fields can be weighed against the field's type.
    fn record_type(
        &mut self,
        name: &str,
        annotation: Option<&Type>,
        value: ExprId,
    ) {
        let held = match annotation {
            Some(declared) => Some(declared.clone()),
            None => value_type(
                self.ast,
                &self.types,
                &self.signatures.fields,
                &self.signatures.returns,
                value,
            ),
        };
        if let Some(held) = held {
            self.types.insert(name.to_string(), held);
        }
    }

    // Whether a value is the address of a binding that already holds a region
    // pointer, which is the only way a dereference can hand one back out.
    fn points_at_region_pointer(&self, value: ExprId) -> bool {
        let ast = self.ast;
        match ast.expr(value) {
            Expression::Unsafe(body) => block_value(ast, *body)
                .is_some_and(|inner| self.points_at_region_pointer(inner)),
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(function) = ast.expr(*callee) else {
                    return false;
                };
                ast.name(*function) == "ptr_to"
                    && ast.exprs_in(*arguments).iter().any(|argument| {
                        root_identifier(ast, *argument)
                            .is_some_and(|root| self.bound.contains(root))
                    })
            }
            // Listed rather than caught by `_`, so a new expression form is a compile error here instead of quietly answering that it points nowhere.
            Expression::Identifier(_)
            | Expression::Literal(_)
            | Expression::ArrayRepeat(..)
            | Expression::Boolean(_)
            | Expression::TypeValue(_)
            | Expression::PackMap(..)
            | Expression::Prefix(..)
            | Expression::Infix(..)
            | Expression::Range(..)
            | Expression::Function(..)
            | Expression::Proc(..)
            | Expression::UnsafeFn(_)
            | Expression::Try(_)
            | Expression::Dereference(_)
            | Expression::Index(..)
            | Expression::FieldAccess(..)
            | Expression::AddressOf(_)
            | Expression::Borrow(_)
            | Expression::BorrowMut(_)
            | Expression::StructInit(..)
            | Expression::EnumVariantInit(..)
            | Expression::Tuple(_)
            | Expression::If(..)
            | Expression::Switch(..) => false,
        }
    }

    fn check(&mut self, block: Range32, root: bool) {
        let ast = self.ast;
        for statement in ast.stmts_in(block) {
            let at = ast.stmt_position(*statement);
            self.check_statement(*statement, at);
        }
        // The block's trailing expression is its value. In a `with` block that
        // value flows to the enclosing scope, so an arena pointer there escapes.
        if root
            && !self.allow_return
            && let Some(last) = ast.stmts_in(block).last()
            && let Statement::Expression(value) = ast.stmt(*last)
            && self.is_region_pointer(*value)
        {
            self.escape("being the block's value", ast.stmt_position(*last));
        }
    }

    fn check_statement(&mut self, statement: StmtId, at: Position) {
        let ast = self.ast;
        match ast.stmt(statement) {
            Statement::Let {
                name,
                value,
                type_annotation,
                ..
            } => {
                let name = ast.name(*name).to_string();
                self.inner.insert(name.clone());
                self.record_type(&name, type_annotation.as_ref(), *value);
                if self.is_region_pointer(*value) {
                    self.bound.insert(name.clone());
                }
                if self.points_at_region_pointer(*value) {
                    self.via_pointer.insert(name);
                }
            }
            Statement::Constant(name, value) => {
                let name = ast.name(*name).to_string();
                self.inner.insert(name.clone());
                self.record_type(&name, None, *value);
                if self.is_region_pointer(*value) {
                    self.bound.insert(name.clone());
                }
                if self.points_at_region_pointer(*value) {
                    self.via_pointer.insert(name);
                }
            }
            Statement::Assignment(place, value) => {
                if self.is_region_pointer(*value) {
                    self.bind_or_escape(*place, at);
                }
            }
            Statement::Return(value) => {
                if self.is_region_pointer(*value) && !self.allow_return {
                    self.escape("being returned", at);
                }
            }
            Statement::While(_, body) => self.check(*body, false),
            Statement::For(variable, _, _, body) => {
                self.inner.insert(ast.name(*variable).to_string());
                self.check(*body, false);
            }
            Statement::With(_, body) => self.check(*body, false),
            Statement::Expression(value) => {
                self.check_conditional(*value);
            }
            // A deferred statement runs at scope exit, inside the region
            // still, so what it writes is written from in here.
            Statement::Defer(inner) | Statement::ErrDefer(inner) => {
                self.check_statement(*inner, at);
            }
            // The multiple-return lowering runs before this check, so this
            // shape is already gone. Named rather than left to a wildcard so
            // it stays that way on purpose.
            Statement::LetMultiple(bindings, value) => {
                let held = self.is_region_pointer(*value);
                for binding in ast.bindings_in(*bindings) {
                    let name = ast.name(binding.name).to_string();
                    self.inner.insert(name.clone());
                    if held {
                        self.bound.insert(name);
                    }
                }
            }
            // Declarations and control transfers store nothing. Listed rather
            // than caught by `_`, so a new statement form is a compile error
            // here instead of a road out of the region nobody walked.
            Statement::Struct(..)
            | Statement::Enum(..)
            | Statement::Flags(..)
            | Statement::TypeAlias(..)
            | Statement::Break
            | Statement::Continue
            | Statement::Import(..)
            | Statement::Extern { .. }
            | Statement::Declared { .. } => {}
        }
    }

    // Storing a region pointer into a place declared inside the region keeps it
    // in the region, and that place now holds it. Storing it anywhere else
    // escapes: out of a `with` block that is a binding the block outlives, and
    // out of a `uses` function it is a parameter, whose frame outlives the call.
    fn bind_or_escape(&mut self, place: ExprId, at: Position) {
        match root_identifier(self.ast, place) {
            Some(root) if self.inner.contains(root) => {
                self.bound.insert(root.to_string());
            }
            _ => {
                let how = if self.allow_return {
                    "being stored into a parameter"
                } else {
                    "being stored outside it"
                };
                self.escape(how, at);
            }
        }
    }

    fn escape(&mut self, how: &str, at: Position) {
        let message = format!(
            "region: a pointer into arena '{}' escapes its region by {how}; it may not outlive the arena",
            self.arena
        );
        record_once(&mut self.diagnostics, at, message);
    }

    // An `if`/`match` used as a statement carries blocks that are still inside
    // the region.
    fn check_conditional(&mut self, expression: ExprId) {
        let ast = self.ast;
        match ast.expr(expression) {
            Expression::If(_, consequence, alternative) => {
                self.check(*consequence, false);
                if let Some(block) = alternative {
                    self.check(*block, false);
                }
            }
            Expression::Switch(_, cases) => {
                for case in ast.cases_in(*cases) {
                    self.check(case.body, false);
                }
            }
            Expression::Unsafe(body) => self.check(*body, false),
            _ => {}
        }
    }

    fn is_region_pointer(&self, expression: ExprId) -> bool {
        let ast = self.ast;
        match ast.expr(expression) {
            Expression::Identifier(name) => {
                self.bound.contains(ast.name(*name))
            }
            Expression::FieldAccess(..) | Expression::Index(..) => {
                self.reads_out(expression)
            }
            // A value built around a region pointer carries it. Reading only
            // the bare pointer let the same pointer out one field down, which
            // is the road a container built in the region takes.
            Expression::StructInit(_, fields)
            | Expression::EnumVariantInit(_, _, fields) => ast
                .named_in(*fields)
                .iter()
                .any(|field| self.is_region_pointer(field.value)),
            Expression::Tuple(items)
            | Expression::Literal(Literal::Array(items)) => ast
                .exprs_in(*items)
                .iter()
                .any(|item| self.is_region_pointer(*item)),
            Expression::ArrayRepeat(value, _) => self.is_region_pointer(*value),
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(function) = ast.expr(*callee) else {
                    return false;
                };
                let function = ast.name(*function);
                // The three primitives that build one. A cast still points
                // where it pointed, so it answers as what it was given does.
                if function == "ptr_to"
                    || function == "slice_from"
                    || function == "ptr_cast"
                {
                    return ast.exprs_in(*arguments).iter().any(|argument| {
                        self.mentions_region(*argument)
                            || self.is_region_pointer(*argument)
                    });
                }
                let returns_view = self
                    .signatures
                    .returns_view
                    .get(function)
                    .copied()
                    .unwrap_or(false);
                if !returns_view {
                    return false;
                }
                // A pointer-returning function hands back an arena pointer only if
                // it draws on this arena. It is a `uses` function, or it is passed
                // the arena (or a value already bound to the region).
                self.signatures.uses_arena.contains(function)
                    || ast
                        .exprs_in(*arguments)
                        .iter()
                        .any(|argument| self.mentions_region(*argument))
            }
            Expression::Unsafe(body) => block_value(ast, *body)
                .is_some_and(|value| self.is_region_pointer(value)),
            // `pp^` where `pp` holds the address of a region pointer reads that
            // pointer back out. `p^` where `p` is the region pointer itself
            // reads the value it names, which is not one.
            Expression::Dereference(inner) => root_identifier(ast, *inner)
                .is_some_and(|root| self.via_pointer.contains(root)),
            Expression::If(_, consequence, alternative) => {
                let branches = [Some(*consequence), *alternative];
                branches.into_iter().flatten().any(|block| {
                    block_value(ast, block)
                        .is_some_and(|value| self.is_region_pointer(value))
                })
            }
            Expression::Switch(_, cases) => {
                ast.cases_in(*cases).iter().any(|case| {
                    block_value(ast, case.body)
                        .is_some_and(|value| self.is_region_pointer(value))
                })
            }
            // Listed rather than caught by `_`, so a new expression form is a compile error here instead of quietly answering that it points nowhere.
            Expression::Literal(_)
            | Expression::Boolean(_)
            | Expression::TypeValue(_)
            | Expression::PackMap(..)
            | Expression::Prefix(..)
            | Expression::Infix(..)
            | Expression::Range(..)
            | Expression::Function(..)
            | Expression::Proc(..)
            | Expression::UnsafeFn(_)
            | Expression::Try(_)
            | Expression::AddressOf(_)
            | Expression::Borrow(_)
            | Expression::BorrowMut(_) => false,
        }
    }

    // Whether an expression reads the arena or a value already bound to the
    // region, so a pointer computed from it belongs to the region.
    fn mentions_region(&self, expression: ExprId) -> bool {
        let ast = self.ast;
        match ast.expr(expression) {
            Expression::Identifier(name) => {
                let name = ast.name(*name);
                name == self.arena || self.bound.contains(name)
            }
            Expression::FieldAccess(base, _)
            | Expression::Dereference(base)
            | Expression::Borrow(base)
            | Expression::BorrowMut(base)
            | Expression::AddressOf(base) => self.mentions_region(*base),
            Expression::Index(base, index) => {
                self.mentions_region(*base) || self.mentions_region(*index)
            }
            Expression::Call(_, arguments) => ast
                .exprs_in(*arguments)
                .iter()
                .any(|argument| self.mentions_region(*argument)),
            Expression::Unsafe(body) => block_value(ast, *body)
                .is_some_and(|value| self.mentions_region(value)),
            // A value built around the arena's storage names it, so a pointer
            // taken of one of those points into the region.
            Expression::StructInit(_, fields)
            | Expression::EnumVariantInit(_, _, fields) => ast
                .named_in(*fields)
                .iter()
                .any(|field| self.mentions_region(field.value)),
            Expression::Tuple(items)
            | Expression::Literal(Literal::Array(items)) => ast
                .exprs_in(*items)
                .iter()
                .any(|item| self.mentions_region(*item)),
            Expression::ArrayRepeat(value, _) => self.mentions_region(*value),
            // Listed rather than caught by `_`, so a new expression form is a compile error here instead of quietly answering that it points nowhere.
            Expression::Literal(_)
            | Expression::Boolean(_)
            | Expression::TypeValue(_)
            | Expression::PackMap(..)
            | Expression::Prefix(..)
            | Expression::Infix(..)
            | Expression::Range(..)
            | Expression::Function(..)
            | Expression::Proc(..)
            | Expression::UnsafeFn(_)
            | Expression::Try(_)
            | Expression::If(..)
            | Expression::Switch(..) => false,
        }
    }
}

// The frame check. A function's locals die when it returns, so a view of one of
// them may not be the thing it answers with. This is the same question the
// region check asks about an arena, asked about the frame instead, and it is
// what stops `ptr_to(local)`, a slice over a local array and a `ref` into one
// from outliving the storage they name.
//
// Provenance rather than rooting, and provenance the walk has to establish
// rather than assume. A value gets one of three answers (`Provenance`), and a
// view leaves the call only on `Outlives`. Answering `Outlives` for a shape
// nobody taught the walk about is what let a frame pointer out through an
// ordinary call, through a function pointer, and out of a `move` parameter: each
// of those is a shape, and a check whose soundness rests on having enumerated
// every shape is a list, not a proof. Refusing what it cannot trace makes the
// enumeration a matter of how much honest code compiles instead.
pub fn check_frame_escapes(ast: &Ast, roots: &[StmtId]) -> Result<()> {
    let diagnostics = check_frame_escapes_recovering(ast, roots);
    if diagnostics.is_empty() {
        return Ok(());
    }
    Err(anyhow::anyhow!(crate::flatten(&diagnostics, "\n")))
}

/// Check every function for a view of its own frame leaving it, reporting each
/// rather than stopping at the first. Frames are independent of one another.
pub fn check_frame_escapes_recovering(
    ast: &Ast,
    roots: &[StmtId],
) -> Vec<Diagnostic> {
    // A callback registration keeps a pointer to its context for as long as it
    // is registered, so the value it answers with names storage in this frame
    // exactly as `ptr_to` does. A context in this frame is the ordinary case
    // and is safe, because `check_linearity` forces the registration to be
    // consumed in the function that made it and this check stops it leaving
    // that function by any other road.
    let registrations =
        crate::lower::callbacks::callback_registrations(ast, roots);
    let fields = collect_field_types(ast, roots);
    let views = collect_view_returns(ast, roots, &fields);
    let externs: HashSet<String> = roots
        .iter()
        .filter_map(|statement| match ast.stmt(*statement) {
            Statement::Extern { name, .. } => Some(ast.name(*name).to_string()),
            _ => None,
        })
        .collect();
    let answer_sources = collect_answer_sources(ast, roots, &externs, &fields);
    let kept =
        collect_kept_parameters(ast, roots, &externs, &fields, &answer_sources);
    let param_modes = collect_param_modes(ast, roots);
    let call_bindings = collect_generic_bindings(ast, roots);
    let return_types = collect_return_types(ast, roots);
    // Which functions answer with a place rather than a value. A `ref T` is the
    // only return that does.
    let ref_returns: HashSet<String> = roots
        .iter()
        .filter_map(|statement| {
            let (name, signature) = match ast.stmt(*statement) {
                Statement::Constant(name, value) => match ast.expr(*value) {
                    Expression::Function(_, signature, _)
                    | Expression::Proc(_, signature, _) => {
                        (name, ast.signature(*signature))
                    }
                    _ => return None,
                },
                Statement::Declared {
                    name, return_sig, ..
                } => (name, ast.signature(*return_sig)),
                _ => return None,
            };
            match &signature.kind {
                ReturnKind::Single(ty) | ReturnKind::Fallible(ty, _)
                    if matches!(ty, Type::Ref(_) | Type::RefMut(_)) =>
                {
                    Some(ast.name(*name).to_string())
                }
                _ => None,
            }
        })
        .collect();
    // A name declared at the top of the file is not this frame's storage. A
    // function's own name is one of these, so is a constant, and neither dies
    // when the call returns.
    let top_level: HashSet<String> = roots
        .iter()
        .filter_map(|statement| match ast.stmt(*statement) {
            Statement::Constant(name, _) => Some(ast.name(*name).to_string()),
            Statement::Extern { name, .. }
            | Statement::Declared { name, .. } => {
                Some(ast.name(*name).to_string())
            }
            _ => None,
        })
        .collect();
    // The type of every constant written as a struct literal. A bundle is one,
    // and a call through one of its fields reads the field's declared signature
    // to say where the answer came from, the same as a call through a field of a
    // parameter does. Without these the field's type could not be found and the
    // walk had nothing to trace, so `ops.pass(p)` on a constant `ops` was
    // refused where the same call on a parameter was allowed.
    let constant_types: HashMap<String, Type> = roots
        .iter()
        .filter_map(|statement| match ast.stmt(*statement) {
            Statement::Constant(name, value) => match ast.expr(*value) {
                Expression::StructInit(held, _) => Some((
                    ast.name(*name).to_string(),
                    Type::Struct(ast.name(*held).to_string()),
                )),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let mut diagnostics = Vec::new();
    for statement in roots {
        if let Statement::Constant(name, value) = ast.stmt(*statement)
            && let Expression::Function(params, signature, body)
            | Expression::Proc(params, signature, body) = ast.expr(*value)
        {
            let signature = ast.signature(*signature);
            let answered = match &signature.kind {
                ReturnKind::Single(ty) | ReturnKind::Fallible(ty, _) => {
                    Some(ty)
                }
                ReturnKind::None | ReturnKind::Multiple(_) => None,
            };
            let mut frame = Frame {
                ast,
                function: ast.name(*name).to_string(),
                storage: HashSet::new(),
                outlives: top_level.clone(),
                locals: HashMap::new(),
                places: HashMap::new(),
                answers_view: answered.is_some_and(|ty| {
                    holds_view(ty, &fields, &mut HashSet::new())
                }),
                answers_place: answered.is_some_and(|ty| {
                    matches!(ty, Type::Ref(_) | Type::RefMut(_))
                }),
                answers: answered.cloned(),
                registrations: &registrations,
                kept: &kept,
                views: &views,
                externs: &externs,
                answer_sources: &answer_sources,
                params: &param_modes,
                bindings: &call_bindings,
                answers_place_by_name: &ref_returns,
                types: constant_types.clone(),
                fields: &fields,
                returns: &return_types,
                diagnostics: Vec::new(),
            };
            for parameter in ast.params_in(*params) {
                // A compile-time parameter is annotated with its own name,
                // which says nothing about what it holds. Where it was declared
                // under a signature or a bundle type, that is what the body
                // has, and it is what says where a call through it can have got
                // its answer from.
                let declared = parameter
                    .compile_time_signature
                    .as_ref()
                    .or(parameter.type_annotation.as_ref());
                if let Some(declared) = declared {
                    frame.types.insert(
                        ast.name(parameter.name).to_string(),
                        declared.clone(),
                    );
                }
                match parameter.mode {
                    // A borrow names the caller's storage, which outlives the
                    // call by definition.
                    ParamMode::Read | ParamMode::Write | ParamMode::Value => {
                        frame
                            .outlives
                            .insert(ast.name(parameter.name).to_string());
                    }
                    // A `move` parameter is in both, because the two sets answer
                    // different questions. Its storage is this call's own copy,
                    // so its address is an address into this frame, and leaving
                    // it out of `storage` is what let
                    // `fn(move p: Point) -> ^i64 { ptr_to(p.x) }` hand back a
                    // pointer at a dead frame. Its value is whatever the caller
                    // handed over, so reading it out and passing it on carries
                    // nothing of this frame with it.
                    ParamMode::Move => {
                        frame
                            .storage
                            .insert(ast.name(parameter.name).to_string());
                        frame
                            .outlives
                            .insert(ast.name(parameter.name).to_string());
                    }
                }
            }
            // An allocation capability is threaded in as a parameter by a
            // lowering that runs after this check, so the body names it and
            // nothing has declared it yet. The arena belongs to the caller's
            // `with` block, where the region half of this file checks it.
            for capability in &signature.uses {
                frame.outlives.insert(capability_binding(capability));
            }
            frame.check(*body, true);
            diagnostics.append(&mut frame.diagnostics);
        }
    }
    diagnostics
}

/// The declared field types of every struct and enum, by the type's name. A
/// return type holds a view when one of its fields does, so the question needs
/// the declarations rather than the type alone.
type FieldTypes = HashMap<String, Vec<(String, Type)>>;

fn collect_field_types(ast: &Ast, roots: &[StmtId]) -> FieldTypes {
    let mut fields: FieldTypes = HashMap::new();
    for statement in roots {
        match ast.stmt(*statement) {
            Statement::Struct(name, _, declared) => {
                fields.insert(
                    ast.name(*name).to_string(),
                    ast.fields_in(*declared)
                        .iter()
                        .map(|field| {
                            (
                                ast.name(field.name).to_string(),
                                field.field_type.clone(),
                            )
                        })
                        .collect(),
                );
            }
            Statement::Enum(name, _, variants) => {
                fields.insert(
                    ast.name(*name).to_string(),
                    ast.variants_in(*variants)
                        .iter()
                        .filter_map(|variant| variant.fields)
                        .flat_map(|held| ast.fields_in(held))
                        .map(|field| {
                            (
                                ast.name(field.name).to_string(),
                                field.field_type.clone(),
                            )
                        })
                        .collect(),
                );
            }
            _ => {}
        }
    }
    fields
}

/// The mode and declared type of every parameter of every function, in order.
/// Positions line up one-to-one with call arguments, including a `$Type`
/// argument against a `$T: Type` parameter.
type ParamModes = HashMap<String, Vec<(ParamMode, Option<Type>)>>;

type CallBindings =
    HashMap<String, Vec<(String, crate::ir::build::GenericBinding)>>;

/// Where every named function's compile-time parameters are bound from.
fn collect_generic_bindings(ast: &Ast, roots: &[StmtId]) -> CallBindings {
    let mut bindings = HashMap::new();
    for statement in roots {
        let (name, params) = match ast.stmt(*statement) {
            Statement::Constant(name, value) => match ast.expr(*value) {
                Expression::Function(params, _, _)
                | Expression::Proc(params, _, _) => (name, params),
                _ => continue,
            },
            Statement::Extern { name, params, .. }
            | Statement::Declared { name, params, .. } => (name, params),
            _ => continue,
        };
        bindings.insert(
            ast.name(*name).to_string(),
            crate::ir::build::generic_bindings(ast, ast.params_in(*params)),
        );
    }
    bindings
}

fn collect_param_modes(ast: &Ast, roots: &[StmtId]) -> ParamModes {
    let mut modes = HashMap::new();
    for statement in roots {
        let (name, params) = match ast.stmt(*statement) {
            Statement::Constant(name, value) => match ast.expr(*value) {
                Expression::Function(params, _, _)
                | Expression::Proc(params, _, _) => (name, params),
                _ => continue,
            },
            Statement::Extern { name, params, .. }
            | Statement::Declared { name, params, .. } => (name, params),
            _ => continue,
        };
        // One entry per argument a call writes, so an index into this is an
        // index into that call's arguments. A compile-time parameter a value
        // parameter settles takes no argument, and leaving it here would line
        // every argument after it up against the parameter beside the one it
        // was written for.
        let settled = crate::ir::build::settled_by(ast, ast.params_in(*params));
        modes.insert(
            ast.name(*name).to_string(),
            ast.params_in(*params)
                .iter()
                .zip(&settled)
                .filter(|(_, settled)| settled.is_none())
                .map(|(parameter, _)| {
                    (parameter.mode, parameter.type_annotation.clone())
                })
                .collect(),
        );
    }
    modes
}

/// What each function answers with, by name.
fn collect_return_types(ast: &Ast, roots: &[StmtId]) -> HashMap<String, Type> {
    let mut returns = HashMap::new();
    for statement in roots {
        match ast.stmt(*statement) {
            Statement::Constant(name, value) => {
                let (Expression::Function(_, signature, _)
                | Expression::Proc(_, signature, _)) = ast.expr(*value)
                else {
                    continue;
                };
                if let Some(ty) =
                    ast.signature_to_type(ast.signature(*signature))
                {
                    returns.insert(ast.name(*name).to_string(), ty);
                }
            }
            Statement::Declared {
                name, return_sig, ..
            } => {
                if let Some(ty) =
                    ast.signature_to_type(ast.signature(*return_sig))
                {
                    returns.insert(ast.name(*name).to_string(), ty);
                }
            }
            Statement::Extern {
                name,
                return_type: Some(return_type),
                ..
            } => {
                returns
                    .insert(ast.name(*name).to_string(), return_type.clone());
            }
            _ => {}
        }
    }
    returns
}

/// Whether each function answers with something that holds a view. A call to one
/// that does not carries no storage out, whatever it was passed, and that is
/// what keeps the rule below from reading every call as a leak.
fn collect_view_returns(
    ast: &Ast,
    roots: &[StmtId],
    fields: &FieldTypes,
) -> HashMap<String, bool> {
    let mut views = HashMap::new();
    for statement in roots {
        let (name, signature) = match ast.stmt(*statement) {
            Statement::Constant(name, value) => match ast.expr(*value) {
                Expression::Function(_, signature, _)
                | Expression::Proc(_, signature, _) => (name, *signature),
                _ => continue,
            },
            Statement::Declared {
                name, return_sig, ..
            } => (name, *return_sig),
            Statement::Extern {
                name, return_type, ..
            } => {
                let holds = return_type.as_ref().is_some_and(|ty| {
                    holds_view(ty, fields, &mut HashSet::new())
                });
                views.insert(ast.name(*name).to_string(), holds);
                continue;
            }
            _ => continue,
        };
        let holds = match &ast.signature(signature).kind {
            ReturnKind::Single(ty) | ReturnKind::Fallible(ty, _) => {
                holds_view(ty, fields, &mut HashSet::new())
            }
            ReturnKind::None | ReturnKind::Multiple(_) => false,
        };
        views.insert(ast.name(*name).to_string(), holds);
    }
    views
}

/// One function as the answer walk reads it: its parameters by name and by
/// declared type, and the body to walk.
struct Declared {
    name: String,
    parameters: Vec<String>,
    types: HashMap<String, Type>,
    answers_place: bool,
    body: Range32,
}

/// What one function's answer walk needs.
struct Answers<'a> {
    ast: &'a Ast,
    parameters: &'a [String],
    declared: &'a HashMap<String, Type>,
    fields: &'a FieldTypes,
    sources: &'a HashMap<String, Vec<bool>>,
    externs: &'a HashSet<String>,
    // Whether the function hands back a place rather than a value. A `ref T`
    // answer is the place itself, so reaching into a parameter to build it
    // names that parameter's storage, where copying a value out of one does
    // not.
    answers_place: bool,
    // Whether reading a field that is itself a view copies the view rather than
    // naming the value it came out of. That is what the answer question wants:
    // the `[]T` inside a container points at storage somebody else owns, so an
    // accessor answering with it names that storage and not the container.
    //
    // The keep question wants the opposite reading of the same expression. A
    // view read out of a container is how a caller writes into the run the
    // container holds, so `held := v.storage` followed by `held[i] = value`
    // does put the value in `v`, and losing that let a view of a local go into
    // a caller's `Vec` unremarked.
    views_copy: bool,
}

/// Which parameters a function's answer can name the storage of.
///
/// A call site weighs the arguments this says reach the answer. Joining over
/// every argument reads a wrapper as handing back whatever it was given, and a
/// wrapper over C hands back the library's storage: `device_create_texture`
/// takes the address of a local descriptor and answers with a handle wgpu owns.
///
/// Grown from nothing to a fixpoint, since one function's answer reaches
/// another's and a set that only gains entries settles. A shape this cannot
/// read answers with every parameter, which is what the join did everywhere.
fn collect_answer_sources(
    ast: &Ast,
    roots: &[StmtId],
    externs: &HashSet<String>,
    fields: &FieldTypes,
) -> HashMap<String, Vec<bool>> {
    let mut functions: Vec<Declared> = Vec::new();
    for statement in roots {
        if let Statement::Constant(name, value) = ast.stmt(*statement)
            && let Expression::Function(parameters, signature, body)
            | Expression::Proc(parameters, signature, body) =
                ast.expr(*value)
        {
            functions.push(Declared {
                name: ast.name(*name).to_string(),
                answers_place: matches!(
                    &ast.signature(*signature).kind,
                    ReturnKind::Single(Type::Ref(_) | Type::RefMut(_))
                        | ReturnKind::Fallible(
                            Type::Ref(_) | Type::RefMut(_),
                            _
                        )
                ),
                parameters: crate::ir::build::argument_names(
                    ast,
                    ast.params_in(*parameters),
                ),
                types: ast
                    .params_in(*parameters)
                    .iter()
                    .filter_map(|one| {
                        one.type_annotation
                            .clone()
                            .map(|ty| (ast.name(one.name).to_string(), ty))
                    })
                    .collect(),
                body: *body,
            });
        }
    }
    let mut sources: HashMap<String, Vec<bool>> = functions
        .iter()
        .map(|one| (one.name.clone(), vec![false; one.parameters.len()]))
        .collect();
    loop {
        let mut grew = false;
        for one in &functions {
            let walk = Answers {
                ast,
                parameters: &one.parameters,
                declared: &one.types,
                fields,
                sources: &sources,
                externs,
                answers_place: one.answers_place,
                views_copy: true,
            };
            let mut environment: HashMap<String, HashSet<usize>> =
                HashMap::new();
            let mut answer: HashSet<usize> = HashSet::new();
            answer_of_block(one.body, &walk, &mut environment, &mut answer);
            let held = sources
                .get_mut(&one.name)
                .expect("every function is listed");
            for index in answer {
                if index < held.len() && !held[index] {
                    held[index] = true;
                    grew = true;
                }
            }
        }
        if !grew {
            return sources;
        }
    }
}

/// The expressions a statement holds directly. Its blocks are not among them:
/// the walk descends into those on its own, and a statement's own expressions
/// are what it evaluates in this frame.
fn statement_expressions(statement: &Statement) -> Vec<ExprId> {
    match statement {
        Statement::Let { value, .. }
        | Statement::Constant(_, value)
        | Statement::LetMultiple(_, value)
        | Statement::Return(value)
        | Statement::Assignment(_, value)
        | Statement::Expression(value) => vec![*value],
        Statement::While(condition, _) => vec![*condition],
        Statement::For(_, _, sequence, _) => vec![*sequence],
        _ => Vec::new(),
    }
}

/// Which parameter's storage a place ultimately sits in. Reaching through a
/// field, an element or a pointer stays with whatever the root named, because
/// the question here is who holds the thing being written into rather than
/// where the bytes are: a slice in a parameter is the parameter's, however far
/// from its frame the block itself lives.
fn container_sources(
    place: ExprId,
    walk: &Answers,
    environment: &HashMap<String, HashSet<usize>>,
) -> HashSet<usize> {
    let ast = walk.ast;
    match ast.expr(place) {
        Expression::FieldAccess(base, _)
        | Expression::Index(base, _)
        | Expression::Dereference(base) => {
            container_sources(*base, walk, environment)
        }
        Expression::Identifier(name) => {
            let name = ast.name(*name);
            if let Some(index) =
                walk.parameters.iter().position(|one| one == name)
            {
                return HashSet::from([index]);
            }
            environment.get(name).cloned().unwrap_or_default()
        }
        Expression::Unsafe(block) => block_value(ast, *block)
            .map_or_else(HashSet::new, |value| {
                container_sources(value, walk, environment)
            }),
        _ => HashSet::new(),
    }
}

/// How many parameters of one call this records keeping for. The self-hosted
/// compiler holds the same table as one mask per parameter, so the width is
/// written here rather than left to differ: a function with more parameters
/// than this records nothing, in both compilers alike.
pub const KEPT_PARAMETERS: usize = 16;

/// Which parameters a call keeps a hold of. `kept[name][i]` is the set of
/// parameter indices whose storage parameter `i` is written into, so a value
/// handed to `i` is reachable from those for as long as they live.
///
/// This is what makes a registration checkable. `graph_pass(g, ..., state)`
/// puts `state` in `g`, so a caller handing over a pointer into its own frame
/// is handing the graph something that dies first, and the frame check refuses
/// it at the call rather than leaving it to be read after the fact.
///
/// A fixpoint, because keeping travels: a wrapper that forwards its argument to
/// something that keeps it keeps it too.
fn collect_kept_parameters(
    ast: &Ast,
    roots: &[StmtId],
    externs: &HashSet<String>,
    fields: &FieldTypes,
    sources: &HashMap<String, Vec<bool>>,
) -> HashMap<String, Vec<HashSet<usize>>> {
    let mut functions: Vec<Declared> = Vec::new();
    for statement in roots {
        if let Statement::Constant(name, value) = ast.stmt(*statement)
            && let Expression::Function(parameters, signature, body)
            | Expression::Proc(parameters, signature, body) =
                ast.expr(*value)
        {
            functions.push(Declared {
                name: ast.name(*name).to_string(),
                answers_place: matches!(
                    &ast.signature(*signature).kind,
                    ReturnKind::Single(Type::Ref(_) | Type::RefMut(_))
                        | ReturnKind::Fallible(
                            Type::Ref(_) | Type::RefMut(_),
                            _
                        )
                ),
                parameters: crate::ir::build::argument_names(
                    ast,
                    ast.params_in(*parameters),
                ),
                types: ast
                    .params_in(*parameters)
                    .iter()
                    .filter_map(|one| {
                        one.type_annotation
                            .clone()
                            .map(|ty| (ast.name(one.name).to_string(), ty))
                    })
                    .collect(),
                body: *body,
            });
        }
    }
    let mut kept: HashMap<String, Vec<HashSet<usize>>> = functions
        .iter()
        .map(|one| {
            (one.name.clone(), vec![HashSet::new(); one.parameters.len()])
        })
        .collect();
    loop {
        let mut grew = false;
        for one in &functions {
            let walk = Answers {
                ast,
                parameters: &one.parameters,
                declared: &one.types,
                fields,
                sources,
                externs,
                answers_place: one.answers_place,
                views_copy: false,
            };
            let mut environment: HashMap<String, HashSet<usize>> =
                HashMap::new();
            let mut found: Vec<(usize, usize)> = Vec::new();
            kept_of_block(one.body, &walk, &kept, &mut environment, &mut found);
            let held =
                kept.get_mut(&one.name).expect("every function is listed");
            for (index, into) in found {
                if index < held.len()
                    && index < KEPT_PARAMETERS
                    && into < KEPT_PARAMETERS
                    && held[index].insert(into)
                {
                    grew = true;
                }
            }
        }
        if !grew {
            return kept;
        }
    }
}

/// Every keep a block makes: an assignment that writes a parameter into
/// something reachable from another, and every call that does the same one
/// level down.
fn kept_of_block(
    block: Range32,
    walk: &Answers,
    kept: &HashMap<String, Vec<HashSet<usize>>>,
    environment: &mut HashMap<String, HashSet<usize>>,
    found: &mut Vec<(usize, usize)>,
) {
    let ast = walk.ast;
    for statement in ast.stmts_in(block) {
        match ast.stmt(*statement) {
            Statement::Let { name, value, .. } => {
                let reached = expression_sources(*value, walk, environment);
                environment.insert(ast.name(*name).to_string(), reached);
                kept_of_expression(*value, walk, kept, environment, found);
            }
            Statement::Assignment(place, value) => {
                let into = container_sources(*place, walk, environment);
                let held = expression_sources(*value, walk, environment);
                for index in &held {
                    for target in &into {
                        if index != target {
                            found.push((*index, *target));
                        }
                    }
                }
                kept_of_expression(*value, walk, kept, environment, found);
            }
            Statement::While(condition, body) => {
                kept_of_expression(*condition, walk, kept, environment, found);
                kept_of_block(*body, walk, kept, environment, found);
            }
            Statement::For(_, _, sequence, body) => {
                kept_of_expression(*sequence, walk, kept, environment, found);
                kept_of_block(*body, walk, kept, environment, found);
            }
            Statement::With(_, body) => {
                kept_of_block(*body, walk, kept, environment, found);
            }
            Statement::Expression(value) | Statement::Return(value) => {
                kept_of_expression(*value, walk, kept, environment, found);
            }
            _ => {}
        }
    }
    // A block's last expression is what it answers with, and a registration
    // written as the whole of a body is exactly that.
    if let Some(value) = block_value(ast, block) {
        kept_of_expression(value, walk, kept, environment, found);
    }
}

/// The calls inside an expression, and what each of them keeps.
fn kept_of_expression(
    value: ExprId,
    walk: &Answers,
    kept: &HashMap<String, Vec<HashSet<usize>>>,
    environment: &HashMap<String, HashSet<usize>>,
    found: &mut Vec<(usize, usize)>,
) {
    let ast = walk.ast;
    if let Expression::Call(callee, arguments) = ast.expr(value)
        && let Expression::Identifier(name) = ast.expr(*callee)
        && let Some(shape) = kept.get(ast.name(*name))
    {
        let arguments = ast.exprs_in(*arguments);
        for (index, into) in shape.iter().enumerate() {
            let Some(argument) = arguments.get(index) else {
                continue;
            };
            let held = expression_sources(*argument, walk, environment);
            for target in into {
                let Some(keeper) = arguments.get(*target) else {
                    continue;
                };
                let receiving = container_sources(*keeper, walk, environment);
                for one in &held {
                    for other in &receiving {
                        if one != other {
                            found.push((*one, *other));
                        }
                    }
                }
            }
        }
    }
    for inner in sub_expressions(ast, value) {
        kept_of_expression(inner, walk, kept, environment, found);
    }
}

/// The expressions one expression holds, for a walk that has to reach every
/// call rather than only the one at the top.
pub(crate) fn sub_expressions(ast: &Ast, value: ExprId) -> Vec<ExprId> {
    match ast.expr(value) {
        Expression::Call(callee, arguments) => {
            let mut held = vec![*callee];
            held.extend(ast.exprs_in(*arguments).iter().copied());
            held
        }
        Expression::FieldAccess(base, _)
        | Expression::Dereference(base)
        | Expression::UnsafeFn(base)
        | Expression::Try(base)
        | Expression::ArrayRepeat(base, _)
        | Expression::Prefix(_, base)
        | Expression::Borrow(base)
        | Expression::BorrowMut(base)
        | Expression::AddressOf(base) => vec![*base],
        Expression::Index(base, index) => vec![*base, *index],
        Expression::Infix(left, _, right)
        | Expression::Range(left, right, _) => {
            vec![*left, *right]
        }
        Expression::Tuple(values)
        | Expression::Literal(Literal::Array(values)) => {
            ast.exprs_in(*values).to_vec()
        }
        Expression::StructInit(_, values)
        | Expression::EnumVariantInit(_, _, values) => {
            ast.named_in(*values).iter().map(|one| one.value).collect()
        }
        _ => Vec::new(),
    }
}

/// The blocks a branching expression holds, walked for the exits inside them.
/// Only the shapes that carry a block need reaching: everything else is a value
/// and is read where it is used.
fn answer_of_branches(
    value: ExprId,
    walk: &Answers,
    environment: &mut HashMap<String, HashSet<usize>>,
    answer: &mut HashSet<usize>,
) {
    let ast = walk.ast;
    match ast.expr(value) {
        Expression::If(_, consequence, alternative) => {
            answer_of_block(*consequence, walk, environment, answer);
            if let Some(alternative) = alternative {
                answer_of_block(*alternative, walk, environment, answer);
            }
        }
        Expression::Switch(_, cases) => {
            for case in ast.cases_in(*cases) {
                answer_of_block(case.body, walk, environment, answer);
            }
        }
        _ => {}
    }
}

/// Walk a body, binding each local to the parameters its value reaches, and
/// gather what every exit answers with.
fn answer_of_block(
    block: Range32,
    walk: &Answers,
    environment: &mut HashMap<String, HashSet<usize>>,
    answer: &mut HashSet<usize>,
) {
    let ast = walk.ast;
    for statement in ast.stmts_in(block) {
        match ast.stmt(*statement) {
            Statement::Let { name, value, .. } => {
                let reached = expression_sources(*value, walk, environment);
                environment.insert(ast.name(*name).to_string(), reached);
            }
            Statement::Return(value) => {
                answer.extend(answer_sources_of(*value, walk, environment));
            }
            Statement::While(_, body)
            | Statement::For(_, _, _, body)
            | Statement::With(_, body) => {
                answer_of_block(*body, walk, environment, answer);
            }
            // A branch is an expression, so a `return` written inside one
            // reaches here as an expression statement rather than as a
            // statement of its own. Walking only the loops meant every exit
            // taken from inside an `if` or a `match` was unread, and a function
            // answering with a view of a parameter from one of those was
            // recorded as naming nothing: the caller was then free to store it
            // somewhere that outlives what it points into.
            Statement::Expression(value) => {
                answer_of_branches(*value, walk, environment, answer);
            }
            _ => {}
        }
    }
    if let Some(value) = block_value(ast, block) {
        answer.extend(answer_sources_of(value, walk, environment));
    }
}

/// What one exit answers with. A function handing back a place is read as a
/// place, so `h.a[i]` names `h`; one handing back a value is read as a value,
/// so the same expression names nothing of `h`.
fn answer_sources_of(
    value: ExprId,
    walk: &Answers,
    environment: &HashMap<String, HashSet<usize>>,
) -> HashSet<usize> {
    if walk.answers_place {
        return place_sources(value, walk, environment);
    }
    expression_sources(value, walk, environment)
}

/// The parameters a place names the storage of. Reaching into one through a
/// field or an element of a fixed array stays inside it; reaching through a
/// slice or a pointer lands on storage allocated elsewhere.
fn place_sources(
    place: ExprId,
    walk: &Answers,
    environment: &HashMap<String, HashSet<usize>>,
) -> HashSet<usize> {
    match walk.ast.expr(place) {
        Expression::FieldAccess(base, _) | Expression::Index(base, _) => {
            if reaches_inline(place, walk) {
                place_sources(*base, walk, environment)
            } else {
                HashSet::new()
            }
        }
        Expression::Dereference(_) => HashSet::new(),
        _ => expression_sources(place, walk, environment),
    }
}

/// The type of a place, read from the parameter declarations and the struct
/// fields, which is as much as this walk needs to tell reaching through an
/// indirection from reaching into storage held inline.
fn declared_place_type(place: ExprId, walk: &Answers) -> Option<Type> {
    let ast = walk.ast;
    match ast.expr(place) {
        Expression::Identifier(name) => {
            walk.declared.get(ast.name(*name)).cloned()
        }
        Expression::FieldAccess(base, field) => {
            let name = match declared_place_type(*base, walk)? {
                Type::Struct(name) | Type::Enum(name) => name,
                Type::Ptr(inner) | Type::Ref(inner) | Type::RefMut(inner) => {
                    match *inner {
                        Type::Struct(name) | Type::Enum(name) => name,
                        _ => return None,
                    }
                }
                _ => return None,
            };
            let field = ast.name(*field);
            walk.fields
                .get(Type::template_of(&name))?
                .iter()
                .find(|(declared, _)| declared == field)
                .map(|(_, ty)| ty.clone())
        }
        Expression::Index(base, _) => match declared_place_type(*base, walk)? {
            Type::Array(inner, _)
            | Type::ArrayGeneric(inner, _)
            | Type::Slice(inner)
            | Type::Ptr(inner) => Some(*inner),
            Type::Str => Some(Type::U8),
            _ => None,
        },
        Expression::Dereference(base) => {
            match declared_place_type(*base, walk)? {
                Type::Ptr(inner) | Type::Ref(inner) | Type::RefMut(inner) => {
                    Some(*inner)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Whether reaching into this place lands on storage held inline by the
/// parameter it is rooted at. An element of a fixed array is, and so is a field.
/// An element of a slice or a raw pointer is not: the run lives wherever it was
/// allocated, which is what a caller handing over a heap slice relies on.
fn reaches_inline(place: ExprId, walk: &Answers) -> bool {
    match walk.ast.expr(place) {
        Expression::Identifier(_) => true,
        Expression::FieldAccess(base, _) => reaches_inline(*base, walk),
        Expression::Index(base, _) => match declared_place_type(*base, walk) {
            Some(Type::Array(..) | Type::ArrayGeneric(..)) => {
                reaches_inline(*base, walk)
            }
            Some(_) => false,
            None => reaches_inline(*base, walk),
        },
        Expression::Dereference(_) => false,
        _ => true,
    }
}

/// The parameters whose storage this expression can name.
fn expression_sources(
    expression: ExprId,
    walk: &Answers,
    environment: &HashMap<String, HashSet<usize>>,
) -> HashSet<usize> {
    let ast = walk.ast;
    let of = |value: ExprId| expression_sources(value, walk, environment);
    let union = |values: &mut dyn Iterator<Item = ExprId>| {
        values.fold(HashSet::new(), |mut held, value| {
            held.extend(expression_sources(value, walk, environment));
            held
        })
    };
    let addressed = |place: ExprId| place_sources(place, walk, environment);
    match ast.expr(expression) {
        Expression::Identifier(name) => {
            let name = ast.name(*name);
            if let Some(index) =
                walk.parameters.iter().position(|one| one == name)
            {
                return HashSet::from([index]);
            }
            environment.get(name).cloned().unwrap_or_default()
        }
        Expression::Literal(Literal::Array(values)) => {
            union(&mut ast.exprs_in(*values).iter().copied())
        }
        Expression::Literal(_) | Expression::Boolean(_) => HashSet::new(),
        // A type argument names no parameter. It is erased before anything runs
        // and there is no storage behind it, the same way a literal has none,
        // which is what `value_provenance` already answers for one. Falling to
        // the arm below instead read `heap_slice($T, count)` as naming every
        // parameter of whatever called it, so a function answering with a
        // struct that holds one heap slice was recorded as naming every
        // argument it was handed, and a caller could not hand that struct back.
        Expression::TypeValue(_) => HashSet::new(),
        Expression::Call(callee, arguments) => {
            let arguments = ast.exprs_in(*arguments);
            let Expression::Identifier(name) = ast.expr(*callee) else {
                return union(&mut arguments.iter().copied());
            };
            let name = ast.name(*name);
            match name {
                "ptr_to" => {
                    arguments.iter().fold(HashSet::new(), |mut held, place| {
                        held.extend(addressed(*place));
                        held
                    })
                }
                "ptr_cast" | "slice_from" => {
                    union(&mut arguments.iter().copied())
                }
                "sizeof" | "alignof" | "type_id" | "offset_of"
                | "slice_len" | "typename" | "name_of" => HashSet::new(),
                // C has global storage, so what an extern answers with is not
                // built from what it was handed.
                _ if walk.externs.contains(name) => HashSet::new(),
                _ => match walk.sources.get(name) {
                    Some(flags) => arguments
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| {
                            flags.get(*index).copied().unwrap_or(true)
                        })
                        .fold(HashSet::new(), |mut held, (_, argument)| {
                            held.extend(of(*argument));
                            held
                        }),
                    None => union(&mut arguments.iter().copied()),
                },
            }
        }
        Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::AddressOf(inner) => addressed(*inner),
        // A place read by value copies what is there. Whatever views the copy
        // holds point where they already pointed, so the container it came out
        // of is not named by the answer.
        //
        // A field that *is* a view is the same case and the common one: the
        // `[]T` inside a container points at storage somebody else owns, so an
        // accessor answering with it names that storage and not the container.
        // Reading it as naming the container refused every accessor called on
        // a local one, `fixed_slice` and `vec_slice` among them.
        Expression::FieldAccess(base, _) | Expression::Index(base, _) => {
            match declared_place_type(expression, walk) {
                Some(ty)
                    if !holds_view(&ty, walk.fields, &mut HashSet::new())
                        || (walk.views_copy && is_direct_view(&ty)) =>
                {
                    HashSet::new()
                }
                _ if reaches_inline(expression, walk) => of(*base),
                _ => HashSet::new(),
            }
        }
        Expression::Dereference(_) => HashSet::new(),
        Expression::UnsafeFn(inner)
        | Expression::Try(inner)
        | Expression::ArrayRepeat(inner, _)
        | Expression::Prefix(_, inner) => of(*inner),
        Expression::Infix(left, _, right)
        | Expression::Range(left, right, _) => {
            let mut held = of(*left);
            held.extend(of(*right));
            held
        }
        Expression::Tuple(values) => {
            union(&mut ast.exprs_in(*values).iter().copied())
        }
        Expression::StructInit(_, values) => {
            union(&mut ast.named_in(*values).iter().map(|one| one.value))
        }
        Expression::EnumVariantInit(_, _, values) => {
            union(&mut ast.named_in(*values).iter().map(|one| one.value))
        }
        Expression::Unsafe(block) => {
            block_value(ast, *block).map_or_else(HashSet::new, of)
        }
        Expression::If(_, then_block, else_block) => {
            let mut held =
                block_value(ast, *then_block).map_or_else(HashSet::new, of);
            if let Some(other) = else_block
                && let Some(value) = block_value(ast, *other)
            {
                held.extend(of(value));
            }
            held
        }
        Expression::Switch(_, cases) => ast.cases_in(*cases).iter().fold(
            HashSet::new(),
            |mut held, case| {
                if let Some(value) = block_value(ast, case.body) {
                    held.extend(of(value));
                }
                held
            },
        ),
        _ => (0..walk.parameters.len()).collect(),
    }
}

/// Whether a value of this type is, or holds, a view of storage it does not own.
/// Only such a value can carry storage out of a call, so only a function
/// answering with one has to account for where that storage came from.
///
/// `seen` stops a type that reaches itself through a field from recurring. A
/// pointer answers true without looking at what it points at, so the walk only
/// descends through storage held inline.
fn holds_view(
    ty: &Type,
    fields: &FieldTypes,
    seen: &mut HashSet<String>,
) -> bool {
    match ty {
        Type::Ptr(_)
        | Type::Slice(_)
        | Type::Ref(_)
        | Type::RefMut(_)
        | Type::Str => true,
        Type::Array(inner, _) | Type::ArrayGeneric(inner, _) => {
            holds_view(inner, fields, seen)
        }
        Type::Distinct(_, inner) => holds_view(inner, fields, seen),
        Type::Struct(name) | Type::Enum(name) => {
            if !seen.insert(name.clone()) {
                return false;
            }
            fields.get(Type::template_of(name)).is_some_and(|held| {
                held.iter()
                    .any(|(_, field)| holds_view(field, fields, seen))
            })
        }
        Type::I8
        | Type::I16
        | Type::I32
        | Type::I64
        | Type::Isize
        | Type::U8
        | Type::U16
        | Type::U32
        | Type::U64
        | Type::Usize
        | Type::F32
        | Type::F64
        | Type::Bool
        | Type::Void
        | Type::ConstUsize(_)
        | Type::ConstFn(_)
        | Type::ConstValue(_)
        | Type::Proc(_, _)
        | Type::Handle(_)
        | Type::TypeParam(_) => false,
        // A type the front end could not name. It might hold anything, and the
        // answer that costs nothing here is the one that asks the question.
        Type::Unknown => true,
    }
}

/// A type with this call's compile-time type arguments put in.
///
/// A generic parameter is written `$T` and says nothing on its own about
/// whether handing it an array forms a view. The call says: `vec_push($[]i64,
/// sink, data)` binds `T` to `[]i64` right there, and reading the parameter
/// unresolved let a view of a local go into a container that outlives it.
fn substituted(ty: &Type, bound: &HashMap<String, Type>) -> Type {
    match ty {
        Type::TypeParam(name) => {
            bound.get(name).cloned().unwrap_or_else(|| ty.clone())
        }
        Type::Slice(inner) => Type::Slice(Box::new(substituted(inner, bound))),
        Type::Ptr(inner) => Type::Ptr(Box::new(substituted(inner, bound))),
        Type::Ref(inner) => Type::Ref(Box::new(substituted(inner, bound))),
        Type::RefMut(inner) => {
            Type::RefMut(Box::new(substituted(inner, bound)))
        }
        Type::Array(inner, count) => {
            Type::Array(Box::new(substituted(inner, bound)), *count)
        }
        Type::ArrayGeneric(inner, count) => Type::ArrayGeneric(
            Box::new(substituted(inner, bound)),
            count.clone(),
        ),
        _ => ty.clone(),
    }
}

/// What one element of a run is, so an array or tuple literal weighs each of its
/// values against what the run is expected to hold.
fn element_type(ty: &Type) -> Option<Type> {
    match ty {
        Type::Array(inner, _)
        | Type::ArrayGeneric(inner, _)
        | Type::Slice(inner) => Some((**inner).clone()),
        _ => None,
    }
}

/// Whether a type is itself a view rather than a value that holds one.
///
/// The difference decides whether handing a place over takes its address.
/// `-> []i64` given an array takes the array's address; `-> Options` given a
/// struct copies the struct, and the pointers inside it keep pointing wherever
/// they already did.
fn is_direct_view(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Ptr(_)
            | Type::Slice(_)
            | Type::Ref(_)
            | Type::RefMut(_)
            | Type::Str
    )
}

/// Where the storage a value names comes from, as far as a walk over one
/// function body can tell.
///
/// The order is what a value built from several parts takes: it is as
/// short-lived as its shortest part, so `Frame` wins over `Unknown` and
/// `Unknown` over `Outlives`. It reads the same way in a diagnostic, since
/// `Frame` is the answer that can say what went wrong and `Unknown` is the one
/// that can only say it could not tell.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Provenance {
    /// Storage this call did not create: a parameter, an allocation capability,
    /// a name declared at the top of the file, or a value holding no storage at
    /// all. Handing a view of it back is what an accessor is for.
    Outlives,
    /// Neither shown to outlive the call nor shown to be this frame's. A view
    /// leaving on this answer is refused, because a walk that reads it as
    /// `Outlives` is reporting what it recognized rather than what holds.
    Unknown,
    /// Storage this frame owns. It dies when the call returns.
    Frame,
}

struct Frame<'a> {
    ast: &'a Ast,
    function: String,
    // Names whose storage is this frame's: every local, every loop variable,
    // and every `move` parameter.
    storage: HashSet<String>,
    // Names whose storage the call did not create: a read or write parameter,
    // an allocation capability, anything declared at the top of the file.
    outlives: HashSet<String>,
    // What each local's value names, for the locals this walk could classify.
    locals: HashMap<String, Provenance>,
    // For a local that is a borrow rather than storage of its own, where the
    // place it names lives. `ref result := a.data[id]` binds one, and so does a
    // binding taken from a call answering with a `ref`. Without this the local
    // reads as this frame's storage and an accessor built on one refuses itself.
    places: HashMap<String, Provenance>,
    // Whether this function answers with something that holds a view. Only then
    // can a value it cannot classify carry storage out, so only then is
    // `Unknown` refused. A function answering with an `i64` hands out no
    // storage however little the walk understood of its body.
    answers_view: bool,
    // Whether it answers with a `ref T`, which is the one return that hands back
    // a place rather than a value.
    answers_place: bool,
    // What it answers with, so the value handed back is weighed against the type
    // the answer expects. A view is *formed* where an array lands somewhere one
    // is wanted, and the type is the only thing that says so.
    answers: Option<Type>,
    // Callback registrations in the program, and which argument of each is the
    // context whose storage it keeps.
    registrations: &'a HashMap<String, crate::lower::callbacks::CallbackShape>,
    // Which parameter of each call keeps which other, so an argument naming
    // this frame is judged where it is handed over.
    kept: &'a HashMap<String, Vec<HashSet<usize>>>,
    // Whether each function answers with something holding a view.
    views: &'a HashMap<String, bool>,
    // The C functions this program declares. What one answers with is storage
    // it was not handed, so the arguments say nothing about it.
    externs: &'a HashSet<String>,
    // Which parameters each function's answer can name the storage of, so a
    // call weighs those arguments and leaves the rest alone.
    answer_sources: &'a HashMap<String, Vec<bool>>,
    // How each function takes its arguments, which says whether a callee was
    // handed the address of what it was passed or a copy of it.
    params: &'a ParamModes,
    // Where each call binds each of a function's compile-time parameters from,
    // so a parameter the signature settles is weighed as the type the argument
    // gives it rather than as its own name.
    bindings: &'a CallBindings,
    // The functions answering with a `ref T`, so a binding taken from one is
    // known to be a borrow of somewhere else.
    answers_place_by_name: &'a HashSet<String>,
    // The declared type of every parameter and of every annotated local, and the
    // declared fields of every struct. Reaching through an indirection lands
    // somewhere this frame does not own, and telling `held[0]` on a `[]T` from
    // `arr[0]` on a `[4]i64` is a question about the base's type.
    types: HashMap<String, Type>,
    fields: &'a FieldTypes,
    // What each function answers with, so a binding takes its type from the call
    // that produced it.
    returns: &'a HashMap<String, Type>,
    diagnostics: Vec<Diagnostic>,
}

impl Frame<'_> {
    /// Walk a block. `answers` says whether this block's trailing value is what
    /// the function hands back, which only the function's own body and a block
    /// in value position can be. Without it, descending into a nested block
    /// treats its last statement as the call's answer and refuses code that
    /// returns nothing of the sort.
    fn check(&mut self, block: Range32, answers: bool) {
        let ast = self.ast;
        let statements = ast.stmts_in(block);
        for (index, statement) in statements.iter().enumerate() {
            let last = index + 1 == statements.len();
            let at = ast.stmt_position(*statement);
            self.check_statement(*statement, last && answers, at);
        }
    }

    fn check_statement(
        &mut self,
        statement: StmtId,
        answers: bool,
        at: Position,
    ) {
        let ast = self.ast;
        // Every statement that holds an expression holds calls, and a call
        // that keeps a pointer keeps it wherever it was written. Judged once
        // here rather than once per statement form, so a form added later
        // is covered without anyone remembering to.
        for value in statement_expressions(ast.stmt(statement)) {
            self.judge_kept(value, at);
        }
        match ast.stmt(statement) {
            Statement::Let {
                name,
                value,
                type_annotation,
                ..
            } => {
                // A binding declared as a view and given a place takes the
                // address of that place, so `view : []i64 = arr` holds a
                // view of the array rather than a copy of it, and the same
                // is true one field down in `h : Holder = { view = arr }`.
                let held =
                    self.expected_provenance(*value, type_annotation.as_ref());
                let name = ast.name(*name).to_string();
                self.storage.insert(name.clone());
                self.locals.insert(name.clone(), held);
                if let Some(borrowed) = self.borrowed_place(*value) {
                    self.places.insert(name.clone(), borrowed);
                }
                if let Some(declared) = type_annotation {
                    self.types.insert(name, declared.clone());
                } else if let Some(inferred) = self.value_type(*value) {
                    self.types.insert(name, inferred);
                }
            }
            Statement::LetMultiple(bindings, value) => {
                let held = self.value_provenance(*value);
                for binding in ast.bindings_in(*bindings) {
                    let name = ast.name(binding.name).to_string();
                    self.storage.insert(name.clone());
                    self.locals.insert(name, held);
                }
            }
            // A function written inside a body is a declaration rather than
            // a value this frame holds. Its own body is checked where the
            // walk reaches it as an item of the program.
            Statement::Constant(_, value)
                if matches!(
                    ast.expr(*value),
                    Expression::Function(..) | Expression::Proc(..)
                ) => {}
            Statement::Constant(name, value) => {
                let held = self.value_provenance(*value);
                let name = ast.name(*name).to_string();
                self.storage.insert(name.clone());
                self.locals.insert(name, held);
            }
            Statement::Return(value) => self.judge(*value, "returned", at),
            Statement::Assignment(place, value) => {
                self.assign(*place, *value, at);
            }
            Statement::While(_, body) | Statement::With(_, body) => {
                self.check(*body, false);
            }
            // A loop variable is this frame's storage the way a local is, so
            // an address of one dies when the call returns. What it holds is
            // an element of the sequence, which is worth whatever the
            // sequence was worth, the same way a `let` takes its worth from
            // the value it was given. Recording only the first half made
            // reading a loop variable answer `Frame`, so walking a slice and
            // keeping the largest element in a local refused to return it.
            Statement::For(name, second, sequence, body) => {
                let held = self.value_provenance(*sequence);
                let name = ast.name(*name).to_string();
                self.storage.insert(name.clone());
                self.locals.insert(name, held);
                if let Some(second) = second {
                    let second = ast.name(*second).to_string();
                    self.storage.insert(second.clone());
                    self.locals.insert(second, held);
                }
                self.check(*body, false);
            }
            Statement::Defer(inner) | Statement::ErrDefer(inner) => {
                self.check_statement(*inner, false, at);
            }
            Statement::Expression(value) => {
                self.check_expression_statement(*value, answers, at);
            }
            // Declarations and control transfers hand nothing to the caller.
            // Listed rather than caught by `_`, so a new statement form is a
            // compile error here rather than a road out of the frame that
            // nobody walked.
            Statement::Struct(..)
            | Statement::Enum(..)
            | Statement::Flags(..)
            | Statement::TypeAlias(..)
            | Statement::Break
            | Statement::Continue
            | Statement::Import(..)
            | Statement::Extern { .. }
            | Statement::Declared { .. } => {}
        }
    }

    /// A statement that is an expression. A block used as a value answers for
    /// the whole function, so its branches are walked and, where the block is
    /// the answer, so is what each one ends with. `match` was missing here, and
    /// a `return` inside an arm was never seen at all.
    fn check_expression_statement(
        &mut self,
        value: ExprId,
        answers: bool,
        at: Position,
    ) {
        let ast = self.ast;
        match ast.expr(value) {
            Expression::If(_, consequence, alternative) => {
                self.answers_here(*consequence, answers);
                if let Some(block) = alternative {
                    self.answers_here(*block, answers);
                }
            }
            Expression::Switch(_, cases) => {
                for case in ast.cases_in(*cases) {
                    self.answers_here(case.body, answers);
                }
            }
            // An `unsafe` block is transparent here. `ptr_to` is refused outside
            // one, so its statements are where a frame pointer is formed, bound
            // and returned, and a walk that steps over the block never sees any
            // of it.
            Expression::Unsafe(body) => self.check(*body, answers),
            // Everything else is a value. It leaves the call only when it is
            // what the call answers with.
            Expression::Identifier(_)
            | Expression::Literal(_)
            | Expression::Boolean(_)
            | Expression::Prefix(..)
            | Expression::Infix(..)
            | Expression::Function(..)
            | Expression::Proc(..)
            | Expression::Call(..)
            | Expression::Index(..)
            | Expression::FieldAccess(..)
            | Expression::AddressOf(_)
            | Expression::Borrow(_)
            | Expression::BorrowMut(_)
            | Expression::Dereference(_)
            | Expression::StructInit(..)
            | Expression::PackMap(..)
            | Expression::Range(..)
            | Expression::Tuple(_)
            | Expression::EnumVariantInit(..)
            | Expression::TypeValue(_)
            | Expression::UnsafeFn(_)
            | Expression::Try(_)
            | Expression::ArrayRepeat(..) => {
                if answers {
                    self.judge(value, "the call's answer", at);
                }
            }
        }
    }

    /// A branch of a block used as a value. Walk it as a block, and where the
    /// block is the function's answer, weigh what the branch ends with too.
    fn answers_here(&mut self, block: Range32, answers: bool) {
        let ast = self.ast;
        self.check(block, answers);
        if answers
            && let Some(last) = ast.stmts_in(block).last()
            && let Statement::Expression(value) = ast.stmt(*last)
        {
            self.judge(*value, "the call's answer", ast.stmt_position(*last));
        }
    }

    /// Writing a view into a place the call cannot see hands it to the caller
    /// just as returning it does, and the caller's frame outlives this one.
    /// Writing one into this frame's own storage keeps it here, and that place
    /// now holds whatever the value named: without recording it, a pointer
    /// stored in a local and returned afterwards left with nobody having asked.
    /// Every call in an expression, judged for what it keeps. A pointer into
    /// this frame handed to a parameter the callee stores in something that
    /// outlives the call is a pointer that will be read after the storage it
    /// names is gone, so it is refused here, where the two arguments are
    /// written next to each other and the mistake is visible.
    ///
    /// Handing it to something that also dies with this frame is the ordinary
    /// case and is allowed: a program that opens an App, spawns its scene, and
    /// registers the two together is registering a state that lives exactly as
    /// long as what holds it.
    fn judge_kept(&mut self, value: ExprId, at: Position) {
        let ast = self.ast;
        if let Expression::Call(callee, arguments) = ast.expr(value)
            && let Expression::Identifier(name) = ast.expr(*callee)
            && let Some(shape) = self.kept.get(ast.name(*name))
        {
            let name = ast.name(*name);
            let arguments = ast.exprs_in(*arguments);
            for (index, into) in shape.iter().enumerate() {
                let Some(argument) = arguments.get(index) else {
                    continue;
                };
                // Against the parameter's declared type, since handing an array
                // to a `[]T` parameter forms a view of it and handing it to a
                // `[N]T` one copies it. With this call's type arguments put in,
                // since a container takes its element as a `$T`.
                let bound = self.type_arguments(name, arguments);
                let declared = self
                    .params
                    .get(name)
                    .and_then(|params| params.get(index))
                    .and_then(|(_, declared)| declared.as_ref())
                    .map(|ty| substituted(ty, &bound));
                if self.expected_provenance(*argument, declared.as_ref())
                    != Provenance::Frame
                {
                    continue;
                }
                let escapes = into.iter().any(|target| {
                    arguments.get(*target).is_some_and(|keeper| {
                        self.place_provenance(*keeper) != Provenance::Frame
                    })
                });
                if escapes {
                    let name =
                        crate::modules::imports::demangle_private_names(name);
                    self.escape(
                        &format!(
                            "handed to '{name}', which keeps it in something \
                             that outlives this frame"
                        ),
                        at,
                    );
                }
            }
        }
        for inner in sub_expressions(ast, value) {
            self.judge_kept(inner, at);
        }
    }

    fn assign(&mut self, place: ExprId, value: ExprId, at: Position) {
        let expected = self.place_type(place);
        let held = self.expected_provenance(value, expected.as_ref());
        if self.place_provenance(place) == Provenance::Frame {
            if let Some(root) = root_identifier(self.ast, place) {
                let root = root.to_string();
                let now = self
                    .locals
                    .get(&root)
                    .copied()
                    .unwrap_or(Provenance::Outlives)
                    .max(held);
                self.locals.insert(root, now);
            }
            return;
        }
        if held == Provenance::Frame {
            self.escape("stored where the call cannot see", at);
        }
    }

    /// What a value leaving the call is worth. `Frame` is refused outright.
    /// `Unknown` is refused wherever the function answers with something that
    /// holds a view, since that is where a value the walk could not classify is
    /// able to carry storage out.
    ///
    /// A function answering with `ref T` answers with a *place*: `local.a[0]`
    /// there hands back the element rather than a copy of it, so the storage the
    /// answer names is what matters and not what happens to sit there. Under
    /// `-> i64` the same expression is a read, and reading a local carries
    /// nothing out. Under `-> ^T` it is a value that happens to be an address,
    /// so a local holding a heap pointer hands back the pointer rather than the
    /// local.
    fn judge(&mut self, value: ExprId, how: &str, at: Position) {
        let answers = self.answers.clone();
        let mut held = self.expected_provenance(value, answers.as_ref());
        if self.answers_place {
            held = held.max(self.place_provenance(value));
        }
        // A borrow named where a value is wanted is read through, so what
        // leaves is a copy of what it points at and nothing points into this
        // frame. `ref b := a` then `b` as the answer of a function answering an
        // `i64` was refused as a pointer escaping, and the same `b` in `b - 41`
        // was taken, which is the same read written two ways.
        //
        // Only a name reaches this: every other shape that carries frame
        // storage out is the pointer itself. The answer has to be a plain
        // value, so a borrow, a raw pointer and anything holding a view are all
        // still weighed.
        if held == Provenance::Frame
            && !self.answers_place
            && !self.answers_view
            && matches!(self.ast.expr(value), Expression::Identifier(_))
            && self
                .answers
                .as_ref()
                .is_some_and(|ty| !matches!(ty, Type::Ptr(_)))
        {
            return;
        }
        match held {
            Provenance::Frame => self.escape(how, at),
            Provenance::Unknown if self.answers_view => self.unproven(at),
            Provenance::Unknown | Provenance::Outlives => {}
        }
    }

    fn escape(&mut self, how: &str, at: Position) {
        let function = self.function.clone();
        let message = format!(
            "region: a pointer into the frame of '{function}' is {how}; the storage it names dies when the call returns"
        );
        record_once(&mut self.diagnostics, at, message);
    }

    fn unproven(&mut self, at: Position) {
        let function = self.function.clone();
        let message = format!(
            "region: '{function}' answers with a borrowed view whose storage cannot be traced to a parameter or an allocation capability; a view that leaves the call has to name storage that outlives it"
        );
        record_once(&mut self.diagnostics, at, message);
    }

    /// Whether a view of this return type could name storage held inside a value
    /// of this parameter type.
    ///
    /// A view points at elements of one type, and it can land inside a value
    /// only where that value holds one of those by value. Anything a parameter
    /// reaches through a pointer, a slice or a `str` is not the parameter's own
    /// storage, so the walk stops at every indirection.
    ///
    /// An aggregate answer is asked the same question about every view it holds
    /// rather than being given up on. Giving up said yes to
    /// `answer.field = build(local)` for every callee answering with a struct,
    /// however that struct was built, because a local's own storage is this
    /// frame: a `Lit` holding a `Cluster` by value took the `Cluster` it was
    /// handed as a pointer into the caller. What the answer holds is what
    /// decides it, and a struct holding a `^Inner` where the parameter is an
    /// `Inner` is still caught by exactly this test.
    fn view_lands_in(&self, answer: &Type, parameter: &Type) -> bool {
        self.view_reaches(answer, parameter, &mut HashSet::new())
    }

    /// The same question, asked through whatever the answer is made of.
    fn view_reaches(
        &self,
        answer: &Type,
        parameter: &Type,
        seen: &mut HashSet<String>,
    ) -> bool {
        match answer {
            Type::Ptr(inner)
            | Type::Slice(inner)
            | Type::Ref(inner)
            | Type::RefMut(inner) => {
                self.holds_inline(inner, parameter, &mut HashSet::new())
            }
            // A `str` views bytes and says nothing about where they are, and a
            // type the walk cannot read could be anything.
            Type::Str | Type::Unknown | Type::TypeParam(_) => true,
            Type::Array(inner, _) | Type::ArrayGeneric(inner, _) => {
                self.view_reaches(inner, parameter, seen)
            }
            Type::Distinct(_, inner) => {
                self.view_reaches(inner, parameter, seen)
            }
            Type::Struct(name) | Type::Enum(name) => {
                if !seen.insert(name.clone()) {
                    return false;
                }
                let Some(declared) = self.fields.get(Type::template_of(name))
                else {
                    // A struct whose fields this walk never saw could hold
                    // anything, so it holds the parameter in.
                    return true;
                };
                declared
                    .iter()
                    .any(|(_, field)| self.view_reaches(field, parameter, seen))
            }
            // A number, a boolean or a function pointer names no storage.
            _ => false,
        }
    }

    fn holds_inline(
        &self,
        element: &Type,
        held: &Type,
        seen: &mut HashSet<String>,
    ) -> bool {
        // A generic that is not yet substituted stands for whatever it will be.
        if matches!(element, Type::TypeParam(_) | Type::Unknown)
            || matches!(held, Type::TypeParam(_) | Type::Unknown)
        {
            return true;
        }
        if element == held {
            return true;
        }
        match held {
            Type::Array(inner, _) | Type::ArrayGeneric(inner, _) => {
                self.holds_inline(element, inner, seen)
            }
            Type::Distinct(_, inner) => self.holds_inline(element, inner, seen),
            Type::Struct(name) | Type::Enum(name) => {
                if !seen.insert(name.clone()) {
                    return false;
                }
                self.fields.get(Type::template_of(name)).is_some_and(
                    |declared| {
                        declared.iter().any(|(_, field)| {
                            self.holds_inline(element, field, seen)
                        })
                    },
                )
            }
            // Reaching through an indirection leaves this value's storage.
            _ => false,
        }
    }

    fn place_type(&self, place: ExprId) -> Option<Type> {
        place_type(self.ast, &self.types, self.fields, place)
    }

    fn value_type(&self, value: ExprId) -> Option<Type> {
        value_type(self.ast, &self.types, self.fields, self.returns, value)
    }
}

/// The declared type of a place, as far as the annotations and the struct
/// declarations give it. `None` means the walk cannot tell, and the rules
/// that ask treat that as the answer that keeps storage in.
fn place_type(
    ast: &Ast,
    locals: &HashMap<String, Type>,
    fields: &FieldTypes,
    place: ExprId,
) -> Option<Type> {
    match ast.expr(place) {
        Expression::Identifier(name) => locals.get(ast.name(*name)).cloned(),
        Expression::FieldAccess(base, field) => {
            let name = match place_type(ast, locals, fields, *base)? {
                Type::Struct(name) | Type::Enum(name) => name,
                Type::Ptr(inner) | Type::Ref(inner) | Type::RefMut(inner) => {
                    match *inner {
                        Type::Struct(name) | Type::Enum(name) => name,
                        _ => return None,
                    }
                }
                _ => return None,
            };
            let field = ast.name(*field);
            fields
                .get(Type::template_of(&name))?
                .iter()
                .find(|(declared, _)| declared == field)
                .map(|(_, ty)| ty.clone())
        }
        Expression::Index(base, _) => {
            match place_type(ast, locals, fields, *base)? {
                Type::Array(inner, _)
                | Type::ArrayGeneric(inner, _)
                | Type::Slice(inner)
                | Type::Ptr(inner) => Some(*inner),
                Type::Str => Some(Type::U8),
                _ => None,
            }
        }
        Expression::Dereference(base) => {
            match place_type(ast, locals, fields, *base)? {
                Type::Ptr(inner) | Type::Ref(inner) | Type::RefMut(inner) => {
                    Some(*inner)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// The type a binding takes from its value, for the shapes that say so
/// plainly. Only enough to answer the indexing question above.
fn value_type(
    ast: &Ast,
    locals: &HashMap<String, Type>,
    fields: &FieldTypes,
    returns: &HashMap<String, Type>,
    value: ExprId,
) -> Option<Type> {
    match ast.expr(value) {
        Expression::Call(callee, arguments) => {
            let arguments = ast.exprs_in(*arguments);
            let Expression::Identifier(name) = ast.expr(*callee) else {
                // A call through a value answers with what the signature it
                // holds says, which is a bundle's field being called.
                let Some(Type::Proc(_, answer)) =
                    place_type(ast, locals, fields, *callee)
                else {
                    return None;
                };
                return Some(*answer);
            };
            match ast.name(*name) {
                // The three that build a view are not declared anywhere, so
                // their types are written here. Without them a pointer bound
                // from `ptr_to` has no type, and the dereference rule cannot
                // tell a scalar read from a view read.
                "ptr_to" => Some(Type::Ptr(Box::new(
                    arguments
                        .first()
                        .and_then(|place| {
                            place_type(ast, locals, fields, *place)
                        })
                        .unwrap_or(Type::Unknown),
                ))),
                "ptr_cast" => {
                    match arguments.first().map(|argument| ast.expr(*argument))
                    {
                        Some(Expression::TypeValue(inner)) => {
                            Some(Type::Ptr(Box::new(inner.clone())))
                        }
                        _ => None,
                    }
                }
                "slice_from" => {
                    match arguments.first().map(|argument| ast.expr(*argument))
                    {
                        Some(Expression::TypeValue(inner)) => {
                            Some(Type::Slice(Box::new(inner.clone())))
                        }
                        _ => None,
                    }
                }
                // A name bound to a function the call site chose has no
                // declaration under that name to read a return type off, and
                // the signature it was declared under is the same promise.
                name => match locals.get(name) {
                    Some(Type::Proc(_, answer)) => Some((**answer).clone()),
                    _ => returns.get(name).cloned(),
                },
            }
        }
        Expression::Unsafe(body) => block_value(ast, *body)
            .and_then(|inner| value_type(ast, locals, fields, returns, inner)),
        Expression::Literal(Literal::String(_)) => Some(Type::Str),
        Expression::Literal(Literal::Array(elements)) => {
            let elements = ast.exprs_in(*elements);
            let element = elements
                .first()
                .and_then(|first| {
                    value_type(ast, locals, fields, returns, *first)
                })
                .unwrap_or(Type::Unknown);
            Some(Type::Array(Box::new(element), elements.len()))
        }
        Expression::ArrayRepeat(inner, count) => Some(Type::ArrayGeneric(
            Box::new(
                value_type(ast, locals, fields, returns, *inner)
                    .unwrap_or(Type::Unknown),
            ),
            SizeExpr::Named(ast.name(*count).to_string()),
        )),
        Expression::StructInit(name, _) => {
            Some(Type::Struct(ast.name(*name).to_string()))
        }
        Expression::EnumVariantInit(name, _, _) => {
            Some(Type::Enum(ast.name(*name).to_string()))
        }
        Expression::Literal(Literal::Integer(_)) => Some(Type::I64),
        Expression::Literal(Literal::Float(_)) => Some(Type::F64),
        Expression::Literal(Literal::Float32(_)) => Some(Type::F32),
        Expression::Literal(Literal::Boolean(_)) | Expression::Boolean(_) => {
            Some(Type::Bool)
        }
        _ => place_type(ast, locals, fields, value),
    }
}

impl Frame<'_> {
    /// What a value is worth where the context expects a particular type.
    ///
    /// A view is *formed* rather than copied wherever an array lands somewhere
    /// a view is wanted, and the storage it then names is the array's. Nothing
    /// about the expression says so: `data` reads the same in
    /// `Holder { view = data }`, `sink.view = data`, `keep(h, data)` and
    /// `-> []i64 { data }`, and only the type on the other side says a view is
    /// being taken. Asking `value_provenance` alone answers with what the array
    /// *holds*, which for a run of numbers is nothing, so every one of those
    /// positions handed a view of a dead frame out while the last of them was
    /// refused.
    ///
    /// So the question is asked against the expected type wherever there is
    /// one, and an aggregate is walked into rather than given up on, since a
    /// struct's field carries its own expectation and that is the road the
    /// escape actually took.
    fn expected_provenance(
        &self,
        value: ExprId,
        expected: Option<&Type>,
    ) -> Provenance {
        let ast = self.ast;
        if let Some(ty) = expected
            && is_direct_view(ty)
        {
            return self
                .value_provenance(value)
                .max(self.coercion_provenance(value));
        }
        match ast.expr(value) {
            Expression::StructInit(name, values)
            | Expression::EnumVariantInit(name, _, values) => {
                self.literal_provenance(ast.name(*name), *values, expected)
            }
            Expression::Tuple(items)
            | Expression::Literal(Literal::Array(items)) => {
                let element = expected.and_then(element_type);
                ast.exprs_in(*items).iter().fold(
                    Provenance::Outlives,
                    |held, item| {
                        held.max(
                            self.expected_provenance(*item, element.as_ref()),
                        )
                    },
                )
            }
            Expression::ArrayRepeat(inner, _) => {
                let element = expected.and_then(element_type);
                self.expected_provenance(*inner, element.as_ref())
            }
            Expression::Unsafe(body) => self.block_expected(*body, expected),
            Expression::If(_, consequence, alternative) => {
                let held = self.block_expected(*consequence, expected);
                match alternative {
                    Some(block) => {
                        held.max(self.block_expected(*block, expected))
                    }
                    None => held,
                }
            }
            Expression::Switch(_, cases) => ast
                .cases_in(*cases)
                .iter()
                .fold(Provenance::Outlives, |held, case| {
                    held.max(self.block_expected(case.body, expected))
                }),
            _ => self.value_provenance(value),
        }
    }

    /// A block's trailing value, weighed against what the block is expected to
    /// answer with.
    fn block_expected(
        &self,
        block: Range32,
        expected: Option<&Type>,
    ) -> Provenance {
        block_value(self.ast, block).map_or(Provenance::Outlives, |value| {
            self.expected_provenance(value, expected)
        })
    }

    /// A struct or enum literal, field by field against the declared types.
    ///
    /// The literal names its own type where the source wrote one, and takes it
    /// from the context where the source left it out, which is what an inferred
    /// literal and the struct behind a multi-return both are.
    fn literal_provenance(
        &self,
        name: &str,
        values: Range32,
        expected: Option<&Type>,
    ) -> Provenance {
        let ast = self.ast;
        let declared =
            self.fields.get(Type::template_of(name)).or_else(
                || match expected {
                    Some(Type::Struct(held) | Type::Enum(held)) => {
                        self.fields.get(Type::template_of(held))
                    }
                    _ => None,
                },
            );
        ast.named_in(values)
            .iter()
            .fold(Provenance::Outlives, |held, field| {
                let expected = declared.and_then(|fields| {
                    fields
                        .iter()
                        .find(|(declared, _)| declared == ast.name(field.name))
                        .map(|(_, ty)| ty)
                });
                held.max(self.expected_provenance(field.value, expected))
            })
    }

    /// The storage a view takes the address of when it is built by coercion.
    ///
    /// `view : []i64 = arr` and `fn() -> []i64 { arr }` both hand back a view of
    /// the array rather than a copy of it, so the array's storage is what the
    /// view names. A value that is already a view carries its own provenance and
    /// no new address is taken, which is what keeps a slice parameter passed
    /// straight back out from reading as this frame's.
    fn coercion_provenance(&self, value: ExprId) -> Provenance {
        match self.value_type(value) {
            Some(
                Type::Ptr(_)
                | Type::Slice(_)
                | Type::Ref(_)
                | Type::RefMut(_)
                | Type::Str,
            ) => Provenance::Outlives,
            _ => self.place_provenance(value),
        }
    }

    /// Where the place a binding names lives, when the binding is a borrow
    /// rather than storage of its own. `ref x := place` parses as a borrow of
    /// that place, and a binding taken from a call answering with a `ref` is one
    /// too. Anything else is `None`: the binding is storage this frame owns.
    fn borrowed_place(&self, value: ExprId) -> Option<Provenance> {
        let ast = self.ast;
        match ast.expr(value) {
            Expression::Borrow(place) | Expression::BorrowMut(place) => {
                Some(self.place_provenance(*place))
            }
            Expression::Unsafe(body) => block_value(ast, *body)
                .and_then(|inner| self.borrowed_place(inner)),
            Expression::Call(callee, _) => {
                let Expression::Identifier(name) = ast.expr(*callee) else {
                    return None;
                };
                self.answers_place_by_name
                    .contains(ast.name(*name))
                    .then(|| self.value_provenance(value))
            }
            _ => None,
        }
    }

    /// What a name's value names. A local answers with whatever its binding was
    /// given, which is how a pointer bound here and handed back later is caught.
    fn name_provenance(&self, name: &str) -> Provenance {
        if let Some(held) = self.locals.get(name) {
            return *held;
        }
        if self.outlives.contains(name) {
            return Provenance::Outlives;
        }
        if self.storage.contains(name) {
            return Provenance::Frame;
        }
        Provenance::Unknown
    }

    /// The storage a place names. Reaching into a place stays in whatever the
    /// root named, and reaching through a pointer lands wherever that pointer
    /// pointed, which is a question about its value rather than about the place.
    fn place_provenance(&self, place: ExprId) -> Provenance {
        let ast = self.ast;
        match ast.expr(place) {
            Expression::Identifier(name) => {
                let name = ast.name(*name);
                // A borrow names storage somewhere else, so it answers with
                // wherever that was and not with the frame it sits in.
                if let Some(held) = self.places.get(name) {
                    *held
                } else if self.storage.contains(name) {
                    Provenance::Frame
                } else if self.outlives.contains(name) {
                    Provenance::Outlives
                } else {
                    Provenance::Unknown
                }
            }
            // Indexing a fixed array stays inside the base's storage. Indexing
            // a slice, a `str` or a raw pointer lands wherever that indirection
            // points, which is a question about the base's value: `held[0]` on a
            // `[]T` parameter names the caller's block, not the slice sitting in
            // this frame.
            Expression::Index(base, _) => match self.place_type(*base) {
                Some(Type::Slice(_) | Type::Str | Type::Ptr(_)) => {
                    self.value_provenance(*base)
                }
                _ => self.place_provenance(*base),
            },
            Expression::FieldAccess(base, _)
            | Expression::Borrow(base)
            | Expression::BorrowMut(base) => self.place_provenance(*base),
            Expression::Dereference(base) => self.value_provenance(*base),
            // Not a place. It names no storage of its own, so it holds nothing
            // to how long a view built from it may live.
            Expression::Literal(_)
            | Expression::Boolean(_)
            | Expression::Prefix(..)
            | Expression::Infix(..)
            | Expression::If(..)
            | Expression::Function(..)
            | Expression::Proc(..)
            | Expression::Call(..)
            | Expression::AddressOf(_)
            | Expression::StructInit(..)
            | Expression::PackMap(..)
            | Expression::Range(..)
            | Expression::Switch(..)
            | Expression::Tuple(_)
            | Expression::EnumVariantInit(..)
            | Expression::TypeValue(_)
            | Expression::Unsafe(_)
            | Expression::UnsafeFn(_)
            | Expression::Try(_)
            | Expression::ArrayRepeat(..) => Provenance::Outlives,
        }
    }

    /// An argument as the callee sees it.
    ///
    /// The callee can build a view of the argument's own storage only when it
    /// was handed that storage's address. A `mut` parameter always is, and a
    /// read parameter of a type that is not copy is too, since an aggregate is
    /// borrowed rather than copied. A read parameter of a copy type gets a
    /// value, and the storage that value came from is nothing to the callee: a
    /// local `i64` passed as a count leaves no way to name the frame it sat in.
    ///
    /// Getting this wrong in the safe direction reads every local passed to
    /// anything as a leak, which is what `heap_slice($T, room)` looked like when
    /// the place was taken unconditionally.
    /// What this call binds each of its compile-time type parameters to, so a
    /// `$T` parameter is weighed as the type the call gives it.
    fn type_arguments(
        &self,
        callee: &str,
        arguments: &[ExprId],
    ) -> HashMap<String, Type> {
        let ast = self.ast;
        let mut bound = HashMap::new();
        let Some(bindings) = self.bindings.get(callee) else {
            return bound;
        };
        for (name, binding) in bindings {
            let Some(argument) = arguments.get(match binding {
                crate::ir::build::GenericBinding::Written(at) => *at,
                crate::ir::build::GenericBinding::Settled(at, _) => *at,
            }) else {
                continue;
            };
            match binding {
                crate::ir::build::GenericBinding::Written(_) => {
                    if let Expression::TypeValue(ty) = ast.expr(*argument) {
                        bound.insert(name.clone(), ty.clone());
                    }
                }
                // The call writes nothing for this one, so what it stands for
                // is the type of the argument the value parameter naming it
                // takes.
                crate::ir::build::GenericBinding::Settled(_, pattern) => {
                    if let Some(held) = value_type(
                        ast,
                        &self.types,
                        self.fields,
                        self.returns,
                        *argument,
                    ) {
                        crate::ir::build::infer_subst_into(
                            pattern,
                            &held,
                            std::slice::from_ref(name),
                            &mut bound,
                        );
                    }
                }
            }
        }
        bound
    }

    fn argument_provenance(
        &self,
        callee: &str,
        index: usize,
        argument: ExprId,
        arguments: &[ExprId],
    ) -> Provenance {
        let bound = self.type_arguments(callee, arguments);
        let held = self
            .params
            .get(callee)
            .and_then(|params| params.get(index))
            .map(|(mode, declared)| {
                (*mode, declared.as_ref().map(|ty| substituted(ty, &bound)))
            });
        let answer = self.returns.get(callee).cloned();
        match (held, answer) {
            (Some((mode, declared)), Some(answer)) => {
                self.held_argument(&answer, declared.as_ref(), mode, argument)
            }
            // A callee whose signature is not in this program. Reading it as
            // taking the address is the answer that lets no storage out.
            _ => self
                .place_provenance(argument)
                .max(self.value_provenance(argument)),
        }
    }

    /// One argument's contribution, given what the callee answers with and how
    /// it takes that argument.
    fn held_argument(
        &self,
        answer: &Type,
        declared: Option<&Type>,
        mode: ParamMode,
        argument: ExprId,
    ) -> Provenance {
        let Some(declared) = declared else {
            return self
                .place_provenance(argument)
                .max(self.value_provenance(argument));
        };
        // The callee was handed the argument's address, so a view it builds can
        // name the argument's own storage. Only where the view could land there:
        // a `[]Entity` cannot point inside a `Query` that holds no Entity by
        // value, whatever address the callee was given, and reading it as though
        // it could made `query_entities(world, q)` a leak of the local `q`.
        let addressed = matches!(mode, ParamMode::Write)
            || (matches!(mode, ParamMode::Read) && !declared.is_copy());
        if addressed && self.view_lands_in(answer, declared) {
            return self
                .place_provenance(argument)
                .max(self.value_provenance(argument));
        }
        // The callee was handed a copy. It can still build a view from what that
        // copy points at, so a `[]T` or a `^T` carries its pointee's provenance
        // across. A parameter that holds no view carries nothing at all: an
        // `i64` count says nothing about how long anything lives, and joining it
        // in is what made `slice_span($T, held, total, 0)` read as a leak.
        //
        // Weighed against the declared type, because the copy the callee gets
        // is a view *of the argument* where the argument was an array: passing
        // a local to `slice_range($T, held, from, count)` hands over the local's
        // own storage, and reading the array as the run of numbers it holds let
        // a slice of it come back out as the call's answer.
        if holds_view(declared, self.fields, &mut HashSet::new()) {
            return self.expected_provenance(argument, Some(declared));
        }
        Provenance::Outlives
    }

    /// The storage a value names.
    fn value_provenance(&self, expression: ExprId) -> Provenance {
        let ast = self.ast;
        match ast.expr(expression) {
            // An address of a place names that place's storage.
            Expression::AddressOf(place)
            | Expression::Borrow(place)
            | Expression::BorrowMut(place) => self.place_provenance(*place),
            Expression::Identifier(name) => {
                self.name_provenance(ast.name(*name))
            }
            // Reaching into a value stays inside whatever it named, unless
            // what is read holds no view of its own. A number read out of a
            // struct that also holds a pointer is a number, and carries
            // nothing however short-lived the struct was. This is the same
            // rule the dereference below runs, asked of a field and an
            // element: without it, `holder.count` and `view[0]` were as
            // short-lived as the storage beside them.
            Expression::Index(base, _) | Expression::FieldAccess(base, _) => {
                match self.place_type(expression) {
                    Some(read)
                        if !holds_view(
                            &read,
                            self.fields,
                            &mut HashSet::new(),
                        ) =>
                    {
                        Provenance::Outlives
                    }
                    _ => self.value_provenance(*base),
                }
            }
            // Reading back through a pointer hands out whatever sits there. A
            // scalar read carries no storage however short-lived the pointer
            // was, which is why `p^` under `-> i64` is the ordinary way to read
            // a local. A read whose type holds a view hands the view out again,
            // which is `pp^` where `pp` was taken from a binding that held a
            // frame pointer.
            Expression::Dereference(base) => {
                let pointee = match self.place_type(*base) {
                    Some(
                        Type::Ptr(inner)
                        | Type::Ref(inner)
                        | Type::RefMut(inner),
                    ) => Some(*inner),
                    _ => None,
                };
                match pointee {
                    Some(inner)
                        if !holds_view(
                            &inner,
                            self.fields,
                            &mut HashSet::new(),
                        ) =>
                    {
                        Provenance::Outlives
                    }
                    _ => self.value_provenance(*base),
                }
            }
            Expression::Try(inner) => self.value_provenance(*inner),
            // A value built around views is as short-lived as the shortest of
            // them, so a struct carrying a frame pointer out is caught the same
            // way the bare pointer is.
            Expression::StructInit(_, fields)
            | Expression::EnumVariantInit(_, _, fields) => self.shortest(
                ast.named_in(*fields).iter().map(|field| field.value),
            ),
            Expression::Tuple(items) => {
                self.shortest(ast.exprs_in(*items).iter().copied())
            }
            Expression::Literal(Literal::Array(elements)) => {
                self.shortest(ast.exprs_in(*elements).iter().copied())
            }
            Expression::ArrayRepeat(value, _) => self.value_provenance(*value),
            Expression::Call(callee, arguments) => {
                self.call_provenance(*callee, ast.exprs_in(*arguments))
            }
            Expression::Unsafe(body) => self.block_provenance(*body),
            // A block used as a value answers with whichever branch runs, so it
            // is worth the shortest-lived of them.
            Expression::If(_, consequence, alternative) => {
                let alternative = alternative
                    .map_or(Provenance::Outlives, |block| {
                        self.block_provenance(block)
                    });
                self.block_provenance(*consequence).max(alternative)
            }
            Expression::Switch(_, cases) => ast
                .cases_in(*cases)
                .iter()
                .fold(Provenance::Outlives, |held, case| {
                    held.max(self.block_provenance(case.body))
                }),
            // A value that names no storage places no constraint on how long a
            // view built from it may live. Arithmetic is among them: there is no
            // pointer arithmetic in the surface, so an infix answers with a
            // number.
            Expression::Literal(_)
            | Expression::Boolean(_)
            | Expression::Prefix(..)
            | Expression::Infix(..)
            | Expression::TypeValue(_)
            | Expression::PackMap(..)
            | Expression::Range(..)
            | Expression::Function(..)
            | Expression::Proc(..)
            | Expression::UnsafeFn(_) => Provenance::Outlives,
        }
    }

    /// The shortest-lived of a run of values, which is what a value built from
    /// all of them is worth.
    fn shortest(&self, values: impl Iterator<Item = ExprId>) -> Provenance {
        values.fold(Provenance::Outlives, |held, value| {
            held.max(self.value_provenance(value))
        })
    }

    /// What a block hands back, which is what its last statement answers with.
    /// A block ending in something that is not a value hands back nothing.
    fn block_provenance(&self, block: Range32) -> Provenance {
        block_value(self.ast, block)
            .map_or(Provenance::Outlives, |value| self.value_provenance(value))
    }

    /// What a call answers with. Frost has no global storage and no closures, so
    /// the only storage a callee can name is storage it was handed, which makes
    /// this the join over the arguments rather than a walk into the callee.
    fn call_provenance(
        &self,
        callee: ExprId,
        arguments: &[ExprId],
    ) -> Provenance {
        let ast = self.ast;
        let Expression::Identifier(name) = ast.expr(callee) else {
            // A call through a value: a function pointer read out of a field or
            // a table. Where the signature cannot be read, nothing says where
            // the answer came from, and answering `Outlives` is what let a
            // frame pointer out through `ops.pass(ptr_to(local))`.
            return self
                .place_type(callee)
                .and_then(|held| self.signature_provenance(&held, arguments))
                .unwrap_or(Provenance::Unknown);
        };
        let name = ast.name(*name);
        match name {
            // The surface address-of. What it answers with is the storage of the
            // place it was given.
            "ptr_to" => {
                arguments.iter().fold(Provenance::Outlives, |held, place| {
                    held.max(self.place_provenance(*place))
                })
            }
            // A cast keeps pointing where it pointed, and a slice wraps a
            // pointer in a length, so both answer with their argument's storage.
            "ptr_cast" | "slice_from" => {
                self.shortest(arguments.iter().copied())
            }
            // A count, an offset and a width are numbers, and a type's name is
            // bytes the compiler wrote. None of them names a caller's storage.
            "sizeof" | "alignof" | "type_id" | "offset_of" | "slice_len"
            | "typename" | "name_of" => Provenance::Outlives,
            // A conversion answers with the type it was given. Where that type
            // holds no view it carries no storage, and where it does the
            // storage is whatever was converted.
            "cast" => match arguments
                .first()
                .map(|argument| ast.expr(*argument))
            {
                Some(Expression::TypeValue(ty))
                    if !holds_view(ty, self.fields, &mut HashSet::new()) =>
                {
                    Provenance::Outlives
                }
                _ => self.shortest(arguments.iter().skip(1).copied()),
            },
            _ => {
                // A registration holds its context for as long as it lives, so
                // it names that storage the way a pointer to it would.
                if let Some(shape) = self.registrations.get(name) {
                    return arguments
                        .get(shape.context)
                        .map_or(Provenance::Unknown, |context| {
                            self.place_provenance(*context)
                        });
                }
                match self.views.get(name) {
                    // A callee answering with no view answers with a value, and
                    // a value carries no storage out however it was built.
                    Some(false) => Provenance::Outlives,
                    // C has global storage, so what an extern answers with is
                    // not built from what it was handed. This gives up catching
                    // a C function that does answer with a pointer into one of
                    // its arguments, which `strchr` is the shape of.
                    Some(true) if self.externs.contains(name) => {
                        Provenance::Outlives
                    }
                    Some(true) => {
                        let reaches = self.answer_sources.get(name);
                        arguments.iter().enumerate().fold(
                            Provenance::Outlives,
                            |held, (index, argument)| {
                                if let Some(flags) = reaches
                                    && !flags
                                        .get(index)
                                        .copied()
                                        .unwrap_or(true)
                                {
                                    return held;
                                }
                                held.max(self.argument_provenance(
                                    name, index, *argument, arguments,
                                ))
                            },
                        )
                    }
                    // A name with no body in this program. A compile-time
                    // parameter standing for a function is one: the call site
                    // decides which function it is, so there is no body here to
                    // walk, and the signature it was declared under is what
                    // says where its answer can have come from. A name the walk
                    // never saw at all has neither.
                    None => self
                        .types
                        .get(name)
                        .and_then(|held| {
                            self.signature_provenance(held, arguments)
                        })
                        .unwrap_or(Provenance::Unknown),
                }
            }
        }
    }

    /// What a call weighed by its signature alone answers with. A callee can
    /// only build a view out of what it was handed or out of storage that
    /// outlives the call, so the answer is worth the shortest-lived argument
    /// that could have reached it. A callee handing back a view of its own
    /// frame is caught where that callee is itself walked.
    fn signature_provenance(
        &self,
        signature: &Type,
        arguments: &[ExprId],
    ) -> Option<Provenance> {
        let Type::Proc(params, answer) = signature else {
            return None;
        };
        if !holds_view(answer, self.fields, &mut HashSet::new()) {
            return Some(Provenance::Outlives);
        }
        Some(arguments.iter().enumerate().fold(
            Provenance::Outlives,
            |held, (index, argument)| {
                held.max(self.held_argument(
                    answer,
                    params.get(index),
                    ParamMode::Read,
                    *argument,
                ))
            },
        ))
    }
}
