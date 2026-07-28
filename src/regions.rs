use std::collections::{HashMap, HashSet};

use anyhow::Result;

use crate::lexer::Position;
use crate::parser::{
    Block, Diagnostic, Expression, Literal, ParamMode, Program, ReturnKind,
    Spanned, Statement, SwitchCase,
};
use crate::types::Type;

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
    returns_pointer: HashMap<String, bool>,
    uses_arena: HashSet<String>,
}

pub fn check_regions(program: &Program) -> Result<()> {
    let diagnostics = check_regions_recovering(program);
    if diagnostics.is_empty() {
        return Ok(());
    }
    Err(anyhow::anyhow!(crate::flatten(&diagnostics, "\n")))
}

/// Check every region in the program, reporting each escape rather than
/// stopping at the first. Regions are independent of one another, and within
/// one an escape does not change what the rest of the block means.
pub fn check_regions_recovering(program: &Program) -> Vec<Diagnostic> {
    let mut signatures = Signatures {
        returns_pointer: HashMap::new(),
        uses_arena: HashSet::new(),
    };
    for statement in program {
        if let Statement::Constant(
            name,
            Expression::Function(_, sig, _) | Expression::Proc(_, sig, _),
        ) = &statement.node
        {
            if matches!(sig.kind, ReturnKind::Single(Type::Ptr(_))) {
                signatures.returns_pointer.insert(name.clone(), true);
            }
            if !sig.uses.is_empty() {
                signatures.uses_arena.insert(name.clone());
            }
        }
    }

    let mut diagnostics = Vec::new();
    for statement in program {
        if let Statement::Constant(
            _,
            Expression::Function(_, sig, body) | Expression::Proc(_, sig, body),
        ) = &statement.node
        {
            // A `uses` function's whole body is a region whose arena is the
            // implicit capability. It may return arena pointers but not leak
            // them into its parameters.
            if let Some(capability) = sig.uses.first() {
                let mut region = Region::new(
                    capability_binding(capability),
                    &signatures,
                    true,
                );
                region.check(body, true);
                diagnostics.append(&mut region.diagnostics);
            }
            find_regions(body, &signatures, &mut diagnostics);
        }
    }
    diagnostics
}

// Walk a block looking for `with` regions to check. An ordinary block imposes no
// region rule of its own.
fn find_regions(
    block: &Block,
    signatures: &Signatures,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for statement in block {
        match &statement.node {
            Statement::With(arena, body) => {
                let mut region = Region::new(arena.clone(), signatures, false);
                region.check(body, true);
                diagnostics.append(&mut region.diagnostics);
                find_regions(body, signatures, diagnostics);
            }
            Statement::While(_, body) | Statement::For(_, _, _, body) => {
                find_regions(body, signatures, diagnostics);
            }
            Statement::Defer(inner) => {
                if let Statement::With(arena, body) = inner.as_ref() {
                    let mut region =
                        Region::new(arena.clone(), signatures, false);
                    region.check(body, true);
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
fn capability_binding(capability: &Type) -> String {
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
        });
    }
}

/// The expression an `unsafe` block answers with.
///
/// `ptr_to` and `slice_from` are refused outside such a block, so every pointer
/// a program can write leaves its frame or its region wrapped in one.
/// A check that does not look through the block does not see the pointer at all.
fn block_value(block: &Block) -> Option<&Expression> {
    match &block.last()?.node {
        Statement::Expression(value) => Some(value),
        _ => None,
    }
}

// The root variable a place is rooted at, so `s.field` and `xs[i]` are rooted at
// `s` and `xs`.
fn root_identifier(place: &Expression) -> Option<&str> {
    match place {
        Expression::Identifier(name) => Some(name),
        Expression::FieldAccess(base, _)
        | Expression::Dereference(base)
        | Expression::Index(base, _) => root_identifier(base),
        _ => None,
    }
}

struct Region<'a> {
    arena: String,
    signatures: &'a Signatures,
    // Whether a returned arena pointer is allowed (true in a `uses` body, false
    // in a `with` block).
    allow_return: bool,
    // Bindings declared inside the region. They die with it, so they may hold a
    // region pointer.
    inner: HashSet<String>,
    // Bindings that currently hold, or transitively contain, a region pointer.
    bound: HashSet<String>,
    // Bindings holding the address of one of those, so reading back through one
    // hands the region pointer out again. This is what tells `pp^` from `p^`
    // without types: `pp` was taken from something already bound, `p` was not.
    via_pointer: HashSet<String>,
    diagnostics: Vec<Diagnostic>,
}

impl<'a> Region<'a> {
    fn new(
        arena: String,
        signatures: &'a Signatures,
        allow_return: bool,
    ) -> Self {
        Region {
            arena,
            signatures,
            allow_return,
            inner: HashSet::new(),
            bound: HashSet::new(),
            via_pointer: HashSet::new(),
            diagnostics: Vec::new(),
        }
    }

    // Whether a value is the address of a binding that already holds a region
    // pointer, which is the only way a dereference can hand one back out.
    fn points_at_region_pointer(&self, value: &Expression) -> bool {
        match value {
            Expression::Unsafe(body) => block_value(body)
                .is_some_and(|inner| self.points_at_region_pointer(inner)),
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(function) = callee.as_ref() else {
                    return false;
                };
                function == "ptr_to"
                    && arguments.iter().any(|argument| {
                        root_identifier(argument)
                            .is_some_and(|root| self.bound.contains(root))
                    })
            }
            // Listed rather than caught by `_`, so a new expression form is a compile error here instead of quietly answering that it points nowhere.
            Expression::Identifier(_)
            | Expression::Literal(_)
            | Expression::ArrayRepeat(..)
            | Expression::Boolean(_)
            | Expression::Sizeof(_)
            | Expression::TypeId(_)
            | Expression::TypeName(_)
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

    fn check(&mut self, block: &Block, root: bool) {
        for statement in block {
            let at = statement.position;
            match &statement.node {
                Statement::Let { name, value, .. }
                | Statement::Constant(name, value) => {
                    self.inner.insert(name.clone());
                    if self.is_region_pointer(value) {
                        self.bound.insert(name.clone());
                    }
                    if self.points_at_region_pointer(value) {
                        self.via_pointer.insert(name.clone());
                    }
                }
                Statement::Assignment(place, value) => {
                    if self.is_region_pointer(value) {
                        self.bind_or_escape(place, "assignment", at);
                    }
                }
                Statement::Return(value) => {
                    if self.is_region_pointer(value) && !self.allow_return {
                        self.escape("being returned", at);
                    }
                }
                Statement::While(_, body) => self.check(body, false),
                Statement::For(variable, _, _, body) => {
                    self.inner.insert(variable.clone());
                    self.check(body, false);
                }
                Statement::With(_, body) => self.check(body, false),
                Statement::Expression(value) => {
                    self.check_conditional(value);
                }
                // A deferred statement runs at scope exit, inside the region
                // still, so what it writes is written from in here.
                Statement::Defer(inner) => {
                    let deferred = vec![Spanned::new((**inner).clone(), at)];
                    self.check(&deferred, false);
                }
                // The multiple-return lowering runs before this check, so this
                // shape is already gone. Named rather than left to a wildcard so
                // it stays that way on purpose.
                Statement::LetMultiple(bindings, value) => {
                    for binding in bindings {
                        self.inner.insert(binding.name.clone());
                        if self.is_region_pointer(value) {
                            self.bound.insert(binding.name.clone());
                        }
                    }
                }
                // `print` writes a value out and stores nothing, and the rest
                // are declarations or control transfers. Listed rather than
                // caught by `_`, so a new statement form is a compile error
                // here instead of a road out of the region nobody walked.
                Statement::Print(..)
                | Statement::Struct(..)
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
        // The block's trailing expression is its value. In a `with` block that
        // value flows to the enclosing scope, so an arena pointer there escapes.
        if root
            && !self.allow_return
            && let Some(last) = block.last()
            && let Statement::Expression(value) = &last.node
            && self.is_region_pointer(value)
        {
            self.escape("being the block's value", last.position);
        }
    }

    // Storing a region pointer into a binding declared inside the region keeps it
    // in the region (and taints that binding). Storing it anywhere else escapes.
    fn bind_or_escape(&mut self, place: &Expression, how: &str, at: Position) {
        match root_identifier(place) {
            Some(name) if self.inner.contains(name) => {
                self.bound.insert(name.to_string());
            }
            _ => self.escape(how, at),
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
    fn check_conditional(&mut self, expression: &Expression) {
        match expression {
            Expression::If(_, consequence, alternative) => {
                self.check(consequence, false);
                if let Some(block) = alternative {
                    self.check(block, false);
                }
            }
            Expression::Switch(_, cases) => {
                for SwitchCase { body, .. } in cases {
                    self.check(body, false);
                }
            }
            Expression::Unsafe(body) => self.check(body, false),
            _ => {}
        }
    }

    fn is_region_pointer(&self, expression: &Expression) -> bool {
        match expression {
            Expression::Identifier(name) => self.bound.contains(name),
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(function) = callee.as_ref() else {
                    return false;
                };
                if function == "ptr_to" || function == "slice_from" {
                    return arguments
                        .iter()
                        .any(|argument| self.mentions_region(argument));
                }
                let returns_pointer = self
                    .signatures
                    .returns_pointer
                    .get(function)
                    .copied()
                    .unwrap_or(false);
                if !returns_pointer {
                    return false;
                }
                // A pointer-returning function hands back an arena pointer only if
                // it draws on this arena. It is a `uses` function, or it is passed
                // the arena (or a value already bound to the region).
                self.signatures.uses_arena.contains(function)
                    || arguments
                        .iter()
                        .any(|argument| self.mentions_region(argument))
            }
            Expression::Unsafe(body) => block_value(body)
                .is_some_and(|value| self.is_region_pointer(value)),
            // `pp^` where `pp` holds the address of a region pointer reads that
            // pointer back out. `p^` where `p` is the region pointer itself
            // reads the value it names, which is not one.
            Expression::Dereference(inner) => root_identifier(inner)
                .is_some_and(|root| self.via_pointer.contains(root)),
            Expression::If(_, consequence, alternative) => {
                let branches = [Some(consequence), alternative.as_ref()];
                branches.into_iter().flatten().any(|block| {
                    block_value(block)
                        .is_some_and(|value| self.is_region_pointer(value))
                })
            }
            Expression::Switch(_, cases) => cases.iter().any(|case| {
                block_value(&case.body)
                    .is_some_and(|value| self.is_region_pointer(value))
            }),
            // Listed rather than caught by `_`, so a new expression form is a compile error here instead of quietly answering that it points nowhere.
            Expression::Literal(_)
            | Expression::ArrayRepeat(..)
            | Expression::Boolean(_)
            | Expression::Sizeof(_)
            | Expression::TypeId(_)
            | Expression::TypeName(_)
            | Expression::TypeValue(_)
            | Expression::PackMap(..)
            | Expression::Prefix(..)
            | Expression::Infix(..)
            | Expression::Range(..)
            | Expression::Function(..)
            | Expression::Proc(..)
            | Expression::UnsafeFn(_)
            | Expression::Try(_)
            | Expression::Index(..)
            | Expression::FieldAccess(..)
            | Expression::AddressOf(_)
            | Expression::Borrow(_)
            | Expression::BorrowMut(_)
            | Expression::StructInit(..)
            | Expression::EnumVariantInit(..)
            | Expression::Tuple(_) => false,
        }
    }

    // Whether an expression reads the arena or a value already bound to the
    // region, so a pointer computed from it belongs to the region.
    fn mentions_region(&self, expression: &Expression) -> bool {
        match expression {
            Expression::Identifier(name) => {
                *name == self.arena || self.bound.contains(name)
            }
            Expression::FieldAccess(base, _)
            | Expression::Dereference(base)
            | Expression::Borrow(base)
            | Expression::BorrowMut(base)
            | Expression::AddressOf(base) => self.mentions_region(base),
            Expression::Index(base, index) => {
                self.mentions_region(base) || self.mentions_region(index)
            }
            Expression::Call(_, arguments) => arguments
                .iter()
                .any(|argument| self.mentions_region(argument)),
            Expression::Unsafe(body) => block_value(body)
                .is_some_and(|value| self.mentions_region(value)),
            // Listed rather than caught by `_`, so a new expression form is a compile error here instead of quietly answering that it points nowhere.
            Expression::Literal(_)
            | Expression::ArrayRepeat(..)
            | Expression::Boolean(_)
            | Expression::Sizeof(_)
            | Expression::TypeId(_)
            | Expression::TypeName(_)
            | Expression::TypeValue(_)
            | Expression::PackMap(..)
            | Expression::Prefix(..)
            | Expression::Infix(..)
            | Expression::Range(..)
            | Expression::Function(..)
            | Expression::Proc(..)
            | Expression::UnsafeFn(_)
            | Expression::Try(_)
            | Expression::StructInit(..)
            | Expression::EnumVariantInit(..)
            | Expression::Tuple(_)
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
pub fn check_frame_escapes(program: &Program) -> Result<()> {
    let diagnostics = check_frame_escapes_recovering(program);
    if diagnostics.is_empty() {
        return Ok(());
    }
    Err(anyhow::anyhow!(crate::flatten(&diagnostics, "\n")))
}

/// Check every function for a view of its own frame leaving it, reporting each
/// rather than stopping at the first. Frames are independent of one another.
pub fn check_frame_escapes_recovering(program: &Program) -> Vec<Diagnostic> {
    // A callback registration keeps a pointer to its context for as long as it
    // is registered, so the value it answers with names storage in this frame
    // exactly as `ptr_to` does. A context in this frame is the ordinary case
    // and is safe, because `check_linearity` forces the registration to be
    // consumed in the function that made it and this check stops it leaving
    // that function by any other road.
    let registrations = crate::callbacks::callback_registrations(program);
    let fields = collect_field_types(program);
    let views = collect_view_returns(program, &fields);
    let param_modes = collect_param_modes(program);
    let return_types = collect_return_types(program);
    // Which functions answer with a place rather than a value. A `ref T` is the
    // only return that does.
    let ref_returns: HashSet<String> = program
        .iter()
        .filter_map(|statement| match &statement.node {
            Statement::Constant(
                name,
                Expression::Function(_, signature, _)
                | Expression::Proc(_, signature, _),
            )
            | Statement::Declared {
                name,
                return_sig: signature,
                ..
            } => match &signature.kind {
                ReturnKind::Single(ty) | ReturnKind::Fallible(ty, _)
                    if matches!(ty, Type::Ref(_) | Type::RefMut(_)) =>
                {
                    Some(name.clone())
                }
                _ => None,
            },
            _ => None,
        })
        .collect();
    // A name declared at the top of the file is not this frame's storage. A
    // function's own name is one of these, so is a constant, and neither dies
    // when the call returns.
    let top_level: HashSet<String> = program
        .iter()
        .filter_map(|statement| match &statement.node {
            Statement::Constant(name, _) => Some(name.clone()),
            Statement::Extern { name, .. }
            | Statement::Declared { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect();
    let mut diagnostics = Vec::new();
    for statement in program {
        if let Statement::Constant(
            name,
            Expression::Function(params, signature, body)
            | Expression::Proc(params, signature, body),
        ) = &statement.node
        {
            let answered = match &signature.kind {
                ReturnKind::Single(ty) | ReturnKind::Fallible(ty, _) => {
                    Some(ty)
                }
                ReturnKind::None | ReturnKind::Multiple(_) => None,
            };
            let mut frame = Frame {
                function: name.clone(),
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
                answers_direct_view: answered.is_some_and(is_direct_view),
                registrations: &registrations,
                views: &views,
                params: &param_modes,
                answers_place_by_name: &ref_returns,
                types: HashMap::new(),
                fields: &fields,
                returns: &return_types,
                diagnostics: Vec::new(),
            };
            for parameter in params {
                if let Some(declared) = &parameter.type_annotation {
                    frame
                        .types
                        .insert(parameter.name.clone(), declared.clone());
                }
                match parameter.mode {
                    // A borrow names the caller's storage, which outlives the
                    // call by definition.
                    ParamMode::Read | ParamMode::Write | ParamMode::Value => {
                        frame.outlives.insert(parameter.name.clone());
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
                        frame.storage.insert(parameter.name.clone());
                        frame.outlives.insert(parameter.name.clone());
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
            frame.check(body, true);
            diagnostics.append(&mut frame.diagnostics);
        }
    }
    diagnostics
}

/// The declared field types of every struct and enum, by the type's name. A
/// return type holds a view when one of its fields does, so the question needs
/// the declarations rather than the type alone.
type FieldTypes = HashMap<String, Vec<(String, Type)>>;

fn collect_field_types(program: &Program) -> FieldTypes {
    let mut fields: FieldTypes = HashMap::new();
    for statement in program {
        match &statement.node {
            Statement::Struct(name, _, declared) => {
                fields.insert(
                    name.clone(),
                    declared
                        .iter()
                        .map(|field| {
                            (field.name.clone(), field.field_type.clone())
                        })
                        .collect(),
                );
            }
            Statement::Enum(name, _, variants) => {
                fields.insert(
                    name.clone(),
                    variants
                        .iter()
                        .filter_map(|variant| variant.fields.as_ref())
                        .flatten()
                        .map(|field| {
                            (field.name.clone(), field.field_type.clone())
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

fn collect_param_modes(program: &Program) -> ParamModes {
    let mut modes = HashMap::new();
    for statement in program {
        let (name, params) = match &statement.node {
            Statement::Constant(
                name,
                Expression::Function(params, _, _)
                | Expression::Proc(params, _, _),
            ) => (name, params),
            Statement::Extern { name, params, .. }
            | Statement::Declared { name, params, .. } => (name, params),
            _ => continue,
        };
        modes.insert(
            name.clone(),
            params
                .iter()
                .map(|parameter| {
                    (parameter.mode, parameter.type_annotation.clone())
                })
                .collect(),
        );
    }
    modes
}

/// What each function answers with, by name.
fn collect_return_types(program: &Program) -> HashMap<String, Type> {
    let mut returns = HashMap::new();
    for statement in program {
        match &statement.node {
            Statement::Constant(
                name,
                Expression::Function(_, signature, _)
                | Expression::Proc(_, signature, _),
            )
            | Statement::Declared {
                name,
                return_sig: signature,
                ..
            } => {
                if let Some(ty) = signature.to_type() {
                    returns.insert(name.clone(), ty);
                }
            }
            Statement::Extern {
                name,
                return_type: Some(return_type),
                ..
            } => {
                returns.insert(name.clone(), return_type.clone());
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
    program: &Program,
    fields: &FieldTypes,
) -> HashMap<String, bool> {
    let mut views = HashMap::new();
    for statement in program {
        let (name, kind) = match &statement.node {
            Statement::Constant(
                name,
                Expression::Function(_, signature, _)
                | Expression::Proc(_, signature, _),
            )
            | Statement::Declared {
                name,
                return_sig: signature,
                ..
            } => (name, signature.kind.clone()),
            Statement::Extern {
                name, return_type, ..
            } => {
                let holds = return_type.as_ref().is_some_and(|ty| {
                    holds_view(ty, fields, &mut HashSet::new())
                });
                views.insert(name.clone(), holds);
                continue;
            }
            _ => continue,
        };
        let holds = match &kind {
            ReturnKind::Single(ty) | ReturnKind::Fallible(ty, _) => {
                holds_view(ty, fields, &mut HashSet::new())
            }
            ReturnKind::None | ReturnKind::Multiple(_) => false,
        };
        views.insert(name.clone(), holds);
    }
    views
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
    // Whether it answers with a view itself rather than with a value holding
    // one, which is what decides whether handing back a place takes its
    // address.
    answers_direct_view: bool,
    // Callback registrations in the program, and which argument of each is the
    // context whose storage it keeps.
    registrations: &'a HashMap<String, crate::callbacks::CallbackShape>,
    // Whether each function answers with something holding a view.
    views: &'a HashMap<String, bool>,
    // How each function takes its arguments, which says whether a callee was
    // handed the address of what it was passed or a copy of it.
    params: &'a ParamModes,
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
    fn check(&mut self, block: &Block, answers: bool) {
        for (index, statement) in block.iter().enumerate() {
            let last = index + 1 == block.len();
            let at = statement.position;
            match &statement.node {
                Statement::Let {
                    name,
                    value,
                    type_annotation,
                    ..
                } => {
                    let mut held = self.value_provenance(value);
                    // A binding declared as a view and given a place takes the
                    // address of that place, so `view : []i64 = arr` holds a
                    // view of the array rather than a copy of it.
                    if let Some(declared) = type_annotation
                        && is_direct_view(declared)
                    {
                        held = held.max(self.coercion_provenance(value));
                    }
                    self.storage.insert(name.clone());
                    self.locals.insert(name.clone(), held);
                    if let Some(borrowed) = self.borrowed_place(value) {
                        self.places.insert(name.clone(), borrowed);
                    }
                    if let Some(declared) = type_annotation {
                        self.types.insert(name.clone(), declared.clone());
                    } else if let Some(inferred) = self.value_type(value) {
                        self.types.insert(name.clone(), inferred);
                    }
                }
                Statement::LetMultiple(bindings, value) => {
                    let held = self.value_provenance(value);
                    for binding in bindings {
                        self.storage.insert(binding.name.clone());
                        self.locals.insert(binding.name.clone(), held);
                    }
                }
                // A function written inside a body is a declaration rather than
                // a value this frame holds. Its own body is checked where the
                // walk reaches it as an item of the program.
                Statement::Constant(
                    _,
                    Expression::Function(..) | Expression::Proc(..),
                ) => {}
                Statement::Constant(name, value) => {
                    let held = self.value_provenance(value);
                    self.storage.insert(name.clone());
                    self.locals.insert(name.clone(), held);
                }
                Statement::Return(value) => self.judge(value, "returned", at),
                Statement::Assignment(place, value) => {
                    self.assign(place, value, at);
                }
                Statement::While(_, body) | Statement::With(_, body) => {
                    self.check(body, false);
                }
                // A loop variable is this frame's storage the way a local is, so
                // an address of one dies when the call returns.
                Statement::For(name, second, _, body) => {
                    self.storage.insert(name.clone());
                    if let Some(second) = second {
                        self.storage.insert(second.clone());
                    }
                    self.check(body, false);
                }
                Statement::Defer(inner) => {
                    let deferred = vec![Spanned::new((**inner).clone(), at)];
                    self.check(&deferred, false);
                }
                Statement::Expression(value) => {
                    self.check_expression_statement(value, last && answers, at);
                }
                // `print` writes a value out and hands nothing to the caller, so
                // no storage leaves the frame through one. The rest are
                // declarations and control transfers. Listed rather than caught
                // by `_`, so a new statement form is a compile error here rather
                // than a road out of the frame that nobody walked.
                Statement::Print(..)
                | Statement::Struct(..)
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
    }

    /// A statement that is an expression. A block used as a value answers for
    /// the whole function, so its branches are walked and, where the block is
    /// the answer, so is what each one ends with. `match` was missing here, and
    /// a `return` inside an arm was never seen at all.
    fn check_expression_statement(
        &mut self,
        value: &Expression,
        answers: bool,
        at: Position,
    ) {
        match value {
            Expression::If(_, consequence, alternative) => {
                self.answers_here(consequence, answers);
                if let Some(block) = alternative {
                    self.answers_here(block, answers);
                }
            }
            Expression::Switch(_, cases) => {
                for case in cases {
                    self.answers_here(&case.body, answers);
                }
            }
            // An `unsafe` block is transparent here. `ptr_to` is refused outside
            // one, so its statements are where a frame pointer is formed, bound
            // and returned, and a walk that steps over the block never sees any
            // of it.
            Expression::Unsafe(body) => self.check(body, answers),
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
            | Expression::Sizeof(_)
            | Expression::TypeName(_)
            | Expression::TypeId(_)
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
    fn answers_here(&mut self, block: &Block, answers: bool) {
        self.check(block, answers);
        if answers
            && let Some(last) = block.last()
            && let Statement::Expression(value) = &last.node
        {
            self.judge(value, "the call's answer", last.position);
        }
    }

    /// Writing a view into a place the call cannot see hands it to the caller
    /// just as returning it does, and the caller's frame outlives this one.
    /// Writing one into this frame's own storage keeps it here, and that place
    /// now holds whatever the value named: without recording it, a pointer
    /// stored in a local and returned afterwards left with nobody having asked.
    fn assign(&mut self, place: &Expression, value: &Expression, at: Position) {
        let held = self.value_provenance(value);
        if self.place_provenance(place) == Provenance::Frame {
            if let Some(root) = root_identifier(place) {
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
    fn judge(&mut self, value: &Expression, how: &str, at: Position) {
        let mut held = self.value_provenance(value);
        if self.answers_place {
            held = held.max(self.place_provenance(value));
        }
        if self.answers_direct_view {
            held = held.max(self.coercion_provenance(value));
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
    fn view_lands_in(&self, answer: &Type, parameter: &Type) -> bool {
        let element = match answer {
            Type::Ptr(inner)
            | Type::Slice(inner)
            | Type::Ref(inner)
            | Type::RefMut(inner) => inner.as_ref(),
            // A `str` views bytes, and an aggregate holding views could hold any
            // of them. Neither says enough to rule the parameter out.
            _ => return true,
        };
        self.holds_inline(element, parameter, &mut HashSet::new())
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

    /// The declared type of a place, as far as the annotations and the struct
    /// declarations give it. `None` means the walk cannot tell, and the rules
    /// that ask treat that as the answer that keeps storage in.
    fn place_type(&self, place: &Expression) -> Option<Type> {
        match place {
            Expression::Identifier(name) => self.types.get(name).cloned(),
            Expression::FieldAccess(base, field) => {
                let name = match self.place_type(base)? {
                    Type::Struct(name) | Type::Enum(name) => name,
                    Type::Ptr(inner)
                    | Type::Ref(inner)
                    | Type::RefMut(inner) => match *inner {
                        Type::Struct(name) | Type::Enum(name) => name,
                        _ => return None,
                    },
                    _ => return None,
                };
                self.fields
                    .get(Type::template_of(&name))?
                    .iter()
                    .find(|(declared, _)| declared == field)
                    .map(|(_, ty)| ty.clone())
            }
            Expression::Index(base, _) => match self.place_type(base)? {
                Type::Array(inner, _)
                | Type::ArrayGeneric(inner, _)
                | Type::Slice(inner)
                | Type::Ptr(inner) => Some(*inner),
                Type::Str => Some(Type::U8),
                _ => None,
            },
            Expression::Dereference(base) => match self.place_type(base)? {
                Type::Ptr(inner) | Type::Ref(inner) | Type::RefMut(inner) => {
                    Some(*inner)
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// The type a binding takes from its value, for the shapes that say so
    /// plainly. Only enough to answer the indexing question above.
    fn value_type(&self, value: &Expression) -> Option<Type> {
        match value {
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(name) = callee.as_ref() else {
                    return None;
                };
                match name.as_str() {
                    // The three that build a view are not declared anywhere, so
                    // their types are written here. Without them a pointer bound
                    // from `ptr_to` has no type, and the dereference rule cannot
                    // tell a scalar read from a view read.
                    "ptr_to" => Some(Type::Ptr(Box::new(
                        arguments
                            .first()
                            .and_then(|place| self.place_type(place))
                            .unwrap_or(Type::Unknown),
                    ))),
                    "ptr_cast" => match arguments.first() {
                        Some(Expression::TypeValue(inner)) => {
                            Some(Type::Ptr(Box::new(inner.clone())))
                        }
                        _ => None,
                    },
                    "slice_from" => match arguments.first() {
                        Some(Expression::TypeValue(inner)) => {
                            Some(Type::Slice(Box::new(inner.clone())))
                        }
                        _ => None,
                    },
                    _ => self.returns.get(name).cloned(),
                }
            }
            Expression::Unsafe(body) => {
                block_value(body).and_then(|inner| self.value_type(inner))
            }
            Expression::Literal(Literal::String(_)) => Some(Type::Str),
            Expression::Literal(Literal::Array(elements)) => {
                let element = elements
                    .first()
                    .and_then(|first| self.value_type(first))
                    .unwrap_or(Type::Unknown);
                Some(Type::Array(Box::new(element), elements.len()))
            }
            Expression::ArrayRepeat(inner, count) => Some(Type::ArrayGeneric(
                Box::new(self.value_type(inner).unwrap_or(Type::Unknown)),
                count.clone(),
            )),
            Expression::StructInit(name, _) => Some(Type::Struct(name.clone())),
            Expression::EnumVariantInit(name, _, _) => {
                Some(Type::Enum(name.clone()))
            }
            Expression::Literal(Literal::Integer(_)) => Some(Type::I64),
            Expression::Literal(Literal::Float(_)) => Some(Type::F64),
            Expression::Literal(Literal::Float32(_)) => Some(Type::F32),
            Expression::Literal(Literal::Boolean(_))
            | Expression::Boolean(_) => Some(Type::Bool),
            _ => self.place_type(value),
        }
    }

    /// The storage a view takes the address of when it is built by coercion.
    ///
    /// `view : []i64 = arr` and `fn() -> []i64 { arr }` both hand back a view of
    /// the array rather than a copy of it, so the array's storage is what the
    /// view names. A value that is already a view carries its own provenance and
    /// no new address is taken, which is what keeps a slice parameter passed
    /// straight back out from reading as this frame's.
    fn coercion_provenance(&self, value: &Expression) -> Provenance {
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
    fn borrowed_place(&self, value: &Expression) -> Option<Provenance> {
        match value {
            Expression::Borrow(place) | Expression::BorrowMut(place) => {
                Some(self.place_provenance(place))
            }
            Expression::Unsafe(body) => {
                block_value(body).and_then(|inner| self.borrowed_place(inner))
            }
            Expression::Call(callee, _) => {
                let Expression::Identifier(name) = callee.as_ref() else {
                    return None;
                };
                self.answers_place_by_name
                    .contains(name)
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
    fn place_provenance(&self, place: &Expression) -> Provenance {
        match place {
            Expression::Identifier(name) => {
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
            Expression::Index(base, _) => match self.place_type(base) {
                Some(Type::Slice(_) | Type::Str | Type::Ptr(_)) => {
                    self.value_provenance(base)
                }
                _ => self.place_provenance(base),
            },
            Expression::FieldAccess(base, _)
            | Expression::Borrow(base)
            | Expression::BorrowMut(base) => self.place_provenance(base),
            Expression::Dereference(base) => self.value_provenance(base),
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
            | Expression::Sizeof(_)
            | Expression::TypeName(_)
            | Expression::TypeId(_)
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
    fn argument_provenance(
        &self,
        callee: &str,
        index: usize,
        argument: &Expression,
    ) -> Provenance {
        let held = self
            .params
            .get(callee)
            .and_then(|params| params.get(index))
            .cloned();
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
        argument: &Expression,
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
        if holds_view(declared, self.fields, &mut HashSet::new()) {
            return self.value_provenance(argument);
        }
        Provenance::Outlives
    }

    /// The storage a value names.
    fn value_provenance(&self, expression: &Expression) -> Provenance {
        match expression {
            // An address of a place names that place's storage.
            Expression::AddressOf(place)
            | Expression::Borrow(place)
            | Expression::BorrowMut(place) => self.place_provenance(place),
            Expression::Identifier(name) => self.name_provenance(name),
            // Reaching into a value stays inside whatever it named.
            Expression::Index(base, _) | Expression::FieldAccess(base, _) => {
                self.value_provenance(base)
            }
            // Reading back through a pointer hands out whatever sits there. A
            // scalar read carries no storage however short-lived the pointer
            // was, which is why `p^` under `-> i64` is the ordinary way to read
            // a local. A read whose type holds a view hands the view out again,
            // which is `pp^` where `pp` was taken from a binding that held a
            // frame pointer.
            Expression::Dereference(base) => {
                let pointee = match self.place_type(base) {
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
                    _ => self.value_provenance(base),
                }
            }
            Expression::Try(inner) => self.value_provenance(inner),
            // A value built around views is as short-lived as the shortest of
            // them, so a struct carrying a frame pointer out is caught the same
            // way the bare pointer is.
            Expression::StructInit(_, fields)
            | Expression::EnumVariantInit(_, _, fields) => {
                self.shortest(fields.iter().map(|(_, value)| value))
            }
            Expression::Tuple(items) => self.shortest(items.iter()),
            Expression::Literal(Literal::Array(elements)) => {
                self.shortest(elements.iter())
            }
            Expression::ArrayRepeat(value, _) => self.value_provenance(value),
            Expression::Call(callee, arguments) => {
                self.call_provenance(callee, arguments)
            }
            Expression::Unsafe(body) => self.block_provenance(body),
            // A block used as a value answers with whichever branch runs, so it
            // is worth the shortest-lived of them.
            Expression::If(_, consequence, alternative) => {
                let alternative = alternative
                    .as_ref()
                    .map_or(Provenance::Outlives, |block| {
                        self.block_provenance(block)
                    });
                self.block_provenance(consequence).max(alternative)
            }
            Expression::Switch(_, cases) => {
                cases.iter().fold(Provenance::Outlives, |held, case| {
                    held.max(self.block_provenance(&case.body))
                })
            }
            // A value that names no storage places no constraint on how long a
            // view built from it may live. Arithmetic is among them: there is no
            // pointer arithmetic in the surface, so an infix answers with a
            // number.
            Expression::Literal(_)
            | Expression::Boolean(_)
            | Expression::Prefix(..)
            | Expression::Infix(..)
            | Expression::Sizeof(_)
            | Expression::TypeName(_)
            | Expression::TypeId(_)
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
    fn shortest<'value>(
        &self,
        values: impl Iterator<Item = &'value Expression>,
    ) -> Provenance {
        values.fold(Provenance::Outlives, |held, value| {
            held.max(self.value_provenance(value))
        })
    }

    /// What a block hands back, which is what its last statement answers with.
    /// A block ending in something that is not a value hands back nothing.
    fn block_provenance(&self, block: &Block) -> Provenance {
        block_value(block)
            .map_or(Provenance::Outlives, |value| self.value_provenance(value))
    }

    /// What a call answers with. Frost has no global storage and no closures, so
    /// the only storage a callee can name is storage it was handed, which makes
    /// this the join over the arguments rather than a walk into the callee.
    fn call_provenance(
        &self,
        callee: &Expression,
        arguments: &[Expression],
    ) -> Provenance {
        let Expression::Identifier(name) = callee else {
            // A call through a value: a function pointer read out of a field or
            // a table. The signature it holds says what it answers with and how
            // it takes each argument, so a call through one is weighed the same
            // way a named call is. Where that signature cannot be read, nothing
            // says where the answer came from, and answering `Outlives` is what
            // let a frame pointer out through `ops.pass(ptr_to(local))`.
            let Some(Type::Proc(params, answer)) = self.place_type(callee)
            else {
                return Provenance::Unknown;
            };
            if !holds_view(&answer, self.fields, &mut HashSet::new()) {
                return Provenance::Outlives;
            }
            return arguments.iter().enumerate().fold(
                Provenance::Outlives,
                |held, (index, argument)| {
                    let declared = params.get(index);
                    held.max(self.held_argument(
                        &answer,
                        declared,
                        ParamMode::Read,
                        argument,
                    ))
                },
            );
        };
        match name.as_str() {
            // The surface address-of. What it answers with is the storage of the
            // place it was given.
            "ptr_to" => {
                arguments.iter().fold(Provenance::Outlives, |held, place| {
                    held.max(self.place_provenance(place))
                })
            }
            // A cast keeps pointing where it pointed, and a slice wraps a
            // pointer in a length, so both answer with their argument's storage.
            "ptr_cast" | "slice_from" => self.shortest(arguments.iter()),
            _ => {
                // A registration holds its context for as long as it lives, so
                // it names that storage the way a pointer to it would.
                if let Some(shape) = self.registrations.get(name) {
                    return arguments
                        .get(shape.context)
                        .map_or(Provenance::Unknown, |context| {
                            self.place_provenance(context)
                        });
                }
                match self.views.get(name) {
                    // A callee answering with no view answers with a value, and
                    // a value carries no storage out however it was built.
                    Some(false) => Provenance::Outlives,
                    Some(true) => arguments.iter().enumerate().fold(
                        Provenance::Outlives,
                        |held, (index, argument)| {
                            held.max(
                                self.argument_provenance(name, index, argument),
                            )
                        },
                    ),
                    // A name with no signature in this program: a compile-time
                    // parameter standing for a function, or one the walk never
                    // saw.
                    None => Provenance::Unknown,
                }
            }
        }
    }
}
