use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::parser::{
    Block, Expression, Program, ReturnKind, Spanned, Statement, SwitchCase,
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
// A pointer confined to a binding declared in the region is fine; that binding
// dies with the region.

struct Signatures {
    returns_pointer: HashMap<String, bool>,
    uses_arena: HashSet<String>,
}

pub fn check_regions(program: &Program) -> Result<()> {
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

    for statement in program {
        if let Statement::Constant(
            _,
            Expression::Function(_, sig, body) | Expression::Proc(_, sig, body),
        ) = &statement.node
        {
            // A `uses` function's whole body is a region whose arena is the
            // implicit capability; it may return arena pointers but not leak
            // them into its parameters.
            if let Some(capability) = sig.uses.first() {
                let mut region = Region::new(
                    capability_binding(capability),
                    &signatures,
                    true,
                );
                region.check(body, true)?;
            }
            find_regions(body, &signatures)?;
        }
    }
    Ok(())
}

// Walk a block looking for `with` regions to check; an ordinary block imposes no
// region rule of its own.
fn find_regions(block: &Block, signatures: &Signatures) -> Result<()> {
    for statement in block {
        match &statement.node {
            Statement::With(arena, body) => {
                let mut region = Region::new(arena.clone(), signatures, false);
                region.check(body, true)?;
                find_regions(body, signatures)?;
            }
            Statement::While(_, body) | Statement::For(_, _, body) => {
                find_regions(body, signatures)?;
            }
            Statement::Defer(inner) => {
                if let Statement::With(arena, body) = inner.as_ref() {
                    let mut region =
                        Region::new(arena.clone(), signatures, false);
                    region.check(body, true)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
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
    // Bindings declared inside the region; they die with it, so they may hold a
    // region pointer.
    inner: HashSet<String>,
    // Bindings that currently hold, or transitively contain, a region pointer.
    bound: HashSet<String>,
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
        }
    }

    fn check(&mut self, block: &Block, root: bool) -> Result<()> {
        for statement in block {
            match &statement.node {
                Statement::Let { name, value, .. }
                | Statement::Constant(name, value) => {
                    self.inner.insert(name.clone());
                    if self.is_region_pointer(value) {
                        self.bound.insert(name.clone());
                    }
                }
                Statement::Assignment(place, value) => {
                    if self.is_region_pointer(value) {
                        self.bind_or_escape(place, "assignment")?;
                    }
                }
                Statement::Return(value) => {
                    if self.is_region_pointer(value) && !self.allow_return {
                        bail!(self.escape("being returned"));
                    }
                }
                Statement::While(_, body) => self.check(body, false)?,
                Statement::For(variable, _, body) => {
                    self.inner.insert(variable.clone());
                    self.check(body, false)?;
                }
                Statement::With(_, body) => self.check(body, false)?,
                Statement::Expression(value) => {
                    self.check_conditional(value)?;
                }
                _ => {}
            }
        }
        // The block's trailing expression is its value; in a `with` block that
        // value flows to the enclosing scope, so an arena pointer there escapes.
        if root
            && !self.allow_return
            && let Some(last) = block.last()
            && let Statement::Expression(value) = &last.node
            && self.is_region_pointer(value)
        {
            bail!(self.escape("being the block's value"));
        }
        Ok(())
    }

    // Storing a region pointer into a binding declared inside the region keeps it
    // in the region (and taints that binding); storing it anywhere else escapes.
    fn bind_or_escape(&mut self, place: &Expression, how: &str) -> Result<()> {
        match root_identifier(place) {
            Some(name) if self.inner.contains(name) => {
                self.bound.insert(name.to_string());
                Ok(())
            }
            _ => bail!(self.escape(how)),
        }
    }

    fn escape(&self, how: &str) -> String {
        format!(
            "region: a pointer into arena '{}' escapes its region by {how}; it may not outlive the arena",
            self.arena
        )
    }

    // An `if`/`match` used as a statement carries blocks that are still inside
    // the region.
    fn check_conditional(&mut self, expression: &Expression) -> Result<()> {
        match expression {
            Expression::If(_, consequence, alternative) => {
                self.check(consequence, false)?;
                if let Some(block) = alternative {
                    self.check(block, false)?;
                }
            }
            Expression::Switch(_, cases) => {
                for SwitchCase { body, .. } in cases {
                    self.check(body, false)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn is_region_pointer(&self, expression: &Expression) -> bool {
        match expression {
            Expression::Identifier(name) => self.bound.contains(name),
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(function) = callee.as_ref() else {
                    return false;
                };
                if function == "ptr_to" {
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
                // it draws on this arena: it is a `uses` function, or it is passed
                // the arena (or a value already bound to the region).
                self.signatures.uses_arena.contains(function)
                    || arguments
                        .iter()
                        .any(|argument| self.mentions_region(argument))
            }
            _ => false,
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
            _ => false,
        }
    }
}

// The frame check. A function's locals die when it returns, so a pointer or a
// slice into one of them may not be the thing it answers with. This is the same
// question the region check asks about an arena, asked about the frame instead,
// and it is what stops `ptr_to(local)` and a slice over a local array from
// outliving the storage they name.
//
// Provenance again rather than rooting: a local holding a pointer it was handed
// is fine to return, and only a pointer formed from this frame's own storage is
// not.
pub fn check_frame_escapes(program: &Program) -> Result<()> {
    // A callback registration keeps a pointer to its context for as long as it
    // is registered, so the value it answers with names storage in this frame
    // exactly as `ptr_to` does. A context in this frame is the ordinary case
    // and is safe, because `check_linearity` forces the registration to be
    // consumed in the function that made it and this check stops it leaving
    // that function by any other road. See docs/callbacks.md.
    let registrations = crate::callbacks::callback_registrations(program);
    for statement in program {
        if let Statement::Constant(
            name,
            Expression::Function(_, signature, body)
            | Expression::Proc(_, signature, body),
        ) = &statement.node
        {
            let escapes = match &signature.kind {
                ReturnKind::Single(ty) => is_borrowed_view(ty),
                ReturnKind::Fallible(ty, _) => is_borrowed_view(ty),
                _ => false,
            };
            let mut frame = Frame {
                function: name.clone(),
                storage: HashSet::new(),
                materialized: HashSet::new(),
                bound: HashSet::new(),
                answers_view: escapes,
                registrations: &registrations,
            };
            frame.check(body)?;
        }
    }
    Ok(())
}

// A type that names storage it does not own: a raw pointer, a slice, or a
// borrow. A returned borrow is held to the frame the same way a returned
// pointer is, so a view of storage built here cannot leave as one.
fn is_borrowed_view(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Ptr(_) | Type::Slice(_) | Type::Ref(_) | Type::RefMut(_)
    )
}

struct Frame<'a> {
    function: String,
    // Locals whose storage is this frame's.
    storage: HashSet<String>,
    // Locals that built their value here rather than being handed one, so a
    // view of them is a view of this frame.
    materialized: HashSet<String>,
    // Locals holding something that points into this frame.
    bound: HashSet<String>,
    // Whether this function answers with a pointer or a slice at all.
    answers_view: bool,
    // Callback registrations in the program, and which argument of each is the
    // context whose storage it keeps.
    registrations: &'a HashMap<String, crate::callbacks::CallbackShape>,
}

impl Frame<'_> {
    fn check(&mut self, block: &Block) -> Result<()> {
        for (index, statement) in block.iter().enumerate() {
            let last = index + 1 == block.len();
            match &statement.node {
                Statement::Let {
                    name,
                    value,
                    type_annotation,
                    ..
                } => {
                    let views_frame =
                        type_annotation.as_ref().is_some_and(is_borrowed_view)
                            && self.views_this_frame(value);
                    if self.points_into_frame(value) || views_frame {
                        self.bound.insert(name.clone());
                    }
                    if materializes(value) {
                        self.materialized.insert(name.clone());
                    }
                    self.storage.insert(name.clone());
                }
                Statement::Return(value) => {
                    if self.points_into_frame(value) {
                        bail!(self.escape("returned"));
                    }
                }
                // Writing one into a parameter hands it to the caller just as
                // returning it does, and the caller's frame outlives this one.
                Statement::Assignment(place, value) => {
                    if self.points_into_frame(value) && !self.rooted_here(place)
                    {
                        bail!(self.escape("stored where the call cannot see"));
                    }
                }
                Statement::While(_, body)
                | Statement::With(_, body)
                | Statement::For(_, _, body) => self.check(body)?,
                Statement::Defer(inner) => {
                    let deferred = vec![Spanned::new(
                        (**inner).clone(),
                        statement.position,
                    )];
                    self.check(&deferred)?;
                }
                // A block used as a value answers for the whole function, so
                // its branches are checked and, when it is the last statement,
                // so is what each branch ends with.
                Statement::Expression(Expression::If(
                    _,
                    consequence,
                    alternative,
                )) => {
                    self.answers_here(consequence, last)?;
                    if let Some(block) = alternative {
                        self.answers_here(block, last)?;
                    }
                }
                Statement::Expression(value)
                    if last && self.points_into_frame(value) =>
                {
                    bail!(self.escape("the call's answer"));
                }
                _ => {}
            }
        }
        Ok(())
    }

    // A branch of a block used as a value: check it as a block, and when the
    // block is the function's answer, check what the branch ends with too.
    fn answers_here(&mut self, block: &Block, answers: bool) -> Result<()> {
        self.check(block)?;
        if answers
            && let Some(last) = block.last()
            && let Statement::Expression(value) = &last.node
            && self.points_into_frame(value)
        {
            bail!(self.escape("the call's answer"));
        }
        Ok(())
    }

    fn escape(&self, how: &str) -> String {
        format!(
            "region: a pointer into the frame of '{}' is {how}; the storage it names dies when the call returns",
            self.function
        )
    }

    // Whether a value points into this frame: an address taken of storage here,
    // a binding already known to hold one, or, when the function answers with a
    // slice, a view of storage here.
    fn points_into_frame(&self, expression: &Expression) -> bool {
        match expression {
            Expression::AddressOf(place)
            | Expression::Borrow(place)
            | Expression::BorrowMut(place) => self.rooted_here(place),
            Expression::Identifier(name) => {
                self.bound.contains(name)
                    || (self.answers_view && self.materialized.contains(name))
            }
            Expression::Index(base, _) | Expression::FieldAccess(base, _) => {
                self.points_into_frame(base)
            }
            // A value built around one carries it out.
            Expression::StructInit(_, fields)
            | Expression::EnumVariantInit(_, _, fields) => fields
                .iter()
                .any(|(_, value)| self.points_into_frame(value)),
            Expression::Tuple(items) => {
                items.iter().any(|item| self.points_into_frame(item))
            }
            // `ptr_to(x)` is how an address is written, and `ptr_cast` keeps
            // pointing where it pointed.
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(name) = callee.as_ref() else {
                    return false;
                };
                match name.as_str() {
                    "ptr_to" => {
                        arguments.iter().any(|place| self.rooted_here(place))
                    }
                    "ptr_cast" => arguments
                        .iter()
                        .any(|inner| self.points_into_frame(inner)),
                    // A registration holds its context for as long as it
                    // lives, so it names that storage the way a pointer to it
                    // would.
                    _ => self
                        .registrations
                        .get(name)
                        .and_then(|shape| arguments.get(shape.context))
                        .is_some_and(|context| self.rooted_here(context)),
                }
            }
            _ => false,
        }
    }

    // Whether a value is a view of storage built in this frame, which is what
    // makes a slice over a local array escape while a slice handed in does not.
    fn views_this_frame(&self, expression: &Expression) -> bool {
        match expression {
            Expression::Identifier(name) => {
                self.materialized.contains(name) || self.bound.contains(name)
            }
            Expression::Index(base, _)
            | Expression::FieldAccess(base, _)
            | Expression::Range(base, _, _) => self.views_this_frame(base),
            _ => false,
        }
    }

    // Whether a place names storage belonging to this frame.
    fn rooted_here(&self, place: &Expression) -> bool {
        match place {
            Expression::Identifier(name) => self.storage.contains(name),
            Expression::Index(base, _)
            | Expression::FieldAccess(base, _)
            | Expression::Borrow(base)
            | Expression::BorrowMut(base) => self.rooted_here(base),
            _ => false,
        }
    }
}

// Whether a binding's value is storage built here rather than one handed in. An
// array or a struct written out lands in this frame; anything else came from
// somewhere that outlives it.
fn materializes(expression: &Expression) -> bool {
    matches!(
        expression,
        Expression::Literal(crate::parser::Literal::Array(_))
            | Expression::Tuple(_)
            | Expression::StructInit(..)
            | Expression::EnumVariantInit(..)
    )
}
