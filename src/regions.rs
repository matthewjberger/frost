use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::parser::{
    Block, Expression, Program, ReturnKind, Statement, SwitchCase,
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
