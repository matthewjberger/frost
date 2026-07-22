use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::parser::{
    Block, Expression, Program, ReturnKind, Statement, SwitchCase,
};
use crate::types::Type;

// The region check. A `with arena { ... }` block is a region: the arena is live
// only inside it. A raw pointer derived from the arena (returned by an allocator
// that takes the arena, or by a `uses` function, or taken with `ptr_to` of arena
// memory) is region-bound and must not outlive the block. Returning such a
// pointer, or storing it in a binding that lives past the block, would leave a
// pointer into memory the arena may reset or free, so it is rejected here, before
// the `with` block is lowered away.

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
            Expression::Function(_, _, body) | Expression::Proc(_, _, body),
        ) = &statement.node
        {
            find_regions(body, &signatures)?;
        }
    }
    Ok(())
}

// Walk a block looking for `with` regions to check; ordinary blocks impose no
// region rule of their own.
fn find_regions(block: &Block, signatures: &Signatures) -> Result<()> {
    for statement in block {
        match &statement.node {
            Statement::With(arena, body) => {
                let mut region = Region {
                    arena: arena.clone(),
                    signatures,
                    inner: HashSet::new(),
                    bound: HashSet::new(),
                };
                region.check(body)?;
                find_regions(body, signatures)?;
            }
            Statement::While(_, body) | Statement::For(_, _, body) => {
                find_regions(body, signatures)?;
            }
            Statement::Defer(inner) => {
                if let Statement::With(arena, body) = inner.as_ref() {
                    let mut region = Region {
                        arena: arena.clone(),
                        signatures,
                        inner: HashSet::new(),
                        bound: HashSet::new(),
                    };
                    region.check(body)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

struct Region<'a> {
    arena: String,
    signatures: &'a Signatures,
    // Bindings declared inside the region; they die with it, so they may hold a
    // region pointer.
    inner: HashSet<String>,
    // Bindings currently holding a region pointer.
    bound: HashSet<String>,
}

impl Region<'_> {
    fn check(&mut self, block: &Block) -> Result<()> {
        for statement in block {
            match &statement.node {
                Statement::Let { name, value, .. } => {
                    self.inner.insert(name.clone());
                    if self.is_region_pointer(value) {
                        self.bound.insert(name.clone());
                    }
                }
                Statement::Constant(name, value) => {
                    self.inner.insert(name.clone());
                    if self.is_region_pointer(value) {
                        self.bound.insert(name.clone());
                    }
                }
                Statement::Assignment(place, value) => {
                    if self.is_region_pointer(value)
                        && !self.is_inner_place(place)
                    {
                        bail!(
                            "region: a pointer into arena '{}' escapes its `with` block by assignment; it may not outlive the region",
                            self.arena
                        );
                    }
                }
                Statement::Return(value) => {
                    if self.is_region_pointer(value) {
                        bail!(
                            "region: a pointer into arena '{}' escapes its `with` block by being returned; it may not outlive the region",
                            self.arena
                        );
                    }
                }
                Statement::While(condition, body) => {
                    let _ = condition;
                    self.check(body)?;
                }
                Statement::For(variable, _, body) => {
                    self.inner.insert(variable.clone());
                    self.check(body)?;
                }
                Statement::With(_, body) => {
                    self.check(body)?;
                }
                Statement::Expression(value) => {
                    self.check_conditional(value)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    // An `if`/`match` used as a statement carries blocks that are still inside
    // the region.
    fn check_conditional(&mut self, expression: &Expression) -> Result<()> {
        match expression {
            Expression::If(_, consequence, alternative) => {
                self.check(consequence)?;
                if let Some(block) = alternative {
                    self.check(block)?;
                }
            }
            Expression::Switch(_, cases) => {
                for SwitchCase { body, .. } in cases {
                    self.check(body)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    // A place that is a binding declared inside the region can hold a region
    // pointer; a parameter, an outer local, or a field of one cannot.
    fn is_inner_place(&self, place: &Expression) -> bool {
        matches!(place, Expression::Identifier(name) if self.inner.contains(name))
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
