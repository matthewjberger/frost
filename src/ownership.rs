use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::lexer::Position;
use crate::parser::{
    Block, Expression, Literal, ParamMode, Parameter, Spanned, Statement,
};
use crate::types::Type;

type Signatures = HashMap<String, Type>;

fn locate<T>(result: Result<T>, position: Position) -> Result<T> {
    result.map_err(|error| {
        let text = crate::imports::demangle_private_names(&error.to_string());
        if position == Position::default() || text.starts_with("at ") {
            anyhow::anyhow!("{text}")
        } else {
            anyhow::anyhow!("at {}: {text}", position.describe())
        }
    })
}

type ParamTypes = HashMap<String, Vec<Option<Type>>>;

/// The declared type of every field of every struct, by the struct's name and
/// the field's. A call through a function held in a field reads its parameter
/// modes from here, since the field is where that signature is written.
type FieldTypes = HashMap<(String, String), Type>;

/// What every check needs to know about the program around the item it is
/// looking at: which types are linear, what each function answers with, and how
/// each function and each field-held signature takes its arguments.
struct Program<'a> {
    linear: &'a HashSet<String>,
    signatures: &'a Signatures,
    param_types: &'a ParamTypes,
    field_types: &'a FieldTypes,
}

fn collect_field_types(statements: &[Spanned<Statement>]) -> FieldTypes {
    let mut fields = HashMap::new();
    for statement in statements {
        if let Statement::Struct(name, _, declared) = &statement.node {
            for field in declared {
                fields.insert(
                    (name.clone(), field.name.clone()),
                    field.field_type.clone(),
                );
            }
        }
    }
    fields
}

pub fn check_ownership(
    statements: &[Spanned<Statement>],
    linear: &HashSet<String>,
) -> Result<()> {
    let reports = check_ownership_recovering(statements, linear);
    if reports.is_empty() {
        return Ok(());
    }
    Err(anyhow::anyhow!(reports.join("\n")))
}

/// Check each top-level item, reporting one failure per item rather than
/// stopping at the first, so a program with a move error in three functions
/// names all three.
///
/// The granularity is the item, not the statement. Within one function the
/// walker marks a name moved as it goes, so continuing past a use-after-move
/// would report every later use of that name as well. Items do not share that
/// state, so stopping at the item boundary accumulates without cascading.
///
/// The messages are already located by `locate`, some of them by an inner
/// position rather than the item's, which is why these are the finished strings
/// rather than `Diagnostic`s.
pub fn check_ownership_recovering(
    statements: &[Spanned<Statement>],
    linear: &HashSet<String>,
) -> Vec<String> {
    let signatures = collect_signatures(statements);
    let param_types = collect_param_types(statements);
    let field_types = collect_field_types(statements);
    let held = linear_closure(linear, &field_types, statements);
    let program = Program {
        linear: &held,
        signatures: &signatures,
        param_types: &param_types,
        field_types: &field_types,
    };
    let mut reports = crate::linear_instances::check_pooled_resources(
        statements,
        &crate::linear_instances::locate_instances(statements),
        &held,
    );
    for statement in statements {
        let outcome = check_statement(&statement.node, &program, &mut reports);
        if let Err(error) = locate(outcome, statement.position) {
            reports.push(error.to_string());
        }
    }
    reports
}

// The declared type of every parameter of every function and extern, in order,
// so a call argument can be told to borrow (a reference parameter) rather than
// move (a value parameter). Positions line up one-to-one with call arguments,
// including a `$Type` argument against a `$T: Type` parameter.
fn collect_param_types(statements: &[Spanned<Statement>]) -> ParamTypes {
    let mut param_types = HashMap::new();
    for statement in statements {
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
        param_types.insert(
            name.clone(),
            params
                .iter()
                .map(|parameter| {
                    let ty = parameter.type_annotation.clone()?;
                    // An extern's parameters are not rewritten by the mode
                    // lowering, so the mode is read here. `value` hands C a
                    // copy, so the caller keeps its own and the argument is
                    // borrowed rather than moved.
                    match parameter.mode {
                        ParamMode::Value => Some(Type::Ref(Box::new(ty))),
                        _ => Some(ty),
                    }
                })
                .collect(),
        );
    }
    param_types
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
            Statement::Declared {
                name, return_sig, ..
            } => {
                signatures.insert(
                    name.clone(),
                    return_sig.to_type().unwrap_or(Type::Void),
                );
            }
            _ => {}
        }
    }
    signatures
}

fn check_statement(
    statement: &Statement,
    program: &Program,
    reports: &mut Vec<String>,
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
        Statement::Enum(name, _, variants) => {
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
            _name,
            Expression::Function(params, _return_sig, body),
        )
        | Statement::Constant(
            _name,
            Expression::Proc(params, _return_sig, body),
        ) => {
            // A reference return is allowed. The frame-escape check holds a
            // borrow to storage that outlives the call, and the region check
            // holds an arena borrow to its region, so returning one is only ever
            // a borrow the caller may keep. `arena_at` is the reason it exists.
            for inner in body {
                check_statement(inner, program, reports)?;
            }
            reports.extend(check_function_moves(params, body, program));
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
    program: &Program,
) -> Vec<String> {
    let mut checker = MoveChecker {
        types: HashMap::new(),
        states: HashMap::new(),
        paths: HashMap::new(),
        linear: program.linear,
        signatures: program.signatures,
        param_types: program.param_types,
        field_types: program.field_types,
        compile_time: params
            .iter()
            .filter(|parameter| {
                matches!(
                    &parameter.type_annotation,
                    Some(Type::TypeParam(name)) if name == &parameter.name
                )
            })
            .map(|parameter| parameter.name.clone())
            .collect(),
        in_defer: false,
        reports: Vec::new(),
        reported: HashSet::new(),
    };
    for parameter in params {
        if let Some(ty) = &parameter.type_annotation {
            checker.note_binding(&parameter.name, Some(ty.clone()));
        }
    }
    checker.check_function_body(body);
    checker.reports
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
    // What each place a body names has been done with, keyed by the place
    // written out. A bare name is a place one step long, so a name and a field
    // of it sit in the same table and are told apart the same way.
    states: HashMap<String, MoveState>,
    // The path behind each of those keys. Two places are the same storage when
    // their paths overlap rather than when they read alike, and the key alone
    // cannot answer that: `xs[i]` and `xs[j]` are written differently and are
    // one element whenever the two numbers agree.
    paths: HashMap<String, Vec<Step>>,
    linear: &'a HashSet<String>,
    signatures: &'a Signatures,
    param_types: &'a ParamTypes,
    field_types: &'a FieldTypes,
    // The enclosing function's compile-time parameters. A call to one of these
    // names a function only once the generic is specialized, so nothing is
    // known about what it does with its arguments until then.
    compile_time: HashSet<String>,
    in_defer: bool,
    reports: Vec<String>,
    // The raw text of what has already been said. Past a move the state stays
    // moved, so every later mention of that name fails the same way, and the
    // second telling is an echo of the first rather than a second mistake.
    reported: HashSet<String>,
}

impl MoveChecker<'_> {
    fn note_binding(&mut self, name: &str, ty: Option<Type>) {
        self.states.insert(name.to_string(), MoveState::Live);
        self.paths.insert(
            name.to_string(),
            vec![Step::Named(name.to_string())],
        );
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
        self.state_of_place(&[Step::Named(name.to_string())], name).0
    }

    /// What has been done with a place, which is what has been done with any
    /// place it shares storage with. Consuming `h.file` consumes part of `h`,
    /// so consuming `h` afterwards is consuming the same resource twice, and
    /// naming `h.file` again is naming what is already gone.
    ///
    /// The exact key is asked first so a place answers for itself, and the rest
    /// of the table is asked only when it has nothing to say.
    ///
    /// The answer carries the place it came from, since that is the one worth
    /// naming: a read of `a.value` refused because `a` was consumed should say
    /// `a`, which is where the value went, rather than the narrower place the
    /// reader happened to write.
    fn state_of_place(
        &self,
        path: &[Step],
        key: &str,
    ) -> (MoveState, String) {
        if let Some(state) = self.states.get(key)
            && *state != MoveState::Live
        {
            return (*state, key.to_string());
        }
        let mut held = (MoveState::Live, key.to_string());
        for (other, state) in &self.states {
            if *state == MoveState::Live || other == key {
                continue;
            }
            let Some(reached) = self.paths.get(other) else {
                continue;
            };
            if places_overlap(reached, path) {
                held = (*state, other.clone());
                if *state == MoveState::Moved {
                    return held;
                }
            }
        }
        held
    }

    /// The key a place is filed under, recording the path behind it so a later
    /// place can be weighed against this one.
    fn place_key(&mut self, path: &[Step]) -> String {
        let key = describe_place(path);
        self.paths.entry(key.clone()).or_insert_with(|| path.to_vec());
        key
    }

    /// A place holds a value again, so nothing sharing its storage is gone.
    fn revive_place(&mut self, path: &[Step], key: &str) {
        self.states.insert(key.to_string(), MoveState::Live);
        // Only what the write covers. Writing `h` gives `h.file` back because the
        // write settles the whole of it; writing `h.file` does not give `h` back,
        // since the rest of `h` is still wherever it went.
        let covered: Vec<String> = self
            .states
            .keys()
            .filter(|other| {
                self.paths
                    .get(*other)
                    .is_some_and(|reached| place_covers(path, reached))
            })
            .cloned()
            .collect();
        for other in covered {
            self.states.insert(other, MoveState::Live);
        }
    }

    /// The place already given away that this one is part of, or `None`. Writing
    /// into storage that was handed to someone else is not giving it back: the
    /// callee owns it and may have released it already.
    fn moved_container_of(&self, path: &[Step]) -> Option<String> {
        self.states
            .iter()
            .filter(|(_, state)| **state != MoveState::Live)
            .filter_map(|(other, _)| {
                let reached = self.paths.get(other)?;
                (reached.len() < path.len()
                    && place_maybe_within(path, reached))
                .then(|| other.clone())
            })
            .min()
    }

    /// What a place reaches through, which is read even where the place itself
    /// is written: the index of `xs[i] = v`, and the base of a field of a
    /// borrow. A bare name reaches through nothing.
    fn visit_beneath(&mut self, target: &Expression) -> Result<()> {
        match target {
            Expression::Identifier(_) => Ok(()),
            Expression::Index(base, index) => {
                self.visit(index, false)?;
                self.visit_beneath(base)
            }
            Expression::FieldAccess(base, _)
            | Expression::Dereference(base) => self.visit_beneath(base),
            other => self.visit(other, false),
        }
    }

    /// A field or an element, reached where a value is wanted. What is asked of
    /// the table is the whole path, since that is what says which storage this
    /// names, and what is recorded is the same path when the value is one that
    /// moves.
    fn visit_place(
        &mut self,
        expression: &Expression,
        path: &[Step],
        moving: bool,
    ) -> Result<()> {
        let key = self.place_key(path);
        // Only a resource. A plain struct read out of a field is a copy the
        // language has always taken, and counting one as a consumption would
        // refuse `bound := o.inner` followed by any other use of `o`. What
        // linearity adds is that a resource has to be consumed exactly once,
        // and it is that count a second consumption breaks.
        //
        // A copy is not a consumption however it is used, so reading the width
        // of a resource waiting on a `defer` is a read like any other. What is
        // gone is gone either way, which is why only the deferred case asks
        // whether this use consumes.
        let consuming = moving
            && self
                .value_type(expression)
                .map(|ty| ty.is_linear_with(self.linear))
                .unwrap_or(false);
        let (state, blamed) = self.state_of_place(path, &key);
        match state {
            MoveState::Live => {}
            MoveState::Deferred if consuming => {
                bail!(
                    "ownership: value '{blamed}' is already scheduled for consumption by a later defer; it cannot be moved again"
                );
            }
            MoveState::Deferred => {}
            MoveState::Moved | MoveState::MaybeMoved => {
                bail!("ownership: use of moved value '{blamed}'");
            }
        }
        if consuming {
            let consumed = if self.in_defer {
                MoveState::Deferred
            } else {
                MoveState::Moved
            };
            self.states.insert(key, consumed);
        }
        Ok(())
    }

    /// Record a failed statement and carry on to the next one, unless the same
    /// thing has already been said.
    fn record(&mut self, outcome: Result<bool>, position: Position) -> bool {
        match outcome {
            Ok(diverges) => diverges,
            Err(error) => {
                if self.reported.insert(error.to_string()) {
                    let located = locate::<bool>(Err(error), position)
                        .expect_err("an error stays an error");
                    self.reports.push(located.to_string());
                }
                false
            }
        }
    }

    fn check_block(&mut self, block: &Block) -> bool {
        let mut diverges = false;
        for statement in block {
            let outcome = self.check_statement(&statement.node);
            diverges = self.record(outcome, statement.position);
            if diverges {
                break;
            }
        }
        diverges
    }

    fn check_function_body(&mut self, block: &Block) -> bool {
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
                    let outcome = self.check_conditional(expression);
                    diverges = self.record(outcome, position);
                } else {
                    let outcome = self.visit(expression, true).map(|()| false);
                    self.record(outcome, position);
                }
            } else {
                let outcome = self.check_statement(&statement.node);
                diverges = self.record(outcome, position);
                if diverges {
                    break;
                }
            }
        }
        diverges
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
                )
                // A binding whose value is a place takes that place's type,
                // which is how a function read out of a table is known to be
                // one: `run := systems[i].run` then `run(world)`.
                .or_else(|| self.value_type(value));
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
                // The target is written, not read. Putting a value into a place
                // is what makes it hold one again, so a place given away and
                // then assigned is live: `vec_free($Table, world.tables)`
                // followed by `world.tables = kept` is a container replacing
                // what it released. Whatever the target reaches through is
                // still a read, which is the index of `xs[i] = v`.
                if let Expression::Identifier(name) = target {
                    self.states.insert(name.clone(), MoveState::Live);
                } else if let Some(path) = self.borrow_place(target) {
                    let key = self.place_key(&path);
                    if let Some(blamed) = self.moved_container_of(&path) {
                        bail!("ownership: use of moved value '{blamed}'");
                    }
                    self.revive_place(&path, &key);
                    self.visit_beneath(target)?;
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
            Statement::For(variable, _, range, body) => {
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
            // `print` takes what it is given the way any other call does, and
            // this pass used to walk straight past it: a value moved into a
            // `print` stayed live and could be used again, which is the one
            // place a use-after-move went unnoticed.
            Statement::Print(expression, arguments) => {
                self.visit(expression, false)?;
                for argument in arguments {
                    self.visit(argument, false)?;
                }
                Ok(false)
            }
            // The allocation-sources lowering runs before this check and leaves
            // no `with` behind, and the multiple-return lowering leaves no
            // `LetMultiple`. Both are walked anyway, so neither becomes a hole
            // if that order ever changes.
            Statement::With(_, body) => {
                self.check_block(body);
                Ok(false)
            }
            Statement::LetMultiple(bindings, value) => {
                self.visit(value, true)?;
                for binding in bindings {
                    self.note_binding(&binding.name, None);
                }
                Ok(false)
            }
            // A declaration holds no expression whose ownership this pass could
            // read. Listed rather than caught by `_`, so a new statement form is
            // a compile error here instead of a move nobody counted.
            Statement::Struct(..)
            | Statement::Enum(..)
            | Statement::Flags(..)
            | Statement::TypeAlias(..)
            | Statement::Import(..)
            | Statement::Extern { .. }
            | Statement::Declared { .. } => Ok(false),
        }
    }

    fn check_loop_body(&mut self, body: &Block) -> Result<()> {
        let before = self.states.clone();
        self.check_block(body);
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
        let diverges = self.check_block(block);
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
        // Every key any arm touched, not just the ones live before them. A
        // field consumed in one branch and not another is moved on one path
        // and not the other, and the key for it exists only inside that arm.
        let mut keys: Vec<String> = before.keys().cloned().collect();
        for states in &live {
            for name in states.keys() {
                if !before.contains_key(name) {
                    keys.push(name.clone());
                }
            }
        }
        keys.sort();
        keys.dedup();
        for name in &keys {
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
            // A field or an element is a place of its own. Consuming one
            // consumes part of what holds it, so the whole path is what the
            // table is asked about rather than the name at its root: reading
            // `h.name` after `h.file` was consumed is reading a different
            // place, and consuming `h` afterwards is consuming the same
            // resource a second time.
            //
            // A place whose type is not known, or which is not a resource, is
            // walked the way it always was. Nothing is recorded for it, so a
            // scalar field costs no bookkeeping.
            Expression::FieldAccess(base, _) => {
                if let Some(path) = self.borrow_place(expression) {
                    self.visit_place(expression, &path, moving)?;
                    return Ok(());
                }
                self.visit(base, false)
            }
            Expression::Index(base, index) => {
                self.visit(index, false)?;
                if let Some(path) = self.borrow_place(expression) {
                    self.visit_place(expression, &path, moving)?;
                    return Ok(());
                }
                self.visit(base, false)
            }
            Expression::PackMap(operand, _, _)
            | Expression::Prefix(_, operand) => self.visit(operand, false),
            Expression::Infix(left, _, right) => {
                self.visit(left, false)?;
                self.visit(right, false)
            }
            Expression::Call(callee, arguments) => {
                self.visit(callee, false)?;
                if let Expression::Identifier(name) = callee.as_ref()
                    && let Some(borrows) = builtin_borrows_first_argument(name)
                {
                    for (index, argument) in arguments.iter().enumerate() {
                        self.visit(argument, !(borrows && index == 0))?;
                    }
                    return Ok(());
                }
                // What the callee does with each argument. A name that is a
                // declared function says so directly. Anything else is called
                // through a value, and the signature that value holds is what
                // says which of its parameters borrow. Without this every
                // argument of an indirect call read as consumed, so a table of
                // systems could not be walked: `systems[i].run(world)` took the
                // world away on the first one.
                let held = self.callee_signature(callee);
                let param_types = match callee.as_ref() {
                    Expression::Identifier(name) => {
                        self.param_types.get(name).or(held.as_ref())
                    }
                    _ => held.as_ref(),
                };
                check_borrow_exclusivity(self, arguments, param_types)?;
                // A call to a compile-time parameter names a function only
                // once the generic is specialized, so it says nothing about
                // ownership yet and the specialized body answers for it
                // afterwards. Any other unknown callee is still read as
                // consuming, so a function pointer does not quietly stop
                // moving what it is given.
                let deferred = matches!(
                    callee.as_ref(),
                    Expression::Identifier(name)
                        if self.compile_time.contains(name)
                );
                let known = !deferred;
                for (index, argument) in arguments.iter().enumerate() {
                    let declared = param_types
                        .and_then(|types| types.get(index))
                        .and_then(|ty| ty.as_ref());
                    let borrows = matches!(
                        declared,
                        Some(Type::Ref(_) | Type::RefMut(_))
                    );
                    self.visit(argument, known && !borrows)?;
                    // What the callee says it takes, rather than what the
                    // argument's own type works out to. A place behind a `mut`
                    // parameter types through a borrow and through the mode
                    // lowering that put it there, and either can leave it
                    // unresolved; the declaration says `move f: File` whatever
                    // the caller wrote, and that is what says a resource was
                    // handed over.
                    if known
                        && !borrows
                        && declared
                            .is_some_and(|ty| ty.is_linear_with(self.linear))
                        && let Some(path) = self.borrow_place(argument)
                    {
                        let key = self.place_key(&path);
                        let consumed = if self.in_defer {
                            MoveState::Deferred
                        } else {
                            MoveState::Moved
                        };
                        self.states.insert(key, consumed);
                    }
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
            Expression::ArrayRepeat(value, _) => self.visit(value, true),
            Expression::If(..) | Expression::Switch(..) => {
                self.check_conditional(expression)?;
                Ok(())
            }
            // An `unsafe` block is a block of ordinary statements. A move made
            // inside one is a move, and not walking in meant a value consumed
            // there stayed live and could be consumed again.
            Expression::Unsafe(body) => {
                self.check_block(body);
                Ok(())
            }
            // Listed rather than caught by `_`, so a new expression form is a
            // compile error here instead of being walked past unexamined.
            Expression::Literal(_)
            | Expression::Boolean(_)
            | Expression::Sizeof(_)
            | Expression::TypeId(_)
            | Expression::TypeName(_)
            | Expression::TypeValue(_)
            | Expression::Range(..)
            | Expression::Function(..)
            | Expression::Proc(..)
            | Expression::UnsafeFn(_)
            | Expression::Try(_)
            | Expression::Tuple(_) => Ok(()),
        }
    }

    fn is_linear_variable(&self, name: &str) -> bool {
        self.types
            .get(name)
            .map(|ty| is_linear_type(ty, self.linear))
            .unwrap_or(false)
    }

    /// The parameter types of what this expression calls, for a callee that is
    /// a value of function-pointer type rather than a declared name.
    fn callee_signature(
        &self,
        callee: &Expression,
    ) -> Option<Vec<Option<Type>>> {
        let Type::Proc(params, _) = self.value_type(callee)? else {
            return None;
        };
        Some(params.into_iter().map(Some).collect())
    }

    /// The type of a place, as far as the names and the struct declarations
    /// give it: a name, an element of one, a field of one, or a field of an
    /// element. This is what a function pointer held in a table is written as.
    fn value_type(&self, expression: &Expression) -> Option<Type> {
        match expression {
            Expression::Identifier(name) => self.types.get(name).cloned(),
            Expression::Index(base, _) => match self.value_type(base)? {
                Type::Array(inner, _) | Type::Slice(inner) => Some(*inner),
                _ => None,
            },
            Expression::FieldAccess(base, field) => {
                // A `mut` parameter of struct type is a borrow of one, and a
                // field of it is the same field. Asking only about the struct
                // itself left every place behind a borrow untyped, so nothing
                // consumed through one was recorded.
                let held = match self.value_type(base)? {
                    Type::Ref(inner) | Type::RefMut(inner) => *inner,
                    other => other,
                };
                let Type::Struct(name) = held else {
                    return None;
                };
                self.field_types.get(&(name, field.clone())).cloned()
            }
            _ => None,
        }
    }

    fn is_move_variable(&self, name: &str) -> bool {
        self.types
            .get(name)
            .map(|ty| !ty.is_copy())
            .unwrap_or(false)
    }

    /// The place a borrowed argument names, as a path from a root variable
    /// through fields, indexes and dereferences. `None` for an expression that
    /// is not a place (a call, a literal), which names no storage a second
    /// borrow could reach.
    fn borrow_place(&self, expression: &Expression) -> Option<Vec<Step>> {
        match expression {
            Expression::Identifier(name) => {
                Some(vec![Step::Named(name.clone())])
            }
            Expression::FieldAccess(base, field) => {
                let mut path = self.borrow_place(base)?;
                path.push(Step::Named(format!(".{field}")));
                Some(path)
            }
            Expression::Index(base, index) => {
                let mut path = self.borrow_place(base)?;
                let literal = match index.as_ref() {
                    Expression::Literal(Literal::Integer(value)) => {
                        Some(*value)
                    }
                    _ => None,
                };
                path.push(Step::Index(literal, format!("[{index}]")));
                Some(path)
            }
            Expression::Dereference(base) => {
                let mut path = self.borrow_place(base)?;
                path.push(Step::Deref(self.reads_raw_pointer(base)));
                Some(path)
            }
            _ => None,
        }
    }

    /// Whether a dereference of this expression goes through a raw pointer
    /// rather than through a borrow.
    ///
    /// A type the walk cannot name answers yes. Where it points is then exactly
    /// what nothing here knows, and this check is about what two places might
    /// share rather than what they are known to share.
    fn reads_raw_pointer(&self, base: &Expression) -> bool {
        match self.value_type(base) {
            Some(Type::Ref(_) | Type::RefMut(_)) => false,
            Some(Type::Ptr(_)) => true,
            Some(_) => false,
            None => true,
        }
    }
}

// One step of the path to a borrowed place.
#[derive(Clone, PartialEq, Eq)]
enum Step {
    // A named field, or a dereference, which is the same question: two of these
    // name the same storage exactly when they are written the same way.
    Named(String),
    // An index. What it selects is decided while the program runs, so two of
    // them are known apart only when both are numbers.
    Index(Option<i64>, String),
    // A dereference, and whether it goes through a raw pointer. Where a
    // dereference lands is decided by what the pointer holds, so two places
    // reached through raw pointers may be one storage however different the
    // names in front of them read.
    //
    // A borrow is not one of those. It was proven safe where it formed, and the
    // parameter-mode lowering that runs before this check rewrites every `mut`
    // scalar parameter to `name^`, so two distinct `mut` parameters reach a call
    // as two dereferences. Reading those as together would refuse ordinary code,
    // which is why the two are told apart rather than counted together.
    Deref(bool),
}

// Whether two steps definitely name different storage.
//
// Two indexes are apart only when both are numbers and the numbers differ.
// `xs[i]` and `xs[j]` are the same element whenever `i == j`, and nothing here
// can rule that out, so they are not apart. Reading them as apart is what let
// two `mut` borrows of one element through: `swap_bump(xs[i], xs[j])` with
// `i == j` handed the same slot to both parameters.
fn steps_apart(left: &Step, right: &Step) -> bool {
    match (left, right) {
        (Step::Named(one), Step::Named(other)) => one != other,
        (Step::Index(Some(one), _), Step::Index(Some(other), _)) => {
            one != other
        }
        (Step::Index(..), Step::Index(..)) => false,
        // A field and an index of the same base name different storage only
        // because one of them is not a place the other could be. They cannot
        // both be written of one type, so this does not arise; reading it as
        // together costs nothing.
        _ => false,
    }
}

// Whether a path reaches storage through a raw pointer at any point.
fn reaches_through_raw(path: &[Step]) -> bool {
    path.iter().any(|step| matches!(step, Step::Deref(true)))
}

// Two places overlap unless some step along their common length is known to name
// different storage. They share the whole of the shorter as a prefix when
// neither differs, so `s` overlaps `s.x`, `s.x` overlaps `s.x.y`, and `s.x` and
// `s.y` are apart. Overlapping places name storage that intersects, which is
// what an exclusive borrow may not share.
//
// A raw dereference breaks the argument the prefix comparison rests on. Every
// step before one says where the pointer was read from, and none of them says
// where it points, so `p^` and `q^` are one place whenever `p` and `q` hold one
// address, and `p^` and `x` are one place whenever `p` holds `x`'s address. The
// names settle nothing either way, so a place reaching through a raw pointer
// overlaps whatever it is weighed against: what the comparison could prove about
// it, it cannot.
//
// Either side rather than both. Refusing only the pair looked like the
// affordable half of the rule, on the reasoning that the rest would refuse
// `f(p^, y)` for every unrelated `y` in a body holding one raw pointer.
// Measuring said otherwise: across the standard library, both compilers and the
// examples, the whole rule refuses nothing.
fn places_overlap(first: &[Step], second: &[Step]) -> bool {
    if reaches_through_raw(first) || reaches_through_raw(second) {
        return true;
    }
    let common = first.len().min(second.len());
    !first[..common]
        .iter()
        .zip(&second[..common])
        .any(|(left, right)| steps_apart(left, right))
}

// Whether two steps definitely name the same storage.
//
// Not the negation of `steps_apart`: an index nothing knows the value of is
// neither definitely together nor definitely apart, and a dereference through a
// raw pointer is the same. Which of the three answers a caller wants depends on
// what it does with it, so the two questions are asked separately.
fn steps_same(left: &Step, right: &Step) -> bool {
    match (left, right) {
        (Step::Named(one), Step::Named(other)) => one == other,
        (Step::Index(Some(one), _), Step::Index(Some(other), _)) => {
            one == other
        }
        (Step::Deref(one), Step::Deref(other)) => !one && !other,
        _ => false,
    }
}

// Whether `outer` definitely names storage `inner` is part of: the same place, or
// one `inner` hangs off.
//
// Unknown answers no. This is what a write consults to decide what it gives back,
// and reviving on a guess is what lets a resource be consumed twice, so a place
// reached through an index nobody knows or a raw pointer is not revived.
fn place_covers(outer: &[Step], inner: &[Step]) -> bool {
    outer.len() <= inner.len()
        && outer
            .iter()
            .zip(inner)
            .all(|(left, right)| steps_same(left, right))
}

// Whether `inner` might be part of some wider place.
//
// Unknown answers yes, which is the other way round from `place_covers` and for
// the same reason. This is what refuses a write into storage that has gone, and
// what cannot be told apart there is what has to be refused.
fn place_maybe_within(inner: &[Step], outer: &[Step]) -> bool {
    outer.len() <= inner.len() && places_overlap(outer, inner)
}

// How a place reads back in a diagnostic.
fn describe_place(path: &[Step]) -> String {
    path.iter()
        .map(|step| match step {
            Step::Named(text) | Step::Index(_, text) => text.as_str(),
            Step::Deref(_) => "^",
        })
        .collect()
}

fn check_borrow_exclusivity(
    checker: &MoveChecker<'_>,
    arguments: &[Expression],
    param_types: Option<&Vec<Option<Type>>>,
) -> Result<()> {
    // Each borrowed argument as (place path, whether it is exclusive). A `mut`
    // parameter and an explicit `&mut` borrow exclusively. A `read` parameter
    // and a `&` share.
    let mut borrows: Vec<(Vec<Step>, bool)> = Vec::new();
    for (index, argument) in arguments.iter().enumerate() {
        let mode = param_types
            .and_then(|types| types.get(index))
            .and_then(|ty| ty.as_ref());
        let borrow = match argument {
            Expression::BorrowMut(inner) => {
                checker.borrow_place(inner).map(|p| (p, true))
            }
            Expression::Borrow(inner) => {
                checker.borrow_place(inner).map(|p| (p, false))
            }
            _ => match mode {
                Some(Type::RefMut(_)) => {
                    checker.borrow_place(argument).map(|p| (p, true))
                }
                Some(Type::Ref(_)) => {
                    checker.borrow_place(argument).map(|p| (p, false))
                }
                _ => None,
            },
        };
        if let Some(borrow) = borrow {
            borrows.push(borrow);
        }
    }
    for (index, (place, exclusive)) in borrows.iter().enumerate() {
        for (other_place, other_exclusive) in borrows.iter().skip(index + 1) {
            if !(*exclusive || *other_exclusive) {
                continue;
            }
            if places_overlap(place, other_place) {
                let name = describe_place(place);
                let other = describe_place(other_place);
                if *exclusive && *other_exclusive {
                    bail!(
                        "ownership: '{name}' and '{other}' are both borrowed as mutable in a single call; mutable borrows are exclusive"
                    );
                }
                bail!(
                    "ownership: '{name}' is borrowed as both shared and mutable in a single call; mutable borrows are exclusive"
                );
            }
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
    ty.is_linear_with(linear)
}

/// Every type that must be consumed: the ones declared `linear`, and the ones
/// that hold such a value in a field. A struct holding a resource is a resource,
/// otherwise wrapping one in an ordinary struct would launder the obligation
/// away. This runs to a fixpoint, since the holder of a holder is one too.
fn linear_closure(
    declared: &HashSet<String>,
    fields: &FieldTypes,
    statements: &[Spanned<Statement>],
) -> HashSet<String> {
    let mut held = declared.clone();
    // The instantiations the program writes. A generic's declared field names a
    // parameter bound to nothing here, so the field table answers for `Slab` and
    // not for `Slab<Node, 2>`, and it is the second that holds the resource.
    let instances = crate::linear_instances::collect_instances(statements);
    let templates = crate::linear_instances::declared_structs(statements);
    loop {
        let mut grew = false;
        for ((owner, _), ty) in fields {
            if held.contains(Type::template_of(owner)) {
                continue;
            }
            if is_linear_type(ty, &held) {
                held.insert(owner.clone());
                grew = true;
            }
        }
        // In the same loop as the holders, since an instance is a resource
        // because of a field and a struct is one because of an instance in a
        // field of its own.
        if crate::linear_instances::note_linear_instances(
            &templates,
            &instances,
            &mut held,
        ) {
            grew = true;
        }
        if !grew {
            return held;
        }
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
        let mut statements = parser.parse()?;
        let linear = parser.linear_types().clone();
        crate::param_modes::lower_param_modes(&mut statements);
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
        let source = "Point :: struct { x: i64, y: i64 }\n\
            read :: fn(p: Point) -> i64 { p.x }";
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
            take :: fn(move p: Point) -> i64 { p.x }\n\
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
            read :: fn(p: Point) -> i64 { p.x }\n\
            run :: fn() -> i64 {\n\
                p := Point { x = 1, y = 2 }\n\
                a := read(p)\n\
                b := read(p)\n\
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
            add :: fn(mut a: i64, mut b: i64) { }\n\
            run :: fn() {\n\
                mut x : i64 = 0\n\
                add(x, x)\n\
            }";
        assert!(check(source).is_err());
    }

    #[test]
    fn shared_and_mutable_borrow_of_same_is_rejected() {
        let source = "\
            Point :: struct { x: i64, y: i64 }\n\
            mix :: fn(a: Point, b: mut Point) { }\n\
            run :: fn() {\n\
                mut x : Point = Point { x = 0, y = 0 }\n\
                mix(x, x)\n\
            }";
        assert!(check(source).is_err());
    }

    // Two index expressions name one element whenever they evaluate the same,
    // and comparing them by how they are written read `xs[i]` and `xs[j]` as
    // apart. With `i == j` both increments landed on the same slot.
    #[test]
    fn mutable_borrows_of_two_unproven_indexes_are_rejected() {
        let source = "\
            bump :: fn(mut a: i64, mut b: i64) { }\n\
            run :: fn(i: i64, j: i64) {\n\
                mut xs : [4]i64 = [0, 0, 0, 0]\n\
                bump(xs[i], xs[j])\n\
            }";
        assert!(check(source).is_err());
    }

    // Two literal indexes are apart, since the numbers say so.
    #[test]
    fn mutable_borrows_of_two_literal_indexes_are_allowed() {
        let source = "\
            bump :: fn(mut a: i64, mut b: i64) { }\n\
            run :: fn() {\n\
                mut xs : [4]i64 = [0, 0, 0, 0]\n\
                bump(xs[0], xs[1])\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn mutable_borrows_of_one_literal_index_twice_are_rejected() {
        let source = "\
            bump :: fn(mut a: i64, mut b: i64) { }\n\
            run :: fn() {\n\
                mut xs : [4]i64 = [0, 0, 0, 0]\n\
                bump(xs[2], xs[2])\n\
            }";
        assert!(check(source).is_err());
    }

    #[test]
    fn distinct_mutable_borrows_are_allowed() {
        let source = "\
            add :: fn(mut a: i64, mut b: i64) { }\n\
            run :: fn() {\n\
                mut x : i64 = 0\n\
                mut y : i64 = 0\n\
                add(x, y)\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn multiple_shared_borrows_are_allowed() {
        let source = "\
            Point :: struct { x: i64, y: i64 }\n\
            sum :: fn(a: Point, b: Point) -> i64 { a.x + b.x }\n\
            run :: fn() -> i64 {\n\
                p : Point = Point { x = 7, y = 0 }\n\
                sum(p, p)\n\
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
        // The destructor takes the linear value by value and unpacks it. The
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

    // A resource reached through a field is a place of its own, and consuming
    // it twice consumes it twice. Moves were tracked by name, so the field was
    // never recorded and nothing said the second consumption was one: with a
    // `close` that frees, this is a double free written without `unsafe`.
    #[test]
    fn consuming_a_resource_through_a_field_twice_is_rejected() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            Holder :: struct { file: File, name: i64 }\n\
            close :: extern fn(f: File)\n\
            run :: fn(move h: Holder) {\n\
                close(h.file)\n\
                close(h.file)\n\
            }";
        assert!(check(source).is_err());
    }

    // The whole contains the part, so consuming the whole after the part is the
    // same resource a second time.
    #[test]
    fn consuming_a_whole_after_its_field_is_rejected() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            Holder :: struct { file: File, name: i64 }\n\
            close :: extern fn(f: File)\n\
            drop_holder :: extern fn(h: Holder)\n\
            run :: fn(move h: Holder) {\n\
                close(h.file)\n\
                drop_holder(h)\n\
            }";
        assert!(check(source).is_err());
    }

    // The same through a borrow. A `mut` parameter of struct type is a borrow
    // of one by the time this runs, so what the callee declared it takes is
    // what says a resource was handed over.
    #[test]
    fn consuming_a_resource_through_a_borrowed_field_twice_is_rejected() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            Holder :: struct { file: File, name: i64 }\n\
            close :: fn(move f: File) -> i64 { f.handle }\n\
            run :: fn(mut h: Holder) -> i64 {\n\
                close(h.file)\n\
                close(h.file)\n\
            }";
        assert!(check(source).is_err());
    }

    // Writing into storage that was handed away is not taking it back. Reviving
    // whatever shared the target's storage let a value be consumed, written into,
    // and consumed again, which is the hole the place tracking was for reached by
    // two steps instead of one.
    #[test]
    fn writing_a_field_of_a_consumed_value_is_rejected() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            Holder :: struct { file: File, name: i64 }\n\
            open :: fn(n: i64) -> File { File { handle = n } }\n\
            drop_holder :: extern fn(h: Holder)\n\
            run :: fn(move h: Holder) {\n\
                drop_holder(h)\n\
                h.file = open(9)\n\
            }";
        assert!(check(source).is_err());
    }

    // Writing the whole of it is, since the write settles every part.
    #[test]
    fn writing_a_whole_after_consuming_a_field_is_accepted() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            Holder :: struct { file: File, name: i64 }\n\
            open :: fn(n: i64) -> File { File { handle = n } }\n\
            close :: extern fn(f: File)\n\
            drop_holder :: extern fn(h: Holder)\n\
            run :: fn(mut h: Holder) {\n\
                close(h.file)\n\
                h = Holder { file = open(9), name = 2 }\n\
                drop_holder(h)\n\
            }";
        assert!(check(source).is_ok());
    }

    // An element answers the same way, and two elements known apart do not.
    #[test]
    fn consuming_an_element_twice_is_rejected() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            close :: extern fn(f: File)\n\
            run :: fn(move run: [2]File) {\n\
                close(run[0])\n\
                close(run[0])\n\
            }";
        assert!(check(source).is_err());
    }

    #[test]
    fn consuming_two_separate_elements_is_accepted() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            close :: extern fn(f: File)\n\
            run :: fn(move run: [2]File) {\n\
                close(run[0])\n\
                close(run[1])\n\
            }";
        assert!(check(source).is_ok());
    }

    // Two fields are different storage, which is what a container releasing
    // each of its own rests on: `world_release` frees several `Vec` fields in a
    // row and every one of them has to be allowed.
    #[test]
    fn consuming_two_separate_fields_is_accepted() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            Pair :: struct { left: File, right: File }\n\
            close :: extern fn(f: File)\n\
            run :: fn(move p: Pair) {\n\
                close(p.left)\n\
                close(p.right)\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn moving_an_owned_value_inside_a_loop_is_rejected() {
        let source = "\
            Point :: struct { x: i64, y: i64 }\n\
            take :: fn(move p: Point) -> i64 { p.x }\n\
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
