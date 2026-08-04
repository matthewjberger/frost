use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::ast::{
    Ast, ExprId, Expression, Literal, Parameter, Range32, Statement, StmtId,
};
use crate::ast_display::display_expr;
use crate::lexer::Position;
use crate::parser::ParamMode;
use crate::types::Type;

type Signatures = HashMap<String, Signature>;

/// What a call to a function answers with, and what a call needs in order to
/// work that out.
///
/// The declared return type on its own is not it. `option_some` is declared
/// `-> Option<T>`, whose template names no resource, and `option_some($File, f)`
/// answers with `Option<File>`, which is one. Reading the declaration as written
/// left a resource put into an option ordinary data, and the obligation went in
/// and did not come out.
struct Signature {
    result: Type,
    /// One slot per parameter: the name where that position declares a
    /// compile-time type parameter, and `None` where it takes a value. A call
    /// binds them positionally, each `$T: Type` taking the type written at its
    /// place, which is how specialization binds them too.
    type_params: Vec<Option<String>>,
}

/// The parameter positions that take a type rather than a value.
fn type_parameter_slots(
    ast: &Ast,
    parameters: &[Parameter],
) -> Vec<Option<String>> {
    parameters
        .iter()
        .map(|parameter| match &parameter.type_annotation {
            Some(Type::TypeParam(name))
                if name.as_str() == ast.name(parameter.name) =>
            {
                Some(name.clone())
            }
            _ => None,
        })
        .collect()
}

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
    summaries: &'a Summaries,
    runs: &'a Runs,
}

/// Everything the move check reads about a program, gathered once so a
/// specialization can be checked at the moment it is made.
///
/// A generic's own body names parameters bound to nothing: `vec_get` reads
/// `v.storage[index]` where the element type is a name standing for anything, so
/// nothing there is a resource and nothing there is a move. `vec_get<File>` is
/// where both are true, and that body exists only while it is being lowered.
/// The self-hosted compiler checks its instances for the same reason.
pub struct Specializations {
    signatures: Signatures,
    param_types: ParamTypes,
    field_types: FieldTypes,
    held: HashSet<String>,
    summaries: Summaries,
    runs: Runs,
}

/// Gather them. A program with nothing to say about ownership still pays for
/// this once, which is the same bargain `check_ownership` strikes.
pub fn specializations(
    ast: &Ast,
    roots: &[StmtId],
    linear: &HashSet<String>,
) -> Specializations {
    let signatures = collect_signatures(ast, roots);
    let param_types = collect_param_types(ast, roots);
    let field_types = collect_field_types(ast, roots);
    let held = linear_closure(linear, &field_types, ast, roots);
    let runs = settle_runs(ast, roots, &field_types);
    let summaries = settle_summaries(
        ast,
        roots,
        &Program {
            linear: &held,
            signatures: &signatures,
            param_types: &param_types,
            field_types: &field_types,
            summaries: &Summaries::new(),
            runs: &runs,
        },
    );
    Specializations {
        signatures,
        param_types,
        field_types,
        held,
        summaries,
        runs,
    }
}

impl Specializations {
    /// What is wrong with one specialized body, which is the same question
    /// asked of any other function.
    pub fn check(
        &self,
        ast: &Ast,
        params: Range32,
        body: Range32,
    ) -> Vec<String> {
        check_function_moves(
            ast,
            params,
            body,
            &Program {
                linear: &self.held,
                signatures: &self.signatures,
                param_types: &self.param_types,
                field_types: &self.field_types,
                summaries: &self.summaries,
                runs: &self.runs,
            },
        )
        .into_iter()
        .map(|held| held.rendered())
        .collect()
    }
}

fn collect_field_types(ast: &Ast, roots: &[StmtId]) -> FieldTypes {
    let mut fields = HashMap::new();
    for statement in roots {
        match ast.stmt(*statement) {
            Statement::Struct(name, _, declared) => {
                for field in ast.fields_in(*declared) {
                    fields.insert(
                        (
                            ast.name(*name).to_string(),
                            ast.name(field.name).to_string(),
                        ),
                        field.field_type.clone(),
                    );
                }
            }
            // A variant's payload is held by the enum exactly as a field is
            // held by a struct, so an enum carrying a resource is one. Reading
            // only the structs left an option holding a file ordinary data, and
            // the obligation went in and did not come out.
            Statement::Enum(name, _, variants) => {
                for variant in ast.variants_in(*variants) {
                    let Some(declared) = variant.fields else {
                        continue;
                    };
                    for field in ast.fields_in(declared) {
                        fields.insert(
                            (
                                ast.name(*name).to_string(),
                                ast.name(field.name).to_string(),
                            ),
                            field.field_type.clone(),
                        );
                    }
                }
            }
            _ => {}
        }
    }
    fields
}

pub fn check_ownership(
    ast: &Ast,
    roots: &[StmtId],
    linear: &HashSet<String>,
) -> Result<()> {
    let reports = check_ownership_recovering(ast, roots, linear);
    if reports.is_empty() {
        return Ok(());
    }
    let rendered: Vec<String> =
        reports.iter().map(|held| held.rendered()).collect();
    Err(anyhow::anyhow!(rendered.join("\n")))
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
/// Each diagnostic's message is already located by `locate`, some by an inner
/// position rather than the item's; the position field anchors the item for a
/// caller that wants structure, and a use of a moved value carries the move
/// as a related place.
pub fn check_ownership_recovering(
    ast: &Ast,
    roots: &[StmtId],
    linear: &HashSet<String>,
) -> Vec<crate::diagnostic::Diagnostic> {
    let signatures = collect_signatures(ast, roots);
    let param_types = collect_param_types(ast, roots);
    let field_types = collect_field_types(ast, roots);
    let held = linear_closure(linear, &field_types, ast, roots);
    let runs = settle_runs(ast, roots, &field_types);
    let summaries = settle_summaries(
        ast,
        roots,
        &Program {
            linear: &held,
            signatures: &signatures,
            param_types: &param_types,
            field_types: &field_types,
            summaries: &Summaries::new(),
            runs: &runs,
        },
    );
    let program = Program {
        linear: &held,
        signatures: &signatures,
        param_types: &param_types,
        field_types: &field_types,
        summaries: &summaries,
        runs: &runs,
    };
    let mut reports: Vec<crate::diagnostic::Diagnostic> =
        crate::linear_instances::check_pooled_resources(
            ast,
            roots,
            &crate::linear_instances::locate_instances(ast, roots),
            &held,
        )
        .into_iter()
        .map(|held| {
            crate::diagnostic::Diagnostic::new(Position::default(), held)
        })
        .collect();
    for statement in roots {
        let outcome = check_statement(ast, *statement, &program, &mut reports);
        if let Err(error) = locate(outcome, ast.stmt_position(*statement)) {
            reports.push(crate::diagnostic::Diagnostic::new(
                ast.stmt_position(*statement),
                error.to_string(),
            ));
        }
    }
    reports
}

// The declared type of every parameter of every function and extern, in order,
// so a call argument can be told to borrow (a reference parameter) rather than
// move (a value parameter). Positions line up one-to-one with call arguments,
// including a `$Type` argument against a `$T: Type` parameter.
fn collect_param_types(ast: &Ast, roots: &[StmtId]) -> ParamTypes {
    let mut param_types = HashMap::new();
    for statement in roots {
        let (name, params) = match ast.stmt(*statement) {
            Statement::Constant(name, value) => match ast.expr(*value) {
                Expression::Function(params, _, _)
                | Expression::Proc(params, _, _) => (*name, *params),
                _ => continue,
            },
            Statement::Extern { name, params, .. }
            | Statement::Declared { name, params, .. } => (*name, *params),
            _ => continue,
        };
        param_types.insert(
            ast.name(name).to_string(),
            ast.params_in(params)
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

fn collect_signatures(ast: &Ast, roots: &[StmtId]) -> Signatures {
    let mut signatures = HashMap::new();
    for statement in roots {
        match ast.stmt(*statement) {
            Statement::Constant(name, value) => {
                let (Expression::Function(parameters, return_sig, _)
                | Expression::Proc(parameters, return_sig, _)) =
                    ast.expr(*value)
                else {
                    continue;
                };
                signatures.insert(
                    ast.name(*name).to_string(),
                    Signature {
                        result: ast
                            .signature_to_type(ast.signature(*return_sig))
                            .unwrap_or(Type::Void),
                        type_params: type_parameter_slots(
                            ast,
                            ast.params_in(*parameters),
                        ),
                    },
                );
            }
            Statement::Extern {
                name,
                return_type,
                params,
                ..
            } => {
                signatures.insert(
                    ast.name(*name).to_string(),
                    Signature {
                        result: return_type.clone().unwrap_or(Type::Void),
                        type_params: type_parameter_slots(
                            ast,
                            ast.params_in(*params),
                        ),
                    },
                );
            }
            Statement::Declared {
                name,
                return_sig,
                params,
                ..
            } => {
                signatures.insert(
                    ast.name(*name).to_string(),
                    Signature {
                        result: ast
                            .signature_to_type(ast.signature(*return_sig))
                            .unwrap_or(Type::Void),
                        type_params: type_parameter_slots(
                            ast,
                            ast.params_in(*params),
                        ),
                    },
                );
            }
            _ => {}
        }
    }
    signatures
}

fn check_statement(
    ast: &Ast,
    statement: StmtId,
    program: &Program,
    reports: &mut Vec<crate::diagnostic::Diagnostic>,
) -> Result<()> {
    match ast.stmt(statement) {
        Statement::Struct(name, _, fields) => {
            for field in ast.fields_in(*fields) {
                if field.field_type.contains_reference() {
                    bail!(
                        "cannot store a reference in struct '{}' (field '{}'); references are second-class",
                        ast.name(*name),
                        ast.name(field.name)
                    );
                }
            }
        }
        Statement::Enum(name, _, variants) => {
            for variant in ast.variants_in(*variants) {
                let Some(fields) = variant.fields else {
                    continue;
                };
                for field in ast.fields_in(fields) {
                    if field.field_type.contains_reference() {
                        bail!(
                            "cannot store a reference in enum '{}' (variant '{}', field '{}'); references are second-class",
                            ast.name(*name),
                            ast.name(variant.name),
                            ast.name(field.name)
                        );
                    }
                }
            }
        }
        Statement::Constant(_name, value) => {
            let (Expression::Function(params, _return_sig, body)
            | Expression::Proc(params, _return_sig, body)) = ast.expr(*value)
            else {
                return Ok(());
            };
            // A reference return is allowed. The frame-escape check holds a
            // borrow to storage that outlives the call, and the region check
            // holds an arena borrow to its region, so returning one is only ever
            // a borrow the caller may keep. `arena_at` is the reason it exists.
            for inner in ast.stmts_in(*body) {
                check_statement(ast, *inner, program, reports)?;
            }
            reports.extend(check_function_moves(ast, *params, *body, program));
        }
        Statement::Extern {
            name, return_type, ..
        } => {
            if let Some(return_type) = return_type
                && return_type.contains_reference()
            {
                bail!(
                    "extern function '{}' cannot return a reference",
                    ast.name(*name)
                );
            }
        }
        _ => {}
    }
    Ok(())
}

/// What each function does to the resources it is lent: for one parameter, the
/// place under it that the body hands on, and whether it does so on every path.
///
/// The count linearity keeps is per place, and a place lives in one function. A
/// callee that consumes part of what it borrows leaves the caller believing that
/// storage is still live, so calling it twice consumes twice. This is what the
/// call site reads to know better.
type Summary = Vec<(usize, Vec<Step>, MoveState)>;
type Summaries = HashMap<String, Summary>;

/// Work out what every function does to what it is lent, until a round learns
/// nothing new.
///
/// A round rather than a pass, because what one function gives away depends on
/// what the functions it calls give away: `outer` consuming through `inner`
/// only shows up once `inner`'s own summary is known. The set only ever grows,
/// so this settles.
///
/// A program with no resources at all pays nothing for this, which is the same
/// bargain every other part of the linear machinery strikes.
fn settle_summaries(ast: &Ast, roots: &[StmtId], seed: &Program) -> Summaries {
    let mut summaries = Summaries::new();
    if seed.linear.is_empty() {
        return summaries;
    }
    loop {
        let program = Program {
            summaries: &summaries,
            ..*seed
        };
        let mut round = Summaries::new();
        for statement in roots {
            let Statement::Constant(name, value) = ast.stmt(*statement) else {
                continue;
            };
            let (Expression::Function(params, _, body)
            | Expression::Proc(params, _, body)) = ast.expr(*value)
            else {
                continue;
            };
            let checker = run_function(ast, *params, *body, &program);
            for entry in summarize(ast, *params, &checker) {
                round
                    .entry(ast.name(*name).to_string())
                    .or_default()
                    .push(entry);
            }
        }
        // Grown rather than replaced, so this settles. A round reads the round
        // before it: applying a summary at a call site marks a place gone,
        // which stops the body recording what it would otherwise have recorded,
        // so a set built fresh each time can lose an entry it had and find it
        // again on the round after, and the two states alternate for ever.
        let mut grew = false;
        for (name, found) in round {
            let held = summaries.entry(name).or_default();
            for entry in found {
                let seen = held.iter().any(|(index, under, _)| {
                    *index == entry.0
                        && describe_place(under) == describe_place(&entry.1)
                });
                if !seen {
                    held.push(entry);
                    grew = true;
                }
            }
        }
        if !grew {
            return summaries;
        }
    }
}

/// The run of steps under a parameter that a summary can name, or `None` where
/// it cannot name this place at all.
///
/// Fields it can name, and the dereference the mode lowering puts in front of a
/// borrowed aggregate. An element it cannot: which element is a number worked
/// out while the program runs, so there is no place to hand a caller that means
/// the one this body took. An empty run is not a place under the parameter but
/// the parameter itself, which the call site already reads from the declaration.
fn nameable_under(path: &[Step]) -> Option<&[Step]> {
    let under = &path[1..];
    let named = under
        .iter()
        .all(|step| matches!(step, Step::Named(_) | Step::Deref(false)));
    (!under.is_empty() && named).then_some(under)
}

/// A resource given away out of a borrowed parameter by a place no summary can
/// name, which is the one shape the count cannot be made to cross a call.
///
/// `vec_get` is the example: it answers with `v.storage[index]`, so the caller
/// is handed a resource out of a container it still believes untouched, and
/// asking twice hands the same one out twice.
///
/// Refused rather than approximated. A summary saying "some element of this
/// went" is the whole container as far as a caller can act on it, and that
/// would refuse a container releasing its own elements one at a time, which is
/// how resources in containers are written. What to do instead is reach the
/// element through a borrow that stays a borrow, or take the container by
/// `move` and answer with it again.
fn handed_out_unnameable(
    ast: &Ast,
    params: Range32,
    checker: &MoveChecker,
) -> Vec<String> {
    let mut reports = Vec::new();
    for parameter in ast.params_in(params) {
        if !matches!(
            parameter.type_annotation,
            Some(Type::Ref(_) | Type::RefMut(_))
        ) {
            continue;
        }
        let parameter_name = ast.name(parameter.name);
        for (key, state) in &checker.states {
            if *state == MoveState::Live {
                continue;
            }
            let Some(path) = checker.paths.get(key) else {
                continue;
            };
            let Some(Step::Named(root)) = path.first() else {
                continue;
            };
            if root.as_str() != parameter_name || path.len() < 2 {
                continue;
            }
            if nameable_under(path).is_some() {
                continue;
            }
            reports.push(format!(
                "'{key}' gives away a resource out of '{}', which \
                 this function only borrows, and names it by an element rather \
                 than by a field. A caller cannot be told which element went, \
                 so nothing stops it asking again and being handed the same one \
                 twice. Reach the element through a borrow that stays a borrow, \
                 or take '{}' by `move` and answer with it.",
                parameter_name, parameter_name
            ));
        }
    }
    reports.sort();
    reports.dedup();
    reports
}

/// The places a body hands on that belong to a borrowed parameter, which is
/// exactly what its caller cannot see for itself.
///
/// A parameter taken by `move` is not one of these. The call site already reads
/// the declaration and marks the whole argument gone, so counting it here would
/// say the same thing twice.
fn summarize(ast: &Ast, params: Range32, checker: &MoveChecker) -> Summary {
    let mut found = Summary::new();
    for (index, parameter) in ast.params_in(params).iter().enumerate() {
        if !matches!(
            parameter.type_annotation,
            Some(Type::Ref(_) | Type::RefMut(_))
        ) {
            continue;
        }
        for (key, state) in &checker.states {
            if *state == MoveState::Live {
                continue;
            }
            let Some(path) = checker.paths.get(key) else {
                continue;
            };
            let Some(Step::Named(root)) = path.first() else {
                continue;
            };
            if root.as_str() != ast.name(parameter.name) {
                continue;
            }
            let Some(under) = nameable_under(path) else {
                continue;
            };
            found.push((index, under.to_vec(), *state));
        }
    }
    found.sort_by(|left, right| {
        (left.0, describe_place(&left.1))
            .cmp(&(right.0, describe_place(&right.1)))
    });
    found
}

/// Where a run of storage sits under a parameter: the parameter's index and the
/// field names beneath it. An empty run of names is the parameter itself, which
/// is what a function taking a `[]T` directly hands back a view of.
type RunSummary = Vec<(usize, Vec<Step>)>;
type RunSummaries = HashMap<String, RunSummary>;

/// What every function does with the runs its callers hold: which one its answer
/// views, and which one it replaces.
///
/// These are what make growth checkable. A container that fills asks the
/// allocator for a wider block and gives the old one back, so a view taken
/// before the growth names storage that is no longer the container's. The caller
/// sees neither half: `vec_slice` looks like an ordinary answer and `vec_push`
/// looks like an ordinary write. Both of these say which run is meant, so the
/// two can be weighed against each other at the call.
///
/// Field-granular on purpose. A container that grows one run while a caller
/// holds a view of another is doing nothing wrong, and the ECS does exactly that
/// on every frame: `group_spawn` grows `g.slots` while a `ref` into `g.members`
/// is live. A summary naming only the parameter cannot tell those apart and
/// refuses the honest one.
#[derive(Default)]
struct Runs {
    /// Which run a call's answer views, for a function that answers with one.
    viewed: RunSummaries,
    /// Which run a call can replace.
    replaced: RunSummaries,
}

/// Whether the program builds anything that holds a run at all. Everything the
/// run summaries do is about a view of a container outliving the block behind
/// it, so a program with no container pays nothing for them.
fn program_holds_runs(fields: &FieldTypes) -> bool {
    fields.values().any(is_view_type)
}

/// Work out, for every function, which run its answer views and which run it
/// replaces, until a round learns nothing new.
///
/// A round rather than a pass, for the reason the move summaries need one: what
/// a wrapper views is what the thing it forwards to views, and what a wrapper
/// replaces is what the thing it calls replaces. `vec_slice` answers with
/// `slice_prefix($T, v.storage, v.len)`, so it is only once `slice_prefix` is
/// known to answer with a view of its own parameter that `v.storage` is the run
/// behind it. The tables only gain entries, so this settles.
fn settle_runs(ast: &Ast, roots: &[StmtId], fields: &FieldTypes) -> Runs {
    let mut runs = Runs::default();
    if !program_holds_runs(fields) {
        return runs;
    }
    loop {
        let mut grew = false;
        for statement in roots {
            let Statement::Constant(name, value) = ast.stmt(*statement) else {
                continue;
            };
            let (Expression::Function(parameters, signature, body)
            | Expression::Proc(parameters, signature, body)) = ast.expr(*value)
            else {
                continue;
            };
            let answers_view = ast
                .signature_to_type(ast.signature(*signature))
                .is_some_and(|result| is_view_type(&result));
            let mut walk = RunWalk {
                ast,
                parameters: ast
                    .params_in(*parameters)
                    .iter()
                    .map(|one| ast.name(one.name).to_string())
                    .collect(),
                declared: ast
                    .params_in(*parameters)
                    .iter()
                    .filter_map(|one| {
                        Some((
                            ast.name(one.name).to_string(),
                            one.type_annotation.clone()?,
                        ))
                    })
                    .collect(),
                fields,
                runs: &runs,
                answers_view,
                locals: HashMap::new(),
                views: HashMap::new(),
                found_viewed: Vec::new(),
                found_replaced: Vec::new(),
            };
            walk.walk_block(*body);
            walk.note_tail(*body);
            let viewed = walk.found_viewed.clone();
            let replaced = walk.found_replaced.clone();
            let parameters = walk.parameters.clone();
            grew |= record_runs(
                &mut runs.viewed,
                ast.name(*name),
                &parameters,
                &viewed,
                true,
            );
            grew |= record_runs(
                &mut runs.replaced,
                ast.name(*name),
                &parameters,
                &replaced,
                false,
            );
        }
        if !grew {
            return runs;
        }
    }
}

/// File the places a walk found under the parameters they are rooted in,
/// answering whether anything was new.
///
/// A place reaching through an element is cut back to the field above it. Which
/// element is a number worked out while the program runs, so there is no place
/// to hand a caller that means the one this body took, and the run holding it is
/// what both sides can name. Cutting back widens the place, which is the
/// direction that refuses rather than the one that lets something through.
fn record_runs(
    summaries: &mut RunSummaries,
    name: &str,
    parameters: &[String],
    found: &[Vec<Step>],
    allow_bare: bool,
) -> bool {
    let mut grew = false;
    for path in found {
        let path = without_borrow_derefs(path.clone());
        let Some(Step::Named(root)) = path.first() else {
            continue;
        };
        let Some(index) = parameters.iter().position(|one| one == root) else {
            continue;
        };
        let under = nameable_prefix(&path);
        if under.is_empty() && !allow_bare {
            continue;
        }
        let held = summaries.entry(name.to_string()).or_default();
        let described = describe_place(under);
        let seen = held.iter().any(|(held_index, held_under)| {
            *held_index == index && describe_place(held_under) == described
        });
        if !seen {
            held.push((index, under.to_vec()));
            grew = true;
        }
    }
    grew
}

/// How many names deep a run summary reaches under a parameter.
///
/// A bound rather than a preference. A function that walks a recursive structure
/// reaches `.next`, then `.next.next`, and a fixpoint over those never settles:
/// each round the entry it learnt last round lets it learn a longer one.
/// Cutting at a fixed depth makes the set of entries finite, and cutting widens
/// what an entry names, which is the direction that refuses rather than the one
/// that lets something through.
///
/// The self-hosted compiler holds the same bound, and the two must keep the same
/// number or they refuse different programs.
pub const RUN_STEPS: usize = 4;

/// The run of names under a parameter that a summary can carry, cut at the first
/// step it cannot name and at the depth it stops reaching.
fn nameable_prefix(path: &[Step]) -> &[Step] {
    let under = &path[1..];
    let end = under
        .iter()
        .position(|step| !matches!(step, Step::Named(_)))
        .unwrap_or(under.len());
    &under[..end.min(RUN_STEPS)]
}

/// A place with the dereferences of borrows taken out of it.
///
/// The mode lowering writes a `^` for every mention of a `mut` parameter, so one
/// side of a comparison can carry a step the other does not: `vec_slice` reads
/// its container and names `v.storage`, and `vec_push` writes to its own and
/// names `v^.storage`. A borrow deref names the storage it is written in front
/// of, so dropping it leaves the place meaning what it meant and lets the two be
/// weighed field against field. A raw dereference is not one of these and stays,
/// since where it lands is exactly what nothing here knows.
fn without_borrow_derefs(path: Vec<Step>) -> Vec<Step> {
    path.into_iter()
        .filter(|step| !matches!(step, Step::Deref(false)))
        .collect()
}

/// One function's body, read for the runs it views and the runs it replaces.
struct RunWalk<'a> {
    ast: &'a Ast,
    parameters: Vec<String>,
    declared: HashMap<String, Type>,
    fields: &'a FieldTypes,
    runs: &'a Runs,
    /// Whether this function answers with a view at all. Only then is there
    /// anything for the answer walk to record.
    answers_view: bool,
    locals: HashMap<String, Type>,
    /// For each local, the runs it views. A local naming a place inside a
    /// parameter is not one of these: what a summary says has to be a place the
    /// caller can name too, and a local is nobody's but this frame's.
    views: HashMap<String, Vec<Vec<Step>>>,
    found_viewed: Vec<Vec<Step>>,
    found_replaced: Vec<Vec<Step>>,
}

impl RunWalk<'_> {
    /// The type of a place, as far as the parameter declarations, the local
    /// bindings and the struct declarations give it.
    fn place_type(&self, expression: ExprId) -> Option<Type> {
        let ast = self.ast;
        match ast.expr(expression) {
            // The borrow a parameter carries is left on, since the mode
            // lowering writes a `^` for every mention of a `mut` parameter and
            // that step is what this type answers for.
            Expression::Identifier(name) => self
                .declared
                .get(ast.name(*name))
                .or_else(|| self.locals.get(ast.name(*name)))
                .cloned(),
            Expression::FieldAccess(base, field) => {
                let held = through_borrow(self.place_type(*base)?);
                let (Type::Struct(owner) | Type::Enum(owner)) = held else {
                    return None;
                };
                // The template as well as the name. A field table gathered over
                // a generic's declaration is filed under `Vec`, and a parameter
                // written `Vec<T>` names the instantiation.
                self.fields
                    .get(&(owner.clone(), ast.name(*field).to_string()))
                    .or_else(|| {
                        self.fields.get(&(
                            Type::template_of(&owner).to_string(),
                            ast.name(*field).to_string(),
                        ))
                    })
                    .cloned()
            }
            Expression::Index(base, _) => {
                match through_borrow(self.place_type(*base)?) {
                    Type::Array(inner, _) | Type::Slice(inner) => Some(*inner),
                    _ => None,
                }
            }
            Expression::Dereference(base) => match self.place_type(*base)? {
                Type::Ptr(inner) | Type::Ref(inner) | Type::RefMut(inner) => {
                    Some(*inner)
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Where a place sits under this function's parameters, and nothing for a
    /// place rooted anywhere else. What a summary says has to be a place the
    /// caller can name too, and a local is nobody's but this frame's.
    fn param_places(&self, expression: ExprId) -> Vec<Vec<Step>> {
        let ast = self.ast;
        match ast.expr(expression) {
            Expression::Identifier(name) => {
                let name = ast.name(*name);
                if self.parameters.iter().any(|one| one == name) {
                    return vec![vec![Step::Named(name.to_string())]];
                }
                Vec::new()
            }
            Expression::FieldAccess(base, field) => self.extend_places(
                *base,
                Step::Named(format!(".{}", ast.name(*field))),
            ),
            Expression::Index(base, index) => {
                let literal = match ast.expr(*index) {
                    Expression::Literal(Literal::Integer(value)) => {
                        Some(*value)
                    }
                    _ => None,
                };
                self.extend_places(
                    *base,
                    Step::Index(
                        literal,
                        format!("[{}]", display_expr(ast, *index)),
                    ),
                )
            }
            Expression::Dereference(base) => {
                let raw = !matches!(
                    self.place_type(*base),
                    Some(Type::Ref(_) | Type::RefMut(_))
                );
                self.extend_places(*base, Step::Deref(raw))
            }
            Expression::Borrow(inner)
            | Expression::BorrowMut(inner)
            | Expression::AddressOf(inner) => self.param_places(*inner),
            Expression::Unsafe(body) => block_tail(ast, *body)
                .map(|value| self.param_places(value))
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    fn extend_places(&self, base: ExprId, step: Step) -> Vec<Vec<Step>> {
        self.param_places(base)
            .into_iter()
            .map(|mut path| {
                path.push(step.clone());
                path
            })
            .collect()
    }

    /// The runs a value names the storage of. A field holding a view is one, an
    /// element of a run belongs to that run, and a call answers with whichever
    /// run its own summary says it does.
    fn run_places(&self, expression: ExprId) -> Vec<Vec<Step>> {
        let ast = self.ast;
        match ast.expr(expression) {
            Expression::Identifier(name) => {
                let name = ast.name(*name);
                if self.parameters.iter().any(|one| one == name) {
                    let held = self
                        .declared
                        .get(name)
                        .cloned()
                        .map(through_borrow)
                        .is_some_and(|ty| is_view_type(&ty));
                    if held {
                        return vec![vec![Step::Named(name.to_string())]];
                    }
                    return Vec::new();
                }
                self.views.get(name).cloned().unwrap_or_default()
            }
            Expression::FieldAccess(..) => match self.place_type(expression) {
                Some(ty) if is_view_type(&ty) => self.param_places(expression),
                _ => Vec::new(),
            },
            Expression::Index(base, _) => self.run_places(*base),
            Expression::Borrow(inner)
            | Expression::BorrowMut(inner)
            | Expression::AddressOf(inner)
            | Expression::Dereference(inner)
            | Expression::Try(inner) => self.run_places(*inner),
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(name) = ast.expr(*callee) else {
                    return Vec::new();
                };
                match self.runs.viewed.get(ast.name(*name)) {
                    Some(summary) => self.against(summary, *arguments),
                    // A run reached through something with no summary of its
                    // own: the builtins that form a view, and the wrappers over
                    // them that this walk cannot see into. What they are given
                    // is what they can be naming, and nothing else is in reach.
                    None if builtin_answers_a_view(ast.name(*name)) => ast
                        .exprs_in(*arguments)
                        .iter()
                        .flat_map(|argument| self.run_places(*argument))
                        .collect(),
                    None => Vec::new(),
                }
            }
            Expression::Unsafe(body) => block_tail(ast, *body)
                .map(|value| self.run_places(value))
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    /// A summary read against the arguments of one call: the place each named
    /// argument sits in, with the summary's own names under it.
    fn against(
        &self,
        summary: &RunSummary,
        arguments: Range32,
    ) -> Vec<Vec<Step>> {
        let mut found = Vec::new();
        for (index, under) in summary {
            let Some(argument) = self.ast.exprs_in(arguments).get(*index)
            else {
                continue;
            };
            // The place the argument sits in, or, where it is a local holding a
            // view or a call of its own, the run it names. The lowering hoists a
            // nested call into a temporary, so a view forwarded to a wrapper
            // reaches here as a name rather than as the call that made it.
            let mut bases = self.param_places(*argument);
            if bases.is_empty() {
                bases = self.run_places(*argument);
            }
            for mut path in bases {
                path.extend(under.iter().cloned());
                found.push(path);
            }
        }
        found
    }

    fn walk_block(&mut self, block: Range32) {
        let ast = self.ast;
        for statement in ast.stmts_in(block) {
            self.walk_statement(*statement);
        }
    }

    fn walk_statement(&mut self, statement: StmtId) {
        let ast = self.ast;
        match ast.stmt(statement) {
            Statement::Let {
                name,
                type_annotation,
                value,
                ..
            } => {
                let name = ast.name(*name).to_string();
                self.walk_expression(*value);
                let held =
                    type_annotation.clone().or_else(|| self.place_type(*value));
                match held {
                    Some(ty) => {
                        self.locals.insert(name.clone(), ty);
                    }
                    None => {
                        self.locals.remove(&name);
                    }
                }
                let views = self.run_places(*value);
                self.views.insert(name, views);
            }
            Statement::Assignment(target, value) => {
                self.walk_expression(*value);
                self.walk_expression(*target);
                // A write that puts a different run in a place is the growth
                // this whole summary is about. Only a run: writing an element,
                // or a length beside the run, leaves the storage where it was,
                // and reading those as growth refuses every container that
                // writes into what a caller is holding a `ref` into.
                if self.place_type(*target).is_some_and(|ty| is_view_type(&ty))
                {
                    let found = self.param_places(*target);
                    self.found_replaced.extend(found);
                }
            }
            Statement::Return(value) => {
                self.walk_expression(*value);
                self.note_answer(*value);
            }
            Statement::Constant(_, value)
            | Statement::LetMultiple(_, value)
            | Statement::Expression(value)
            | Statement::Print(value, _) => self.walk_expression(*value),
            Statement::While(condition, body) => {
                self.walk_expression(*condition);
                self.walk_block(*body);
            }
            Statement::For(_, _, sequence, body) => {
                self.walk_expression(*sequence);
                self.walk_block(*body);
            }
            Statement::With(_, body) => self.walk_block(*body),
            Statement::Defer(inner) => self.walk_statement(*inner),
            _ => {}
        }
    }

    /// The calls inside a value, and the blocks they hold. A container grows
    /// inside `if (v.len >= v.cap)`, and an `if` is an expression here, so a
    /// walk that reads only statements misses the one assignment this is for.
    fn walk_expression(&mut self, expression: ExprId) {
        let ast = self.ast;
        if let Expression::Call(callee, arguments) = ast.expr(expression)
            && let Expression::Identifier(name) = ast.expr(*callee)
            && let Some(summary) = self.runs.replaced.get(ast.name(*name))
        {
            let found = self.against(summary, *arguments);
            self.found_replaced.extend(found);
        }
        match ast.expr(expression) {
            Expression::Unsafe(body) => self.walk_block(*body),
            Expression::If(condition, consequence, alternative) => {
                self.walk_expression(*condition);
                self.walk_block(*consequence);
                if let Some(block) = alternative {
                    self.walk_block(*block);
                }
            }
            Expression::Switch(scrutinee, cases) => {
                self.walk_expression(*scrutinee);
                for case in ast.cases_in(*cases) {
                    self.walk_block(case.body);
                }
            }
            _ => {
                for inner in crate::regions::sub_expressions(ast, expression) {
                    self.walk_expression(inner);
                }
            }
        }
    }

    /// What a body hands back, where it hands back a view. A branch in answer
    /// position hands back whatever its arms do, so each arm's own last value is
    /// an answer of its own.
    fn note_answer(&mut self, value: ExprId) {
        if !self.answers_view {
            return;
        }
        let ast = self.ast;
        match ast.expr(value) {
            Expression::If(_, consequence, alternative) => {
                self.note_block_answer(*consequence);
                if let Some(block) = alternative {
                    self.note_block_answer(*block);
                }
            }
            Expression::Switch(_, cases) => {
                for case in ast.cases_in(*cases) {
                    self.note_block_answer(case.body);
                }
            }
            _ => {
                let found = self.run_places(value);
                self.found_viewed.extend(found);
            }
        }
    }

    fn note_block_answer(&mut self, block: Range32) {
        if let Some(value) = block_tail(self.ast, block) {
            self.note_answer(value);
        }
    }

    /// The value a body falls out of its end with, which is an answer the same
    /// as a `return` is. The emitters are what turn the last statement into a
    /// return, so nothing before them has marked it as one.
    fn note_tail(&mut self, body: Range32) {
        self.note_block_answer(body);
    }
}

/// The value a block falls out of its end with.
fn block_tail(ast: &Ast, block: Range32) -> Option<ExprId> {
    match ast.stmt(*ast.stmts_in(block).last()?) {
        Statement::Expression(value) => Some(*value),
        _ => None,
    }
}

/// Through the borrow a parameter that reads an aggregate carries. The question
/// everywhere here is about what it refers to.
fn through_borrow(ty: Type) -> Type {
    match ty {
        Type::Ref(inner) | Type::RefMut(inner) => *inner,
        other => other,
    }
}

/// The builtins whose answer is a view of what they were given. These have no
/// body to read a summary off, and the standard library reaches every run
/// through one of them: `slice_prefix` is `slice_from($T, ptr_to(held[0]), n)`,
/// and it is that chain that says the answer names `held`.
///
/// The self-hosted compiler parses all three into nodes of their own rather than
/// into calls, so what is a name here is a shape there.
fn builtin_answers_a_view(name: &str) -> bool {
    matches!(name, "slice_from" | "ptr_to" | "ptr_cast")
}

fn check_function_moves(
    ast: &Ast,
    params: Range32,
    body: Range32,
    program: &Program,
) -> Vec<crate::diagnostic::Diagnostic> {
    let checker = run_function(ast, params, body, program);
    let unnameable = handed_out_unnameable(ast, params, &checker);
    let mut reports = checker.reports;
    reports.extend(unnameable.into_iter().map(|held| {
        crate::diagnostic::Diagnostic::new(Position::default(), held)
    }));
    reports
}

fn run_function<'a>(
    ast: &'a Ast,
    params: Range32,
    body: Range32,
    program: &Program<'a>,
) -> MoveChecker<'a> {
    let mut checker = MoveChecker {
        ast,
        types: HashMap::new(),
        states: HashMap::new(),
        paths: HashMap::new(),
        linear: program.linear,
        signatures: program.signatures,
        param_types: program.param_types,
        field_types: program.field_types,
        summaries: program.summaries,
        compile_time: ast
            .params_in(params)
            .iter()
            .filter(|parameter| {
                matches!(
                    &parameter.type_annotation,
                    Some(Type::TypeParam(name))
                        if name.as_str() == ast.name(parameter.name)
                )
            })
            .map(|parameter| ast.name(parameter.name).to_string())
            .collect(),
        views: HashMap::new(),
        runs: program.runs,
        view_runs: HashMap::new(),
        view_borrows: HashSet::new(),
        stale: HashMap::new(),
        replacements: 0,
        in_defer: false,
        reports: Vec::new(),
        reported: HashSet::new(),
        moved_at: HashMap::new(),
        at: Position::default(),
    };
    for parameter in ast.params_in(params) {
        if let Some(ty) = &parameter.type_annotation {
            checker.note_binding(ast.name(parameter.name), Some(ty.clone()));
        }
    }
    checker.check_function_body(body);
    checker
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
    ast: &'a Ast,
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
    // What each function already worked out does to the resources it is lent.
    // Empty while the summaries are still settling, which is why they settle
    // before anything is reported.
    summaries: &'a Summaries,
    // The enclosing function's compile-time parameters. A call to one of these
    // names a function only once the generic is specialized, so nothing is
    // known about what it does with its arguments until then.
    compile_time: HashSet<String>,
    // For a binding that views a container's storage rather than its own, the
    // place that container sits in. `view := vec_slice($i64, v)` names the block
    // `v` points at, and the frame check traces that to `v` and stops there,
    // since `v` is alive. What it cannot ask is whether the block is still the
    // one it was: giving it back leaves the view naming storage the allocator
    // has taken, and every read through it is bounds-checked against a length
    // that describes what used to be there.
    views: HashMap<String, (Vec<Step>, String)>,
    /// What each function views and replaces, which is what makes growth
    /// visible from a call site.
    runs: &'a Runs,
    /// The run each binding that views one names, written as a place in this
    /// frame. `view := vec_slice($T, v)` names `v.storage`, and that is the run
    /// a later `vec_push` gives back to the allocator.
    view_runs: HashMap<String, Vec<Vec<Step>>>,
    /// Which of those were taken by a borrow. `ref e := vec_slice($T, v)[i]`
    /// binds one, and `e = 999` writes through it into the container rather
    /// than binding the name to something else.
    view_borrows: HashSet<String>,
    /// The bindings whose run has been replaced since they were taken, and the
    /// place that replaced it. Reading one is reading the block the allocator
    /// has taken.
    stale: HashMap<String, String>,
    /// How many times a run under a live view has been replaced. A loop body is
    /// walked twice only when this moved, since the second walk is what asks
    /// what the top of the loop reads on the turn after.
    replacements: usize,
    in_defer: bool,
    reports: Vec<crate::diagnostic::Diagnostic>,
    // The raw text of what has already been said. Past a move the state stays
    // moved, so every later mention of that name fails the same way, and the
    // second telling is an echo of the first rather than a second mistake.
    reported: HashSet<String>,
    // Where each place was moved, keyed the way `states` is. A use after a
    // move points back here, so the report shows the move as well as the use.
    moved_at: HashMap<String, Position>,
    // The statement being walked, which is what a move records as its place.
    at: Position,
}

impl MoveChecker<'_> {
    fn note_binding(&mut self, name: &str, ty: Option<Type>) {
        self.states.insert(name.to_string(), MoveState::Live);
        self.paths
            .insert(name.to_string(), vec![Step::Named(name.to_string())]);
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
        self.state_of_place(&[Step::Named(name.to_string())], name)
            .0
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
    fn state_of_place(&self, path: &[Step], key: &str) -> (MoveState, String) {
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
    /// The container a value views, where it views one.
    ///
    /// A call answering with a view can only name storage it was handed, so the
    /// container is among its arguments: the one that is a place and holds the
    /// run rather than being it. A `[]T` argument passed straight through is not
    /// one, since the block it names was allocated somewhere else and giving
    /// that argument away does not free it. `ref e := vec_slice($T, v)[i]` views
    /// the same container as the slice it indexes, so the walk reaches through a
    /// borrow and an element to the call underneath.
    fn viewed_container(
        &mut self,
        value: ExprId,
    ) -> Option<(Vec<Step>, String)> {
        let ast = self.ast;
        match ast.expr(value) {
            Expression::Borrow(inner)
            | Expression::BorrowMut(inner)
            | Expression::Index(inner, _)
            | Expression::FieldAccess(inner, _) => {
                self.viewed_container(*inner)
            }
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(name) = ast.expr(*callee) else {
                    return None;
                };
                if !self
                    .signatures
                    .get(ast.name(*name))
                    .is_some_and(|held| is_view_type(&held.result))
                {
                    return None;
                }
                // Which argument the container is comes from the callee's own
                // declaration rather than from the argument's type: a
                // parameter holding a run is what a view can be taken of, and
                // the declaration says so without the walk having to know what
                // every local here is. The self-hosted compiler has no local
                // types at all at this point, so asking the declaration is also
                // what lets both compilers ask the same question.
                let declared = self.param_types.get(ast.name(*name))?;
                let held =
                    ast.exprs_in(*arguments).iter().enumerate().find_map(
                        |(index, argument)| {
                            let ty = declared.get(index)?.as_ref()?;
                            // Through the borrow the mode lowering put there. A
                            // parameter that reads an aggregate is a reference by
                            // the time this runs, and the question is about what it
                            // refers to.
                            let ty = match ty {
                                Type::Ref(inner)
                                | Type::RefMut(inner)
                                | Type::Ptr(inner) => inner.as_ref(),
                                other => other,
                            };
                            // A struct counts whether or not its declared fields
                            // still show the run. The self-hosted compiler reads a
                            // generic's parameter as whatever instantiation was
                            // made last, and taking a view of something and then
                            // giving that thing away is the same mistake whichever
                            // instantiation the node happens to name.
                            let container =
                                matches!(ty, Type::Struct(_) | Type::Enum(_))
                                    || holds_run(ty, self.field_types);
                            container.then_some(*argument)
                        },
                    )?;
                let path = self.borrow_place(held)?;
                let key = self.place_key(&path);
                Some((path, key))
            }
            _ => None,
        }
    }

    /// The runs a value views, written as places in this frame.
    ///
    /// `vec_slice($T, v)` views `v.storage`, and `ref e := vec_slice($T, v)[i]`
    /// views the same run: a `ref` binding is a borrow of the place, so it holds
    /// on to the storage rather than taking a copy out of it.
    ///
    /// Reaching an element or a field without a borrow in front of it is that
    /// copy, and a copy is a value of its own from the moment it is made. The
    /// ECS reads `vec_slice($EntitySlot, world.slots)[id].generation` into an
    /// `i64` and pushes to the same container two lines later, and reading the
    /// number as a view of the block it came out of refused that.
    fn viewed_runs(&self, value: ExprId) -> Vec<Vec<Step>> {
        match self.ast.expr(value) {
            Expression::Borrow(inner) | Expression::BorrowMut(inner) => {
                self.run_behind(*inner)
            }
            Expression::Call(..) => self.run_behind(value),
            _ => Vec::new(),
        }
    }

    /// The run a place sits in, reaching through the borrows and the elements
    /// in front of the call that formed the view.
    fn run_behind(&self, value: ExprId) -> Vec<Vec<Step>> {
        let ast = self.ast;
        match ast.expr(value) {
            Expression::Borrow(inner)
            | Expression::BorrowMut(inner)
            | Expression::Index(inner, _)
            | Expression::FieldAccess(inner, _) => self.run_behind(*inner),
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(name) = ast.expr(*callee) else {
                    return Vec::new();
                };
                let Some(summary) = self.runs.viewed.get(ast.name(*name))
                else {
                    return Vec::new();
                };
                let mut found = Vec::new();
                for (index, under) in summary {
                    let Some(argument) = ast.exprs_in(*arguments).get(*index)
                    else {
                        continue;
                    };
                    for mut path in self.argument_bases(*argument) {
                        path.extend(under.iter().cloned());
                        found.push(path);
                    }
                }
                found
            }
            _ => Vec::new(),
        }
    }

    /// The places an argument can be naming: the place it is, or, where it is a
    /// call of its own, the runs that call answers with.
    ///
    /// `passthrough(vec_slice($T, v))` hands on the run `vec_slice` named, and
    /// asking only for a place gave up there and tracked nothing, so a view
    /// taken through one wrapper was invisible to both the growth rule and the
    /// container-release one.
    fn argument_bases(&self, argument: ExprId) -> Vec<Vec<Step>> {
        // A binding that views a run stands for that run rather than for
        // storage of its own. `held := vec_slice($T, v)` handed to a wrapper is
        // the run inside `v`, and reading `held` as a place loses it.
        if let Expression::Identifier(name) = self.ast.expr(argument)
            && let Some(runs) = self.view_runs.get(self.ast.name(*name))
            && !runs.is_empty()
        {
            return runs.clone();
        }
        match self.borrow_place(argument) {
            Some(path) => vec![without_borrow_derefs(path)],
            None => self.run_behind(argument),
        }
    }

    /// A run has been given back to the allocator and a different one put in its
    /// place, so every view still naming it names storage that is no longer this
    /// program's.
    ///
    /// The replaced place has to be the run itself or something the run hangs
    /// off, never something inside it. `ecs_add` grows a run within a `World`
    /// that a `ref` into `g.members` is pointing at, and the block that moved is
    /// inside an element of the viewed run rather than the run: the `[]World`
    /// still has the pointer and the length it had. Reading those as one refused
    /// the ECS and every program built on it.
    fn note_replaced(&mut self, place: &[Step]) {
        // A place reached through a raw dereference is not one this answers
        // for. Where it lands is exactly what nothing here knows, so the
        // overlap test reads it as possibly anything, and a single
        // `unsafe { p^ = 42 }` would leave every view in the frame stale.
        // Raw pointers are the explicit escape hatch and what they reach is
        // the caller's responsibility, so this declines rather than saying
        // everything moved.
        if reaches_through_raw(place) {
            return;
        }
        let marked: Vec<String> = self
            .view_runs
            .iter()
            .filter(|(name, runs)| {
                !self.stale.contains_key(name.as_str())
                    && runs.iter().any(|run| {
                        !reaches_through_raw(run)
                            && place_maybe_within(run, place)
                    })
            })
            .map(|(name, _)| name.clone())
            .collect();
        if marked.is_empty() {
            return;
        }
        self.replacements += 1;
        let blamed = describe_place(place);
        for name in marked {
            self.stale.insert(name, blamed.clone());
        }
    }

    /// What a call replaces, read against the places the caller wrote. The
    /// argument as written with the callee's names under it, which is the same
    /// reading `apply_summary` makes of what a call consumes.
    fn apply_replacements(&mut self, callee: ExprId, arguments: Range32) {
        let ast = self.ast;
        let Expression::Identifier(name) = ast.expr(callee) else {
            return;
        };
        let Some(summary) = self.runs.replaced.get(ast.name(*name)).cloned()
        else {
            return;
        };
        for (index, under) in summary {
            let Some(argument) = ast.exprs_in(arguments).get(index) else {
                continue;
            };
            for mut path in self.argument_bases(*argument) {
                path.extend(under.iter().cloned());
                self.note_replaced(&path);
            }
        }
    }

    /// A binding takes a view, or stops being one. Rebinding is what taking the
    /// view again after a push amounts to, and it is what clears the staleness a
    /// push left.
    fn note_view_runs(&mut self, name: &str, value: ExprId) {
        let runs = self.viewed_runs(value);
        if runs.is_empty() {
            self.view_runs.remove(name);
            self.view_borrows.remove(name);
        } else {
            self.view_runs.insert(name.to_string(), runs);
            if matches!(
                self.ast.expr(value),
                Expression::Borrow(_) | Expression::BorrowMut(_)
            ) {
                self.view_borrows.insert(name.to_string());
            } else {
                self.view_borrows.remove(name);
            }
        }
        self.stale.remove(name);
    }

    /// Whether a binding names a run that has since been replaced.
    fn views_replaced_run(&self, name: &str) -> Option<String> {
        self.stale.get(name).cloned()
    }

    /// Whether a binding that views a container is naming storage the container
    /// has since given away.
    ///
    /// Two tables answer for it. The container walk names what the call was
    /// handed, and the run summaries name the field under it, which is narrower
    /// and reaches a view taken through a wrapper: `passthrough(vec_slice($T, v))`
    /// hands the call a view rather than the container, so no parameter of it
    /// holds the run and the container walk has nothing to say. Giving the
    /// container away is the same mistake whichever of the two found the view.
    fn views_gone_storage(&self, name: &str) -> Option<String> {
        if let Some((path, key)) = self.views.get(name)
            && self.state_of_place(path, key).0 != MoveState::Live
        {
            return Some(key.clone());
        }
        for run in self.view_runs.get(name)? {
            let key = describe_place(run);
            let (state, blamed) = self.state_of_place(run, &key);
            if state != MoveState::Live {
                return Some(blamed);
            }
        }
        None
    }

    fn place_key(&mut self, path: &[Step]) -> String {
        let key = describe_place(path);
        self.paths
            .entry(key.clone())
            .or_insert_with(|| path.to_vec());
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
    fn visit_beneath(&mut self, target: ExprId) -> Result<()> {
        match self.ast.expr(target) {
            Expression::Identifier(_) => Ok(()),
            Expression::Index(base, index) => {
                let base = *base;
                self.visit(*index, false)?;
                self.visit_beneath(base)
            }
            Expression::FieldAccess(base, _)
            | Expression::Dereference(base) => self.visit_beneath(*base),
            _ => self.visit(target, false),
        }
    }

    /// A field or an element, reached where a value is wanted. What is asked of
    /// the table is the whole path, since that is what says which storage this
    /// names, and what is recorded is the same path when the value is one that
    /// moves.
    fn visit_place(
        &mut self,
        expression: ExprId,
        path: &[Step],
        moving: bool,
    ) -> Result<()> {
        // Reaching into a view is reading through it, so the same question is
        // asked of the name at the root: an element of a view of a container
        // that has given its block back is the freed block.
        if let Some(Step::Named(root)) = path.first()
            && let Some(container) = self.views_gone_storage(root)
        {
            let root = root.clone();
            bail!(
                "'{root}' views storage held by '{container}', which has been given away; the block it names is not the caller's to read"
            );
        }
        if let Some(Step::Named(root)) = path.first()
            && let Some(replaced) = self.views_replaced_run(root)
        {
            let root = root.clone();
            bail!(
                "'{root}' views a run that '{replaced}' has since replaced; growing a container gives its old block back, so the storage this names is not the caller's to read. Take the view again after the growth"
            );
        }
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
                    "value '{blamed}' is already scheduled for consumption by a later defer; it cannot be moved again"
                );
            }
            MoveState::Deferred => {}
            MoveState::Moved | MoveState::MaybeMoved => {
                bail!("use of moved value '{blamed}'");
            }
        }
        if consuming {
            let consumed = if self.in_defer {
                MoveState::Deferred
            } else {
                self.moved_at.insert(key.clone(), self.at);
                MoveState::Moved
            };
            self.states.insert(key, consumed);
        }
        Ok(())
    }

    /// What the callee did to the resources it was lent, read against the
    /// places the caller wrote.
    ///
    /// The place is the argument as written with the callee's path under it, so
    /// `once(h)` where `once` consumes its parameter's `.file` gives up `h.file`
    /// here. That is what makes the count hold across a call: a second `once(h)`
    /// names storage that is already gone.
    ///
    /// The argument as written is deliberate. A loop that binds each element of
    /// a container by `ref` and releases it hands over a name it rebinds every
    /// turn, and rebinding gives that name back, so releasing a container's
    /// elements one at a time stays what it always was.
    fn apply_summary(
        &mut self,
        callee: ExprId,
        arguments: Range32,
    ) -> Result<()> {
        let ast = self.ast;
        let Expression::Identifier(name) = ast.expr(callee) else {
            return Ok(());
        };
        let Some(summary) = self.summaries.get(ast.name(*name)) else {
            return Ok(());
        };
        for (index, under, state) in summary.clone() {
            let Some(argument) = ast.exprs_in(arguments).get(index) else {
                continue;
            };
            let Some(mut path) = self.borrow_place(*argument) else {
                continue;
            };
            path.extend(under);
            let key = self.place_key(&path);
            let (held, blamed) = self.state_of_place(&path, &key);
            if held != MoveState::Live {
                bail!("use of moved value '{blamed}'");
            }
            let consumed = if self.in_defer {
                MoveState::Deferred
            } else {
                if matches!(state, MoveState::Moved | MoveState::MaybeMoved) {
                    self.moved_at.insert(key.clone(), self.at);
                }
                state
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
                    let message = located.to_string();
                    // A use of a moved value points back at the move. The
                    // name is quoted in the message it was just given, and
                    // the move recorded where it happened under the same key.
                    let related = message
                        .split_once("use of moved value '")
                        .and_then(|(_, rest)| rest.split_once('\''))
                        .and_then(|(name, _)| {
                            let moved = self.moved_at.get(name)?;
                            Some(vec![(
                                *moved,
                                format!("'{name}' was moved here"),
                            )])
                        })
                        .unwrap_or_default();
                    self.reports.push(crate::diagnostic::Diagnostic {
                        position,
                        message,
                        related,
                    });
                }
                false
            }
        }
    }

    fn check_block(&mut self, block: Range32) -> bool {
        let ast = self.ast;
        let mut diverges = false;
        for statement in ast.stmts_in(block) {
            self.at = ast.stmt_position(*statement);
            let outcome = self.check_statement(*statement);
            diverges = self.record(outcome, ast.stmt_position(*statement));
            if diverges {
                break;
            }
        }
        diverges
    }

    fn check_function_body(&mut self, block: Range32) -> bool {
        let ast = self.ast;
        let statements = ast.stmts_in(block);
        let mut diverges = false;
        for (index, statement) in statements.iter().enumerate() {
            let is_last = index + 1 == statements.len();
            let position = ast.stmt_position(*statement);
            self.at = position;
            if is_last
                && let Statement::Expression(expression) = ast.stmt(*statement)
            {
                if matches!(
                    ast.expr(*expression),
                    Expression::If(..) | Expression::Switch(..)
                ) {
                    let outcome = self.check_conditional(*expression);
                    diverges = self.record(outcome, position);
                } else {
                    let outcome = self.visit(*expression, true).map(|()| false);
                    self.record(outcome, position);
                }
            } else {
                let outcome = self.check_statement(*statement);
                diverges = self.record(outcome, position);
                if diverges {
                    break;
                }
            }
        }
        diverges
    }

    fn check_statement(&mut self, statement: StmtId) -> Result<bool> {
        let ast = self.ast;
        match ast.stmt(statement) {
            Statement::Let {
                name,
                type_annotation,
                value,
                ..
            } => {
                let name = ast.name(*name).to_string();
                let value = *value;
                self.visit(value, true)?;
                let inferred = infer_type(
                    ast,
                    type_annotation.as_ref(),
                    value,
                    &self.types,
                    self.signatures,
                )
                // A binding whose value is a place takes that place's type,
                // which is how a function read out of a table is known to be
                // one: `run := systems[i].run` then `run(world)`.
                .or_else(|| self.value_type(value));
                self.note_binding(&name, inferred);
                match self.viewed_container(value) {
                    Some(held) => {
                        self.views.insert(name.clone(), held);
                    }
                    // Rebinding replaces whatever the name viewed before, which
                    // is what taking the view again after a push amounts to.
                    None => {
                        self.views.remove(&name);
                    }
                }
                self.note_view_runs(&name, value);
                Ok(false)
            }
            Statement::Constant(_, value)
                if matches!(
                    ast.expr(*value),
                    Expression::Function(..) | Expression::Proc(..)
                ) =>
            {
                Ok(false)
            }
            Statement::Constant(name, value) => {
                let value = *value;
                self.visit(value, true)?;
                let inferred =
                    infer_type(ast, None, value, &self.types, self.signatures);
                self.note_binding(ast.name(*name), inferred);
                Ok(false)
            }
            Statement::Assignment(target, value) => {
                let target = *target;
                let value = *value;
                self.visit(value, true)?;
                // A write that puts a different run in a place is the growth a
                // container does inside itself, seen from the frame that holds
                // it. `vec_push` writes `v.storage` because handing the borrow
                // to a helper would copy the header and lose the new block, so
                // a body that grows a run and reads a view of it never calls
                // anything and no summary carries it.
                //
                // Which write replaces a run is asked of the places, not of the
                // types: a write to the run, or to something the run hangs off,
                // replaces it, and a write to an element inside it does not.
                // That is the same relation a call's replacement is weighed by,
                // and it needs no type table, which is what had left this open.
                if let Some(path) = self.borrow_place(target) {
                    self.note_replaced(&without_borrow_derefs(path));
                }
                // Writing to a `ref` writes through it into the container, so
                // it is a use of the run rather than a rebinding of the name,
                // and a stale one lands in the block that was given back.
                if let Expression::Identifier(name) = ast.expr(target)
                    && self.view_borrows.contains(ast.name(*name))
                    && let Some(replaced) =
                        self.views_replaced_run(ast.name(*name))
                {
                    let name = ast.name(*name);
                    bail!(
                        "'{name}' views a run that '{replaced}' has since replaced; growing a container gives its old block back, so the storage this names is not the caller's to write. Take the view again after the growth"
                    );
                }
                // A binding that is given a view again names the run it was
                // just handed rather than the one it named before, which is what
                // taking the view again after a push amounts to. A write that
                // hands it no view is a write through it, and leaves what it
                // views alone.
                if let Expression::Identifier(name) = ast.expr(target)
                    && !self.viewed_runs(value).is_empty()
                {
                    self.note_view_runs(ast.name(*name), value);
                }
                // The target is written, not read. Putting a value into a place
                // is what makes it hold one again, so a place given away and
                // then assigned is live: `vec_free($Table, world.tables)`
                // followed by `world.tables = kept` is a container replacing
                // what it released. Whatever the target reaches through is
                // still a read, which is the index of `xs[i] = v`.
                if let Expression::Identifier(name) = ast.expr(target) {
                    self.states
                        .insert(ast.name(*name).to_string(), MoveState::Live);
                } else if let Some(path) = self.borrow_place(target) {
                    let key = self.place_key(&path);
                    if let Some(blamed) = self.moved_container_of(&path) {
                        bail!("use of moved value '{blamed}'");
                    }
                    self.revive_place(&path, &key);
                    self.visit_beneath(target)?;
                } else {
                    self.visit(target, false)?;
                }
                Ok(false)
            }
            Statement::Return(expression) => {
                self.visit(*expression, true)?;
                Ok(true)
            }
            Statement::Expression(expression) => {
                if matches!(
                    ast.expr(*expression),
                    Expression::If(..) | Expression::Switch(..)
                ) {
                    self.check_conditional(*expression)
                } else {
                    self.visit(*expression, false)?;
                    Ok(false)
                }
            }
            Statement::While(condition, body) => {
                self.visit(*condition, false)?;
                self.check_loop_body(*body)?;
                Ok(false)
            }
            Statement::For(variable, _, range, body) => {
                self.visit(*range, false)?;
                self.note_binding(ast.name(*variable), Some(Type::I64));
                self.check_loop_body(*body)?;
                Ok(false)
            }
            Statement::Defer(inner) => {
                let was_in_defer = self.in_defer;
                self.in_defer = true;
                let result = self.check_statement(*inner);
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
                self.visit(*expression, false)?;
                for argument in ast.exprs_in(*arguments) {
                    self.visit(*argument, false)?;
                }
                Ok(false)
            }
            // The allocation-sources lowering runs before this check and leaves
            // no `with` behind, and the multiple-return lowering leaves no
            // `LetMultiple`. Both are walked anyway, so neither becomes a hole
            // if that order ever changes.
            Statement::With(_, body) => {
                self.check_block(*body);
                Ok(false)
            }
            Statement::LetMultiple(bindings, value) => {
                self.visit(*value, true)?;
                for binding in ast.bindings_in(*bindings) {
                    self.note_binding(ast.name(binding.name), None);
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

    fn check_loop_body(&mut self, body: Range32) -> Result<()> {
        let before = self.states.clone();
        let replacements = self.replacements;
        self.check_block(body);
        // A second turn of the loop begins where the first one left off, so a
        // view taken above the loop and read at the top of the body is read
        // after the body has already replaced the run behind it. Walking once
        // cannot see that, and walking a body that binds the view fresh every
        // turn twice sees nothing wrong, since the binding clears what the last
        // turn left.
        //
        // Only where a run under a live view was replaced. That is rare, and
        // without the guard every nested loop in the program would be walked
        // twice for each level of nesting.
        if self.replacements != replacements {
            let after = self.states.clone();
            // The move states are put back first, so this walk says nothing
            // about them the first one did not already say: a value the body
            // binds fresh reads as moved from the state the first walk left, and
            // that is the shape reported as a use after move.
            self.states = before.clone();
            self.check_block(body);
            self.states = after;
        }
        for name in before.keys() {
            let previous = before.get(name).copied().unwrap_or(MoveState::Live);
            if previous == MoveState::Live
                && self.state_of(name) != MoveState::Live
                && self.is_move_variable(name)
            {
                if self.is_linear_variable(name) {
                    bail!(
                        "linear value '{name}' is consumed inside a loop; a linear resource must be consumed exactly once, not once per iteration"
                    );
                }
                bail!(
                    "value '{name}' is moved inside a loop; it would be used after move on a later iteration"
                );
            }
        }
        self.states = before;
        Ok(())
    }

    fn check_conditional(&mut self, expression: ExprId) -> Result<bool> {
        match self.ast.expr(expression) {
            Expression::If(condition, consequence, alternative) => {
                self.check_if(*condition, *consequence, *alternative)
            }
            Expression::Switch(scrutinee, cases) => {
                self.check_switch(*scrutinee, *cases)
            }
            _ => {
                self.visit(expression, false)?;
                Ok(false)
            }
        }
    }

    fn check_arm(
        &mut self,
        block: Range32,
    ) -> Result<(HashMap<String, MoveState>, bool)> {
        let diverges = self.check_block(block);
        let states = self.states.clone();
        Ok((states, diverges))
    }

    /// Every arm reads the views as they were before the branch, and after it a
    /// view is stale if any arm replaced the run behind it. Growth on one path
    /// is growth as far as the code below can tell.
    fn merge_stale(
        &mut self,
        before: &HashMap<String, String>,
        arms: Vec<HashMap<String, String>>,
    ) {
        self.stale = before.clone();
        for arm in arms {
            for (name, blamed) in arm {
                self.stale.entry(name).or_insert(blamed);
            }
        }
    }

    fn check_if(
        &mut self,
        condition: ExprId,
        consequence: Range32,
        alternative: Option<Range32>,
    ) -> Result<bool> {
        self.visit(condition, false)?;
        let before = self.states.clone();
        let stale_before = self.stale.clone();

        let (then_states, then_diverges) = self.check_arm(consequence)?;
        let then_stale =
            std::mem::replace(&mut self.stale, stale_before.clone());

        self.states = before.clone();
        let (else_states, else_diverges) = match alternative {
            Some(block) => self.check_arm(block)?,
            None => (before.clone(), false),
        };
        let else_stale = std::mem::take(&mut self.stale);

        self.merge_stale(&stale_before, vec![then_stale, else_stale]);
        self.states = self.merge_arms(
            &before,
            &[(then_states, then_diverges), (else_states, else_diverges)],
        );
        Ok(then_diverges && else_diverges)
    }

    fn check_switch(
        &mut self,
        scrutinee: ExprId,
        cases: Range32,
    ) -> Result<bool> {
        let ast = self.ast;
        self.visit(scrutinee, false)?;
        if let Expression::Identifier(name) = ast.expr(scrutinee)
            && self.is_linear_variable(ast.name(*name))
        {
            self.states
                .insert(ast.name(*name).to_string(), MoveState::Moved);
        }
        let before = self.states.clone();
        let stale_before = self.stale.clone();
        let mut arms = Vec::new();
        let mut stale_arms = Vec::new();
        for case in ast.cases_in(cases) {
            self.states = before.clone();
            self.stale = stale_before.clone();
            arms.push(self.check_arm(case.body)?);
            stale_arms.push(std::mem::take(&mut self.stale));
        }
        self.merge_stale(&stale_before, stale_arms);
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

    fn visit(&mut self, expression: ExprId, moving: bool) -> Result<()> {
        let ast = self.ast;
        match ast.expr(expression) {
            Expression::Identifier(name) => {
                let name = ast.name(*name);
                if let Some(container) = self.views_gone_storage(name) {
                    bail!(
                        "'{name}' views storage held by '{container}', which has been given away; the block it names is not the caller's to read"
                    );
                }
                if let Some(replaced) = self.views_replaced_run(name) {
                    bail!(
                        "'{name}' views a run that '{replaced}' has since replaced; growing a container gives its old block back, so the storage this names is not the caller's to read. Take the view again after the growth"
                    );
                }
                match self.state_of(name) {
                    MoveState::Live => {
                        if moving && self.is_move_variable(name) {
                            let consumed = if self.in_defer {
                                MoveState::Deferred
                            } else {
                                self.moved_at.insert(name.to_string(), self.at);
                                MoveState::Moved
                            };
                            self.states.insert(name.to_string(), consumed);
                        }
                    }
                    MoveState::Deferred => {
                        if moving {
                            bail!(
                                "value '{name}' is already scheduled for consumption by a later defer; it cannot be moved again"
                            );
                        }
                    }
                    MoveState::Moved | MoveState::MaybeMoved => {
                        bail!("use of moved value '{name}'");
                    }
                }
                Ok(())
            }
            Expression::Borrow(inner)
            | Expression::BorrowMut(inner)
            | Expression::AddressOf(inner)
            | Expression::Dereference(inner) => self.visit(*inner, false),
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
                self.visit(*base, false)
            }
            Expression::Index(base, index) => {
                let base = *base;
                self.visit(*index, false)?;
                if let Some(path) = self.borrow_place(expression) {
                    self.visit_place(expression, &path, moving)?;
                    return Ok(());
                }
                self.visit(base, false)
            }
            Expression::PackMap(operand, _, _)
            | Expression::Prefix(_, operand) => self.visit(*operand, false),
            Expression::Infix(left, _, right) => {
                let right = *right;
                self.visit(*left, false)?;
                self.visit(right, false)
            }
            Expression::Call(callee, arguments) => {
                let callee = *callee;
                let arguments = *arguments;
                self.visit(callee, false)?;
                if let Expression::Identifier(name) = ast.expr(callee)
                    && let Some(borrows) =
                        builtin_borrows_first_argument(ast.name(*name))
                {
                    for (index, argument) in
                        ast.exprs_in(arguments).iter().enumerate()
                    {
                        self.visit(*argument, !(borrows && index == 0))?;
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
                let param_types = match ast.expr(callee) {
                    Expression::Identifier(name) => {
                        self.param_types.get(ast.name(*name)).or(held.as_ref())
                    }
                    _ => held.as_ref(),
                };
                check_borrow_exclusivity(
                    self,
                    ast.exprs_in(arguments),
                    param_types,
                )?;
                // A call to a compile-time parameter names a function only
                // once the generic is specialized, so it says nothing about
                // ownership yet and the specialized body answers for it
                // afterwards. Any other unknown callee is still read as
                // consuming, so a function pointer does not quietly stop
                // moving what it is given.
                let deferred = matches!(
                    ast.expr(callee),
                    Expression::Identifier(name)
                        if self.compile_time.contains(ast.name(*name))
                );
                let known = !deferred;
                for (index, argument) in
                    ast.exprs_in(arguments).iter().enumerate()
                {
                    let declared = param_types
                        .and_then(|types| types.get(index))
                        .and_then(|ty| ty.as_ref());
                    let borrows = matches!(
                        declared,
                        Some(Type::Ref(_) | Type::RefMut(_))
                    );
                    self.visit(*argument, known && !borrows)?;
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
                        && let Some(path) = self.borrow_place(*argument)
                    {
                        let key = self.place_key(&path);
                        let consumed = if self.in_defer {
                            MoveState::Deferred
                        } else {
                            self.moved_at.insert(key.clone(), self.at);
                            MoveState::Moved
                        };
                        self.states.insert(key, consumed);
                    }
                }
                if known {
                    self.apply_summary(callee, arguments)?;
                    self.apply_replacements(callee, arguments);
                }
                Ok(())
            }
            Expression::StructInit(_, fields) => {
                for field in ast.named_in(*fields) {
                    self.visit(field.value, true)?;
                }
                Ok(())
            }
            Expression::EnumVariantInit(_, _, fields) => {
                for field in ast.named_in(*fields) {
                    self.visit(field.value, true)?;
                }
                Ok(())
            }
            Expression::Literal(Literal::Array(elements)) => {
                for element in ast.exprs_in(*elements) {
                    self.visit(*element, true)?;
                }
                Ok(())
            }
            Expression::ArrayRepeat(value, _) => self.visit(*value, true),
            Expression::If(..) | Expression::Switch(..) => {
                self.check_conditional(expression)?;
                Ok(())
            }
            // An `unsafe` block is a block of ordinary statements. A move made
            // inside one is a move, and not walking in meant a value consumed
            // there stayed live and could be consumed again.
            Expression::Unsafe(body) => {
                self.check_block(*body);
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
    fn callee_signature(&self, callee: ExprId) -> Option<Vec<Option<Type>>> {
        let Type::Proc(params, _) = self.value_type(callee)? else {
            return None;
        };
        Some(params.into_iter().map(Some).collect())
    }

    /// The type of a place, as far as the names and the struct declarations
    /// give it: a name, an element of one, a field of one, or a field of an
    /// element. This is what a function pointer held in a table is written as.
    fn value_type(&self, expression: ExprId) -> Option<Type> {
        let ast = self.ast;
        match ast.expr(expression) {
            Expression::Identifier(name) => {
                self.types.get(ast.name(*name)).cloned()
            }
            Expression::Index(base, _) => match self.value_type(*base)? {
                Type::Array(inner, _) | Type::Slice(inner) => Some(*inner),
                _ => None,
            },
            Expression::FieldAccess(base, field) => {
                // A `mut` parameter of struct type is a borrow of one, and a
                // field of it is the same field. Asking only about the struct
                // itself left every place behind a borrow untyped, so nothing
                // consumed through one was recorded.
                let held = match self.value_type(*base)? {
                    Type::Ref(inner) | Type::RefMut(inner) => *inner,
                    other => other,
                };
                let Type::Struct(name) = held else {
                    return None;
                };
                self.field_types
                    .get(&(name, ast.name(*field).to_string()))
                    .cloned()
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
    fn borrow_place(&self, expression: ExprId) -> Option<Vec<Step>> {
        let ast = self.ast;
        match ast.expr(expression) {
            Expression::Identifier(name) => {
                Some(vec![Step::Named(ast.name(*name).to_string())])
            }
            Expression::FieldAccess(base, field) => {
                let mut path = self.borrow_place(*base)?;
                path.push(Step::Named(format!(".{}", ast.name(*field))));
                Some(path)
            }
            Expression::Index(base, index) => {
                let mut path = self.borrow_place(*base)?;
                let literal = match ast.expr(*index) {
                    Expression::Literal(Literal::Integer(value)) => {
                        Some(*value)
                    }
                    _ => None,
                };
                path.push(Step::Index(
                    literal,
                    format!("[{}]", display_expr(ast, *index)),
                ));
                Some(path)
            }
            Expression::Dereference(base) => {
                let mut path = self.borrow_place(*base)?;
                path.push(Step::Deref(self.reads_raw_pointer(*base)));
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
    fn reads_raw_pointer(&self, base: ExprId) -> bool {
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
/// Whether a type is a view of storage rather than storage of its own.
fn is_view_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Slice(_)
            | Type::Str
            | Type::Ptr(_)
            | Type::Ref(_)
            | Type::RefMut(_)
    )
}

/// Whether a value of this type owns a run that something else can view: a
/// struct holding a slice is one, and a slice is not, since the block a slice
/// names was allocated elsewhere and handing the slice on does not free it.
fn holds_run(ty: &Type, fields: &FieldTypes) -> bool {
    match ty {
        Type::Struct(name) | Type::Enum(name) => {
            let template = Type::template_of(name);
            fields.iter().any(|((held, _), field)| {
                Type::template_of(held) == template && is_view_type(field)
            })
        }
        Type::Distinct(_, inner) => holds_run(inner, fields),
        _ => false,
    }
}

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
    arguments: &[ExprId],
    param_types: Option<&Vec<Option<Type>>>,
) -> Result<()> {
    let ast = checker.ast;
    // Each borrowed argument as (place path, whether it is exclusive). A `mut`
    // parameter and an explicit `&mut` borrow exclusively. A `read` parameter
    // and a `&` share.
    let mut borrows: Vec<(Vec<Step>, bool)> = Vec::new();
    for (index, argument) in arguments.iter().enumerate() {
        let mode = param_types
            .and_then(|types| types.get(index))
            .and_then(|ty| ty.as_ref());
        let borrow = match ast.expr(*argument) {
            Expression::BorrowMut(inner) => {
                checker.borrow_place(*inner).map(|p| (p, true))
            }
            Expression::Borrow(inner) => {
                checker.borrow_place(*inner).map(|p| (p, false))
            }
            _ => match mode {
                Some(Type::RefMut(_)) => {
                    checker.borrow_place(*argument).map(|p| (p, true))
                }
                Some(Type::Ref(_)) => {
                    checker.borrow_place(*argument).map(|p| (p, false))
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
                        "'{name}' and '{other}' are both borrowed as mutable in a single call; mutable borrows are exclusive"
                    );
                }
                bail!(
                    "'{name}' is borrowed as both shared and mutable in a single call; mutable borrows are exclusive"
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
    ast: &Ast,
    roots: &[StmtId],
) -> HashSet<String> {
    let mut held = declared.clone();
    // The instantiations the program writes. A generic's declared field names a
    // parameter bound to nothing here, so the field table answers for `Slab` and
    // not for `Slab<Node, 2>`, and it is the second that holds the resource.
    let instances = crate::linear_instances::collect_instances(ast, roots);
    let templates = crate::linear_instances::declared_structs(ast, roots);
    loop {
        let mut grew = false;
        for ((owner, _), ty) in fields {
            // The name itself as well as the template it came from. A field
            // table gathered over monomorphized declarations owns names like
            // `Option<File>`, whose template is `Option`, so a guard reading
            // only the template answers about a name that was never going to be
            // inserted and reports growth on every round for ever.
            if held.contains(owner.as_str())
                || held.contains(Type::template_of(owner))
            {
                continue;
            }
            if is_linear_type(ty, &held) && held.insert(owner.clone()) {
                grew = true;
            }
        }
        // In the same loop as the holders, since an instance is a resource
        // because of a field and a struct is one because of an instance in a
        // field of its own.
        if crate::linear_instances::note_linear_instances(
            &templates, &instances, &mut held,
        ) {
            grew = true;
        }
        if !grew {
            return held;
        }
    }
}

fn infer_type(
    ast: &Ast,
    annotation: Option<&Type>,
    value: ExprId,
    types: &HashMap<String, Type>,
    signatures: &Signatures,
) -> Option<Type> {
    if let Some(ty) = annotation {
        return Some(ty.clone());
    }
    match ast.expr(value) {
        Expression::StructInit(name, _) => {
            Some(Type::Struct(ast.name(*name).to_string()))
        }
        Expression::EnumVariantInit(name, _, _) => {
            Some(Type::Enum(ast.name(*name).to_string()))
        }
        Expression::Literal(Literal::String(_)) => Some(Type::Str),
        Expression::Literal(Literal::Integer(_)) => Some(Type::I64),
        Expression::Literal(Literal::Float(_)) => Some(Type::F64),
        Expression::Literal(Literal::Float32(_)) => Some(Type::F32),
        Expression::Literal(Literal::Boolean(_)) | Expression::Boolean(_) => {
            Some(Type::Bool)
        }
        Expression::Identifier(name) => types.get(ast.name(*name)).cloned(),
        // The declared result with this call's type arguments put in, so a call
        // that answers with an instantiation is typed as the one it makes
        // rather than as the template it was declared from.
        Expression::Call(callee, arguments) => {
            let Expression::Identifier(name) = ast.expr(*callee) else {
                return None;
            };
            let signature = signatures.get(ast.name(*name))?;
            let mut subst: HashMap<String, Type> = HashMap::new();
            for (slot, argument) in
                signature.type_params.iter().zip(ast.exprs_in(*arguments))
            {
                if let Some(parameter) = slot
                    && let Expression::TypeValue(ty) = ast.expr(*argument)
                {
                    subst.insert(parameter.clone(), ty.clone());
                }
            }
            if subst.is_empty() {
                return Some(signature.result.clone());
            }
            Some(crate::ir_build::substitute_type(&signature.result, &subst))
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
        let mut module = parser.parse()?;
        let linear = parser.linear_types().clone();
        crate::param_modes::lower_param_modes(&mut module.ast, &module.roots);
        check_ownership(&module.ast, &module.roots, &linear)
    }

    // A use of a moved value carries the move as a related place: the
    // diagnostic answers where the use is, and points back at the line the
    // value left on.
    #[test]
    fn a_use_after_move_points_back_at_the_move() {
        let source = "\
            Pack :: struct { weight: i64 }\n\
            take :: fn(move p: Pack) -> i64 { p.weight }\n\
            run :: fn() -> i64 {\n\
                held := Pack { weight = 1 }\n\
                first := take(held)\n\
                take(held)\n\
            }\n";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let positions = lexer.positions().to_vec();
        let mut parser = Parser::with_positions(&tokens, &positions);
        let mut module = parser.parse().unwrap();
        let linear = parser.linear_types().clone();
        crate::param_modes::lower_param_modes(&mut module.ast, &module.roots);
        let reports =
            check_ownership_recovering(&module.ast, &module.roots, &linear);
        let moved = reports
            .iter()
            .find(|held| held.message.contains("use of moved value"))
            .expect("the second take reports a use after move");
        assert_eq!(moved.related.len(), 1, "{moved:?}");
        assert_eq!(moved.related[0].0.line, 5, "{moved:?}");
        assert!(moved.related[0].1.contains("was moved here"));
        let error = check_ownership(&module.ast, &module.roots, &linear)
            .unwrap_err()
            .to_string();
        assert!(error.contains("was moved here"), "{error}");
    }

    #[test]
    fn a_view_read_after_its_run_was_replaced_is_rejected() {
        let source = "\
            Bag :: struct { room: []i64, len: i64 }\n\
            bag_slice :: fn(b: Bag) -> []i64 { b.room }\n\
            bag_grow :: fn(mut b: Bag, fresh: []i64) { b.room = fresh }\n\
            run :: fn(mut b: Bag, fresh: []i64) -> i64 {\n\
                view := bag_slice(b)\n\
                bag_grow(b, fresh)\n\
                view[0]\n\
            }";
        assert!(check(source).is_err());
    }

    #[test]
    fn a_view_taken_again_after_the_growth_is_allowed() {
        let source = "\
            Bag :: struct { room: []i64, len: i64 }\n\
            bag_slice :: fn(b: Bag) -> []i64 { b.room }\n\
            bag_grow :: fn(mut b: Bag, fresh: []i64) { b.room = fresh }\n\
            run :: fn(mut b: Bag, fresh: []i64) -> i64 {\n\
                mut view := bag_slice(b)\n\
                bag_grow(b, fresh)\n\
                view = bag_slice(b)\n\
                view[0]\n\
            }";
        assert!(check(source).is_ok());
    }

    // The three writes that neighbour the one which replaces a run. Reading
    // which is which off the places rather than off the types is what makes
    // these three cheap to tell apart: a different field is apart at its first
    // step, an element sits below the run, and reading before the write happens
    // before anything moved.
    #[test]
    fn a_write_beside_a_run_leaves_a_view_of_it_alone() {
        let source = "\
            Bag :: struct { room: []i64, len: i64 }\n\
            bag_slice :: fn(b: Bag) -> []i64 { b.room }\n\
            grow_other :: fn(mut b: Bag, count: i64) -> i64 {\n\
                view := bag_slice(b)\n\
                b.len = count\n\
                view[0]\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn a_write_into_a_run_leaves_a_view_of_it_alone() {
        let source = "\
            Bag :: struct { room: []i64, len: i64 }\n\
            bag_slice :: fn(b: Bag) -> []i64 { b.room }\n\
            write_element :: fn(mut b: Bag, value: i64) -> i64 {\n\
                view := bag_slice(b)\n\
                b.room[0] = value\n\
                view[0]\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn a_view_read_before_the_same_body_replaces_its_run_is_allowed() {
        let source = "\
            Bag :: struct { room: []i64, len: i64 }\n\
            bag_slice :: fn(b: Bag) -> []i64 { b.room }\n\
            read_then_grow :: fn(mut b: Bag, fresh: []i64) -> i64 {\n\
                view := bag_slice(b)\n\
                held := view[0]\n\
                b.room = fresh\n\
                held\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn growing_one_run_leaves_a_view_of_another_alone() {
        let source = "\
            Pair :: struct { left: []i64, right: []i64 }\n\
            pair_right :: fn(p: Pair) -> []i64 { p.right }\n\
            pair_grow :: fn(mut p: Pair, fresh: []i64) { p.left = fresh }\n\
            run :: fn(mut p: Pair, fresh: []i64) -> i64 {\n\
                view := pair_right(p)\n\
                pair_grow(p, fresh)\n\
                view[0]\n\
            }";
        assert!(check(source).is_ok());
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
