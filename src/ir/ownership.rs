use std::collections::{HashMap, HashSet};

use anyhow::Result;

use crate::diagnostic::Diagnostic;
use crate::ir::{
    BlockId, IrFunction, IrModule, IrOperand, IrRvalue, IrStatement,
    IrTerminator, LocalId,
};

const UNOWNED: u8 = 1;
const OWNED: u8 = 2;
const CONSUMED: u8 = 4;

type State = HashMap<LocalId, u8>;

pub fn check_linearity(module: &IrModule) -> Result<()> {
    let reports = check_linearity_recovering(module, &HashSet::new());
    if reports.is_empty() {
        return Ok(());
    }
    let rendered: Vec<String> =
        reports.iter().map(|held| held.rendered()).collect();
    Err(anyhow::anyhow!(rendered.join("\n")))
}

/// Check each function, reporting one failure per function rather than
/// stopping at the first. The ownership state a walk builds belongs to the
/// function it walked, so a fault in one says nothing about the next.
/// `pooled` names the container types already refused as pools of resources.
/// Nothing consumes such a value the way the language asks, which is the
/// refusal, so telling its holder to consume it is telling them to do something
/// that fixes nothing.
pub fn check_linearity_recovering(
    module: &IrModule,
    pooled: &HashSet<String>,
) -> Vec<crate::diagnostic::Diagnostic> {
    let mut reports = Vec::new();
    for function in &module.functions {
        if let Err(fault) = check_function(function, pooled) {
            reports.push(fault);
        }
    }
    reports
}

fn check_function(
    function: &IrFunction,
    pooled: &HashSet<String>,
) -> Result<(), Diagnostic> {
    let linear_locals: Vec<LocalId> = (0..function.locals.len())
        .filter(|&local| function.locals[local].linear)
        .collect();
    if linear_locals.is_empty() {
        return Ok(());
    }

    let seed: State = linear_locals
        .iter()
        .map(|&local| {
            let state = if local < function.param_count {
                OWNED
            } else {
                UNOWNED
            };
            (local, state)
        })
        .collect();

    let block_count = function.blocks.len();
    let mut block_entry: Vec<Option<State>> = vec![None; block_count];
    block_entry[function.entry] = Some(seed);

    let mut worklist: Vec<BlockId> = vec![function.entry];
    while let Some(block_id) = worklist.pop() {
        let entry = block_entry[block_id]
            .clone()
            .expect("worklist blocks always have an entry state");
        let exit = transfer_block(function, block_id, entry);
        for successor in successors(&function.blocks[block_id].terminator) {
            let merged = match &block_entry[successor] {
                Some(existing) => join(existing, &exit, &linear_locals),
                None => exit.clone(),
            };
            if block_entry[successor].as_ref() != Some(&merged) {
                block_entry[successor] = Some(merged);
                worklist.push(successor);
            }
        }
    }

    let referenced = referenced_locals(function);
    for (block_id, entry) in block_entry.iter().enumerate() {
        if let Some(entry) = entry {
            report_block(
                function,
                block_id,
                entry.clone(),
                &referenced,
                pooled,
            )?;
        }
    }
    Ok(())
}

fn referenced_locals(function: &IrFunction) -> HashSet<LocalId> {
    let mut referenced = HashSet::new();
    for block in &function.blocks {
        for statement in &block.statements {
            match statement {
                IrStatement::Assign(_, rvalue) => {
                    collect_rvalue(rvalue, &mut referenced);
                }
                IrStatement::Store { address, value } => {
                    collect_operand(address, &mut referenced);
                    collect_operand(value, &mut referenced);
                }
                IrStatement::Copy {
                    destination,
                    source,
                    ..
                } => {
                    collect_operand(destination, &mut referenced);
                    collect_operand(source, &mut referenced);
                }
                IrStatement::Own(_) | IrStatement::Consume(_) => {}
            }
        }
        match &block.terminator {
            IrTerminator::Return(Some(operand)) => {
                collect_operand(operand, &mut referenced);
            }
            IrTerminator::Branch { condition, .. } => {
                collect_operand(condition, &mut referenced);
            }
            _ => {}
        }
    }
    referenced
}

fn collect_operand(operand: &IrOperand, referenced: &mut HashSet<LocalId>) {
    if let IrOperand::Local(local) = operand {
        referenced.insert(*local);
    }
}

fn collect_rvalue(rvalue: &IrRvalue, referenced: &mut HashSet<LocalId>) {
    match rvalue {
        IrRvalue::Use(operand)
        | IrRvalue::Unary(_, operand)
        | IrRvalue::Cast(operand, _) => collect_operand(operand, referenced),
        IrRvalue::Binary(_, left, right) => {
            collect_operand(left, referenced);
            collect_operand(right, referenced);
        }
        IrRvalue::AddressOf { local, .. } => {
            referenced.insert(*local);
        }
        IrRvalue::FieldAddress { base, .. } => {
            collect_operand(base, referenced);
        }
        IrRvalue::ElementAddress { base, index, .. } => {
            collect_operand(base, referenced);
            collect_operand(index, referenced);
        }
        IrRvalue::Load { address, .. } => collect_operand(address, referenced),
        IrRvalue::Call { arguments, .. } => {
            for argument in arguments {
                collect_operand(argument, referenced);
            }
        }
        IrRvalue::FunctionAddress(_) => {}
    }
}

fn transfer_block(
    function: &IrFunction,
    block_id: BlockId,
    mut state: State,
) -> State {
    for statement in &function.blocks[block_id].statements {
        apply(&mut state, statement);
    }
    state
}

fn apply(state: &mut State, statement: &IrStatement) {
    match statement {
        IrStatement::Assign(local, _) | IrStatement::Own(local) => {
            if state.contains_key(local) {
                state.insert(*local, OWNED);
            }
        }
        IrStatement::Consume(local) => {
            state.insert(*local, CONSUMED);
        }
        IrStatement::Store { .. } | IrStatement::Copy { .. } => {}
    }
}

fn report_block(
    function: &IrFunction,
    block_id: BlockId,
    mut state: State,
    referenced: &HashSet<LocalId>,
    pooled: &HashSet<String>,
) -> Result<(), Diagnostic> {
    for statement in &function.blocks[block_id].statements {
        if let IrStatement::Consume(local) = statement {
            let current = state.get(local).copied().unwrap_or(UNOWNED);
            if current != OWNED {
                let name = local_name(function, *local);
                let message = if current == CONSUMED {
                    format!("linear value {name} is consumed more than once")
                } else {
                    format!(
                        "linear value {name} may be consumed more than once or before it holds a resource"
                    )
                };
                return Err(located(function, *local, message));
            }
        }
        apply(&mut state, statement);
    }

    if let IrTerminator::Return(_) = &function.blocks[block_id].terminator {
        for (&local, &owned) in state.iter() {
            if local < function.param_count {
                continue;
            }
            if owned & OWNED == 0 {
                continue;
            }
            // A pool of resources is refused where it is written, and the
            // obligation it carries cannot be answered: no consumer discharges
            // it. Telling its holder to consume it as well points at a second
            // line with nothing the reader can do about it.
            if pooled.contains(&function.locals[local].ty.to_string()) {
                continue;
            }
            if let Some(held) = &function.locals[local].name {
                // The storage a `_` was given. Naming it points the reader at a
                // word nothing in the program spells, so the complaint is the
                // one the `_` earns: it took a resource and let it go.
                if names_a_discard(held) {
                    return Err(located(
                        function,
                        local,
                        format!(
                            "this `_` drops a '{}', which is consumed exactly once; bind it to a name and consume it",
                            function.locals[local].ty
                        ),
                    ));
                }
                let name = local_name(function, local);
                return Err(located(
                    function,
                    local,
                    format!(
                        "linear value {name} is not consumed on every path before return"
                    ),
                ));
            }
            if !referenced.contains(&local) {
                return Err(located(
                    function,
                    local,
                    "a linear value is created but never consumed".to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn located(
    function: &IrFunction,
    local: LocalId,
    message: String,
) -> Diagnostic {
    Diagnostic::new(
        function.locals[local].position,
        crate::modules::imports::demangle_private_names(&message),
    )
}

fn local_name(function: &IrFunction, local: LocalId) -> String {
    match &function.locals[local].name {
        Some(name) => format!("'{name}'"),
        None => format!("_{local}"),
    }
}

fn join(left: &State, right: &State, linear_locals: &[LocalId]) -> State {
    linear_locals
        .iter()
        .map(|&local| {
            let a = left.get(&local).copied().unwrap_or(0);
            let b = right.get(&local).copied().unwrap_or(0);
            (local, a | b)
        })
        .collect()
}

fn successors(terminator: &IrTerminator) -> Vec<BlockId> {
    match terminator {
        IrTerminator::Jump(block) => vec![*block],
        IrTerminator::Branch {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        IrTerminator::Return(_) | IrTerminator::Unreachable => Vec::new(),
    }
}

/// Whether this local is the storage a `_` was given.
///
/// Told by the name, because that is the whole of what the lowering leaves
/// behind: the parser writes `__discard0` for a discard and a program cannot
/// spell the prefix, which is reserved. A flag on the local would say it
/// outright, and the name is what crosses from the parser to here.
fn names_a_discard(name: &str) -> bool {
    name.starts_with("__discard")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Lexer, Parser, build_module};

    fn check(source: &str) -> Result<()> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let mut module = parser.parse()?;
        let linear = parser.linear_types().clone();
        let module = build_module(&mut module.ast, &module.roots, &linear)?;
        // The walk a build runs, with the set of pooled types it is handed
        // there. A program here declares none, so it is empty, and the path
        // under test is the path that runs.
        let reports = check_linearity_recovering(&module, &HashSet::new());
        match reports.is_empty() {
            true => Ok(()),
            false => Err(anyhow::anyhow!(
                reports
                    .iter()
                    .map(|held| held.rendered())
                    .collect::<Vec<_>>()
                    .join("\n")
            )),
        }
    }

    const PRELUDE: &str = "\
        File :: linear struct { handle: i64 }\n\
        open :: fn() -> File { File { handle = 1 } }\n\
        close :: extern fn(f: File)\n";

    #[test]
    fn ir_accepts_a_consumed_linear() {
        let source =
            format!("{PRELUDE}run :: fn() {{ f := open()  close(f) }}");
        assert!(check(&source).is_ok());
    }

    #[test]
    fn ir_rejects_a_leaked_linear() {
        let source = format!("{PRELUDE}run :: fn() {{ f := open() }}");
        assert!(check(&source).is_err());
    }

    #[test]
    fn ir_rejects_a_double_consumed_linear() {
        let source = format!(
            "{PRELUDE}run :: fn() {{ f := open()  close(f)  close(f) }}"
        );
        assert!(check(&source).is_err());
    }

    #[test]
    fn ir_rejects_consumption_on_only_one_branch() {
        let source = format!(
            "{PRELUDE}run :: fn() {{ f := open()  if (1 == 1) {{ close(f) }} }}"
        );
        assert!(check(&source).is_err());
    }

    #[test]
    fn ir_accepts_consumption_on_every_branch() {
        let source = format!(
            "{PRELUDE}run :: fn() {{ f := open()  if (1 == 1) {{ close(f) }} else {{ close(f) }} }}"
        );
        assert!(check(&source).is_ok());
    }

    #[test]
    fn ir_rejects_consumption_inside_a_loop() {
        let source = format!(
            "{PRELUDE}run :: fn() {{ f := open()  mut i : i64 = 0  while (i < 3) {{ close(f)  i = i + 1 }} }}"
        );
        assert!(check(&source).is_err());
    }

    #[test]
    fn ir_rejects_a_discarded_linear() {
        let source = format!("{PRELUDE}run :: fn() {{ open() }}");
        assert!(check(&source).is_err());
    }

    #[test]
    fn ir_rejects_a_discarded_linear_in_the_middle_of_a_body() {
        let source =
            format!("{PRELUDE}run :: fn() {{ f := open()  open()  close(f) }}");
        assert!(check(&source).is_err());
    }

    #[test]
    fn ir_accepts_a_temporary_passed_straight_by_value() {
        let source = format!("{PRELUDE}run :: fn() {{ close(open()) }}");
        assert!(check(&source).is_ok());
    }

    #[test]
    fn ir_accepts_returning_a_fresh_temporary() {
        let source = format!(
            "{PRELUDE}make :: fn() -> File {{ open() }}\nrun :: fn() {{ close(make()) }}"
        );
        assert!(check(&source).is_ok());
    }

    #[test]
    fn ir_accepts_a_linear_moved_into_a_field_then_consumed() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            Box :: linear struct { inner: File }\n\
            open :: fn() -> File { File { handle = 1 } }\n\
            sink :: extern fn(b: Box)\n\
            run :: fn() { f := open()  b := Box { inner = f }  sink(b) }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn ir_rejects_leaking_an_aggregate_that_holds_a_linear() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            Box :: linear struct { inner: File }\n\
            open :: fn() -> File { File { handle = 1 } }\n\
            run :: fn() { f := open()  b := Box { inner = f } }";
        assert!(check(source).is_err());
    }

    #[test]
    fn ir_rejects_an_ignored_linear_enum_result() {
        let source = "\
            Outcome :: linear enum { Ok { value: i64 }, Err { code: i64 } }\n\
            run_step :: fn() -> Outcome { Outcome::Ok { value = 1 } }\n\
            caller :: fn() -> i64 { result := run_step()  7 }";
        assert!(check(source).is_err());
    }

    #[test]
    fn ir_rejects_consumption_on_only_one_match_arm() {
        let source = "\
            File :: linear struct { handle: i64 }\n\
            Flag :: enum { A, B }\n\
            open :: fn() -> File { File { handle = 1 } }\n\
            close :: extern fn(f: File)\n\
            noop :: extern fn()\n\
            run :: fn() { f := open()  flag := Flag::A  match flag { case .A: close(f)  case .B: noop() } }";
        assert!(check(source).is_err());
    }

    #[test]
    fn ir_rejects_a_leak_on_an_early_return() {
        let source = format!(
            "{PRELUDE}run :: fn() -> i64 {{ f := open()  if (1 == 1) {{ return 0 }}  close(f)  0 }}"
        );
        assert!(check(&source).is_err());
    }

    #[test]
    fn ir_rejects_a_linear_left_unconsumed_by_a_break() {
        let source = format!(
            "{PRELUDE}run :: fn() {{ f := open()  mut i : i64 = 0  while (i < 3) {{ break  close(f) }} }}"
        );
        assert!(check(&source).is_err());
    }
}
