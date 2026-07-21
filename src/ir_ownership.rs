use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::ir::{
    BlockId, IrFunction, IrModule, IrStatement, IrTerminator, LocalId,
};

const UNOWNED: u8 = 1;
const OWNED: u8 = 2;
const CONSUMED: u8 = 4;

type State = HashMap<LocalId, u8>;

pub fn check_linearity(module: &IrModule) -> Result<()> {
    for function in &module.functions {
        check_function(function)?;
    }
    Ok(())
}

fn check_function(function: &IrFunction) -> Result<()> {
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

    for (block_id, entry) in block_entry.iter().enumerate() {
        if let Some(entry) = entry {
            report_block(function, block_id, entry.clone())?;
        }
    }
    Ok(())
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
        IrStatement::Assign(local, _) => {
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
) -> Result<()> {
    for statement in &function.blocks[block_id].statements {
        if let IrStatement::Consume(local) = statement {
            let current = state.get(local).copied().unwrap_or(UNOWNED);
            if current != OWNED {
                let name = local_name(function, *local);
                if current == CONSUMED {
                    bail!(
                        "linearity: linear value {name} is consumed more than once"
                    );
                }
                bail!(
                    "linearity: linear value {name} may be consumed more than once or before it holds a resource"
                );
            }
        }
        apply(&mut state, statement);
    }

    if let IrTerminator::Return(_) = &function.blocks[block_id].terminator {
        for (&local, &owned) in state.iter() {
            if local < function.param_count {
                continue;
            }
            if function.locals[local].name.is_none() {
                continue;
            }
            if owned & OWNED != 0 {
                let name = local_name(function, local);
                bail!(
                    "linearity: linear value {name} is not consumed on every path before return"
                );
            }
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Lexer, Parser, build_module};

    fn check(source: &str) -> Result<()> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let statements = parser.parse()?;
        let linear = parser.linear_types().clone();
        let module = build_module(&statements, &linear)?;
        check_linearity(&module)
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
}
