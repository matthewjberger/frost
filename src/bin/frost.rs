use std::fs;

use anyhow::{Context, Result};
use clap::Parser;
use frost::{Compiler, Lexer, Parser as FrostParser, VirtualMachine};

#[derive(Parser)]
#[command(name = "frost")]
#[command(about = "The Frost programming language")]
struct Cli {
    file: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let source = fs::read_to_string(&cli.file)
        .with_context(|| format!("Failed to read file: {}", cli.file))?;

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().context("Lexer error")?;

    let mut parser = FrostParser::new(&tokens);
    let statements = parser.parse().context("Parser error")?;

    let mut compiler = Compiler::new(&statements);
    let bytecode = compiler.compile().context("Compiler error")?;

    let mut vm = VirtualMachine::new(
        bytecode.constants,
        bytecode.functions,
        bytecode.heap,
    );
    vm.run(&bytecode.instructions).context("Runtime error")?;

    Ok(())
}
