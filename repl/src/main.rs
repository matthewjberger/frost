use anyhow::Result;
use monkey::{Lexer, Parser};
use rustyline::{error::ReadlineError, Editor};

fn main() -> Result<()> {
    println!();
    println!("Welcome to the Monkey programming language REPL!");
    println!("You may type Monkey code below for evaluation.");
    println!("Enter \"exit\" or press \"CTRL+C\" to exit the REPL.");
    println!();

    let mut rl = Editor::<()>::new();
    if rl.load_history("history.txt").is_err() {
        println!("No previous history.");
    }
    loop {
        let readline = rl.readline("monkey >> ");
        match readline {
            Ok(line) => match line.as_ref() {
                "exit" => break,
                line => {
                    rl.add_history_entry(line);

                    // Lexing
                    let mut lexer = Lexer::new(line);
                    let tokens = match lexer.tokenize() {
                        Ok(tokens) => tokens,
                        Err(error) => {
                            eprintln!("Error lexing: {}", error);
                            continue;
                        }
                    };
                    println!("--- Tokens ---");
                    println!("{:?}", tokens);

                    // Parsing
                    let mut parser = Parser::new(&tokens);
                    let program = match parser.parse() {
                        Ok(program) => program,
                        Err(error) => {
                            eprintln!("Error parsing: {}", error);
                            continue;
                        }
                    };
                    println!("--- Statements ---");
                    for statement in program.iter() {
                        println!("{}", statement);
                        println!("Debug: {:?}", statement);
                    }
                }
            },
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }
    rl.save_history("history.txt")?;
    Ok(())
}
