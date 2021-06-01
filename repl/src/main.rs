use anyhow::Result;
use frost::{Evaluator, Lexer, Object, Parser};
use rustyline::{error::ReadlineError, Editor};

fn main() -> Result<()> {
    println!();
    println!("Welcome to the Monkey programming language REPL!");
    println!("You may type Monkey code below for evaluation.");
    println!("Enter \"exit\" or press \"CTRL+C\" to exit the REPL.");
    println!();

    let mut evaluator = Evaluator::default();

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

                    let mut lexer = Lexer::new(line);
                    let tokens = match lexer.tokenize() {
                        Ok(tokens) => tokens,
                        Err(error) => {
                            eprintln!("Error lexing: {}", error);
                            continue;
                        }
                    };

                    let mut parser = Parser::new(&tokens);
                    let program = match parser.parse() {
                        Ok(program) => program,
                        Err(error) => {
                            eprintln!("Error parsing: {}", error);
                            continue;
                        }
                    };

                    let result = match evaluator.evaluate_program(&program) {
                        Ok(program) => program,
                        Err(error) => {
                            eprintln!("Error evaluating: {}", error);
                            continue;
                        }
                    };

                    // Leaving this in for debugging purposes
                    let verbose = false;
                    if verbose {
                        println!("--- Tokens ---");
                        println!("{:?}", tokens);
                        println!("--- Statements ---");
                        for statement in program.iter() {
                            println!("{}", statement);
                            println!("Debug: {:?}", statement);
                        }
                        println!("--- Result ---");
                    }

                    if result != Object::Empty {
                        println!("{}", result);
                    }
                }
            },
            Err(ReadlineError::Interrupted) => break,
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }
    rl.save_history("history.txt")?;
    Ok(())
}
