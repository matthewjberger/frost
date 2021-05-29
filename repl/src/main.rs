use anyhow::Result;
use monkey::Lexer;
use rustyline::{error::ReadlineError, Editor};

fn main() -> Result<()> {
    println!();
    println!("Welcome to the Monkey programming language REPL!");
    println!("You may type Monkey code below for evaluation.");
    println!("Enter \"exit\" or press \"CTRL+C\" to exit the REPL.");
    println!();

    let mut rl = Editor::<()>::new();
    loop {
        let readline = rl.readline("monkey >> ");
        match readline {
            Ok(line) => match line.as_ref() {
                "exit" => break,
                line => {
                    let mut lexer = Lexer::new(line);
                    for token in lexer.tokenize()? {
                        println!("{:?}", token)
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
    Ok(())
}
