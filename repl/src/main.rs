use anyhow::Result;
use frost::{Compiler, Lexer, Parser, Value64, VirtualMachine};
use rustyline::{error::ReadlineError, Editor};

fn main() -> Result<()> {
    println!(
        r"
Welcome to the Frost programming language REPL!
You may type Frost code below for evaluation.
Enter 'exit' or press 'CTRL+C' to exit the REPL.
    "
    );

    let mut rl = Editor::<()>::new();
    if rl.load_history("history.txt").is_err() {
        println!("No previous history.");
    }

    let mut accumulated_code = String::new();

    loop {
        let readline = rl.readline("> ");
        match readline {
            Ok(line) => match line.as_ref() {
                "exit" => break,
                line => {
                    rl.add_history_entry(line);

                    let test_code = format!("{}\n{}", accumulated_code, line);

                    let mut lexer = Lexer::new(&test_code);
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

                    let mut compiler = Compiler::new(&program);
                    let bytecode = match compiler.compile() {
                        Ok(bytecode) => bytecode,
                        Err(error) => {
                            eprintln!("Error compiling: {}", error);
                            continue;
                        }
                    };

                    let mut vm = VirtualMachine::new(
                        bytecode.constants,
                        bytecode.functions,
                        bytecode.heap,
                    );

                    if let Err(error) = vm.run(&bytecode.instructions) {
                        eprintln!("Error running: {}", error);
                        continue;
                    }

                    accumulated_code = test_code;

                    let result = match vm.last_popped() {
                        Ok(value) => value,
                        Err(_) => continue,
                    };

                    if result != Value64::Null {
                        print_value(&vm, result);
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

fn print_value(vm: &VirtualMachine, value: Value64) {
    match value {
        Value64::Integer(n) => println!("{}", n),
        Value64::Float(f) => println!("{}", f),
        Value64::Float32(f) => println!("{}f32", f),
        Value64::Bool(b) => println!("{}", b),
        Value64::Null => println!("null"),
        Value64::HeapRef(index) => match &vm.heap[index as usize] {
            frost::HeapObject::String(s) => println!("\"{}\"", s),
            frost::HeapObject::Array(arr) => {
                print!("[");
                for (index, val) in arr.iter().enumerate() {
                    if index > 0 {
                        print!(", ");
                    }
                    print_value_inline(vm, *val);
                }
                println!("]");
            }
            frost::HeapObject::HashMap(map) => {
                print!("{{");
                for (index, (key, val)) in map.iter().enumerate() {
                    if index > 0 {
                        print!(", ");
                    }
                    print!("{}: ", key);
                    print_value_inline(vm, *val);
                }
                println!("}}");
            }
            frost::HeapObject::Closure(closure) => {
                println!("<closure fn#{}>", closure.function_index);
            }
            frost::HeapObject::Struct(name, _fields) => {
                println!("<struct {}>", name);
            }
            frost::HeapObject::BuiltIn(builtin) => {
                println!("<builtin {:?}>", builtin);
            }
            frost::HeapObject::NativeFunction(name) => {
                println!("<native {}>", name);
            }
            frost::HeapObject::NativeHandle(idx) => {
                println!("<native handle #{}>", idx);
            }
            frost::HeapObject::TaggedUnion(tag, fields) => {
                print!("<tagged union tag={} ", tag);
                print!("{{");
                for (index, val) in fields.iter().enumerate() {
                    if index > 0 {
                        print!(", ");
                    }
                    print_value_inline(vm, *val);
                }
                println!("}}>");
            }
            frost::HeapObject::Vec2(x, y) => {
                println!("Vec2({}, {})", x, y);
            }
            frost::HeapObject::Vec3(x, y, z) => {
                println!("Vec3({}, {}, {})", x, y, z);
            }
            frost::HeapObject::Vec4(x, y, z, w) => {
                println!("Vec4({}, {}, {}, {})", x, y, z, w);
            }
            frost::HeapObject::Quat(x, y, z, w) => {
                println!("Quat({}, {}, {}, {})", x, y, z, w);
            }
            frost::HeapObject::Mat4(m) => {
                println!("Mat4({:?})", m);
            }
        },
    }
}

fn print_value_inline(vm: &VirtualMachine, value: Value64) {
    match value {
        Value64::Integer(n) => print!("{}", n),
        Value64::Float(f) => print!("{}", f),
        Value64::Float32(f) => print!("{}f32", f),
        Value64::Bool(b) => print!("{}", b),
        Value64::Null => print!("null"),
        Value64::HeapRef(index) => match &vm.heap[index as usize] {
            frost::HeapObject::String(s) => print!("\"{}\"", s),
            frost::HeapObject::Array(_) => print!("[...]"),
            frost::HeapObject::HashMap(_) => print!("{{...}}"),
            frost::HeapObject::Closure(closure) => {
                print!("<closure fn#{}>", closure.function_index);
            }
            frost::HeapObject::Struct(name, _) => {
                print!("<struct {}>", name);
            }
            frost::HeapObject::BuiltIn(builtin) => {
                print!("<builtin {:?}>", builtin);
            }
            frost::HeapObject::NativeFunction(name) => {
                print!("<native {}>", name);
            }
            frost::HeapObject::NativeHandle(idx) => {
                print!("<native handle #{}>", idx);
            }
            frost::HeapObject::TaggedUnion(tag, _) => {
                print!("<tagged union tag={}>", tag);
            }
            frost::HeapObject::Vec2(x, y) => {
                print!("Vec2({}, {})", x, y);
            }
            frost::HeapObject::Vec3(x, y, z) => {
                print!("Vec3({}, {}, {})", x, y, z);
            }
            frost::HeapObject::Vec4(x, y, z, w) => {
                print!("Vec4({}, {}, {}, {})", x, y, z, w);
            }
            frost::HeapObject::Quat(x, y, z, w) => {
                print!("Quat({}, {}, {}, {})", x, y, z, w);
            }
            frost::HeapObject::Mat4(m) => {
                print!("Mat4({:?})", m);
            }
        },
    }
}
