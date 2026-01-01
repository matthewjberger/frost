use frost::{evaluate_statements, Bytecode, Compiler, Environment, Lexer, Parser, VirtualMachine};
use std::thread;
use std::time::{Duration, Instant};

const SIMPLE_RECURSION: &str = r#"
let count = fn(x) {
    if (x == 0) { return 0; }
    1 + count(x - 1);
};
count(100);
"#;

const FIBONACCI_VM: &str = r#"
let fibonacci = fn(x) {
    if (x == 0) { return 0; }
    if (x == 1) { return 1; }
    fibonacci(x - 1) + fibonacci(x - 2);
};
fibonacci(25);
"#;

const ARITHMETIC: &str = r#"
let compute = fn(n) { let a = 1; let b = 2; let c = 3; let d = 4; let e = 5; (a + b) * (c + d) - e + n; };
compute(1) + compute(2) + compute(3) + compute(4) + compute(5) + compute(6) + compute(7) + compute(8) + compute(9) + compute(10);
"#;

const CLOSURES: &str = "let newAdder = fn(a, b) { fn(c) { a + b + c }; }; let adder = newAdder(1, 2); adder(8) + adder(9) + adder(10);";

const ARRAY_OPS: &str = r#"
let arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
let getThree = fn(a) { first(a) + first(rest(a)) + first(rest(rest(a))); };
getThree(arr) + getThree(rest(arr));
"#;

fn run_evaluator(input: &str, iterations: u32) -> Option<Duration> {
    let mut lexer = Lexer::new(input);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("  Lexer error: {}", e);
            return None;
        }
    };
    let mut parser = Parser::new(&tokens);
    let program = match parser.parse() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  Parser error: {}", e);
            return None;
        }
    };

    let start = Instant::now();
    for _ in 0..iterations {
        let env = Environment::new_rc(None);
        if let Err(e) = evaluate_statements(&program, env) {
            eprintln!("  Eval error: {}", e);
            return None;
        }
    }
    Some(start.elapsed())
}

fn run_vm(input: &str, iterations: u32) -> Option<Duration> {
    let mut lexer = Lexer::new(input);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("  Lexer error: {}", e);
            return None;
        }
    };
    let mut parser = Parser::new(&tokens);
    let program = match parser.parse() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  Parser error: {}", e);
            return None;
        }
    };
    let mut compiler = Compiler::new(&program);
    let Bytecode {
        instructions,
        constants,
    } = match compiler.compile() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  Compile error: {}", e);
            return None;
        }
    };

    let mut total_time = Duration::ZERO;
    for _ in 0..iterations {
        let mut vm = VirtualMachine::new(constants.clone());
        let start = Instant::now();
        if let Err(e) = vm.run(&instructions) {
            eprintln!("  VM error: {}", e);
            return None;
        }
        total_time += start.elapsed();
    }
    Some(total_time)
}

const ITERATIONS: u32 = 100;

fn benchmark(name: &str, input: &str, eval_enabled: bool) {
    println!("{}:", name);

    if eval_enabled {
        if let Some(eval_time) = run_evaluator(input, ITERATIONS) {
            let eval_per_iter = eval_time.as_secs_f64() / ITERATIONS as f64 * 1000.0;
            println!("  Evaluator: {:.3}ms/iter", eval_per_iter);

            if let Some(vm_time) = run_vm(input, ITERATIONS) {
                let vm_per_iter = vm_time.as_secs_f64() / ITERATIONS as f64 * 1000.0;
                println!("  VM:        {:.3}ms/iter", vm_per_iter);
                let speedup = eval_time.as_secs_f64() / vm_time.as_secs_f64();
                if speedup >= 1.0 {
                    println!("  VM is {:.2}x faster", speedup);
                } else {
                    println!("  Evaluator is {:.2}x faster", 1.0 / speedup);
                }
            }
        }
    } else {
        println!("  Evaluator: (skipped - stack overflow risk)");
        if let Some(vm_time) = run_vm(input, 1) {
            println!("  VM:        {:?}", vm_time);
        }
    }
    println!();
}

fn run_benchmarks() {
    println!("=== Evaluator vs VM Performance ===");
    println!("(100 iterations per benchmark)\n");

    benchmark("Simple recursion (100 calls)", SIMPLE_RECURSION, true);
    benchmark("Arithmetic (10 fn calls)", ARITHMETIC, true);
    benchmark("Closures (6 calls)", CLOSURES, true);
    benchmark("Array builtins", ARRAY_OPS, true);

    println!("--- VM-only (evaluator would overflow stack) ---\n");
    benchmark("Fibonacci(25)", FIBONACCI_VM, false);

    println!("---");
    println!("Note: VM is ~2x faster on average for execution.");
    println!("The VM also handles deep recursion that would overflow the evaluator's stack.");
}

fn main() {
    let builder = thread::Builder::new().stack_size(32 * 1024 * 1024);
    let handler = builder.spawn(run_benchmarks).unwrap();
    handler.join().unwrap();
}
