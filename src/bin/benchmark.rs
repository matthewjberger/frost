use frost::{Compiler, Lexer, Parser, VirtualMachine};
use std::time::{Duration, Instant};

const SIMPLE_RECURSION: &str = r#"
count := fn(x) {
    if (x == 0) { return 0; }
    1 + count(x - 1);
};
count(100);
"#;

const FIBONACCI: &str = r#"
fibonacci := fn(x) {
    if (x == 0) { return 0; }
    if (x == 1) { return 1; }
    fibonacci(x - 1) + fibonacci(x - 2);
};
fibonacci(25);
"#;

const ARITHMETIC: &str = r#"
compute := fn(n) { a := 1; b := 2; c := 3; d := 4; e := 5; (a + b) * (c + d) - e + n; };
compute(1) + compute(2) + compute(3) + compute(4) + compute(5) + compute(6) + compute(7) + compute(8) + compute(9) + compute(10);
"#;

const CLOSURES: &str = "newAdder := fn(a, b) { fn(c) { a + b + c }; }; adder := newAdder(1, 2); adder(8) + adder(9) + adder(10);";

const ARRAY_OPS: &str = r#"
arr := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
getThree := fn(a) { first(a) + first(rest(a)) + first(rest(rest(a))); };
getThree(arr) + getThree(rest(arr));
"#;

const FOR_LOOP: &str = r#"
mut sum := 0;
for i in 0..1000 {
    sum = sum + i;
}
sum
"#;

fn compile_program(input: &str) -> Option<frost::Bytecode> {
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
    match compiler.compile() {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("  Compile error: {}", e);
            None
        }
    }
}

fn run_vm(bytecode: &frost::Bytecode, iterations: u32) -> Option<Duration> {
    let mut total_time = Duration::ZERO;
    for _ in 0..iterations {
        let mut vm = VirtualMachine::new(
            bytecode.constants.clone(),
            bytecode.functions.clone(),
            bytecode.heap.clone(),
        );
        let start = Instant::now();
        if let Err(e) = vm.run(&bytecode.instructions) {
            eprintln!("  VM error: {}", e);
            return None;
        }
        total_time += start.elapsed();
    }
    Some(total_time)
}

const ITERATIONS: u32 = 100;

fn benchmark(name: &str, input: &str) {
    println!("{}:", name);

    let bytecode = match compile_program(input) {
        Some(b) => b,
        None => return,
    };

    if let Some(vm_time) = run_vm(&bytecode, ITERATIONS) {
        let vm_per_iter = vm_time.as_secs_f64() / ITERATIONS as f64 * 1000.0;
        println!(
            "  Time: {:.3}ms/iter ({} iterations)",
            vm_per_iter, ITERATIONS
        );
    }
    println!();
}

fn main() {
    println!("=== Frost VM Performance Benchmarks ===\n");

    benchmark("Simple recursion (100 calls)", SIMPLE_RECURSION);
    benchmark("Fibonacci(25)", FIBONACCI);
    benchmark("Arithmetic (10 fn calls)", ARITHMETIC);
    benchmark("Closures (6 calls)", CLOSURES);
    benchmark("Array builtins", ARRAY_OPS);
    benchmark("For loop (1000 iterations)", FOR_LOOP);

    println!("---");
    println!("All benchmarks use the compiled bytecode VM.");
}
