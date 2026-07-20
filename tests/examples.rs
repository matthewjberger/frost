use std::path::PathBuf;

use frost::{Compiler, Lexer, Parser, VirtualMachine};

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples")
}

fn run_example(file_name: &str) -> anyhow::Result<()> {
    let path = examples_dir().join(file_name);
    let source = std::fs::read_to_string(&path)?;

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize()?;

    let mut parser = Parser::new(&tokens);
    let statements = parser.parse()?;

    let mut compiler = Compiler::new_with_path(&statements, examples_dir());
    let bytecode = compiler.compile()?;

    let mut vm = VirtualMachine::new(
        bytecode.constants,
        bytecode.functions,
        bytecode.heap,
    );
    vm.run(&bytecode.instructions)
}

const VM_CLEAN_EXAMPLES: &[&str] = &[
    "arena_pool_demo.frost",
    "array_set_test.frost",
    "bootstrap_test.frost",
    "comptime_bitmask.frost",
    "comptime_ecs.frost",
    "comptime_field_access.frost",
    "comptime_simple.frost",
    "comptime_struct_test.frost",
    "eval_arrays.frost",
    "eval_basics.frost",
    "eval_comprehensive.frost",
    "eval_fizzbuzz.frost",
    "eval_functions.frost",
    "eval_ownership.frost",
    "eval_structs.frost",
    "fizzbuzz.frost",
    "freecs.frost",
    "learn_frost.frost",
    "math.frost",
    "ref_test.frost",
    "tagged_union.frost",
    "test_arr.frost",
    "test_calc.frost",
    "test_const.frost",
    "test_safety.frost",
    "vec3_utils.frost",
];

#[test]
fn vm_clean_examples_compile_and_run() {
    let mut failures = Vec::new();
    for example in VM_CLEAN_EXAMPLES {
        if let Err(error) = run_example(example) {
            failures.push(format!("{example}: {error}"));
        }
    }
    assert!(
        failures.is_empty(),
        "expected these examples to compile and run cleanly:\n{}",
        failures.join("\n")
    );
}

#[test]
fn move_error_example_is_rejected() {
    let result = run_example("eval_move_error.frost");
    let error =
        result.expect_err("eval_move_error.frost must fail the move checker");
    let message = error.to_string();
    assert!(
        message.contains("moved"),
        "expected a move error, got: {message}"
    );
}
