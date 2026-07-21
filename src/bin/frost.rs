use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::Parser;
use frost::{
    Compiler, Expression, Lexer, Literal, Parameter, Parser as FrostParser,
    Position, ReturnSignature, RunOutcome, Spanned, Statement, Type,
    VirtualMachine, build_module, check_module, check_ownership,
    compile_ir_to_object, emit_c, resolve_imports, run_module,
};

#[derive(Parser)]
#[command(name = "frost")]
#[command(about = "The Frost programming language")]
struct Cli {
    file: String,

    #[arg(short, long, help = "Compile to native object file")]
    native: bool,

    #[arg(short, long, help = "Output file path for native compilation")]
    output: Option<String>,

    #[arg(short, long, help = "Link into executable (implies --native)")]
    link: bool,

    #[arg(long, help = "Additional object files or libraries to link")]
    libs: Vec<String>,

    #[arg(long, help = "Emit C source instead of using the Cranelift backend")]
    emit_c: bool,

    #[arg(
        long,
        help = "Interpret the typed IR directly (reference oracle for scalar programs)"
    )]
    run_ir: bool,

    #[arg(long, help = "Compile and run the file's `test` blocks")]
    test: bool,
}

fn test_harness(tests: &[(String, String)]) -> Vec<Spanned<Statement>> {
    let spanned = |statement| Spanned::new(statement, Position::default());
    let call = |name: &str, arguments: Vec<Expression>| {
        Expression::Call(
            Box::new(Expression::Identifier(name.to_string())),
            arguments,
        )
    };
    let call_stmt = |name: &str, arguments: Vec<Expression>| {
        spanned(Statement::Expression(call(name, arguments)))
    };
    let param = |name: &str, ty: Type| Parameter {
        name: name.to_string(),
        type_annotation: Some(ty),
        mutable: false,
    };
    let external = |name: &str, params: Vec<Parameter>| {
        spanned(Statement::Extern {
            name: name.to_string(),
            params,
            return_type: None,
        })
    };

    let mut items = vec![
        external(
            "frost_test_start",
            vec![param("name", Type::Ptr(Box::new(Type::I8)))],
        ),
        external("frost_test_ok", Vec::new()),
        external("frost_assert", vec![param("cond", Type::Bool)]),
    ];

    let mut body = Vec::new();
    for (test_name, function_name) in tests {
        body.push(call_stmt(
            "frost_test_start",
            vec![Expression::Literal(Literal::String(test_name.clone()))],
        ));
        body.push(call_stmt(function_name, Vec::new()));
        body.push(call_stmt("frost_test_ok", Vec::new()));
    }
    body.push(spanned(Statement::Expression(Expression::Literal(
        Literal::Integer(0),
    ))));

    items.push(spanned(Statement::Constant(
        "main".to_string(),
        Expression::Function(
            Vec::new(),
            ReturnSignature::Single(Type::I64),
            body,
        ),
    )));
    items
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let source = fs::read_to_string(&cli.file)
        .with_context(|| format!("Failed to read file: {}", cli.file))?;

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().context("Lexer error")?;
    let positions = lexer.positions().to_vec();

    let mut parser = FrostParser::with_positions(&tokens, &positions);
    let parsed = parser.parse().context("Parser error")?;

    let base_dir = Path::new(&cli.file)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let resolved = resolve_imports(
        parsed,
        &base_dir,
        parser.linear_types().clone(),
        parser.tests().to_vec(),
    )
    .context("Import error")?;
    let statements = resolved.statements;
    let linear_types = resolved.linear_types;
    let tests = resolved.tests;
    check_ownership(&statements, &linear_types).context("Ownership error")?;

    if cli.test {
        if tests.is_empty() {
            println!("no tests found in {}", cli.file);
            return Ok(());
        }
        let mut augmented = statements.clone();
        augmented.extend(test_harness(&tests));
        check_ownership(&augmented, &linear_types)
            .context("Ownership error")?;
        let module = build_module(&augmented).context("IR lowering error")?;
        check_module(&module).context("IR type error")?;
        let object_bytes = compile_ir_to_object(&module)
            .context("Native compilation error")?;

        let stem = Path::new(&cli.file)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let directory = std::env::temp_dir();
        let object_path = directory.join(format!("frost_test_{stem}.o"));
        let exe_path = directory
            .join(format!("frost_test_{stem}{}", std::env::consts::EXE_SUFFIX));
        fs::write(&object_path, object_bytes)?;
        link_executable(
            &object_path.to_string_lossy(),
            &exe_path.to_string_lossy(),
            &cli.libs,
        )?;
        fs::remove_file(&object_path).ok();

        println!("running {} test(s)", tests.len());
        let status = Command::new(&exe_path)
            .status()
            .context("Failed to run test executable")?;
        fs::remove_file(&exe_path).ok();
        if status.success() {
            println!("all tests passed");
            return Ok(());
        }
        std::process::exit(1);
    }

    if cli.run_ir {
        let module = build_module(&statements).context("IR lowering error")?;
        check_module(&module).context("IR type error")?;
        match run_module(&module) {
            RunOutcome::Output(output) => {
                print!("{output}");
                return Ok(());
            }
            RunOutcome::Unsupported(reason) => {
                eprintln!("frost: ir interpreter declined: {reason}");
                std::process::exit(3);
            }
        }
    }

    if cli.emit_c {
        let module = build_module(&statements).context("IR lowering error")?;
        check_module(&module).context("IR type error")?;
        let c_source = emit_c(&module).context("C emission error")?;

        let input_path = Path::new(&cli.file);
        let stem = input_path.file_stem().unwrap_or_default().to_string_lossy();

        if cli.link {
            let c_path = format!("{}.c", stem);
            fs::write(&c_path, c_source).with_context(|| {
                format!("Failed to write C file: {}", c_path)
            })?;
            let exe_path = cli.output.clone().unwrap_or_else(|| {
                if cfg!(windows) {
                    format!("{}.exe", stem)
                } else {
                    stem.to_string()
                }
            });
            compile_c(&c_path, &exe_path, &cli.libs)?;
            fs::remove_file(&c_path).ok();
            println!("Linked executable: {}", exe_path);
        } else {
            let c_path =
                cli.output.clone().unwrap_or_else(|| format!("{}.c", stem));
            fs::write(&c_path, c_source).with_context(|| {
                format!("Failed to write C file: {}", c_path)
            })?;
            println!("Emitted C: {}", c_path);
        }
        return Ok(());
    }

    if cli.native || cli.link {
        let module = build_module(&statements).context("IR lowering error")?;
        check_module(&module).context("IR type error")?;
        let object_bytes = compile_ir_to_object(&module)
            .context("Native compilation error")?;

        let input_path = Path::new(&cli.file);
        let stem = input_path.file_stem().unwrap_or_default().to_string_lossy();

        let object_path = if cli.link {
            format!("{}.o", stem)
        } else {
            cli.output.clone().unwrap_or_else(|| format!("{}.o", stem))
        };

        fs::write(&object_path, object_bytes).with_context(|| {
            format!("Failed to write object file: {}", object_path)
        })?;

        if cli.link {
            let exe_path = cli.output.clone().unwrap_or_else(|| {
                if cfg!(windows) {
                    format!("{}.exe", stem)
                } else {
                    stem.to_string()
                }
            });

            link_executable(&object_path, &exe_path, &cli.libs)?;

            fs::remove_file(&object_path).ok();

            println!("Linked executable: {}", exe_path);
        } else {
            println!("Compiled to: {}", object_path);
        }
    } else {
        let base_path = PathBuf::from(&cli.file)
            .canonicalize()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()));
        let mut compiler = if let Some(base) = base_path {
            Compiler::new_with_path(&statements, base)
        } else {
            Compiler::new(&statements)
        };
        let bytecode = compiler.compile().context("Compiler error")?;

        let mut vm = VirtualMachine::new(
            bytecode.constants,
            bytecode.functions,
            bytecode.heap,
        );
        vm.run(&bytecode.instructions).context("Runtime error")?;
    }

    Ok(())
}

const RUNTIME_SOURCE: &str = include_str!("../../runtime/frost_runtime.c");

fn write_runtime_source() -> Result<PathBuf> {
    let path = std::env::temp_dir()
        .join(format!("frost_runtime_{}.c", std::process::id()));
    fs::write(&path, RUNTIME_SOURCE).with_context(|| {
        format!("Failed to write runtime: {}", path.display())
    })?;
    Ok(path)
}

fn compile_c(
    c_path: &str,
    exe_path: &str,
    extra_libs: &[String],
) -> Result<()> {
    let compiler = find_linker().ok_or_else(|| {
        anyhow::anyhow!("No C compiler found. Please install gcc or clang.")
    })?;

    let runtime_path = write_runtime_source()?;

    let mut cmd = Command::new(compiler);
    if compiler == "cl" {
        cmd.arg(c_path);
        cmd.arg(&runtime_path);
        cmd.arg(format!("/Fe:{}", exe_path));
        for lib in extra_libs {
            cmd.arg(lib);
        }
    } else {
        cmd.arg("-std=c11");
        cmd.arg(c_path);
        cmd.arg(&runtime_path);
        cmd.arg("-o");
        cmd.arg(exe_path);
        for lib in extra_libs {
            cmd.arg(lib);
        }
    }

    let output = cmd.output().context("Failed to run C compiler")?;
    fs::remove_file(&runtime_path).ok();
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("C compiler failed: {}", stderr);
    }
    Ok(())
}

fn find_linker() -> Option<&'static str> {
    let linkers = if cfg!(windows) {
        vec!["gcc", "clang", "cl"]
    } else {
        vec!["cc", "gcc", "clang"]
    };

    for linker in linkers {
        if Command::new(linker).arg("--version").output().is_ok() {
            return Some(match linker {
                "gcc" => "gcc",
                "clang" => "clang",
                "cc" => "cc",
                "cl" => "cl",
                _ => linker,
            });
        }
    }
    None
}

fn link_executable(
    object_path: &str,
    exe_path: &str,
    extra_libs: &[String],
) -> Result<()> {
    let linker = find_linker().ok_or_else(|| {
        anyhow::anyhow!(
            "No suitable linker found. Please install gcc, clang, or MSVC."
        )
    })?;

    let runtime_path = write_runtime_source()?;

    let mut cmd = Command::new(linker);

    if linker == "cl" {
        cmd.arg(object_path);
        cmd.arg(&runtime_path);
        cmd.arg(format!("/Fe:{}", exe_path));
        for lib in extra_libs {
            cmd.arg(lib);
        }
    } else {
        cmd.arg(object_path);
        cmd.arg(&runtime_path);
        cmd.arg("-o");
        cmd.arg(exe_path);
        for lib in extra_libs {
            cmd.arg(lib);
        }
    }

    let output = cmd.output().context("Failed to run linker")?;
    fs::remove_file(&runtime_path).ok();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Linker failed: {}", stderr);
    }

    Ok(())
}
