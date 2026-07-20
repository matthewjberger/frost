use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::Parser;
use frost::{
    Compiler, Lexer, Parser as FrostParser, VirtualMachine, build_module,
    check_ownership, compile_ir_to_object, emit_c,
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let source = fs::read_to_string(&cli.file)
        .with_context(|| format!("Failed to read file: {}", cli.file))?;

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().context("Lexer error")?;
    let positions = lexer.positions().to_vec();

    let mut parser = FrostParser::with_positions(&tokens, &positions);
    let statements = parser.parse().context("Parser error")?;

    let linear_types = parser.linear_types().clone();
    check_ownership(&statements, &linear_types).context("Ownership error")?;

    if cli.emit_c {
        let module = build_module(&statements).context("IR lowering error")?;
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
