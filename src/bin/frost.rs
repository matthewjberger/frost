use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use clap::Parser;
use frost::{compile_to_object, Compiler, Lexer, Parser as FrostParser, VirtualMachine};

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
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let source = fs::read_to_string(&cli.file)
        .with_context(|| format!("Failed to read file: {}", cli.file))?;

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().context("Lexer error")?;

    let mut parser = FrostParser::new(&tokens);
    let statements = parser.parse().context("Parser error")?;

    if cli.native || cli.link {
        let object_bytes = compile_to_object(&statements).context("Native compilation error")?;

        let input_path = Path::new(&cli.file);
        let stem = input_path.file_stem().unwrap_or_default().to_string_lossy();

        let object_path = if cli.link {
            format!("{}.o", stem)
        } else {
            cli.output.clone().unwrap_or_else(|| format!("{}.o", stem))
        };

        fs::write(&object_path, object_bytes)
            .with_context(|| format!("Failed to write object file: {}", object_path))?;

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

fn link_executable(object_path: &str, exe_path: &str, extra_libs: &[String]) -> Result<()> {
    let linker = find_linker().ok_or_else(|| {
        anyhow::anyhow!("No suitable linker found. Please install gcc, clang, or MSVC.")
    })?;

    let mut cmd = Command::new(linker);

    if linker == "cl" {
        cmd.arg(object_path);
        cmd.arg(format!("/Fe:{}", exe_path));
        for lib in extra_libs {
            cmd.arg(lib);
        }
    } else {
        cmd.arg(object_path);
        cmd.arg("-o");
        cmd.arg(exe_path);
        for lib in extra_libs {
            cmd.arg(lib);
        }
    }

    let output = cmd.output().context("Failed to run linker")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Linker failed: {}", stderr);
    }

    Ok(())
}
