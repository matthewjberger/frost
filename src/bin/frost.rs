use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::Parser;
use frost::{
    BuildCache, Expression, Lexer, Literal, Parameter, Parser as FrostParser,
    Position, ReturnKind, ReturnSignature, RunOutcome, Spanned, Statement,
    Type, build_module, build_module_per_module, check_callback_declarations,
    check_frame_escapes, check_linearity, check_module, check_ownership,
    check_regions, compile_ir_to_object, emit_c, lower_allocation_sources,
    lower_failure_sets, lower_param_modes, register_entry_file,
    resolve_imports_cached, run_module,
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

    #[arg(
        long,
        help = "Link with no libc: a minimal freestanding runtime and a custom entry point"
    )]
    freestanding: bool,

    #[arg(
        long,
        help = "Reuse a module's cached object unless its source or an imported interface changed"
    )]
    incremental: bool,

    #[arg(
        long,
        default_value = ".frost-build",
        help = "Where --incremental keeps interfaces and objects"
    )]
    build_dir: String,
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
        mode: frost::ParamMode::Read,
        compile_time_signature: None,
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
            ReturnSignature::plain(ReturnKind::Single(Type::I64)),
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

    let base_dir = Path::new(&cli.file)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    // The entry file is a file like any other, so a diagnostic from it should
    // name one rather than a bare line number.
    let entry = register_entry_file(Path::new(&cli.file), &base_dir);
    let positions: Vec<Position> = lexer
        .positions()
        .iter()
        .map(|position| Position {
            file: entry,
            ..*position
        })
        .collect();

    let mut parser = FrostParser::with_positions(&tokens, &positions);
    let parsed = parser.parse().context("Parser error")?;
    // A module's object is only its own on the link path, so that is the only
    // place a cached one can be linked instead of built. `--test` needs every
    // module's `test` blocks, which a module answered for from the cache is
    // never read far enough to have.
    let cache = if cli.incremental {
        if !cli.link {
            bail!(
                "--incremental needs --link, since a module is a compilation unit only when the objects are linked"
            );
        }
        if cli.test || cli.emit_c || cli.run_ir {
            bail!(
                "--incremental applies to native linking, not --test, --emit-c or --run-ir"
            );
        }
        Some(BuildCache::open(Path::new(&cli.build_dir))?)
    } else {
        None
    };

    let resolved = resolve_imports_cached(
        parsed,
        &base_dir,
        parser.linear_types().clone(),
        parser.tests().to_vec(),
        cache.as_ref(),
    )
    .context("Import error")?;
    let mut statements = resolved.statements;
    let linear_types = resolved.linear_types;
    let tests = resolved.tests;
    let mut modules = resolved.modules;
    check_callback_declarations(&statements).context("Callback error")?;
    check_regions(&statements).context("Region error")?;
    check_frame_escapes(&statements).context("Region error")?;
    lower_allocation_sources(&mut statements)
        .context("Allocation source error")?;
    lower_failure_sets(&mut statements).context("Failure set error")?;
    lower_param_modes(&mut statements);
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
        let module = build_module(&augmented, &linear_types)
            .context("IR lowering error")?;
        check_module(&module).context("IR type error")?;
        check_linearity(&module).context("Linearity error")?;
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
            &[object_path.to_string_lossy().into_owned()],
            &exe_path.to_string_lossy(),
            &cli.libs,
            false,
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
        let module = build_module(&statements, &linear_types)
            .context("IR lowering error")?;
        check_module(&module).context("IR type error")?;
        check_linearity(&module).context("Linearity error")?;
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
        let module = build_module(&statements, &linear_types)
            .context("IR lowering error")?;
        check_module(&module).context("IR type error")?;
        check_linearity(&module).context("Linearity error")?;
        let c_source = emit_c(&module).context("C emission error")?;

        let input_path = Path::new(&cli.file);
        let stem = input_path.file_stem().unwrap_or_default().to_string_lossy();

        if cli.link {
            // The C is an intermediate on the way to the executable, so it
            // belongs in the temp directory rather than the working one, where
            // it would collide with a concurrent build and outlive a failure.
            let c_path = std::env::temp_dir()
                .join(format!("{stem}_{}.c", std::process::id()))
                .to_string_lossy()
                .into_owned();
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
        // Linking is where a module can be its own compilation unit, so it is
        // also where a specialization is emitted once per module that asked for
        // it rather than once per program.
        let module = if cli.link {
            build_module_per_module(&statements, &linear_types)
        } else {
            build_module(&statements, &linear_types)
        }
        .context("IR lowering error")?;
        check_module(&module).context("IR type error")?;
        check_linearity(&module).context("Linearity error")?;
        let input_path = Path::new(&cli.file);
        let stem = input_path.file_stem().unwrap_or_default().to_string_lossy();

        // Linking is where a module can be its own compilation unit, so the IR
        // is split per module and each part becomes its own object. `--native`
        // without `--link` still writes the one object file its `-o` names,
        // since that output is a single file by contract.
        let parts = if cli.link {
            module.split_by_module()
        } else {
            vec![module]
        };
        let mut object_paths = Vec::with_capacity(parts.len());
        // Objects for cached modules are named for the fingerprint that
        // produced them and outlive the build; everything else is an
        // intermediate that goes away with it.
        let mut temporary: Vec<String> = Vec::new();
        for (index, part) in parts.iter().enumerate() {
            let file = part
                .functions
                .first()
                .map(|function| function.module)
                .unwrap_or_default();
            let planned = modules
                .iter()
                .position(|plan| plan.file == file && !plan.reused);
            if let Some(planned) = planned {
                let plan = &modules[planned];
                let object_bytes = compile_ir_to_object(part)
                    .context("Native compilation error")?;
                fs::write(&plan.object, object_bytes).with_context(|| {
                    format!(
                        "Failed to write object file: {}",
                        plan.object.display()
                    )
                })?;
                if let Some(cache) = &cache {
                    cache.discard_other_objects(&plan.tag, &plan.object);
                }
                object_paths.push(plan.object.to_string_lossy().into_owned());
                modules[planned].record.emits_object = true;
                continue;
            }
            if modules.iter().any(|plan| plan.file == file && plan.reused) {
                continue;
            }
            let object_bytes = compile_ir_to_object(part)
                .context("Native compilation error")?;
            let object_path = if cli.link {
                format!("{}.{}.o", stem, index)
            } else {
                cli.output.clone().unwrap_or_else(|| format!("{}.o", stem))
            };
            fs::write(&object_path, object_bytes).with_context(|| {
                format!("Failed to write object file: {}", object_path)
            })?;
            object_paths.push(object_path.clone());
            temporary.push(object_path);
        }

        // A module the cache answered for is linked whether or not this build
        // produced anything for it, since its object holds code no other part
        // does.
        for plan in &modules {
            if !plan.reused {
                continue;
            }
            println!("Reused {}", plan.module);
            if plan.record.emits_object {
                object_paths.push(plan.object.to_string_lossy().into_owned());
            }
        }

        if cli.link {
            let exe_path = cli.output.clone().unwrap_or_else(|| {
                if cfg!(windows) {
                    format!("{}.exe", stem)
                } else {
                    stem.to_string()
                }
            });

            link_executable(
                &object_paths,
                &exe_path,
                &cli.libs,
                cli.freestanding,
            )?;

            for object_path in &temporary {
                fs::remove_file(object_path).ok();
            }

            // Written after the link, so a record never claims an object a
            // failed build did not finish producing.
            if let Some(cache) = &cache {
                for plan in &modules {
                    if !plan.reused {
                        cache.store(&plan.tag, &plan.record)?;
                    }
                }
            }

            println!("Linked executable: {}", exe_path);
        } else {
            for object_path in &object_paths {
                println!("Compiled to: {}", object_path);
            }
        }
    } else {
        let module = build_module(&statements, &linear_types)
            .context("IR lowering error")?;
        check_module(&module).context("IR type error")?;
        check_linearity(&module).context("Linearity error")?;
        let object_bytes = compile_ir_to_object(&module)
            .context("Native compilation error")?;
        let stem = Path::new(&cli.file)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let directory = std::env::temp_dir();
        let object_path = directory.join(format!("frost_run_{stem}.o"));
        let exe_path = directory
            .join(format!("frost_run_{stem}{}", std::env::consts::EXE_SUFFIX));
        fs::write(&object_path, object_bytes)?;
        link_executable(
            &[object_path.to_string_lossy().into_owned()],
            &exe_path.to_string_lossy(),
            &cli.libs,
            false,
        )?;
        fs::remove_file(&object_path).ok();
        let status = Command::new(&exe_path)
            .status()
            .context("Failed to run executable")?;
        fs::remove_file(&exe_path).ok();
        if !status.success() {
            std::process::exit(status.code().unwrap_or(1));
        }
    }

    Ok(())
}

const RUNTIME_SOURCE: &str = include_str!("../../runtime/frost_runtime.c");
const FREESTANDING_SOURCE: &str =
    include_str!("../../runtime/frost_freestanding.c");

fn write_runtime_source_named(source: &str, name: &str) -> Result<PathBuf> {
    let path =
        std::env::temp_dir().join(format!("{name}_{}.c", std::process::id()));
    fs::write(&path, source).with_context(|| {
        format!("Failed to write runtime: {}", path.display())
    })?;
    Ok(path)
}

// The runtime is a fixed piece of C that does not vary with the program being
// compiled, so recompiling it on every build is wasted work. Build it once into
// an object cached in the temp directory, keyed by a hash of the source and the
// tool that built it, and link that object thereafter. On the native backend
// this takes the C compiler out of the per-build path entirely.
fn runtime_object(tool: &str, freestanding: bool) -> Result<PathBuf> {
    let source = if freestanding {
        FREESTANDING_SOURCE
    } else {
        RUNTIME_SOURCE
    };
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    tool.hash(&mut hasher);
    freestanding.hash(&mut hasher);
    let key = hasher.finish();

    let extension = if tool == "cl" { "obj" } else { "o" };
    let directory = std::env::temp_dir();
    let cached =
        directory.join(format!("frost_runtime_{key:016x}.{extension}"));
    if cached.exists() {
        return Ok(cached);
    }

    let name = if freestanding {
        "frost_freestanding"
    } else {
        "frost_runtime"
    };
    let source_path = write_runtime_source_named(source, name)?;
    let pending = directory.join(format!(
        "frost_runtime_{key:016x}_{}.{extension}",
        std::process::id()
    ));

    let mut command = Command::new(tool);
    if tool == "cl" {
        command.arg("/c");
        command.arg(&source_path);
        command.arg(format!("/Fo:{}", pending.display()));
    } else {
        command.arg("-std=c11");
        command.arg("-c");
        command.arg(&source_path);
        command.arg("-o");
        command.arg(&pending);
    }
    let output = command
        .output()
        .context("Failed to compile the Frost runtime")?;
    fs::remove_file(&source_path).ok();
    if !output.status.success() {
        bail!(
            "Frost runtime failed to compile: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // A rename is atomic, so builds racing to fill the cache all end up with the
    // same object rather than a half-written one.
    if fs::rename(&pending, &cached).is_err() {
        fs::copy(&pending, &cached).with_context(|| {
            format!("Failed to cache runtime: {}", cached.display())
        })?;
        fs::remove_file(&pending).ok();
    }
    Ok(cached)
}

fn compile_c(
    c_path: &str,
    exe_path: &str,
    extra_libs: &[String],
) -> Result<()> {
    let compiler = find_linker().ok_or_else(|| {
        anyhow::anyhow!("No C compiler found. Please install gcc or clang.")
    })?;

    let runtime_path = runtime_object(compiler, false)?;

    // The C is an intermediate, so it is compiled the way an intermediate
    // should be. Without this the C path ran unoptimized while the Cranelift
    // path asked for speed, which made the two backends answer the same thing
    // at very different cost and made the C path a poor measurement.
    let mut cmd = Command::new(compiler);
    if compiler == "cl" {
        cmd.arg("/O2");
        cmd.arg(c_path);
        cmd.arg(&runtime_path);
        cmd.arg(format!("/Fe:{}", exe_path));
        for lib in extra_libs {
            cmd.arg(lib);
        }
    } else {
        cmd.arg("-std=c11");
        cmd.arg("-O2");
        cmd.arg(c_path);
        cmd.arg(&runtime_path);
        cmd.arg("-o");
        cmd.arg(exe_path);
        for lib in extra_libs {
            cmd.arg(lib);
        }
    }

    let output = cmd.output().context("Failed to run C compiler")?;
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

#[cfg(target_os = "windows")]
fn add_freestanding_link_args(cmd: &mut Command) {
    // Windows: exit through kernel32, entry mainCRTStartup. No C runtime.
    cmd.arg("-lkernel32");
    cmd.arg("-e").arg("mainCRTStartup");
}

#[cfg(target_os = "linux")]
fn add_freestanding_link_args(cmd: &mut Command) {
    // Linux: the runtime's _start is the entry, exit is a raw syscall. No libc.
    cmd.arg("-e").arg("_start");
}

#[cfg(target_os = "macos")]
fn add_freestanding_link_args(cmd: &mut Command) {
    // macOS: entry _start, exit via syscall, but macOS always routes syscalls
    // through libSystem, so link that one library and nothing else.
    cmd.arg("-e").arg("_start");
    cmd.arg("-lSystem");
}

#[cfg(not(any(
    target_os = "windows",
    target_os = "linux",
    target_os = "macos"
)))]
fn add_freestanding_link_args(_cmd: &mut Command) {}

fn link_executable(
    object_paths: &[String],
    exe_path: &str,
    extra_libs: &[String],
    freestanding: bool,
) -> Result<()> {
    let linker = find_linker().ok_or_else(|| {
        anyhow::anyhow!(
            "No suitable linker found. Please install gcc, clang, or MSVC."
        )
    })?;

    if freestanding && linker == "cl" {
        bail!("--freestanding is supported with gcc or clang, not MSVC");
    }

    let runtime_path = runtime_object(linker, freestanding)?;

    let mut cmd = Command::new(linker);

    if linker == "cl" {
        cmd.args(object_paths);
        cmd.arg(&runtime_path);
        cmd.arg(format!("/Fe:{}", exe_path));
        for lib in extra_libs {
            cmd.arg(lib);
        }
    } else {
        if freestanding {
            cmd.arg("-nostdlib");
        }
        cmd.args(object_paths);
        cmd.arg(&runtime_path);
        cmd.arg("-o");
        cmd.arg(exe_path);
        for lib in extra_libs {
            cmd.arg(lib);
        }
        if freestanding {
            // The freestanding runtime supplies the platform's entry point; the
            // linker just needs the matching entry symbol and, where the OS
            // requires it, the one library that exposes process exit. This is the
            // per-target floor, the same shape Rust's targets use.
            add_freestanding_link_args(&mut cmd);
        }
    }

    let output = cmd.output().context("Failed to run linker")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Linker failed: {}", stderr);
    }

    Ok(())
}
