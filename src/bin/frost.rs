use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use frost::{
    Ast, BuildCache, ExprId, Expression as AstExpression, Layer, Lexer,
    Literal as AstLiteral, Manifest, Module, Parameter as AstParameter,
    Parser as FrostParser, Position, Range32, Resolution, ReturnKind,
    ReturnSignature, RunOutcome, SearchRoot, Statement, TEST_PREFIX, TokenSpan,
    Type, compile_ir_to_object, emit_c, lower_allocation_sources,
    lower_failure_sets, lower_multiple_returns, lower_param_modes,
    register_entry_file, resolve_distinct_types, resolve_imports_cached,
    run_module, strip_unsafe_fns,
};

// The words that are read off the arguments before clap sees them, listed here
// because clap cannot know about them and a reader who cannot see one in `--help`
// concludes it does not exist.
const SUBCOMMANDS: &str = "Words read before the options, each taking files or directories:
  run <file> [args] compile the file, run it with those arguments, and exit on what it returned
  fmt <path>...     write each file the way the formatter renders it, `-` for stdin
  lint <path>...    findings a build does not refuse on: `--diagnostics=json` for one object a line
  fix <file>        apply the fixes the diagnostics offered
  api <prefix>      the exported surface under a name prefix, with signatures
  generate [--check]  write the files frost.json says a program of this project
                    writes, or check that none of them is stale";

#[derive(Parser)]
#[command(name = "frost")]
#[command(about = "The Frost programming language")]
#[command(after_help = SUBCOMMANDS)]
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
        help = "Fail the build when an `unsafe` block vouches for nothing. Every build warns about one; this is what holds a tree to zero of them"
    )]
    audit_unsafe: bool,

    #[arg(
        long,
        default_value = ".frost-build",
        help = "Where --incremental keeps interfaces and objects"
    )]
    build_dir: String,

    #[arg(
        short = 'L',
        long = "lib-path",
        value_name = "DIR",
        help = "Directory to search for imports, after the importing file's own. Repeatable"
    )]
    lib_path: Vec<String>,

    #[arg(
        long,
        value_name = "FORMAT",
        help = "How diagnostics are written: `caret` for a reader, `json` for one object per report per line"
    )]
    diagnostics: Option<String>,
}

/// Whether this run writes its reports as JSON.
///
/// Read from the arguments directly rather than from the parsed command line,
/// because the top of the program has to know before `clap` has run: a file that
/// cannot be read is a report too, and the caller asked for JSON for all of
/// them.
fn wants_json() -> bool {
    let mut arguments = std::env::args();
    while let Some(argument) = arguments.next() {
        if argument == "--diagnostics" {
            return arguments.next().as_deref() == Some("json");
        }
        if let Some(format) = argument.strip_prefix("--diagnostics=") {
            return format == "json";
        }
    }
    false
}

// Where an import may be found, most specific first.
//
// The importing file's own directory is always tried before any of these and is
// not in the list. See `find_import`. After it: directories named on the command
// line, then `FROST_PATH`, then whatever the project's manifest declares, then
// the standard library. Command line beats environment beats project file, which
// is the order of how deliberately each was said.
fn search_roots(
    cli: &Cli,
    project_root: &Path,
) -> Result<(Vec<SearchRoot>, Vec<Layer>)> {
    let mut roots = Vec::new();
    let mut layers = Vec::new();
    for directory in &cli.lib_path {
        roots.push(SearchRoot::project(PathBuf::from(directory)));
    }
    for directory in frost::path_from_environment() {
        roots.push(SearchRoot::project(directory));
    }
    // The nearest manifest at or above the entry file. A build's entry is any
    // file in the project, so what the project declares is found by walking up
    // rather than by being repeated in every directory a build might start in.
    if let Some((manifest, directory)) = Manifest::find_upward(project_root)? {
        for search in manifest.search_paths(&directory) {
            roots.push(SearchRoot::project(search));
        }
        for (name, path) in
            manifest.layers.iter().zip(manifest.layer_paths(&directory))
        {
            layers.push(Layer::new(name, &path));
        }
    }
    if let Some(standard) = frost::bundled_std() {
        roots.push(SearchRoot::named("std", standard));
    }
    Ok((roots, layers))
}

/// Whether the program already declares this runtime entry point itself, which
/// the compiler's own source does. Declaring it twice is a redefinition.
fn declares_extern(program: &Module, wanted: &str) -> bool {
    program.roots.iter().any(|statement| {
        matches!(
            program.ast.stmt(*statement),
            Statement::Extern { name, .. }
                if program.ast.name(*name) == wanted
        )
    })
}

fn harness_parameter(ast: &mut Ast, name: &str, ty: Type) -> AstParameter {
    let name = ast.intern(name);
    AstParameter {
        // Made for the test harness, which nobody wrote.
        at: frost::Position::default(),
        name,
        type_annotation: Some(ty),
        mutable: false,
        mode: frost::ParamMode::Read,
        compile_time_signature: None,
        compile_time_default: None,
        pack: false,
        format: false,
        capability: false,
    }
}

fn push_extern(
    program: &mut Module,
    name: &str,
    params: Vec<AstParameter>,
    return_type: Option<Type>,
) {
    let name = program.ast.intern(name);
    let params = program.ast.add_parameters(params);
    let id = program.ast.push_stmt(
        Statement::Extern {
            name,
            params,
            return_type,
            // The harness's own runtime entry points, generated here and
            // audited by construction, so the generated body needs no
            // `unsafe` block around a call the compiler wrote itself.
            safe: true,
        },
        TokenSpan::NONE,
    );
    program.roots.push(id);
}

/// What `assert` lowers to. Generated here and audited by construction, so a
/// call the compiler wrote itself needs no `unsafe` block around it.
fn push_assert_declaration(program: &mut Module) {
    let cond = harness_parameter(&mut program.ast, "cond", Type::Bool);
    let place = harness_parameter(
        &mut program.ast,
        "where",
        Type::Ptr(Box::new(Type::I8)),
    );
    push_extern(program, "frost_rt_assert_at", vec![cond, place], None);
}

// The runner takes each test body as a function pointer rather than the
// harness calling it directly, because a failing assertion has to escape
// back into the runner without ending the run, and the setjmp that makes
// that possible has to own the call. See runtime/frost_runtime.c.
fn push_test_harness(program: &mut Module, tests: &[(String, String)]) {
    let name = harness_parameter(
        &mut program.ast,
        "name",
        Type::Ptr(Box::new(Type::I8)),
    );
    let body_parameter = harness_parameter(
        &mut program.ast,
        "body",
        Type::Proc(Vec::new(), Box::new(Type::Void)),
    );
    push_extern(
        program,
        "frost_rt_test_run",
        vec![name, body_parameter],
        None,
    );
    push_extern(
        program,
        "frost_rt_test_summary",
        Vec::new(),
        Some(Type::I64),
    );
    let cond = harness_parameter(&mut program.ast, "cond", Type::Bool);
    let place = harness_parameter(
        &mut program.ast,
        "where",
        Type::Ptr(Box::new(Type::I8)),
    );
    push_extern(program, "frost_rt_assert_at", vec![cond, place], None);

    let ast = &mut program.ast;
    let call = |ast: &mut Ast, name: &str, arguments: Vec<ExprId>| {
        let callee = ast.intern(name);
        let callee =
            ast.push_expr(AstExpression::Identifier(callee), TokenSpan::NONE);
        let arguments = ast.add_expr_list(&arguments);
        ast.push_expr(AstExpression::Call(callee, arguments), TokenSpan::NONE)
    };

    let mut body = Vec::new();
    for (test_name, function_name) in tests {
        let test_name = ast.push_expr(
            AstExpression::Literal(AstLiteral::String(test_name.clone())),
            TokenSpan::NONE,
        );
        let function = ast.intern(function_name);
        let function =
            ast.push_expr(AstExpression::Identifier(function), TokenSpan::NONE);
        let run = call(ast, "frost_rt_test_run", vec![test_name, function]);
        body.push(ast.push_stmt(Statement::Expression(run), TokenSpan::NONE));
    }
    // The summary is the last expression, so its failure count is what `main`
    // answers with and what the process exits on.
    let summary = call(ast, "frost_rt_test_summary", Vec::new());
    body.push(ast.push_stmt(Statement::Expression(summary), TokenSpan::NONE));

    let body = ast.add_stmt_list(&body);
    let signature = ast
        .push_signature(ReturnSignature::plain(ReturnKind::Single(Type::I64)));
    let main_function = ast.push_expr(
        AstExpression::Function(Range32::EMPTY, signature, body),
        TokenSpan::NONE,
    );
    let main_name = ast.intern("main");
    let main_id = ast.push_stmt(
        Statement::Constant(main_name, main_function),
        TokenSpan::NONE,
    );
    program.roots.push(main_id);
}

// Every `.frost` file under a directory, in a stable order.
fn frost_files(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(next) = stack.pop() {
        for entry in fs::read_dir(&next)
            .with_context(|| format!("reading {}", next.display()))?
        {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|kind| kind == "frost") {
                found.push(path);
            }
        }
    }
    found.sort();
    Ok(found)
}

// `--test` on a directory runs every file under it, each in its own process.
// Separate processes rather than one program, because a test that manages to
// crash outright takes down only its own file, and because two files are two
// programs that may define the same names.
fn test_directory(directory: &Path, arguments: &[String]) -> Result<()> {
    let files = frost_files(directory)?;
    if files.is_empty() {
        println!("no .frost files under {}", directory.display());
        return Ok(());
    }
    let executable = std::env::current_exe()
        .context("finding the compiler to run tests with")?;
    let mut failed = 0;
    for file in &files {
        println!("== {}", file.display());
        let status = Command::new(&executable)
            .arg("--test")
            .args(arguments)
            .arg(file)
            .status()
            .with_context(|| format!("running tests in {}", file.display()))?;
        if !status.success() {
            failed += 1;
        }
    }
    println!(
        "
{} of {} file(s) failed",
        failed,
        files.len()
    );
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// Why a compile was refused, as the reports themselves rather than as text.
///
/// The reports reach the top of the program this way so that what is printed is
/// chosen there: the same faults are a caret report for a reader and a line of
/// JSON each for a program. Displaying it gives the caret report's text, which
/// is what an error printed any other way already gave.
#[derive(Debug)]
struct Refused(Vec<frost::Diagnostic>);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lines: Vec<String> =
            self.0.iter().map(frost::Diagnostic::rendered).collect();
        write!(f, "{}", lines.join("\n"))
    }
}

impl std::error::Error for Refused {}

/// The quote a report puts around a name.
const QUOTE: char = 0x27 as char;

/// A name that is not there, answered with the nearest one that is.
///
/// The suggestion rides on the report as an edit, so a reader applying edits
/// gets it and a reader reading them sees what was meant. Only an unambiguous
/// nearest name is offered.
fn suggest_names(
    program: &Module,
    roots: &[frost::StmtId],
    collected: &mut [frost::Diagnostic],
) {
    // An imported name is carried under a private tag, so the names compared
    // against are the ones a reader writes.
    let mut held: Vec<String> = Vec::new();
    for statement in roots {
        match program.ast.stmt(*statement) {
            Statement::Constant(name, _)
            | Statement::Struct(name, ..)
            | Statement::Enum(name, ..)
            | Statement::Flags(name, ..)
            | Statement::TypeAlias(name, _)
            | Statement::Extern { name, .. }
            | Statement::Declared { name, .. } => {
                held.push(frost::demangle_private_names(
                    program.ast.name(*name),
                ));
            }
            _ => {}
        }
    }
    let declares = declared_files(program, roots);
    let known: Vec<&str> = held.iter().map(String::as_str).collect();
    for report in collected.iter_mut() {
        let Some(wanted) = names_a_missing_name(&report.message) else {
            continue;
        };
        // A name the program declares and this file cannot reach. Saying it is
        // not there is false, and the reader who believes it writes the
        // declaration a second time instead of adding one word to an export
        // line. Answered before the nearest name, since the nearest name to a
        // name that exists is itself.
        if let Some(file) = declares.get(&wanted) {
            report.message = format!(
                "{} (declared in {file}; add it to the export line there)",
                report.message
            );
            continue;
        }
        let Some(near) = frost::nearest(&wanted, &known) else {
            continue;
        };
        report.message = format!("{} (did you mean '{near}'?)", report.message);
    }
}

/// The file each declared name is written in, by the name a reader writes.
///
/// A module's names are mangled as they are spliced, exported ones as well as
/// private ones, so the tag says nothing about whether a name is offered. What
/// says it is the report: a name the program declares, that a file was still
/// refused for naming, is one that file cannot reach.
fn declared_files(
    program: &Module,
    roots: &[frost::StmtId],
) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for statement in roots {
        let name = match program.ast.stmt(*statement) {
            Statement::Constant(name, _)
            | Statement::Struct(name, ..)
            | Statement::Enum(name, ..)
            | Statement::Flags(name, ..)
            | Statement::TypeAlias(name, _)
            | Statement::Extern { name, .. }
            | Statement::Declared { name, .. } => *name,
            _ => continue,
        };
        let plain = frost::demangle_private_names(program.ast.name(name));
        // By file name. What a module is called otherwise depends on where the
        // build was started from, and this sentence is compared word for word.
        let position = program.ast.stmt_position(*statement);
        let Some(file) = position.file_name() else {
            continue;
        };
        out.entry(plain).or_insert(file);
    }
    out
}

/// The name a report says is not there, out of the reports that say so.
fn names_a_missing_name(message: &str) -> Option<String> {
    for opening in [
        "unknown variable '",
        "call to undefined function '",
        "unknown function '",
    ] {
        if let Some(rest) = message.split_once(opening)
            && let Some((name, _)) = rest.1.split_once(QUOTE)
        {
            return Some(name.to_string());
        }
    }
    if let Some(rest) = message.split_once("' is not a type this program")
        && let Some(at) = rest.0.rfind(QUOTE)
    {
        return Some(rest.0[at + 1..].to_string());
    }
    None
}

/// Everything the run collected, as the one error a refused compile prints.
fn refuse(collected: &[frost::Diagnostic]) -> Result<()> {
    if collected.is_empty() {
        return Ok(());
    }
    Err(anyhow!(Refused(frost::grouped_diagnostics(
        collected.to_vec()
    ))))
}

/// A rewrite's fault, printed beside every fault the checks had already found.
///
/// The chain's innermost message is the one with something to say, which is
/// also the one `render_diagnostic` prints, so joining that with the collected
/// reports gives the same text either way.
fn beside<T>(collected: &[frost::Diagnostic], outcome: Result<T>) -> Result<T> {
    let fault = match outcome {
        Ok(held) => return Ok(held),
        Err(fault) => fault,
    };
    let mut held = collected.to_vec();
    // A rewrite that knows where it is says so, and the report keeps that
    // position rather than rendering the location into the message and then
    // reporting it as coming from nowhere. Without this a fault from one of
    // these passes named no file, so a reader was told what was wrong and left
    // to find it.
    if let Some(located) = fault
        .chain()
        .find_map(|error| error.downcast_ref::<frost::LocatedError>())
    {
        held.push(frost::Diagnostic::new(
            located.position,
            located.message.clone(),
        ));
        return Err(anyhow!(Refused(frost::grouped_diagnostics(held))));
    }
    // A rewrite's fault has no report of its own, so it becomes one: the
    // innermost message is the one with something to say, and it says where it
    // is itself when it knows.
    if let Some(innermost) = fault.chain().last() {
        // A message a pass joined from several faults is several faults, so
        // each line becomes a report of its own rather than one report holding
        // a paragraph.
        for line in innermost.to_string().lines() {
            held.push(frost::Diagnostic::new(
                frost::Position::default(),
                line.to_string(),
            ));
        }
    }
    Err(anyhow!(Refused(frost::grouped_diagnostics(held))))
}

/// Lower to IR and check the IR, beside the faults the walks over the syntax
/// found.
///
/// A function that failed to lower contributes no IR, so the two IR checks run
/// only when lowering was clean: asked about a module with a hole where a
/// function should be, they report the hole in every function that calls it
/// rather than what is wrong. Whatever they do find joins the collected
/// reports, since a region escape in one function and a type error in another
/// are two faults and one run should name both.
fn lowered_and_checked(
    program: &mut Module,
    linear_types: &std::collections::HashSet<String>,
    per_module: bool,
    collected: &[frost::Diagnostic],
    idle: &[frost::Diagnostic],
) -> Result<frost::IrModule> {
    // A type nothing declares stops the run here. Lowering asks every type for
    // a width and a layout, so what it says about one that is not there is the
    // same fault told a second way: `sizeof(Widget)` reported the missing name
    // and then reported that `sizeof` has no layout for it.
    if collected
        .iter()
        .any(|held| held.message.contains("is not a type this program declares"))
    {
        let mut faults = collected.to_vec();
        suggest_names(program, &program.roots.clone(), &mut faults);
        refuse(&faults)?;
    }
    let lowered = beside(
        collected,
        frost::build_module_recovering(
            &mut program.ast,
            &program.roots,
            linear_types,
            per_module,
        ),
    )?;
    let module = lowered.module;
    let lowering = lowered.failures;
    let mut faults = collected.to_vec();
    faults.extend(lowering.iter().cloned());

    // A type nothing declares leaves every local written with it carrying a
    // type the program does not have, so the IR check reports the value
    // assigned into one rather than the name that is not there. The same reason
    // the checks wait on a clean lowering: what they say is the consequence.
    let types_are_known = !collected.iter().any(|held| {
        held.message.contains("is not a type this program declares")
    });
    if lowering.is_empty() && types_are_known {
        faults.extend(frost::check_module_recovering(&module));
        // The linearity check and the ownership rules say the same kind of
        // thing about the same values, so a function both have something to say
        // about is one mistake described twice: 'held' consumed more than once,
        // and 'held' used after it was moved. The ownership rules say it at the
        // use, which is the line to change.
        if lowered.ownership.is_empty() {
            faults.extend(frost::check_linearity_recovering(
                &module,
                &frost::pooled_instance_names(
                    &program.ast,
                    &program.roots,
                    linear_types,
                ),
            ));
        }
    }
    // What the ownership rules said, once the checks that read the module have
    // spoken. A name whose value came from a call to a function nothing
    // declares is moved out of a call that is not there, so the missing name is
    // the fault to report and the move is what follows from it. The same reason
    // an undeclared type holds those checks back.
    let calls_are_known = !faults
        .iter()
        .any(|held| held.message.contains("call to undefined function"));
    if calls_are_known {
        faults.extend(frost::check_ownership_recovering(
            &program.ast,
            &program.roots,
            linear_types,
        ));
        faults.extend(lowered.ownership);
    }
    suggest_names(program, &program.roots.clone(), &mut faults);
    refuse(&faults)?;
    // A build that is refused says what it refused and nothing else, so this is
    // past the last of them. A warning is a report too, and a caller reading
    // JSON gets it as one rather than as a line in the middle of the stream.
    // A warning is a report, so it comes out where the file puts it rather than
    // where the pass that found it ran.
    let warnings = frost::in_source_order(idle.to_vec());
    if wants_json() {
        eprint!("{}", frost::diagnostics_as_json(&warnings, "warning"));
    } else {
        eprint!("{}", frost::render_warnings(&warnings));
    }
    Ok(module)
}

/// `frost fmt <paths...>`: write the one rendering of every file named.
///
/// A directory is every `.frost` file under it. `-` is standard input, whose
/// rendering goes to standard output, which is what an editor calls. `--check`
/// writes nothing and names the files that are not already formatted, so a build
/// can refuse on it.
fn format_paths(arguments: &[String]) -> Result<bool> {
    let check = arguments.iter().any(|held| held == "--check");
    let named: Vec<&String> = arguments
        .iter()
        .filter(|held| !held.starts_with("--"))
        .collect();
    if named.is_empty() {
        bail!("frost fmt: which files?");
    }
    if named.len() == 1 && named[0] == "-" {
        let mut source = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut source)
            .context("reading standard input")?;
        let formatted = frost::format_source(&source);
        if check {
            return Ok(formatted == source);
        }
        print!("{formatted}");
        return Ok(true);
    }

    let mut files = Vec::new();
    for path in named {
        let path = Path::new(path);
        if path.is_dir() {
            files.extend(frost_files(path)?);
        } else {
            files.push(path.to_path_buf());
        }
    }
    let mut clean = true;
    for file in &files {
        let source = fs::read_to_string(file)
            .with_context(|| format!("reading {}", file.display()))?;
        let formatted = frost::format_source(&source);
        if formatted == source {
            continue;
        }
        clean = false;
        if check {
            println!("{}", file.display());
            continue;
        }
        fs::write(file, &formatted)
            .with_context(|| format!("writing {}", file.display()))?;
        println!("formatted {}", file.display());
    }
    // Writing the files is the work, so doing it is success. Only `--check`
    // answers with whether they were already what it writes.
    Ok(clean || !check)
}

/// `frost api <prefix> [paths...]`: the exported surface a prefix names.
///
/// A flat namespace has no `.` to narrow a guess with, and a prefix is what a
/// family is named by here, so this is the narrowing asked for directly.
fn print_api(arguments: &[String]) -> Result<bool> {
    let json = arguments.iter().any(|held| held == "--diagnostics=json")
        || arguments.iter().any(|held| held == "--json");
    let named: Vec<&String> = arguments
        .iter()
        .filter(|held| !held.starts_with("--"))
        .collect();
    let Some(prefix) = named.first() else {
        bail!("frost api: which prefix?");
    };
    let mut files = Vec::new();
    if named.len() > 1 {
        for path in &named[1..] {
            let path = Path::new(path);
            if path.is_dir() {
                files.extend(frost::sources(path));
            } else {
                files.push(path.to_path_buf());
            }
        }
    } else {
        files = frost::sources(Path::new("."));
    }
    let found = frost::exported(&files, prefix);
    if json {
        for held in &found {
            println!(
                "{}",
                serde_json::to_string(held).unwrap_or_else(|_| String::new())
            );
        }
    } else {
        for held in &found {
            println!("{}:{}", held.file, held.line);
            println!("{}", held.signature);
            println!();
        }
        println!("{} exported name(s) beginning with '{prefix}'", found.len());
    }
    Ok(true)
}

/// `frost lint <paths...>`: report what is worth looking at, and refuse nothing.
///
/// A finding is advisory: a build never fails on one, and `frost <file>` says
/// exactly what it said before. Exits nonzero when there are findings, so a
/// build that wants to hold a tree to none of them can.
fn lint_paths(arguments: &[String]) -> Result<bool> {
    let named: Vec<&String> = arguments
        .iter()
        .filter(|held| !held.starts_with("--"))
        .collect();
    if named.is_empty() {
        bail!("frost lint: which files?");
    }
    let mut files = Vec::new();
    for path in named {
        let path = Path::new(path);
        if path.is_dir() {
            files.extend(frost_files(path)?);
        } else {
            files.push(path.to_path_buf());
        }
    }
    let mut clean = true;
    for file in &files {
        let source = fs::read_to_string(file)
            .with_context(|| format!("reading {}", file.display()))?;
        let base = file.parent().map(Path::to_path_buf).unwrap_or_default();
        let entry = register_entry_file(file);
        let mut lexer = Lexer::new(&source);
        let Ok(tokens) = lexer.tokenize() else {
            continue;
        };
        let positions: Vec<Position> = lexer
            .positions()
            .iter()
            .map(|position| Position {
                file: entry,
                ..*position
            })
            .collect();
        let mut parser = FrostParser::with_positions(&tokens, &positions);
        let (parsed, faulted) = parser.parse_recovering();
        // A file that does not parse is a file the compiler has something to
        // say about, and saying it twice helps nobody.
        if !faulted.is_empty() {
            continue;
        }
        let exports: Vec<String> = parser.exports().to_vec();
        // Over the program the file becomes once its imports are resolved,
        // which is what a build reads. Asked about the file alone, the unsafety
        // walk cannot see the declaration that says a called extern is
        // unchecked, and reports every block that wraps one.
        let mut roots = Vec::new();
        if let Some(standard) = frost::bundled_std() {
            roots.push(SearchRoot::named("std", standard));
        }
        let project = base.canonicalize().unwrap_or_else(|_| base.clone());
        let mut layers = Vec::new();
        if let Ok(Some((manifest, directory))) = Manifest::find_upward(&project)
        {
            for search in manifest.search_paths(&directory) {
                roots.push(SearchRoot::project(search));
            }
            for (name, path) in
                manifest.layers.iter().zip(manifest.layer_paths(&directory))
            {
                layers.push(Layer::new(name, &path));
            }
        }
        let Ok(resolved) = resolve_imports_cached(
            parsed,
            &base,
            parser.linear_types().clone(),
            parser.tests().to_vec(),
            Resolution {
                cache: None,
                roots: &roots,
                layers: &layers,
            },
        ) else {
            continue;
        };
        let whole = resolved.program;
        // The prefix this file's directory declares, if the project declares
        // one for it.
        let mut wanted: Option<String> = None;
        if let Ok(Some((manifest, directory))) = Manifest::find_upward(&project)
        {
            let here = file.canonicalize().unwrap_or_else(|_| file.clone());
            for (under, prefix) in &manifest.prefixes {
                let root = directory.join(under);
                let root = root.canonicalize().unwrap_or(root);
                if here.starts_with(&root) {
                    wanted = Some(prefix.clone());
                }
            }
        }
        let found: Vec<frost::Diagnostic> = frost::lint(
            &whole.ast,
            &whole.roots,
            &exports,
            &tokens,
            wanted.as_deref(),
        )
        .into_iter()
        .filter(|held| held.position.file == entry)
        .collect();
        if found.is_empty() {
            continue;
        }
        clean = false;
        if wants_json() {
            print!("{}", frost::diagnostics_as_json(&found, "warning"));
        } else {
            for held in &found {
                print!(
                    "{}",
                    frost::render_diagnostic(&anyhow!(held.rendered()))
                );
            }
        }
    }
    Ok(clean)
}

/// `frost fix <file>`: apply every edit the reports carry that can be applied
/// unread.
///
/// The reports are read back out of `--diagnostics=json` rather than taken from
/// a call into the checks, because that channel is what this is here to make
/// usable. An edit that cannot be read out of it and applied is a hole in the
/// channel, and going through it is what shows the hole.
///
/// Fixing what one round finds can uncover the next: a file whose parse was
/// refused has never been checked, so its faults are not knowable until the
/// parse is fixed. Rounds continue while each one applies something, and a
/// bound stops a rule that would undo its own edit from running forever.
fn apply_fixes(file: &str) -> Result<()> {
    const ROUNDS: usize = 8;
    let mut applied = 0usize;
    for _ in 0..ROUNDS {
        let reports = reports_for(file)?;
        let mut edits: Vec<frost::Replacement> = reports
            .into_iter()
            .filter_map(|report| report.fix)
            .filter(|fix| fix.certain)
            .collect();
        if edits.is_empty() {
            break;
        }
        // Highest offset first, so applying one leaves the offsets of the ones
        // not yet applied standing.
        edits.sort_by_key(|fix| std::cmp::Reverse(fix.span.0));
        let mut text = fs::read_to_string(file)
            .with_context(|| format!("reading {file}"))?;
        let mut written = 0usize;
        let mut last = usize::MAX;
        for fix in &edits {
            // Two edits over the same bytes are one edit twice, and the second
            // would be applied to text the first has already replaced.
            if fix.span.1 > last {
                continue;
            }
            if fix.span.1 > text.len() || fix.span.0 > fix.span.1 {
                continue;
            }
            text.replace_range(fix.span.0..fix.span.1, &fix.replacement);
            last = fix.span.0;
            written += 1;
        }
        if written == 0 {
            break;
        }
        fs::write(file, &text).with_context(|| format!("writing {file}"))?;
        applied += written;
    }
    match applied {
        0 => println!("frost fix: nothing to apply in {file}"),
        1 => println!("frost fix: applied 1 edit to {file}"),
        many => println!("frost fix: applied {many} edits to {file}"),
    }
    Ok(())
}

/// `frost generate [--check]`: write every file the project's manifest says a
/// program of its own writes.
///
/// The generator is compiled and run by this compiler, so a checkout holding two
/// compilers regenerates with whichever one was asked. Its arguments are the
/// output path first and the declared inputs after, which is the order a program
/// written before any manifest existed already took.
///
/// A step always writes, and `--check` writes somewhere else and compares the
/// bytes. Staleness is decided from content because a checkout stamps every file
/// with the time it was made, so a timestamp says nothing about whether the
/// generator would write something different.
fn run_generators(arguments: &[String]) -> Result<bool> {
    let checking = arguments.iter().any(|held| held == "--check");
    if let Some(unknown) = arguments
        .iter()
        .find(|held| held.starts_with('-') && *held != "--check")
    {
        // Said here rather than raised, so it reads the same as every other
        // line this command writes and as what the self-hosted compiler says.
        eprintln!("frost generate: unknown option '{unknown}'");
        return Ok(false);
    }
    let here = std::env::current_dir().context("finding the project")?;
    let Some((manifest, root)) = Manifest::find_upward(&here)? else {
        eprintln!("frost generate: no frost.json declares this project");
        return Ok(true);
    };
    // Every member is read before any of them runs. A project holding a half
    // declared step beside a good one writes nothing, where running them in
    // turn would leave a file on disk that the refusal never mentioned.
    if manifest
        .generated
        .iter()
        .any(|step| step.output.is_empty() || step.from.is_empty())
    {
        eprintln!(
            "frost generate: a member of 'generated' in frost.json does \
             not name both an output and what writes it"
        );
        return Ok(false);
    }
    if manifest.generated.is_empty() {
        eprintln!("frost generate: this project declares nothing to generate");
        return Ok(true);
    }
    let compiler =
        std::env::current_exe().context("finding the compiler to run")?;
    let mut settled = true;
    for step in &manifest.generated {
        let (output, from, inputs) = step.resolved(&root);
        let written = if checking {
            scratch_for(&output)
        } else {
            output.clone()
        };
        let mut running = Command::new(&compiler);
        running.arg("run").arg(&from).arg(&written);
        running.args(&inputs);
        let status = running
            .status()
            .with_context(|| format!("running {}", from.display()))?;
        if !status.success() {
            // What a check wrote is a temporary file, and nothing reads it
            // once the step it belongs to has stopped.
            if checking {
                fs::remove_file(&written).ok();
            }
            // What went wrong was said by whatever ran, on the stream this
            // shares, so this names the step rather than repeating it.
            eprintln!(
                "frost generate: {} did not write {}",
                step.from, step.output
            );
            return Ok(false);
        }
        // A generator lays its output out to be read while it assembles it, and
        // the tree holds every Frost file to one rendering. Formatting here is
        // what makes those two agree, so `--check` compares what a build leaves
        // on disk rather than the form on the way there.
        if written.extension().is_some_and(|held| held == "frost") {
            let rendered = Command::new(&compiler)
                .arg("fmt")
                .arg(&written)
                .status()
                .with_context(|| format!("formatting {}", written.display()))?;
            if !rendered.success() {
                if checking {
                    fs::remove_file(&written).ok();
                }
                eprintln!(
                    "frost generate: the formatter refused {}",
                    step.output
                );
                return Ok(false);
            }
        }
        if !checking {
            eprintln!("frost generate: wrote {}", step.output);
            continue;
        }
        let fresh = fs::read(&written)
            .with_context(|| format!("reading {}", written.display()))?;
        let current = fs::read(&output).unwrap_or_default();
        fs::remove_file(&written).ok();
        if fresh == current {
            eprintln!("frost generate: {} is up to date", step.output);
            continue;
        }
        settled = false;
        eprintln!(
            "frost generate: {} is not what {} writes. Run `frost generate`",
            step.output, step.from
        );
    }
    Ok(settled)
}

/// Where a `--check` writes instead of over the declared output.
///
/// In the temporary directory, so a check that dies partway leaves nothing in
/// the tree, and named after the output's own file name so the extension
/// survives and the formatter reads it as the kind of file it is. The number in
/// front is the whole output path, so two projects generating a file of the same
/// name do not check each other's.
fn scratch_for(output: &Path) -> PathBuf {
    let mut hasher = DefaultHasher::new();
    output.hash(&mut hasher);
    let name = output
        .file_name()
        .map(|held| held.to_string_lossy().into_owned())
        .unwrap_or_else(|| "generated".to_string());
    std::env::temp_dir()
        .join(format!("frost_generate_{:016x}_{name}", hasher.finish()))
}

/// What this compiler says about a file, as reports.
fn reports_for(file: &str) -> Result<Vec<frost::Report>> {
    let executable =
        std::env::current_exe().context("finding the compiler to ask")?;
    let object = std::env::temp_dir()
        .join(format!("frost_fix_{}.o", std::process::id()));
    let asked = Command::new(&executable)
        .arg("--diagnostics=json")
        .arg("--native")
        .arg("-o")
        .arg(&object)
        .arg(file)
        .output()
        .with_context(|| format!("asking about {file}"))?;
    fs::remove_file(&object).ok();
    let mut reports = Vec::new();
    for line in String::from_utf8_lossy(&asked.stderr).lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // A line that is not a report is this compiler saying something in its
        // own voice, which is worth passing on rather than swallowing.
        match serde_json::from_str::<frost::Report>(line) {
            Ok(report) => reports.push(report),
            Err(_) => eprintln!("{line}"),
        }
    }
    Ok(reports)
}

/// How much stack the compiler runs on.
///
/// Every pass over a program recurses over its syntax, so how deep the compiler
/// goes follows how deeply the program nests, and the compiler's own source is
/// the deepest program it is asked about. Windows gives a main thread one
/// megabyte, which a debug build of these passes spends before reaching the
/// bottom of the self-hosted compiler: it dies with a stack overflow while a
/// release build of the same code, with smaller frames, finishes.
///
/// That difference is the reason this is not left to the default. A limit that
/// only a debug build reaches turns every test that compiles a large program
/// into a failure that does not reproduce under `--release`.
const COMPILER_STACK_BYTES: usize = 256 * 1024 * 1024;

fn main() -> std::process::ExitCode {
    // `fix` is read off the arguments rather than declared as a subcommand,
    // because the compiler takes a file as its first argument and has since
    // before it took anything else. A file really named `fix` is still
    // compilable by writing its extension, which every Frost file has.
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.first().is_some_and(|held| held == "fmt") {
        return match format_paths(&arguments[1..]) {
            Ok(true) => std::process::ExitCode::SUCCESS,
            Ok(false) => std::process::ExitCode::FAILURE,
            Err(error) => {
                eprint!("{}", frost::render_diagnostic(&error));
                std::process::ExitCode::FAILURE
            }
        };
    }
    if arguments.first().is_some_and(|held| held == "api") {
        return match print_api(&arguments[1..]) {
            Ok(_) => std::process::ExitCode::SUCCESS,
            Err(error) => {
                eprint!("{}", frost::render_diagnostic(&error));
                std::process::ExitCode::FAILURE
            }
        };
    }
    if arguments.first().is_some_and(|held| held == "lint") {
        return match lint_paths(&arguments[1..]) {
            Ok(true) => std::process::ExitCode::SUCCESS,
            Ok(false) => std::process::ExitCode::FAILURE,
            Err(error) => {
                eprint!("{}", frost::render_diagnostic(&error));
                std::process::ExitCode::FAILURE
            }
        };
    }
    if arguments.first().is_some_and(|held| held == "generate") {
        return match run_generators(&arguments[1..]) {
            Ok(true) => std::process::ExitCode::SUCCESS,
            Ok(false) => std::process::ExitCode::FAILURE,
            Err(error) => {
                eprint!("{}", frost::render_diagnostic(&error));
                std::process::ExitCode::FAILURE
            }
        };
    }
    if arguments.first().is_some_and(|held| held == "fix") {
        let Some(file) = arguments.get(1) else {
            eprintln!("frost fix: which file?");
            return std::process::ExitCode::FAILURE;
        };
        return match apply_fixes(file) {
            Ok(()) => std::process::ExitCode::SUCCESS,
            Err(error) => {
                eprint!("{}", frost::render_diagnostic(&error));
                std::process::ExitCode::FAILURE
            }
        };
    }

    // `frost run <file> [args...]`: the file is the only thing the compiler
    // reads off the line, and everything after it is the program's own. The
    // split is what puts a program's `--check` in front of the program rather
    // than in front of the compiler.
    let mut parsed: Vec<String> = std::env::args().collect();
    let mut forwarded: Vec<String> = Vec::new();
    if arguments.first().is_some_and(|held| held == "run") {
        let Some(file) = arguments.get(1) else {
            eprintln!("frost run: which file?");
            return std::process::ExitCode::FAILURE;
        };
        parsed = vec![parsed[0].clone(), file.clone()];
        forwarded = arguments[2..].to_vec();
    }

    let outcome = std::thread::Builder::new()
        .stack_size(COMPILER_STACK_BYTES)
        .spawn(move || compile(parsed, forwarded))
        .map_err(|error| {
            anyhow!("failed to start the compiler thread: {error}")
        })
        .and_then(|handle| {
            handle
                .join()
                .map_err(|_| anyhow!("the compiler panicked"))?
        });
    match outcome {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            // Rendered rather than printed as a chain: a message that knows
            // where it is gets the line it is about and a caret under the
            // column, which is what the self-hosted compiler has always done
            // and there is no reason for two formats.
            //
            // A caller that asked for JSON gets the reports themselves, which
            // is why they travel to here as reports. A fault raised as text
            // rather than as a report still becomes one, so the stream holds
            // every fault whatever raised it.
            if wants_json() {
                // A fault raised as text becomes a report per line of the
                // innermost message, which is the one with something to say and
                // the one the caret report prints.
                let refused = match error.downcast_ref::<Refused>() {
                    Some(refused) => refused.0.clone(),
                    None => error
                        .chain()
                        .last()
                        .map(|innermost| innermost.to_string())
                        .unwrap_or_default()
                        .lines()
                        .map(|line| {
                            frost::Diagnostic::new(
                                frost::Position::default(),
                                line.to_string(),
                            )
                        })
                        .collect(),
                };
                eprint!("{}", frost::diagnostics_as_json(&refused, "error"));
            } else {
                eprint!("{}", frost::render_diagnostic(&error));
            }
            std::process::ExitCode::FAILURE
        }
    }
}

fn compile(parsed: Vec<String>, forwarded: Vec<String>) -> Result<()> {
    let cli = Cli::parse_from(parsed);

    // A directory is a suite rather than a program, so it never reaches the
    // rest of this.
    let entry = Path::new(&cli.file);
    if cli.test && entry.is_dir() {
        let mut forwarded = Vec::new();
        for directory in &cli.lib_path {
            forwarded.push("-L".to_string());
            forwarded.push(directory.clone());
        }
        for library in &cli.libs {
            forwarded.push("--libs".to_string());
            forwarded.push(library.clone());
        }
        return test_directory(entry, &forwarded);
    }

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
    let entry = register_entry_file(Path::new(&cli.file));
    let positions: Vec<Position> = lexer
        .positions()
        .iter()
        .map(|position| Position {
            file: entry,
            ..*position
        })
        .collect();
    let (tokens, positions) =
        frost::expand_includes(tokens, positions, &base_dir)
            .with_context(|| format!("in {}", cli.file))?;
    let (tokens, positions, lifted) = frost::resolve_when(tokens, positions)
        .with_context(|| format!("in {}", cli.file))?;

    // Where the search for the project's manifest starts, which is the entry
    // file's directory as an absolute path. A file named with no directory at
    // all sits in the one the build was started in, and the empty path that
    // leaves behind canonicalizes to nothing, so the search would start nowhere
    // and a build run from inside a project would not find it.
    let named = if base_dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        base_dir.as_path()
    };
    let project_root =
        named.canonicalize().unwrap_or_else(|_| base_dir.clone());
    let (roots, layers) = search_roots(&cli, &project_root)?;

    let mut parser = FrostParser::with_positions(&tokens, &positions);
    parser.also_lifted_lines(lifted);
    // The runtime is the one file that may define names in the runtime's own
    // name space, and it is the one this compiler resolved as the runtime.
    if is_the_runtime(&cli.file) {
        parser.compiling_the_runtime();
    }
    // A generic type this file imports may be named in a literal here, so which
    // names can start one is settled from the files it imports as well as from
    // this one.
    let imported = frost::imported_generic_types(
        &source,
        &base_dir,
        &roots,
        &layers,
        &project_root,
    );
    parser.also_generic(imported.generic_types);
    parser.also_const_functions(imported.const_functions);
    parser.preload_diagnostics(lexer.diagnostics_in_file(entry));
    // The reports themselves rather than the text they render as, so a caller
    // reading JSON gets the parse's faults as reports and the edits they carry.
    // A parse fault is where an edit is most often the whole answer, and it is
    // also where the program stops: recovery skipped the statement it was in,
    // so a name that statement declared reads as undeclared everywhere after,
    // and what the checks would say about it is about a program the reader did
    // not write.
    let (parsed, faulted) = parser.parse_recovering();
    refuse(&faulted)?;
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
        Resolution {
            cache: cache.as_ref(),
            roots: &roots,
            layers: &layers,
        },
    )
    .context("Import error")?;
    let mut program = resolved.program;
    let mut linear_types = resolved.linear_types;
    let tests = resolved.tests;
    let mut modules = resolved.modules;
    if !cli.test {
        let ast = &program.ast;
        program.roots.retain(|statement| {
            !matches!(
                ast.stmt(*statement),
                Statement::Constant(name, _)
                    if ast.name(*name).contains(TEST_PREFIX)
            )
        });
    }
    // Every check reports what it found and the next one still runs, so one
    // invocation names every independent fault rather than the first pass's
    // worth. A rewrite is the exception: it edits the program, and no pass
    // after a failed edit can be trusted to read what it left behind, so a
    // rewrite that fails ends the run with whatever the checks have collected.
    let mut faults: Vec<frost::Diagnostic> = Vec::new();
    faults.extend(frost::check_callback_declarations_recovering(
        &program.ast,
        &program.roots,
    ));
    faults.extend(frost::check_declared_types(&program.ast, &program.roots));
    faults.extend(frost::check_entry_point(&program.ast, &program.roots));
    let (unchecked, idle) =
        frost::check_unsafety_and_audit(&program.ast, &program.roots);
    faults.extend(unchecked);
    // A block holding nothing unchecked is reported on every build, since the
    // list of `unsafe` blocks is only worth reading while every one of them
    // earns its place. `--audit-unsafe` turns the report into a failure, which
    // is what holds a tree to it.
    // Held until the build is known to have one thing to say. A reader shown a
    // block that vouches for nothing beside the fault that stopped the build
    // has been handed a second thing to read and one thing to do, so the
    // warnings go out where the last check has passed.
    if cli.audit_unsafe && !idle.is_empty() {
        // The one build where an idle block is what stops it, so this is where
        // the blocks are named: the summary below counts them and the reader
        // needs the lines.
        if wants_json() {
            eprint!("{}", frost::diagnostics_as_json(&idle, "warning"));
        } else {
            eprint!(
                "{}",
                frost::render_warnings(&frost::in_source_order(idle.clone()))
            );
        }
        anyhow::bail!(
            "Unsafe audit: {} block(s) vouch for nothing",
            idle.len()
        );
    }
    // `unsafe fn` is only meaningful to the unsafety check. Strip it to the
    // plain function it wraps before any later pass or backend sees one.
    strip_unsafe_fns(&mut program.ast, &program.roots);
    resolve_distinct_types(&mut program.ast, &program.roots);
    // The transitive set, since a struct holding a resource is one. The
    // lowering asks it about a `_` taking a value that has to be consumed, and
    // asking the declared set alone would have missed a struct holding a file.
    let held_linear =
        frost::linear_with_holders(&linear_types, &program.ast, &program.roots);
    beside(
        &faults,
        lower_multiple_returns(
            &mut program.ast,
            &mut program.roots,
            &held_linear,
        ),
    )?;
    faults.extend(frost::check_regions_recovering(
        &program.ast,
        &program.roots,
    ));
    faults.extend(frost::check_frame_escapes_recovering(
        &program.ast,
        &program.roots,
    ));
    // Threading a capability through the calls that draw one is a rewrite, so a
    // failure here ends the run. What it finds and rewrites nothing for — a
    // function that draws a capability taken as a value — joins the checks
    // above instead, so a program with one of those and an unrelated fault
    // elsewhere names both.
    let taken_as_values = beside(
        &faults,
        lower_allocation_sources(&mut program.ast, &program.roots),
    )?;
    faults.extend(taken_as_values);
    // A failure set's result is linear when what it carries is, so the set of
    // linear types grows here and the ownership check below sees the whole of
    // it.
    beside(&faults, lower_failure_sets(&mut program, &mut linear_types))?;
    lower_param_modes(&mut program.ast, &program.roots);
    // `assert` is a builtin, so it belongs to every program rather than to the
    // test harness that used to be the only thing declaring what it lowers to.
    // Without this it read as an unknown variable outside a test, which is a
    // different language from the one the self-hosted compiler accepts.
    if !cli.test && !declares_extern(&program, "frost_rt_assert_at") {
        push_assert_declaration(&mut program);
    }
    // The ownership check reads the program that is about to be lowered, which
    // under `--test` is the one the harness has been added to, so each path
    // runs it over its own and none runs it twice.
    if cli.test {
        if tests.is_empty() {
            println!("no tests found in {}", cli.file);
            return Ok(());
        }
        let mut augmented = program.clone();
        push_test_harness(&mut augmented, &tests);
        let module = lowered_and_checked(
            &mut augmented,
            &linear_types,
            false,
            &faults,
            &idle,
        )?;
        let stem = Path::new(&cli.file)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let directory = std::env::temp_dir();
        let exe_path = directory
            .join(format!("frost_test_{stem}{}", std::env::consts::EXE_SUFFIX));
        // `--emit-c` picks the backend a test body runs through, the same way it
        // does for a program. A suite that asked for both and got the native one
        // twice is a suite that was only ever testing one of them.
        if cli.emit_c {
            let c_source = emit_c(&module).context("C emission error")?;
            let c_path = directory.join(format!("frost_test_{stem}.c"));
            fs::write(&c_path, c_source).with_context(|| {
                format!("Failed to write C file: {}", c_path.display())
            })?;
            compile_c(
                &c_path.to_string_lossy(),
                &exe_path.to_string_lossy(),
                &cli.libs,
            )?;
            fs::remove_file(&c_path).ok();
        } else {
            let object_bytes = compile_ir_to_object(&module)
                .context("Native compilation error")?;
            let object_path = directory.join(format!("frost_test_{stem}.o"));
            fs::write(&object_path, object_bytes)?;
            link_executable(
                &[object_path.to_string_lossy().into_owned()],
                &exe_path.to_string_lossy(),
                &cli.libs,
                false,
            )?;
            fs::remove_file(&object_path).ok();
        }

        println!("running {} test(s)", tests.len());
        let status = Command::new(&exe_path)
            .status()
            .context("Failed to run test executable")?;
        fs::remove_file(&exe_path).ok();
        // The runner prints its own summary, so there is nothing to add here
        // beyond making the process exit on a failure.
        if status.success() {
            return Ok(());
        }
        std::process::exit(1);
    }


    if cli.run_ir {
        let module = lowered_and_checked(
            &mut program,
            &linear_types,
            false,
            &faults,
            &idle,
        )?;
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
        let module = lowered_and_checked(
            &mut program,
            &linear_types,
            false,
            &faults,
            &idle,
        )?;
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
        let module = lowered_and_checked(
            &mut program,
            &linear_types,
            cli.link,
            &faults,
            &idle,
        )?;
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
        // Intermediate objects are named for the executable rather than for the
        // source, since two builds of the same program to different outputs run
        // at once and one naming them for the source would delete the other's.
        let exe_path = cli.output.clone().unwrap_or_else(|| {
            if cfg!(windows) {
                format!("{}.exe", stem)
            } else {
                stem.to_string()
            }
        });

        let mut object_paths = Vec::with_capacity(parts.len());
        // Objects for cached modules are named for the fingerprint that
        // produced them and outlive the build. Everything else is an
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
                format!("{}.{}.o", exe_path, index)
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
        let module = lowered_and_checked(
            &mut program,
            &linear_types,
            false,
            &faults,
            &idle,
        )?;
        let object_bytes = compile_ir_to_object(&module)
            .context("Native compilation error")?;
        // Named after the whole path rather than after the file's own name.
        // Two projects each running a `tools/writer.frost` are two programs,
        // and a name taken from the last part of the path is one name for
        // both, so one run would build over the other's program and then
        // execute it. The self-hosted compiler names this file after the whole
        // path for the same reason.
        let stem = Path::new(&cli.file)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let mut hasher = DefaultHasher::new();
        std::fs::canonicalize(&cli.file)
            .unwrap_or_else(|_| PathBuf::from(&cli.file))
            .hash(&mut hasher);
        let stem = format!("{stem}_{:016x}", hasher.finish());
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
        // A program the compiler ran can reach the compiler that ran it, which
        // is what a build program written in Frost needs and what it would
        // otherwise have to be told on its own command line.
        let mut running = Command::new(&exe_path);
        running.args(&forwarded);
        if let Ok(self_path) = std::env::current_exe() {
            running.env("FROST_COMPILER", self_path);
        }
        let status = running.status().context("Failed to run executable")?;
        fs::remove_file(&exe_path).ok();
        if !status.success() {
            std::process::exit(status.code().unwrap_or(1));
        }
    }

    Ok(())
}

const RUNTIME_SOURCE: &str = include_str!("../../runtime/frost_runtime.c");
const RUNTIME_FROST_SOURCE: &str = include_str!("../../runtime/runtime.frost");
const FREESTANDING_SOURCE: &str =
    include_str!("../../runtime/frost_freestanding.c");

/// What the copy of the runtime compiled into this binary hashes to. The
/// stand-in file is named for it, so every process derives the same path from
/// the same bytes and one of them can be recognised as the runtime by the
/// others.
fn frost_runtime_key() -> u64 {
    let mut hasher = DefaultHasher::new();
    RUNTIME_FROST_SOURCE.hash(&mut hasher);
    hasher.finish()
}

/// Where the Frost half of the runtime is.
///
/// FROST_RUNTIME_FROST names it. Otherwise it is looked for beside the compiler
/// and then up the directories a checkout puts it under, which is the search the
/// self-hosted compiler does and the search the standard library gets, so a
/// compiler run from somewhere other than where it was built still finds it.
/// Found nowhere, the copy compiled into this binary stands in.
fn frost_runtime_path() -> PathBuf {
    if let Ok(named) = std::env::var("FROST_RUNTIME_FROST")
        && !named.is_empty()
    {
        return PathBuf::from(named);
    }
    if let Ok(executable) = std::env::current_exe() {
        let mut directory = executable.parent().map(Path::to_path_buf);
        for _ in 0..3 {
            let Some(here) = directory else {
                break;
            };
            let candidate = here.join("runtime").join("runtime.frost");
            if candidate.exists() {
                return candidate;
            }
            directory = here.parent().map(Path::to_path_buf);
        }
    }
    std::env::temp_dir()
        .join(format!("frost_runtime_{:016x}.frost", frost_runtime_key()))
}

/// Whether the file being compiled is that runtime, which is the one file that
/// may define names in the runtime's own name space.
fn is_the_runtime(file: &str) -> bool {
    let wanted = frost_runtime_path();
    let here = Path::new(file);
    match (here.canonicalize(), wanted.canonicalize()) {
        (Ok(here), Ok(wanted)) => here == wanted,
        _ => here == wanted,
    }
}

/// The Frost half of the runtime, compiled into an object every link puts
/// beside the C stub's.
///
/// Cached the way the C half is, keyed on the source, so it is built once and
/// linked thereafter. Compiled by running this compiler again rather than by
/// calling into the passes directly, for the same reason the self-hosted one
/// does it that way: reaching an object from a path is what the command line
/// already is, and a second way in would be a second thing to keep in step with
/// the checks the first one runs.
///
/// `--native` writes an object and links nothing, so a build of the runtime
/// never asks for a runtime and the recursion is one level deep.
fn frost_runtime_object() -> Result<PathBuf> {
    let key = frost_runtime_key();

    let directory = std::env::temp_dir();
    let cached = directory.join(format!("frost_runtime_frost_{key:016x}.o"));
    if cached.exists() {
        return Ok(cached);
    }

    let source_path = frost_runtime_path();
    if !source_path.exists() {
        fs::write(&source_path, RUNTIME_FROST_SOURCE).with_context(|| {
            format!("Failed to write the runtime: {}", source_path.display())
        })?;
    }
    let pending = directory.join(format!(
        "frost_runtime_frost_{key:016x}_{}.o",
        std::process::id()
    ));
    let compiler = std::env::current_exe()
        .context("finding the compiler to build the runtime with")?;
    let output = Command::new(&compiler)
        .arg("--native")
        .arg("-o")
        .arg(&pending)
        .arg(&source_path)
        .output()
        .context("Failed to compile the Frost runtime")?;
    if !output.status.success() {
        bail!(
            "The Frost runtime failed to compile: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // A rename is atomic, so builds racing to fill the cache all end up with the
    // same object rather than a half-written one.
    if fs::rename(&pending, &cached).is_err() {
        fs::copy(&pending, &cached).with_context(|| {
            format!("Failed to cache the runtime: {}", cached.display())
        })?;
        fs::remove_file(&pending).ok();
    }
    Ok(cached)
}

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
    let frost_runtime_path = frost_runtime_object()?;

    // The C is an intermediate, so it is compiled the way an intermediate
    // should be. Without this the C path ran unoptimized while the Cranelift
    // path asked for speed, which made the two backends answer the same thing
    // at different cost and made the C path a poor measurement.
    //
    // The two flags beside the optimizer are not tuning. They hold the C
    // compiler to what Frost means: an integer wraps on overflow here rather
    // than being undefined, and `ptr_cast` reads the same bytes through another
    // type on purpose. A compiler entitled to assume neither ever happens is
    // entitled to miscompile this, and no comparison against another backend
    // would show it, since a differential compiles both sides the same way.
    // MSVC assumes neither by default, so the `cl` branch needs no flag.
    let mut cmd = Command::new(compiler);
    if compiler == "cl" {
        cmd.arg("/O2");
        cmd.arg(c_path);
        cmd.arg(&runtime_path);
        cmd.arg(&frost_runtime_path);
        cmd.arg(format!("/Fe:{}", exe_path));
        for lib in extra_libs {
            cmd.arg(lib);
        }
    } else {
        cmd.arg("-std=c11");
        cmd.arg("-O2");
        cmd.arg("-fwrapv");
        cmd.arg("-fno-strict-aliasing");
        cmd.arg(c_path);
        cmd.arg(&runtime_path);
        cmd.arg(&frost_runtime_path);
        cmd.arg("-o");
        cmd.arg(exe_path);
        for lib in extra_libs {
            cmd.arg(lib);
        }
        // The C math functions std/math.frost calls (sqrtf and the rest) live in
        // libm on Linux and the BSDs. On macOS and mingw the flag is a harmless
        // no-op, and MSVC keeps them in the CRT and is the `cl` branch above.
        cmd.arg("-lm");
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
    // Windows. Exit through kernel32, entry mainCRTStartup. No C runtime.
    cmd.arg("-lkernel32");
    cmd.arg("-e").arg("mainCRTStartup");
}

#[cfg(target_os = "linux")]
fn add_freestanding_link_args(cmd: &mut Command) {
    // Linux. The runtime's _start is the entry, exit is a raw syscall. No libc.
    cmd.arg("-e").arg("_start");
}

#[cfg(target_os = "macos")]
fn add_freestanding_link_args(cmd: &mut Command) {
    // macOS. Entry _start, exit via syscall, but macOS always routes syscalls
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
    // A freestanding link has no libc, and the Frost half of the runtime reports
    // through stdio, so that target keeps its own silent copies of the checks in
    // `frost_freestanding.c` and this object is not what it links.
    let frost_runtime_path = if freestanding {
        None
    } else {
        Some(frost_runtime_object()?)
    };

    let mut cmd = Command::new(linker);

    if linker == "cl" {
        cmd.args(object_paths);
        cmd.arg(&runtime_path);
        if let Some(path) = &frost_runtime_path {
            cmd.arg(path);
        }
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
        if let Some(path) = &frost_runtime_path {
            cmd.arg(path);
        }
        cmd.arg("-o");
        cmd.arg(exe_path);
        for lib in extra_libs {
            cmd.arg(lib);
        }
        if !freestanding {
            // A program that calls the C math functions (sqrtf and the rest,
            // used by std/math.frost) needs libm on the platforms that keep it
            // out of the C runtime. Linux and the BSDs do. On macOS and mingw it
            // is folded in and the flag is a harmless no-op. MSVC keeps them in
            // the CRT and is the `cl` branch above.
            cmd.arg("-lm");
        }
        if freestanding {
            // The freestanding runtime supplies the platform's entry point. The
            // linker needs the matching entry symbol and, where the OS
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
