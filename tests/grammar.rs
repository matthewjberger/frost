use frost::{Lexer, Parser};

fn parses(source: &str) -> bool {
    let mut lexer = Lexer::new(source);
    let Ok(tokens) = lexer.tokenize() else {
        return false;
    };
    let mut parser = Parser::new(&tokens);
    parser.parse().is_ok()
}

#[test]
fn grammar_accepts_specified_constructs() {
    let valid = [
        "MAX :: 10",
        "PI :: 3.14",
        "GREETING :: \"hello\"",
        "add :: fn(a: i64, b: i64) -> i64 { a + b }",
        "noop :: fn() { }",
        "Point :: struct { x: i64, y: i64 }",
        "Pair :: struct($T: Type) { first: T, second: T }",
        "Shape :: enum { Circle { r: i64 }, Rect { w: i64, h: i64 } }",
        "Kind :: enum { A, B { n: i64 } }",
        "File :: linear struct { fd: i64 }",
        "Meters :: distinct i64",
        "printf :: extern fn(fmt: ^i8, value: i64) -> i32",
        "close :: extern fn(f: i64)",
        // An extern takes parameter modes and compile-time parameters, which
        // is what a callback registration is written with.
        "consume :: extern fn(move f: i64)",
        "reg :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> i64",
        "make :: extern fn(v: i64) -> Ctx",
        // A function type may say `mut`, because a `mut` parameter is a
        // reference in the signature and the surface has no reference type.
        "hold :: fn(f: fn(mut Ctx, i64)) { }",
        "held :: fn(f: fn(move Ctx) -> i64) { }",
        // A value parameter is a compile-time integer, on a function as well
        // as on a struct, and stands for its value in the body.
        "Slab :: struct($T: Type, $N: usize) { storage: [N]T }",
        "reset :: fn($T: Type, $N: usize, mut s: Slab<T, N>) { }",
        "size :: fn($N: usize) -> i64 { N }",
        // An enum takes type parameters the same way a struct does.
        "Option :: enum($T: Type) { None, Some { value: T } }",
        "Either :: enum($L: Type, $R: Type) { Left { v: L }, Right { v: R } }",
        "main :: fn() -> i64 {\n x := 5\n mut y : i64 = 0\n y = y + 1\n 0\n }",
        "cond :: fn() -> i64 { if (1 < 2) { 1 } else { 0 } }",
        "loop :: fn() -> i64 {\n mut i : i64 = 0\n while (i < 3) { i = i + 1 }\n i\n }",
        "counted :: fn() { for i in 0..10 { } }",
        "area :: fn(s: i64) -> i64 {\n match s {\n case 0: 1\n case _: 0\n }\n }",
        "shape :: fn(s: i64) -> i64 {\n match s {\n case .Circle { r }: r\n case .Rect { w, h }: w\n case _: 0\n }\n }",
        "tup :: fn(x: i64) -> i64 {\n match (x % 3, x % 5) {\n case (0, 0): 1\n case (_, _): 0\n }\n }",
        "ptrs :: fn(a: ^i8, b: i64, mut c: i64, d: []i64, e: [4]i64, h: Handle<i64>) -> i64 { 0 }",
        "nested :: fn(p: Pair<Pair<i64>>) -> i64 { 0 }",
        "make :: fn($T: Type, n: i64) -> i64 { sizeof(T) }",
        "callit :: fn() -> i64 { make(8) }",
        "deref :: fn(p: ^i8) -> i8 { p^ }",
        "field :: fn() -> i64 {\n pt := Point { x = 1, y = 2 }\n pt.x\n }",
        "variant :: fn() -> i64 { unwrap(Shape::Circle { radius = 5 }) }",
        "unit :: fn() -> i64 { pick(Kind::A) }",
        "deferred :: fn() -> i64 {\n defer noop()\n 0\n }",
        "arr :: fn() -> i64 {\n xs := [1, 2, 3]\n xs[0]\n }",
        "fnptr :: fn(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }",
        "ranges :: fn() -> i64 {\n for i in 0..=5 { }\n 0\n }",
    ];
    for source in valid {
        assert!(parses(source), "grammar should accept:\n{source}");
    }
}

#[test]
fn grammar_rejects_malformed_input() {
    let invalid = [
        "cond :: fn() -> i64 { if 1 < 2 { 1 } else { 0 } }",
        "loop :: fn() { while i < 3 { } }",
        "Point :: struct { x i64 }",
        "bad :: fn() -> i64 { 1 + }",
        // Truncated inputs must reject, not hang (regression: the parameter
        // loop used to spin forever at end of input).
        "trunc :: fn(a: i64",
        "trunc2 :: fn(a: i64, b",
        "truncstruct :: struct { x: i64",
        "truncmatch :: fn() -> i64 { match x { case 0: 1",
        "nofields :: struct {",
    ];
    for source in invalid {
        assert!(!parses(source), "grammar should reject:\n{source}");
    }
}

// Every `.frost` file in the repository, which is what the formatter's
// invariants are held over. The corpus is the style definition, so it is also
// the only honest test set.
fn corpus() -> Vec<std::path::PathBuf> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut found = Vec::new();
    let mut stack = vec![
        root.join("std"),
        root.join("lib"),
        root.join("selfhosted"),
        root.join("examples"),
        root.join("tools"),
        root.join("bench"),
    ];
    while let Some(next) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&next) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|kind| kind == "frost") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

// The invariant everything else stands on: the token extents and the gaps
// between them are the whole file. If this holds, no comment and no piece of
// whitespace is invisible to the formatter, whatever it then decides to write.
#[test]
fn every_source_is_its_tokens_and_gaps() {
    let files = corpus();
    assert!(files.len() > 100, "the corpus should be the whole tree");
    for file in &files {
        let source = std::fs::read_to_string(file).unwrap();
        let pieces = frost::tokens_and_gaps(&source)
            .unwrap_or_else(|| panic!("{} could not be lexed", file.display()));
        assert_eq!(
            pieces.concat(),
            source,
            "{} is not its tokens and its gaps",
            file.display()
        );
    }
}

#[test]
fn formatting_the_corpus_is_idempotent() {
    for file in corpus() {
        let source = std::fs::read_to_string(&file).unwrap();
        let once = frost::format_source(&source);
        let twice = frost::format_source(&once);
        assert_eq!(
            once,
            twice,
            "{} formats differently the second time",
            file.display()
        );
    }
}

// Formatting changes no token. The bytes between them are this formatter's to
// settle; the tokens are the program.
#[test]
fn formatting_the_corpus_changes_no_token() {
    for file in corpus() {
        let source = std::fs::read_to_string(&file).unwrap();
        let formatted = frost::format_source(&source);
        let before = frost::tokens_and_gaps(&source).unwrap();
        let after = frost::tokens_and_gaps(&formatted)
            .unwrap_or_else(|| panic!("{} did not lex after", file.display()));
        let tokens = |pieces: &Vec<String>| -> Vec<String> {
            pieces.iter().skip(1).step_by(2).cloned().collect()
        };
        assert_eq!(
            tokens(&before),
            tokens(&after),
            "{} has different tokens after formatting",
            file.display()
        );
    }
}

// The tree is what the formatter writes. A build fails on a file that is not,
// which is what keeps one rendering the only rendering.
#[test]
fn the_corpus_is_formatted() {
    let unformatted: Vec<String> = corpus()
        .into_iter()
        .filter(|file| {
            std::fs::read_to_string(file)
                .map(|source| frost::format_source(&source) != source)
                .unwrap_or(false)
        })
        .map(|file| file.display().to_string())
        .collect();
    assert!(
        unformatted.is_empty(),
        "run `frost fmt` over these:\n{}",
        unformatted.join("\n")
    );
}

// Where a line opens with an operator while no bracket is open, and where the
// expression it continues began.
struct Continuation {
    /// The byte the expression starts at, where an opening bracket goes.
    opens: usize,
    /// The byte after the last token of the run, where the closing one goes.
    closes: usize,
}

fn continuations_outside_brackets(source: &str) -> Vec<Continuation> {
    use frost::{Lexer, Token};
    let mut lexer = Lexer::new(source);
    let Ok(tokens) = lexer.tokenize() else {
        return Vec::new();
    };
    let starts = lexer.positions().to_vec();
    let ends = lexer.ends().to_vec();
    let offset = |place: &frost::Position| -> usize {
        let mut at = 0usize;
        for (number, line) in source.split_inclusive('\n').enumerate() {
            if number + 1 == place.line {
                let within = line
                    .char_indices()
                    .nth(place.column.saturating_sub(1))
                    .map(|(at, _)| at)
                    .unwrap_or(line.len());
                return at + within;
            }
            at += line.len();
        }
        source.len()
    };

    let continues = |token: &Token| {
        matches!(
            token,
            Token::Plus
                | Token::Asterisk
                | Token::Slash
                | Token::Percent
                | Token::And
                | Token::Or
                | Token::Pipe
                | Token::Ampersand
                | Token::Equal
                | Token::NotEqual
                | Token::LessThan
                | Token::LessThanOrEqual
                | Token::GreaterThan
                | Token::GreaterThanOrEqual
                | Token::Dot
        )
    };

    // The token each line begins at, and the bracket depth in front of it.
    let mut depth = 0i32;
    let mut depths = vec![0i32; tokens.len()];
    let mut opens_line = vec![false; tokens.len()];
    let mut previous = 0usize;
    for (index, token) in tokens.iter().enumerate() {
        opens_line[index] = starts[index].line != previous;
        previous = starts[index].line;
        depths[index] = depth;
        match token {
            Token::LeftParentheses | Token::LeftBracket => depth += 1,
            Token::RightParentheses | Token::RightBracket => depth -= 1,
            _ => {}
        }
    }

    let mut found = Vec::new();
    let mut index = 0usize;
    while index < tokens.len() {
        if !(opens_line[index]
            && continues(&tokens[index])
            && depths[index] == 0)
        {
            index += 1;
            continue;
        }
        // Back to the first token of the statement this continues: the last
        // line opener that was not itself a continuation.
        let mut head = index;
        while head > 0 {
            head -= 1;
            if opens_line[head] && !continues(&tokens[head]) {
                break;
            }
        }
        // The expression begins after the last `:=`, `=` or `return` on that
        // line; a block's trailing value begins at the line's first token.
        let mut begins = head;
        let mut at = head;
        while at < tokens.len() && starts[at].line == starts[head].line {
            if matches!(
                tokens[at],
                Token::ColonAssign | Token::Assign | Token::Return
            ) {
                begins = at + 1;
            }
            at += 1;
        }
        // Forward over every further continuation line of this run.
        let mut last = index;
        let mut ahead = index;
        while ahead < tokens.len() {
            if opens_line[ahead] && !continues(&tokens[ahead]) && ahead > index
            {
                break;
            }
            if depths[ahead] == 0
                && opens_line[ahead]
                && continues(&tokens[ahead])
            {
                last = ahead;
            }
            ahead += 1;
        }
        // The run ends at the token before the next one that opens a line with
        // no bracket held open: everything up to there, nested calls and their
        // own wrapped arguments included, belongs to this expression.
        let mut end = last;
        while end + 1 < tokens.len()
            && !(opens_line[end + 1] && depths[end + 1] == 0)
        {
            end += 1;
        }
        found.push(Continuation {
            opens: offset(&starts[begins]),
            closes: offset(&ends[end]),
        });
        index = end + 1;
    }
    found
}

// Run once with `cargo test -r --test grammar -- --ignored migrate`. Kept as
// the record of how the corpus was moved off leading-operator continuation.
#[test]
#[ignore]
fn migrate_continuations_into_brackets() {
    let mut moved = 0usize;
    for file in corpus() {
        let Ok(source) = std::fs::read_to_string(&file) else {
            continue;
        };
        let found = continuations_outside_brackets(&source);
        if found.is_empty() {
            continue;
        }
        let mut text = source.clone();
        // Latest first, so an earlier offset stays where it is.
        for held in found.iter().rev() {
            text.insert(held.closes, ')');
            text.insert(held.opens, '(');
        }
        std::fs::write(&file, text).unwrap();
        moved += found.len();
    }
    println!("wrapped {moved} continuations");
}
