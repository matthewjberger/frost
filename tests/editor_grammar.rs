use std::collections::BTreeSet;

const GRAMMAR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/.vscode/frost/syntaxes/frost.tmLanguage.json"
);

const CONTEXTUAL: &[&str] = &[
    "align",
    "export",
    "false",
    "flags",
    "include_str",
    "packed",
    "test",
    "true",
    "value",
];

const BUILTIN_TYPES: &[&str] = &[
    "Handle", "Type", "bool", "columns", "f32", "f64", "i16", "i32", "i64",
    "i8", "isize", "str", "u16", "u32", "u64", "u8", "usize",
];

const FUNCTION_MARKERS: &[&str] = &["extern", "inline", "safe", "unsafe"];

const TYPE_HEAD_WORDS: &[&str] = &[
    "distinct", "enum", "flags", "linear", "packed", "struct", "type",
];

fn collect_matches(value: &serde_json::Value, into: &mut Vec<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, held) in map {
                if key == "match"
                    && let Some(text) = held.as_str()
                {
                    into.push(text.to_string());
                } else {
                    collect_matches(held, into);
                }
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_matches(item, into);
            }
        }
        _ => {}
    }
}

fn words_of(pattern: &str) -> Vec<String> {
    let Some(open) = pattern.find("\\b(") else {
        return Vec::new();
    };
    let rest = &pattern[open + 3..];
    let Some(close) = rest.find(')') else {
        return Vec::new();
    };
    rest[..close].split('|').map(String::from).collect()
}

fn lookahead_words(pattern: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut rest = pattern;
    while let Some(open) = rest.find("(?:") {
        let body = &rest[open + 3..];
        let Some(close) = body.find(')') else {
            break;
        };
        for piece in body[..close].split('|') {
            let word = piece.trim_end_matches("\\s+");
            if !word.is_empty()
                && word.chars().all(|character| character.is_ascii_lowercase())
            {
                words.push(word.to_string());
            }
        }
        rest = &body[close..];
    }
    words
}

#[test]
fn the_editor_grammar_lists_exactly_the_words_the_compiler_knows() {
    let text =
        std::fs::read_to_string(GRAMMAR).expect("the grammar is in the tree");
    let grammar: serde_json::Value =
        serde_json::from_str(&text).expect("the grammar is valid JSON");

    let mut found = BTreeSet::new();
    for rule in [
        "keywords",
        "builtin-functions",
        "builtin-constants",
        "builtin-types",
    ] {
        let mut patterns = Vec::new();
        collect_matches(&grammar["repository"][rule], &mut patterns);
        assert!(!patterns.is_empty(), "the grammar has no '{rule}' rule");
        for pattern in &patterns {
            found.extend(words_of(pattern));
        }
    }

    let mut expected: BTreeSet<String> = frost::KEYWORD_NAMES
        .iter()
        .copied()
        .map(String::from)
        .collect();
    expected.extend(frost::COMPILER_NAMES.iter().copied().map(String::from));
    expected.extend(CONTEXTUAL.iter().copied().map(String::from));
    expected.extend(BUILTIN_TYPES.iter().copied().map(String::from));

    let missing = expected.difference(&found).collect::<Vec<_>>();
    let extra = found.difference(&expected).collect::<Vec<_>>();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "the editor grammar has drifted from the compiler\n  \
         the compiler knows these and the grammar does not: {missing:?}\n  \
         the grammar claims these and the compiler does not: {extra:?}"
    );
}

#[test]
fn the_declaration_lookaheads_hold_the_markers_the_parser_accepts() {
    let text =
        std::fs::read_to_string(GRAMMAR).expect("the grammar is in the tree");
    let grammar: serde_json::Value =
        serde_json::from_str(&text).expect("the grammar is valid JSON");

    let mut patterns = Vec::new();
    collect_matches(&grammar["repository"]["declarations"], &mut patterns);
    assert!(
        !patterns.is_empty(),
        "the grammar has no 'declarations' rule"
    );

    for (label, needle, pinned) in [
        ("function head", "fn\\b", FUNCTION_MARKERS),
        ("type head", "struct", TYPE_HEAD_WORDS),
    ] {
        let Some(rule) =
            patterns.iter().find(|pattern| pattern.contains(needle))
        else {
            panic!("no declaration rule mentions '{needle}'");
        };
        let found: BTreeSet<String> =
            lookahead_words(rule).into_iter().collect();
        let expected: BTreeSet<String> =
            pinned.iter().copied().map(String::from).collect();
        for word in pinned {
            assert!(
                frost::KEYWORD_NAMES.contains(word)
                    || CONTEXTUAL.contains(word),
                "'{word}' is pinned as a {label} word and the compiler does \
                 not know it"
            );
        }
        let missing = expected.difference(&found).collect::<Vec<_>>();
        let extra = found.difference(&expected).collect::<Vec<_>>();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "the {label} rule has drifted from the parser\n  \
             the parser accepts these and the grammar does not: {missing:?}\n  \
             the grammar claims these and the parser does not: {extra:?}"
        );
    }
}
