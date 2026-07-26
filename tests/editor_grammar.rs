use std::collections::BTreeSet;

const GRAMMAR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/.vscode/frost/syntaxes/frost.tmLanguage.json"
);

const CONTEXTUAL: &[&str] = &["export", "flags", "test"];

const BUILTIN_TYPES: &[&str] = &["Handle", "Type", "columns"];

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

#[test]
fn the_editor_grammar_lists_exactly_the_words_the_compiler_knows() {
    let text =
        std::fs::read_to_string(GRAMMAR).expect("the grammar is in the tree");
    let grammar: serde_json::Value =
        serde_json::from_str(&text).expect("the grammar is valid JSON");

    let mut found = BTreeSet::new();
    for rule in ["keywords", "builtin-functions", "builtin-types"] {
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
    expected.extend(frost::BUILTIN_FUNCTIONS.iter().copied().map(String::from));
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
