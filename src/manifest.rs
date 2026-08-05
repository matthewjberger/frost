use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

// A project's `frost.json`, sitting beside the file named on the command line.
//
// It exists to answer one question that a relative `import` cannot: where a
// library this project depends on lives. Everything else a manifest could grow
// into (versions, dependencies fetched from somewhere) is deliberately absent,
// because none of it is needed to compile a program and each of it is a
// decision that would be hard to take back.
//
// JSON rather than a bespoke format, because interfaces and build records are
// already serde and JSON and a second format would be a second thing to learn.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]
pub struct Manifest {
    // What the project calls itself. Not used for resolution. It is here so a
    // diagnostic can name the project rather than a directory.
    #[serde(default)]
    pub name: String,
    // Directories to search for an import, relative to the manifest.
    #[serde(default)]
    pub paths: Vec<String>,
    // The project's layers, lowest first, relative to the manifest. A file
    // under one of these may import from its own layer or from any listed
    // before it, and importing from one listed after is refused. A file under
    // none of them is unconstrained both as an importer and as a target, which
    // is what leaves the standard library and a one-file program alone.
    #[serde(default)]
    pub layers: Vec<String>,
    // The prefix a file's exported names must share, by the directory the file
    // sits under. A flat namespace is navigated by prefix, so a family that
    // does not share one cannot be asked for with `frost api`.
    #[serde(default)]
    pub prefixes: std::collections::BTreeMap<String, String>,
}

pub const MANIFEST_NAME: &str = "frost.json";

impl Manifest {
    fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", path.display()))
    }

    // The nearest manifest at or above a directory, and the directory it sits
    // in. A layer list describes a whole project, and the entry file of a build
    // is any file in it, so the declaration is found by walking up rather than
    // by being repeated in every directory a build might start from.
    pub fn find_upward(start: &Path) -> Result<Option<(Self, PathBuf)>> {
        let mut directory = start.to_path_buf();
        loop {
            let path = directory.join(MANIFEST_NAME);
            if path.exists() {
                return Ok(Some((Self::read(&path)?, directory)));
            }
            if !directory.pop() {
                return Ok(None);
            }
        }
    }

    // The layer directories it declares, resolved against the manifest.
    pub fn layer_paths(&self, project_root: &Path) -> Vec<PathBuf> {
        self.layers
            .iter()
            .map(|entry| project_root.join(entry))
            .collect()
    }

    // The search directories it declares, resolved against the manifest.
    pub fn search_paths(&self, project_root: &Path) -> Vec<PathBuf> {
        self.paths
            .iter()
            .map(|entry| project_root.join(entry))
            .collect()
    }
}

// Where the standard library lives, in the order worth trying.
//
// `FROST_STD` wins, because someone who says exactly where it is means it. Then
// a `std` beside the compiler, which is what an installed layout looks like.
// Then two directories up from the compiler, which is what a `cargo build`
// layout looks like. The binary lands in `target/debug` and the library is at
// the repository root.
pub fn bundled_std() -> Option<PathBuf> {
    if let Ok(named) = std::env::var("FROST_STD") {
        let path = PathBuf::from(named);
        return path.is_dir().then_some(path);
    }
    let executable = std::env::current_exe().ok()?;
    let directory = executable.parent()?;
    let installed = directory.join("std");
    if installed.is_dir() {
        return Some(installed);
    }
    let from_build = directory.parent()?.parent()?.join("std");
    from_build.is_dir().then_some(from_build)
}

// The directories named by `FROST_PATH`, split the way the platform splits a
// path list.
pub fn path_from_environment() -> Vec<PathBuf> {
    let Ok(value) = std::env::var("FROST_PATH") else {
        return Vec::new();
    };
    std::env::split_paths(&value)
        .filter(|entry| !entry.as_os_str().is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_manifest_declares_where_libraries_live() {
        let manifest: Manifest = serde_json::from_str(
            r#"{ "name": "demo", "paths": ["lib", "vendor/things"] }"#,
        )
        .unwrap();
        assert_eq!(manifest.name, "demo");
        let roots = manifest.search_paths(Path::new("/project"));
        assert_eq!(roots.len(), 2);
        assert!(roots[0].ends_with("lib"));
        assert!(roots[1].ends_with("vendor/things"));
    }

    // Both fields are optional, so the smallest useful manifest is one line.
    #[test]
    fn a_manifest_may_declare_almost_nothing() {
        let manifest: Manifest =
            serde_json::from_str(r#"{ "paths": ["lib"] }"#).unwrap();
        assert_eq!(manifest.name, "");
        assert_eq!(manifest.paths, vec!["lib".to_string()]);
    }
}
