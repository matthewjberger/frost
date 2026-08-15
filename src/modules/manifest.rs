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
    // The files this project writes with a program of its own, in the order
    // they are written. `frost generate` runs them.
    #[serde(default, deserialize_with = "read_generated")]
    pub generated: Vec<Generated>,
}

// Read whatever shape is written, so that a build reading `paths` and `layers`
// beside this gets its answer. A member that is not an object, one that names
// its output with a number, and a `generated` that is not a list at all all
// arrive with nothing named, and `frost generate` is what refuses them.
fn read_generated<'de, D>(reader: D) -> Result<Vec<Generated>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let held = <serde_json::Value as serde::Deserialize>::deserialize(reader)?;
    Ok(held
        .as_array()
        .map(|members| members.iter().map(read_one_generated).collect())
        .unwrap_or_default())
}

fn read_one_generated(member: &serde_json::Value) -> Generated {
    Generated {
        output: text_at(member, "output"),
        from: text_at(member, "from"),
        inputs: member
            .get("inputs")
            .and_then(serde_json::Value::as_array)
            .map(|held| {
                held.iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn text_at(member: &serde_json::Value, name: &str) -> String {
    member
        .get(name)
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string()
}

// One file a program in this project writes.
//
// The program is an ordinary Frost program taking the output path first and its
// inputs after, so it can be compiled, run and read on its own without knowing
// a manifest exists. Declaring it here is what lets a checkout regenerate every
// such file with one command and check that none of them is stale, rather than
// each generator needing a build script that knows where its own inputs are.
//
// What it writes stays a file in the tree: a reader can open it, a diff shows
// what a schema change did to it, and the compiler that reads it afterward is
// the ordinary one. That is the whole reason generation lives here rather than
// inside the compiler as a step that injects declarations.
// Each path is optional to read and required to use, so a member missing one is
// refused by `frost generate` rather than by whatever read the manifest next. A
// build reads `paths` and `layers` and never this, and refusing to compile a
// program because a build step beside it is half declared would be refusing it
// for a reason that has nothing to do with it.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]
pub struct Generated {
    // The file the program writes, relative to the manifest.
    #[serde(default)]
    pub output: String,
    // The program that writes it, relative to the manifest.
    #[serde(default)]
    pub from: String,
    // What that program reads, relative to the manifest, handed to it after the
    // output path in the order written here.
    #[serde(default)]
    pub inputs: Vec<String>,
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

impl Generated {
    // The three paths it declares, resolved against the manifest, so a command
    // run from anywhere reaches the same files a command run from the project
    // root does.
    pub fn resolved(
        &self,
        project_root: &Path,
    ) -> (PathBuf, PathBuf, Vec<PathBuf>) {
        (
            project_root.join(&self.output),
            project_root.join(&self.from),
            self.inputs
                .iter()
                .map(|entry| project_root.join(entry))
                .collect(),
        )
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

    #[test]
    fn a_manifest_declares_the_files_a_program_of_its_own_writes() {
        let manifest: Manifest = serde_json::from_str(
            r#"{ "generated": [
                   { "output": "lib/wgpu.frost",
                     "from": "tools/bindgen.frost",
                     "inputs": ["vendor/webgpu.json"] }] }"#,
        )
        .unwrap();
        assert_eq!(manifest.generated.len(), 1);
        let (output, from, inputs) =
            manifest.generated[0].resolved(Path::new("/project"));
        assert!(output.ends_with("lib/wgpu.frost"));
        assert!(from.ends_with("tools/bindgen.frost"));
        assert_eq!(inputs.len(), 1);
        assert!(inputs[0].ends_with("vendor/webgpu.json"));
    }

    // A generator that reads nothing still writes something, so the inputs are
    // what may be left out rather than what has to be there.
    #[test]
    fn a_generator_may_read_nothing() {
        let manifest: Manifest = serde_json::from_str(
            r#"{ "generated": [
                   { "output": "a.frost", "from": "b.frost" }] }"#,
        )
        .unwrap();
        assert!(manifest.generated[0].inputs.is_empty());
    }

    // A build reads `paths` and `layers` and never reads this, so a member of
    // any shape is read rather than refused. What it names comes out empty,
    // which is what `frost generate` has something to say about.
    #[test]
    fn a_generated_member_of_any_shape_leaves_a_build_alone() {
        let manifest: Manifest = serde_json::from_str(
            r#"{ "paths": ["lib"],
                 "generated": [
                   "oops",
                   7,
                   { "output": 3, "from": "w.frost" },
                   { "output": "a", "from": "b", "inputs": [1, "c"] }] }"#,
        )
        .expect("a member of any shape is read");
        assert_eq!(manifest.paths, vec!["lib".to_string()]);
        assert_eq!(manifest.generated.len(), 4);
        assert!(manifest.generated[0].output.is_empty());
        assert!(manifest.generated[0].from.is_empty());
        assert!(manifest.generated[1].output.is_empty());
        assert!(manifest.generated[2].output.is_empty());
        assert_eq!(manifest.generated[2].from, "w.frost");
        assert_eq!(manifest.generated[3].inputs, vec!["c".to_string()]);
    }

    #[test]
    fn a_generated_that_is_not_a_list_leaves_a_build_alone() {
        let manifest: Manifest =
            serde_json::from_str(r#"{ "layers": ["a"], "generated": "oops" }"#)
                .expect("a 'generated' of any shape is read");
        assert_eq!(manifest.layers, vec!["a".to_string()]);
        assert!(manifest.generated.is_empty());
    }
}
