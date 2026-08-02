use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};

fn declared() -> &'static RwLock<Vec<(String, PathBuf)>> {
    static LAYERS: OnceLock<RwLock<Vec<(String, PathBuf)>>> = OnceLock::new();
    LAYERS.get_or_init(|| RwLock::new(Vec::new()))
}

pub fn declare(named: &[String], directories: &[PathBuf]) {
    let mut held = declared().write().expect("the layer list is readable");
    held.clear();
    for (name, directory) in named.iter().zip(directories) {
        let resolved = directory
            .canonicalize()
            .unwrap_or_else(|_| directory.clone());
        held.push((name.clone(), resolved));
    }
}

fn layer_of(file: &Path) -> Option<(usize, String)> {
    let resolved = file.canonicalize().unwrap_or_else(|_| file.to_path_buf());
    let held = declared().read().expect("the layer list is readable");
    let mut best: Option<(usize, String, usize)> = None;
    for (index, (name, directory)) in held.iter().enumerate() {
        if !resolved.starts_with(directory) {
            continue;
        }
        let depth = directory.components().count();
        if best.as_ref().is_none_or(|(_, _, held)| depth > *held) {
            best = Some((index, name.clone(), depth));
        }
    }
    best.map(|(index, name, _)| (index, name))
}

/// Whether an import reaches from one layer into a later one, and what to say
/// about it.
///
/// The paths are the ones resolution settled on, so `../engine/../renderer/x`
/// has already become the file it names and cannot spell its way past this.
/// Where either file is under no declared layer there is nothing to compare, so
/// the import stands.
pub fn reaching_upward(importing: &Path, imported: &Path) -> Option<String> {
    let (from, from_name) = layer_of(importing)?;
    let (into, into_name) = layer_of(imported)?;
    if into <= from {
        return None;
    }
    // The importing side is named by its layer rather than by its file, since
    // resolution knows which directory wrote the import and not which file in
    // it. The imported side is a file, because that is what was resolved.
    Some(format!(
        "layer: '{from_name}' may not reach '{into_name}': a file in \
         '{from_name}' imports '{}'",
        shown(imported)
    ))
}

// A path as a reader would write it: relative to the layer it sits in, with the
// layer's own name in front, so a diagnostic says `lib/engine/world.frost`
// wherever the build was started from.
fn shown(file: &Path) -> String {
    let resolved = file.canonicalize().unwrap_or_else(|_| file.to_path_buf());
    let held = declared().read().expect("the layer list is readable");
    for (name, directory) in held.iter() {
        if let Ok(rest) = resolved.strip_prefix(directory) {
            return format!("{name}/{}", rest.display()).replace('\\', "/");
        }
    }
    resolved.display().to_string().replace('\\', "/")
}
