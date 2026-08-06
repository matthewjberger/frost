use std::path::{Path, PathBuf};

// One layer of a project: the name the manifest gave it and the directory that
// name resolves to. The order they are declared in is the order they may reach,
// so a layer's index is its rank and comparing two ranks is the whole rule.
//
// Carried as a slice beside the search roots, which is the other list an import
// is resolved against, and for the same reason: what resolution is allowed to
// use travels with the call rather than sitting somewhere the call can reach.
#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub directory: PathBuf,
}

impl Layer {
    pub fn new(name: &str, directory: &Path) -> Self {
        Layer {
            name: name.to_string(),
            directory: directory
                .canonicalize()
                .unwrap_or_else(|_| directory.to_path_buf()),
        }
    }
}

// Which layer a file is in, or none. The deepest declared layer holding it
// wins, so a layer declared inside another answers for the files under it.
fn layer_of(layers: &[Layer], file: &Path) -> Option<usize> {
    if layers.is_empty() {
        return None;
    }
    let resolved = file.canonicalize().unwrap_or_else(|_| file.to_path_buf());
    layers
        .iter()
        .enumerate()
        .filter(|(_, layer)| resolved.starts_with(&layer.directory))
        .max_by_key(|(_, layer)| layer.directory.components().count())
        .map(|(index, _)| index)
}

/// Whether an import reaches from one layer into a later one, and what to say
/// about it.
///
/// The paths are the ones resolution settled on, so `../engine/../renderer/x`
/// has already become the file it names and cannot spell its way past this.
/// Where either file is under no declared layer there is nothing to compare, so
/// the import stands.
pub fn reaching_upward(
    layers: &[Layer],
    importing: &Path,
    imported: &Path,
) -> Option<String> {
    let from = layer_of(layers, importing)?;
    let into = layer_of(layers, imported)?;
    if into <= from {
        return None;
    }
    // The importing side is named by its layer rather than by its file, since
    // resolution knows which directory wrote the import and not which file in
    // it. The imported side is a file, because that is what was resolved.
    Some(format!(
        "layer: '{}' may not reach '{}': a file in '{}' imports '{}'",
        layers[from].name,
        layers[into].name,
        layers[from].name,
        shown(layers, imported)
    ))
}

// A path as a reader would write it: the layer's own name, then the rest of the
// path under it, so a diagnostic says `lib/engine/world.frost` wherever the
// build was started from.
fn shown(layers: &[Layer], file: &Path) -> String {
    let resolved = file.canonicalize().unwrap_or_else(|_| file.to_path_buf());
    for layer in layers {
        if let Ok(rest) = resolved.strip_prefix(&layer.directory) {
            return format!("{}/{}", layer.name, rest.display())
                .replace('\\', "/");
        }
    }
    resolved.display().to_string().replace('\\', "/")
}
