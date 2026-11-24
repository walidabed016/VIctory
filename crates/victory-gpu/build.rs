//! The translator's fingerprint (`VICTORY_TRANSLATOR`): what the shader
//! cache keys translations by. It covers victory-shader's sources, the
//! dependency versions (`Cargo.lock`) and the cache's own encoding, so a
//! changed translator never finds an old translation, and a build that
//! leaves the translator alone finds the last build's.

use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("set by cargo"));
    let shader = manifest.join("../victory-shader");
    let inputs = [
        shader.join("Cargo.toml"),
        shader.join("src"),
        manifest.join("../../Cargo.lock"),
        manifest.join("src/shader_cache.rs"),
    ];
    let mut files = Vec::new();
    for input in &inputs {
        println!("cargo:rerun-if-changed={}", input.display());
        walk(input, &mut files);
    }
    files.sort();
    let mut h = blake3::Hasher::new();
    for file in &files {
        let name = file
            .strip_prefix(&manifest)
            .unwrap_or(file)
            .to_string_lossy()
            .replace('\\', "/");
        let bytes = std::fs::read(file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
        for part in [name.as_bytes(), &bytes] {
            h.update(&(part.len() as u64).to_le_bytes());
            h.update(part);
        }
    }
    println!(
        "cargo:rustc-env=VICTORY_TRANSLATOR={}",
        h.finalize().to_hex()
    );
}

fn walk(path: &Path, files: &mut Vec<PathBuf>) {
    if path.is_dir() {
        for entry in std::fs::read_dir(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())) {
            walk(&entry.expect("a directory entry").path(), files);
        }
    } else {
        files.push(path.to_owned());
    }
}
