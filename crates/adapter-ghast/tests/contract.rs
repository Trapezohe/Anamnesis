//! `ghast` adapter satisfies the shared `anamnesis_core::contract`
//! invariants. Mirrors the pattern from `adapter-codex` / `adapter-mem0`.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anamnesis_adapter_ghast::{ghast_adapter, GhastAdapter};
use anamnesis_core::contract::AdapterContract;

static NONCE: AtomicU64 = AtomicU64::new(0);

fn fixture_root() -> PathBuf {
    let n = NONCE.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let root = std::env::temp_dir().join(format!("anamnesis-ghast-contract-{pid}-{n}"));
    fs::create_dir_all(root.join("prompts/coding")).unwrap();
    fs::create_dir_all(root.join("resources/bundled-skills/memory-management")).unwrap();
    fs::write(
        root.join("prompts/coding/default.md"),
        "default coding prompt",
    )
    .unwrap();
    fs::write(
        root.join("resources/bundled-skills/memory-management/SKILL.md"),
        "skill body",
    )
    .unwrap();
    fs::write(
        root.join("resources/bundled-skills/memory-management/REFERENCES.md"),
        "refs",
    )
    .unwrap();
    root
}

#[tokio::test]
async fn ghast_satisfies_adapter_contract() {
    let root = fixture_root();
    let contract = AdapterContract::new(move || -> GhastAdapter {
        ghast_adapter(root.clone(), Some("default"))
    });
    contract.run_all().await;
}

#[tokio::test]
async fn ghast_no_instance_satisfies_contract() {
    let root = fixture_root();
    let contract =
        AdapterContract::new(move || -> GhastAdapter { ghast_adapter(root.clone(), None) });
    contract.run_all().await;
}
