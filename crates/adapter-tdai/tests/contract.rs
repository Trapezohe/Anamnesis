//! `tdai` adapter satisfies the shared `anamnesis_core::contract`
//! invariants. Mirrors the pattern from `adapter-codex` / `adapter-mem0`.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anamnesis_adapter_tdai::{tdai_adapter, TdaiAdapter};
use anamnesis_core::contract::AdapterContract;

static NONCE: AtomicU64 = AtomicU64::new(0);

fn fixture_root() -> PathBuf {
    let n = NONCE.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("anamnesis-tdai-contract-{pid}-{n}"));
    fs::create_dir_all(dir.join("refs")).unwrap();
    fs::write(dir.join("persona.md"), "I am a senior engineer.").unwrap();
    fs::write(dir.join("refs/conv-1.md"), "raw conversation").unwrap();
    fs::write(
        dir.join("facts.jsonl"),
        "{\"f\":\"likes rust\"}\n{\"f\":\"hates mocks\"}\n",
    )
    .unwrap();
    fs::write(dir.join("scenario.md"), "scenario body").unwrap();
    dir
}

#[tokio::test]
async fn tdai_satisfies_adapter_contract() {
    let root = fixture_root();
    let contract = AdapterContract::new(move || -> TdaiAdapter {
        tdai_adapter(root.clone(), Some("default"))
    });
    contract.run_all().await;
}

#[tokio::test]
async fn tdai_no_instance_satisfies_contract() {
    let root = fixture_root();
    let contract =
        AdapterContract::new(move || -> TdaiAdapter { tdai_adapter(root.clone(), None) });
    contract.run_all().await;
}
