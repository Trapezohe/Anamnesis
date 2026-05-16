//! `codex` adapter satisfies the shared `anamnesis_core::contract`
//! invariants.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anamnesis_adapter_codex::{codex_adapter, CodexAdapter};
use anamnesis_core::contract::AdapterContract;

static NONCE: AtomicU64 = AtomicU64::new(0);

fn fixture_root() -> PathBuf {
    let n = NONCE.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("anamnesis-codex-contract-{pid}-{n}"));
    fs::create_dir_all(dir.join("sessions")).unwrap();
    fs::write(
        dir.join("sessions/s-1.jsonl"),
        "{\"role\":\"user\",\"content\":\"first\"}\n",
    )
    .unwrap();
    fs::write(
        dir.join("sessions/s-2.json"),
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
    )
    .unwrap();
    dir
}

#[tokio::test]
async fn codex_satisfies_adapter_contract() {
    let root = fixture_root();
    let contract = AdapterContract::new(move || -> CodexAdapter {
        codex_adapter(root.clone(), Some("default"))
    });
    contract.run_all().await;
}

#[tokio::test]
async fn codex_no_instance_satisfies_contract() {
    let root = fixture_root();
    let contract =
        AdapterContract::new(move || -> CodexAdapter { codex_adapter(root.clone(), None) });
    contract.run_all().await;
}
