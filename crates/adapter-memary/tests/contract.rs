//! `memary` adapter satisfies the shared `anamnesis_core::contract`
//! invariants. Mirrors the pattern from `adapter-codex` / `adapter-mem0`.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anamnesis_adapter_memary::{memary_adapter, MemaryAdapter};
use anamnesis_core::contract::AdapterContract;

static NONCE: AtomicU64 = AtomicU64::new(0);

fn fixture_root() -> PathBuf {
    let n = NONCE.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("anamnesis-memary-contract-{pid}-{n}"));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("memory_stream.json"),
        r#"[{"entity":"Alice","date":"2026-05-01T10:00:00"},{"entity":"Paris","date":"2026-05-01T10:05:00"}]"#,
    )
    .unwrap();
    fs::write(
        dir.join("entity_knowledge_store.json"),
        r#"[{"entity":"Alice","count":3,"date":"2026-05-01T10:05:00"}]"#,
    )
    .unwrap();
    fs::write(
        dir.join("past_chat.json"),
        r#"[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]"#,
    )
    .unwrap();
    fs::write(dir.join("system_persona.txt"), "I am a helpful agent.").unwrap();
    fs::write(dir.join("user_persona.txt"), "user is a senior eng").unwrap();
    dir
}

#[tokio::test]
async fn memary_satisfies_adapter_contract() {
    let root = fixture_root();
    let contract = AdapterContract::new(move || -> MemaryAdapter {
        memary_adapter(root.clone(), Some("default"))
    });
    contract.run_all().await;
}

#[tokio::test]
async fn memary_no_instance_satisfies_contract() {
    let root = fixture_root();
    let contract =
        AdapterContract::new(move || -> MemaryAdapter { memary_adapter(root.clone(), None) });
    contract.run_all().await;
}
