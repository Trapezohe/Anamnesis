//! `memos` adapter satisfies the shared `anamnesis_core::contract`
//! invariants. Mirrors the pattern from `adapter-codex` / `adapter-mem0`.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anamnesis_adapter_memos::{memos_adapter, MemosAdapter};
use anamnesis_core::contract::AdapterContract;

static NONCE: AtomicU64 = AtomicU64::new(0);

fn fixture_root() -> PathBuf {
    let n = NONCE.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let root = std::env::temp_dir().join(format!("anamnesis-memos-contract-{pid}-{n}"));
    let cube = root.join("cube-1");
    fs::create_dir_all(&cube).unwrap();
    let payload = serde_json::json!([
        {
            "id": "i-1",
            "memory": "user prefers Rust",
            "metadata": {
                "memory_type": "UserMemory",
                "user_id": "u-1",
                "session_id": "s-1",
                "source": "conversation",
                "status": "activated",
                "updated_at": "2026-05-01T10:00:00"
            }
        },
        {
            "id": "i-2",
            "memory": "Paris is the capital",
            "metadata": {
                "memory_type": "LongTermMemory",
                "status": "activated",
                "updated_at": "2026-05-02T10:00:00"
            }
        }
    ]);
    fs::write(cube.join("textual_memory.json"), payload.to_string()).unwrap();
    root
}

#[tokio::test]
async fn memos_satisfies_adapter_contract() {
    let root = fixture_root();
    let contract = AdapterContract::new(move || -> MemosAdapter {
        memos_adapter(root.clone(), Some("default"))
    });
    contract.run_all().await;
}

#[tokio::test]
async fn memos_no_instance_satisfies_contract() {
    let root = fixture_root();
    let contract =
        AdapterContract::new(move || -> MemosAdapter { memos_adapter(root.clone(), None) });
    contract.run_all().await;
}
