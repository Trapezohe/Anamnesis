//! `openclaw` adapter satisfies the shared `anamnesis_core::contract`
//! invariants. Mirrors the pattern from `adapter-codex` / `adapter-mem0`.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anamnesis_adapter_openclaw::{openclaw_adapter, OpenClawAdapter};
use anamnesis_core::contract::AdapterContract;

static NONCE: AtomicU64 = AtomicU64::new(0);

fn fixture_root() -> PathBuf {
    let n = NONCE.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("anamnesis-openclaw-contract-{pid}-{n}"));
    let ws = dir.join("workspace");
    let skills = ws.join("skills");
    let sess = ws.join("sessions");
    fs::create_dir_all(skills.join("write-code")).unwrap();
    fs::create_dir_all(&sess).unwrap();
    fs::write(dir.join("openclaw.json"), "{}").unwrap();
    fs::write(ws.join("AGENTS.md"), "agents config").unwrap();
    fs::write(ws.join("SOUL.md"), "system persona").unwrap();
    fs::write(skills.join("write-code/SKILL.md"), "produce rust").unwrap();
    fs::write(sess.join("a.jsonl"), "{\"k\":1}\n").unwrap();
    dir
}

#[tokio::test]
async fn openclaw_satisfies_adapter_contract() {
    let root = fixture_root();
    let contract = AdapterContract::new(move || -> OpenClawAdapter {
        openclaw_adapter(root.clone(), Some("default"))
    });
    contract.run_all().await;
}

#[tokio::test]
async fn openclaw_no_instance_satisfies_contract() {
    let root = fixture_root();
    let contract =
        AdapterContract::new(move || -> OpenClawAdapter { openclaw_adapter(root.clone(), None) });
    contract.run_all().await;
}
