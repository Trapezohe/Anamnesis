//! `openviking` adapter satisfies the shared `anamnesis_core::contract`
//! invariants. Mirrors the pattern from `adapter-codex` / `adapter-mem0`.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anamnesis_adapter_openviking::{openviking_adapter, OpenVikingAdapter};
use anamnesis_core::contract::AdapterContract;

static NONCE: AtomicU64 = AtomicU64::new(0);

fn fixture_workspace() -> PathBuf {
    let n = NONCE.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let workspace = std::env::temp_dir().join(format!("anamnesis-openviking-contract-{pid}-{n}"));
    let acct = workspace.join("local/acct-1");

    // Resource scope.
    fs::create_dir_all(acct.join("resources/docs/auth")).unwrap();
    fs::write(acct.join("resources/docs/auth/.abstract.md"), "abs").unwrap();
    fs::write(acct.join("resources/docs/auth/.overview.md"), "over").unwrap();
    fs::write(acct.join("resources/docs/auth/oauth.md"), "oauth body").unwrap();
    fs::write(acct.join("resources/docs/auth/.relations.json"), "{}").unwrap();

    // User memories.
    fs::create_dir_all(acct.join("user/u-1/memories/preferences")).unwrap();
    fs::create_dir_all(acct.join("user/u-1/memories/entities")).unwrap();
    fs::create_dir_all(acct.join("user/u-1/memories/events")).unwrap();
    fs::write(acct.join("user/u-1/memories/profile.md"), "profile").unwrap();
    fs::write(
        acct.join("user/u-1/memories/preferences/coding.md"),
        "uses rust",
    )
    .unwrap();
    fs::write(
        acct.join("user/u-1/memories/entities/alice.md"),
        "alice is pm",
    )
    .unwrap();
    fs::write(
        acct.join("user/u-1/memories/events/2026-05-01.md"),
        "shipped v1",
    )
    .unwrap();

    // Agent memories + skills + instructions.
    fs::create_dir_all(acct.join("agent/a-1/memories/cases")).unwrap();
    fs::create_dir_all(acct.join("agent/a-1/memories/patterns")).unwrap();
    fs::create_dir_all(acct.join("agent/a-1/skills/search-web")).unwrap();
    fs::create_dir_all(acct.join("agent/a-1/instructions")).unwrap();
    fs::write(
        acct.join("agent/a-1/memories/cases/c-1.md"),
        "fixed auth bug",
    )
    .unwrap();
    fs::write(
        acct.join("agent/a-1/memories/patterns/p-1.md"),
        "retry-on-5xx",
    )
    .unwrap();
    fs::write(
        acct.join("agent/a-1/skills/search-web/SKILL.md"),
        "search overview",
    )
    .unwrap();
    fs::write(
        acct.join("agent/a-1/skills/search-web/.abstract.md"),
        "search abs",
    )
    .unwrap();
    fs::write(acct.join("agent/a-1/instructions/system.md"), "be helpful").unwrap();

    // Session.
    fs::create_dir_all(acct.join("session/s-1")).unwrap();
    fs::write(acct.join("session/s-1/.abstract.md"), "sess abs").unwrap();
    fs::write(acct.join("session/s-1/.overview.md"), "sess over").unwrap();
    fs::write(acct.join("session/s-1/.meta.json"), "{}").unwrap();
    fs::write(
        acct.join("session/s-1/messages.jsonl"),
        "{\"id\":\"m1\",\"role\":\"user\",\"parts\":[{\"type\":\"text\",\"text\":\"hi\"}]}\n\
         {\"id\":\"m2\",\"role\":\"assistant\",\"parts\":[{\"type\":\"text\",\"text\":\"hello\"}]}\n",
    )
    .unwrap();

    workspace
}

#[tokio::test]
async fn openviking_satisfies_adapter_contract() {
    let root = fixture_workspace();
    let contract = AdapterContract::new(move || -> OpenVikingAdapter {
        openviking_adapter(root.clone(), Some("default"))
    });
    contract.run_all().await;
}

#[tokio::test]
async fn openviking_no_instance_satisfies_contract() {
    let root = fixture_workspace();
    let contract = AdapterContract::new(move || -> OpenVikingAdapter {
        openviking_adapter(root.clone(), None)
    });
    contract.run_all().await;
}
