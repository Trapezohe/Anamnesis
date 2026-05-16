//! `claude-code` adapter against the shared `anamnesis_core::contract`
//! invariants. Lives as an integration test (separate target) so the
//! contract sees the adapter as an external consumer would.

use std::fs;
use std::path::PathBuf;

use anamnesis_adapter_claude_code::{ClaudeCodeAdapter, ClaudeCodeConfig};
use anamnesis_core::contract::AdapterContract;

fn fixture_root() -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("anamnesis-claude-contract-{nonce}"));
    let proj = root.join("project-abc");
    fs::create_dir_all(proj.join("memory")).unwrap();
    fs::write(
        proj.join("memory").join("user_role.md"),
        "---\nname: senior\ndescription: long-rust\nmetadata:\n  type: user\n---\n\nbody one",
    )
    .unwrap();
    fs::write(
        proj.join("memory").join("feedback_a.md"),
        "---\nname: no-mocks\nmetadata:\n  type: feedback\n---\n\nbody two",
    )
    .unwrap();
    fs::write(proj.join("memory").join("MEMORY.md"), "index").unwrap();
    fs::write(
        proj.join("session-1.jsonl"),
        "{\"role\":\"user\",\"content\":\"hi\"}\n",
    )
    .unwrap();
    root
}

#[tokio::test]
async fn claude_code_satisfies_adapter_contract() {
    let root = fixture_root();
    let contract = AdapterContract::new(|| {
        ClaudeCodeAdapter::new(ClaudeCodeConfig {
            projects_root: root.clone(),
            instance: Some("default".into()),
        })
    });
    contract.run_all().await;
}

/// Same contract, but with `instance = None` to verify the instance-None
/// dedup pitfall (BLUEPRINT §10 #2) is handled correctly.
#[tokio::test]
async fn claude_code_no_instance_still_satisfies_contract() {
    let root = fixture_root();
    let contract = AdapterContract::new(|| {
        ClaudeCodeAdapter::new(ClaudeCodeConfig {
            projects_root: root.clone(),
            instance: None,
        })
    });
    contract.run_all().await;
}
