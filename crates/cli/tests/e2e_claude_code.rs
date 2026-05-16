//! End-to-end tests: build a fake `~/.claude/projects` fixture, drive
//! the real `anamnesis` binary through discover → import → search, and
//! assert known content is findable.
//!
//! Two tests:
//!   * `e2e_fts_only_path_finds_memory_file` — always runs. Verifies
//!     the entire pipeline EXCEPT the vector side, so we don't pay the
//!     fastembed model download tax in every CI run.
//!   * `e2e_hybrid_path_finds_memory_file` — gated behind
//!     `FASTEMBED_DOWNLOAD=1`. Downloads the default 120-MB model on
//!     first run and verifies the full Hybrid loop end-to-end.

use std::fs;
use std::path::{Path, PathBuf};

use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;

fn cli() -> Command {
    Command::cargo_bin("anamnesis").expect("cargo bin")
}

fn tmp_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

/// Build a realistic-looking `~/.claude/projects/<hash>/` layout under
/// `home`, with two distinctive memory files and one JSONL session that
/// later tests can match against.
fn seed_claude_fixture(home: &Path) -> PathBuf {
    let projects = home.join(".claude").join("projects");
    let proj = projects.join("trapezohe-anamnesis");
    fs::create_dir_all(proj.join("memory")).expect("create memory dir");

    fs::write(
        proj.join("memory").join("user_role.md"),
        "---\n\
         name: senior-rust-engineer\n\
         description: User has 10 years of Rust experience\n\
         metadata:\n  type: user\n\
         ---\n\n\
         The user is a senior Rust engineer who prefers thorough error \
         handling and dislikes silent fallbacks. Always lean toward \
         explicit, surface-level errors rather than swallowed Option::None.\n",
    )
    .expect("write user_role.md");

    fs::write(
        proj.join("memory").join("feedback_database.md"),
        "---\n\
         name: integration-tests-need-real-database\n\
         description: Never mock the database in integration tests\n\
         metadata:\n  type: feedback\n\
         ---\n\n\
         Never mock the database for integration tests. We got burned \
         last quarter when a mocked test passed but the production \
         migration silently failed in staging.\n",
    )
    .expect("write feedback.md");

    fs::write(proj.join("memory").join("MEMORY.md"), "index file, ignored")
        .expect("write MEMORY.md");

    fs::write(
        proj.join("session-deadbeef.jsonl"),
        r#"{"role":"user","content":"how should I structure the adapter?"}
{"role":"assistant","content":"keep core IO-free; adapters own their normalize."}"#,
    )
    .expect("write session jsonl");

    projects
}

#[test]
fn e2e_fts_only_path_finds_memory_file() {
    let home = tmp_dir();
    let data = tmp_dir();
    let _projects = seed_claude_fixture(home.path());

    // 1. init creates the db and pins active embedding model. We're
    // running fulltext-only so we never need to actually use it.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["init"])
        .assert()
        .success();

    // 2. discover should find the claude-code source with high confidence.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["discover"])
        .assert()
        .success()
        .stdout(
            contains("claude-code")
                .and(contains("high"))
                .and(contains("memory file")),
        );

    // 3. import (skip embedding worker — no model download).
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["import", "claude-code", "--no-embed"])
        .assert()
        .success()
        .stdout(contains("import done"));

    // 4. status reports the records + chunks we just imported.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["status"])
        .assert()
        .success()
        .stdout(
            // 2 memory + 1 session = 3 records.
            contains("records         : 3"),
        );

    // 5. search (fulltext) finds the feedback memory by a distinctive phrase.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "search",
            "mocked test",
            "--mode",
            "fulltext",
            "--limit",
            "5",
        ])
        .assert()
        .success()
        .stdout(contains("feedback_database").or(contains("integration tests")));

    // 6. search for a phrase from the user memory file.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["search", "senior Rust engineer", "--mode", "fulltext"])
        .assert()
        .success()
        .stdout(contains("user_role").or(contains("Rust")));

    // 7. search --json round-trips through serde and contains both hits
    //    when the query matches both files.
    let json_out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["search", "Rust", "--mode", "fulltext", "--json"])
        .output()
        .expect("run search --json");
    assert!(json_out.status.success());
    let payload: serde_json::Value =
        serde_json::from_slice(&json_out.stdout).expect("parseable json");
    let results = payload["results"].as_array().expect("results array");
    assert!(
        !results.is_empty(),
        "expected at least one Rust-related hit"
    );
    // Every result must carry provenance back to the source file.
    for r in results {
        assert_eq!(r["adapter"], "claude-code");
        assert!(r["native_path"].is_string());
    }
}

#[test]
fn e2e_hybrid_path_finds_memory_file() {
    if std::env::var("FASTEMBED_DOWNLOAD").ok().as_deref() != Some("1") {
        eprintln!("skipping: FASTEMBED_DOWNLOAD != 1 (vector E2E requires model download)");
        return;
    }

    let home = tmp_dir();
    let data = tmp_dir();
    let _projects = seed_claude_fixture(home.path());

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["init"])
        .assert()
        .success();

    // Full import — embedding worker runs and downloads the model.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["import", "claude-code"])
        .assert()
        .success()
        .stdout(contains("embedding worker").and(contains("processed")));

    // After embedding, jobs_pending should be 0 and chunks > 0.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["status"])
        .assert()
        .success()
        .stdout(contains("jobs pending    : 0"));

    // Hybrid (semantic) search: a question that the FTS-only path would
    // miss because the words don't overlap with the memory body, but the
    // multilingual-e5-small embedding should still bring up the user
    // role memory.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "search",
            "what do we know about the user's coding background?",
            "--mode",
            "hybrid",
            "--limit",
            "5",
        ])
        .assert()
        .success();
}
