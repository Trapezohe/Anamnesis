//! `anamnesis import <target> --reconcile-export-*` post-import hook —
//! R151 derive-format behaviour. The hook is always bucket=only-left
//! (imported side = left), so the lagging adapter fed the export is
//! always `--reconcile-export-against`. Parallels the standalone
//! `reconcile-export` parity in reconcile_export.rs.

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use predicates::str::contains;

fn cli() -> Command {
    Command::cargo_bin("anamnesis").expect("cargo bin")
}

fn seed_claude_fixture(home: &Path) {
    let proj = home.join(".claude").join("projects").join("proj");
    fs::create_dir_all(proj.join("memory")).unwrap();
    fs::write(
        proj.join("memory").join("note.md"),
        "---\nname: a-note\ndescription: a note\nmetadata:\n  type: user\n---\n\nA distinctive imported memory body.\n",
    )
    .unwrap();
}

fn init_and_seed(home: &Path, data: &Path) {
    seed_claude_fixture(home);
    cli()
        .env("HOME", home)
        .env("ANAMNESIS_DATA_DIR", data)
        .args(["init"])
        .assert()
        .success();
}

#[test]
fn derives_against_format_when_omitted() {
    let home = tempfile::tempdir().unwrap();
    let data = tempfile::tempdir().unwrap();
    init_and_seed(home.path(), data.path());
    let out = data.path().join("drift.db");
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "import",
            "claude-code",
            "--no-embed",
            "--reconcile-export-against",
            "mem0",
            "--reconcile-export-out",
            out.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(contains("lagging=mem0"))
        .stdout(contains("format=mem0-sqlite [derived]"));
    assert!(out.is_file());
}

#[test]
fn explicit_mismatch_against_succeeds_with_warning() {
    let home = tempfile::tempdir().unwrap();
    let data = tempfile::tempdir().unwrap();
    init_and_seed(home.path(), data.path());
    let out = data.path().join("forced.jsonl");
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "import",
            "claude-code",
            "--no-embed",
            "--reconcile-export-against",
            "mem0",
            "--reconcile-export-out",
            out.to_str().unwrap(),
            "--reconcile-export-format",
            "jsonl",
        ])
        .assert()
        .success()
        .stderr(contains("differs"))
        .stdout(contains("format=jsonl [explicit]"));
}

#[test]
fn omitted_format_errors_before_import_when_against_has_no_target() {
    let home = tempfile::tempdir().unwrap();
    let data = tempfile::tempdir().unwrap();
    init_and_seed(home.path(), data.path());
    let out = data.path().join("nope.db");
    // against=claude-code has no round-trip target → preflight error.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "import",
            "claude-code",
            "--no-embed",
            "--reconcile-export-against",
            "claude-code",
            "--reconcile-export-out",
            out.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("no round-trip export format"));
    assert!(!out.exists());
    // The import must not have committed.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["status"])
        .assert()
        .success()
        .stdout(contains("records         : 0"));
}

#[test]
fn reconcile_export_with_dry_run_is_rejected() {
    let home = tempfile::tempdir().unwrap();
    let data = tempfile::tempdir().unwrap();
    init_and_seed(home.path(), data.path());
    let out = data.path().join("nope.db");
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "import",
            "claude-code",
            "--no-embed",
            "--dry-run",
            "--reconcile-export-against",
            "mem0",
            "--reconcile-export-out",
            out.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("incompatible with --dry-run"));
    assert!(!out.exists());
}

#[test]
fn format_without_against_is_rejected_by_clap() {
    let home = tempfile::tempdir().unwrap();
    let data = tempfile::tempdir().unwrap();
    init_and_seed(home.path(), data.path());
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "import",
            "claude-code",
            "--no-embed",
            "--reconcile-export-format",
            "jsonl",
        ])
        .assert()
        .failure();
}
