//! Round-70 `anamnesis eval-quality` end-to-end.
//!
//! Two things this checks that the unit-tests in `anamnesis-search`
//! can't:
//!
//!   1. The CLI subcommand correctly threads JSONL judgments through
//!      the *production* `HybridSearcher::search_filtered` + `pack`
//!      pipeline — no per-test shortcut.
//!   2. `--min-mrr` and `--min-ndcg` actually gate the process exit
//!      code, so CI can use this as a pass/fail check on retrieval
//!      regressions.
//!
//! Pure-fulltext mode (the CLI's default) so the test never needs to
//! download a fastembed model.

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;

fn cli() -> Command {
    Command::cargo_bin("anamnesis").expect("cargo bin")
}

fn tmp_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

/// Lay down a minimal claude-code-shaped fixture with three records,
/// each containing a unique marker phrase that the FTS path can lock
/// onto. The fixture is intentionally small — enough rows to make
/// MRR / nDCG non-trivial without dragging in real corpora.
fn seed_fixture(home: &Path) {
    let proj = home.join(".claude").join("projects").join("eval-proj");
    let memdir = proj.join("memory");
    fs::create_dir_all(&memdir).unwrap();
    fs::write(
        memdir.join("alpha.md"),
        "---\n\
         name: alpha-record\n\
         description: anchored on the marker crocodileTeaRocket\n\
         metadata:\n  type: user\n\
         ---\n\n\
         A user fact about preferring the marker crocodileTeaRocket.\n",
    )
    .unwrap();
    fs::write(
        memdir.join("bravo.md"),
        "---\n\
         name: bravo-record\n\
         description: distinct marker platypusBanjoComet only\n\
         metadata:\n  type: user\n\
         ---\n\n\
         A fact for the marker platypusBanjoComet only — unrelated to alpha.\n",
    )
    .unwrap();
    fs::write(
        memdir.join("charlie.md"),
        "---\n\
         name: charlie-record\n\
         description: third anchor only the wombatFlute query should hit\n\
         metadata:\n  type: user\n\
         ---\n\n\
         A third memory about the marker wombatFlute. Nothing else here.\n",
    )
    .unwrap();
}

/// Build the judgments JSONL using the *real* `native_id` shape the
/// claude-code adapter synthesises (`{instance}|memory|{abs_path}`).
/// Anchoring on the natural-key match path is deliberate — it's the
/// curator-friendly mode and the path we want covered.
fn judgments_jsonl(home: &Path) -> String {
    let proj = home.join(".claude").join("projects").join("eval-proj");
    let memdir = proj.join("memory");
    let alpha = format!("default|memory|{}", memdir.join("alpha.md").display());
    let bravo = format!("default|memory|{}", memdir.join("bravo.md").display());
    let charlie = format!("default|memory|{}", memdir.join("charlie.md").display());
    // Omit `instance` on both sides — the store normalises the default
    // instance to `""` and the JudgedRecordRef matcher does the same on
    // `None`, so they meet at "" naturally. Adding `"instance": "default"`
    // here would *create* a mismatch.
    //
    // Build the JSON via `serde_json::json!` so Windows-style native_ids
    // (which contain backslashes from `path::Display`) get properly
    // escaped — hand-formatting blew up on the Windows CI runner.
    let line = |id: &str, q: &str, nid: &str| {
        let v = serde_json::json!({
            "id": id,
            "query": q,
            "relevant": [{
                "adapter": "claude-code",
                "native_id": nid,
                "grade": 3,
            }],
        });
        format!("{}\n", serde_json::to_string(&v).unwrap())
    };
    let mut out = String::new();
    out.push_str(&line("q-alpha", "crocodileTeaRocket", &alpha));
    out.push_str(&line("q-bravo", "platypusBanjoComet", &bravo));
    out.push_str(&line("q-charlie", "wombatFlute", &charlie));
    out
}

/// Set up: HOME with the three claude-code records, data dir
/// initialised + claude-code source registered + imported.
fn arrange() -> (tempfile::TempDir, tempfile::TempDir, std::path::PathBuf) {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "source",
            "add",
            "claude-code",
            "--instance",
            "default",
            "--path",
        ])
        .arg(home.path().join(".claude").join("projects"))
        .assert()
        .success();
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["import", "claude-code", "--no-embed"])
        .assert()
        .success();

    let judg_path = data.path().join("judgments.jsonl");
    fs::write(&judg_path, judgments_jsonl(home.path())).unwrap();
    (home, data, judg_path)
}

#[test]
fn eval_quality_passes_when_thresholds_met() {
    let (home, data, judg) = arrange();
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["eval-quality"])
        .arg("--judgments")
        .arg(&judg)
        .args([
            "--mode",
            "fulltext",
            "--min-mrr",
            "0.99",
            "--min-ndcg",
            "0.99",
            "--json",
        ])
        .assert()
        .success()
        .stdout(
            contains("\"queries\": 3")
                .and(contains("\"mrr_at_k\""))
                .and(contains("\"ndcg_at_k\""))
                .and(contains("\"failed_thresholds\": []")),
        );
}

#[test]
fn eval_quality_fails_when_mrr_threshold_unreachable() {
    let (home, data, judg) = arrange();
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["eval-quality"])
        .arg("--judgments")
        .arg(&judg)
        .args(["--mode", "fulltext", "--min-mrr", "1.01"])
        .assert()
        .failure()
        .stderr(contains("quality below threshold"));
}

#[test]
fn eval_quality_errors_on_empty_judgments_file() {
    let (home, data, _judg) = arrange();
    let empty = data.path().join("empty.jsonl");
    fs::write(&empty, "").unwrap();
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["eval-quality"])
        .arg("--judgments")
        .arg(&empty)
        .args(["--mode", "fulltext"])
        .assert()
        .failure()
        .stderr(contains("no queries"));
}
