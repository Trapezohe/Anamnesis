//! `anamnesis watch` — install-and-auto-sync daemon (R151, PR 1 of N).
//!
//! Long-running foreground process that keeps the local store current
//! with the user's memory frameworks without re-running `import` by hand.
//! Watches every registered source's filesystem location; a change
//! debounces into one incremental import (the same pipeline `import`
//! drives). Operator backgrounds it via nohup / systemd / launchd; PR 2
//! wires "auto-start on install".
//!
//! ## Layering (keeps CI non-flaky)
//!
//! The decision logic is split out as pure, clock-free state machines —
//! [`DebouncePlanner`] (collapse bursts → one import) and [`PathRouter`]
//! (map a changed path back to its source). They're unit-tested with
//! injected events + caller-supplied `Instant`s, never a real `notify`
//! watcher (macOS FSEvents coalescing / Windows latency would make
//! event-driven tests flaky). The thin IO layer ([`run_watch`]) wires a
//! real watcher into those machines and is covered by cross-platform
//! compilation, not by unit tests.
//!
//! ## SQLite sources
//!
//! mem0 / letta write through `-wal` / `-shm` sidecars and atomic
//! renames, so watching the `.db` file directly misses writes. We watch
//! the **parent directory** and re-scan on any change; the importer's
//! `raw_hash` dedup turns unchanged rows into no-op upserts.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anamnesis_core::adapter::ScanOpts;
use anamnesis_store::Store;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};

/// Quiet window: a source is imported once no further change arrives for
/// this long. Collapses editor save-storms / SQLite WAL churn into one run.
const DEBOUNCE_WINDOW: Duration = Duration::from_secs(2);

/// Incremental imports rewind `since` by this much before `last_import_at`
/// so deltas written in the same wall-clock second as the previous import
/// are never skipped. Re-imported rows are no-op upserts (raw_hash dedup),
/// so the overlap is free correctness insurance.
const SINCE_OVERLAP_SECS: i64 = 5;

/// Identifies a registered source: `(adapter, instance)`. Empty instance
/// is the default instance (mirrors `SourceRow.instance`).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SourceKey {
    /// Adapter id (`"mem0"`, `"claude-code"`, …).
    pub adapter: String,
    /// Instance discriminator; `""` for the default instance.
    pub instance: String,
}

impl SourceKey {
    /// Human label: `adapter` or `adapter:instance`.
    pub fn label(&self) -> String {
        if self.instance.is_empty() {
            self.adapter.clone()
        } else {
            format!("{}:{}", self.adapter, self.instance)
        }
    }
}

/// Collapse a burst of change events per source into one "import due"
/// decision once the quiet window elapses. Pure: no clock, no fs, no
/// store — the caller supplies `now` so the machine is deterministic.
#[derive(Debug)]
pub struct DebouncePlanner {
    window: Duration,
    /// source → deadline (`last_event + window`). Each new event pushes
    /// the deadline out; `take_due` fires once `now >= deadline`.
    pending: HashMap<SourceKey, Instant>,
}

impl DebouncePlanner {
    /// New planner with the given quiet window.
    pub fn new(window: Duration) -> Self {
        Self {
            window,
            pending: HashMap::new(),
        }
    }

    /// Record a change event for `source` observed at `now`. Resets the
    /// source's quiet-window deadline.
    pub fn observe(&mut self, source: SourceKey, now: Instant) {
        self.pending.insert(source, now + self.window);
    }

    /// Drain every source whose quiet window has elapsed by `now`.
    /// Returns them sorted by label for deterministic ordering.
    pub fn take_due(&mut self, now: Instant) -> Vec<SourceKey> {
        let mut due: Vec<SourceKey> = self
            .pending
            .iter()
            .filter(|(_, &deadline)| now >= deadline)
            .map(|(k, _)| k.clone())
            .collect();
        for k in &due {
            self.pending.remove(k);
        }
        due.sort_by_key(|k| k.label());
        due
    }

    /// Earliest pending deadline — for sleeping until the next import is
    /// due. `None` when nothing is pending.
    pub fn next_deadline(&self) -> Option<Instant> {
        self.pending.values().copied().min()
    }

    /// How many sources are currently waiting out their quiet window.
    /// Test-only introspection (the daemon loop reads `next_deadline`).
    #[cfg(test)]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

/// Map a changed filesystem path back to the source whose watched root
/// contains it. Longest-prefix wins (so a nested source root takes
/// priority over a broader one). Pure.
#[derive(Debug, Default)]
pub struct PathRouter {
    /// `(canonical watched root, source)`, longest root first.
    roots: Vec<(PathBuf, SourceKey)>,
}

impl PathRouter {
    /// Register a watched root for a source. Re-sorts longest-first.
    pub fn insert(&mut self, root: PathBuf, source: SourceKey) {
        self.roots.push((root, source));
        // Longest path component-count first → most specific match wins.
        self.roots
            .sort_by_key(|(p, _)| std::cmp::Reverse(p.components().count()));
    }

    /// Route a changed path to its source, if any watched root is a
    /// prefix of (or equal to) it.
    pub fn route(&self, changed: &Path) -> Option<&SourceKey> {
        self.roots
            .iter()
            .find(|(root, _)| changed == root || changed.starts_with(root))
            .map(|(_, key)| key)
    }

    /// All distinct watched roots (for arming the fs watcher).
    pub fn roots(&self) -> impl Iterator<Item = &Path> {
        self.roots.iter().map(|(p, _)| p.as_path())
    }

    /// True when no roots are registered (nothing to watch).
    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }
}

/// The directory a source's location should be *watched* at. Files
/// (mem0/letta `.db`) watch their parent dir (WAL/rename safety);
/// directories watch themselves.
fn watch_root_for(location: &Path) -> PathBuf {
    if location.is_dir() {
        location.to_path_buf()
    } else {
        location
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| location.to_path_buf())
    }
}

/// `since` bound for a watch-triggered incremental import: the source's
/// `last_import_at` rewound by [`SINCE_OVERLAP_SECS`]. `None` (full
/// catch-up) when the source has never imported.
fn incremental_since(last_import_at: Option<i64>) -> Option<DateTime<Utc>> {
    let ts = last_import_at?;
    DateTime::<Utc>::from_timestamp((ts - SINCE_OVERLAP_SECS).max(0), 0)
}

/// fs-watchable sources only — `generic-mcp` (URL) has no local path and
/// is deferred to PR 2's interval poller.
fn is_fs_watchable(adapter: &str) -> bool {
    adapter != anamnesis_adapter_generic_mcp::ADAPTER_ID
}

/// Build the [`PathRouter`] from the registered sources, skipping URL
/// adapters and sources whose location doesn't exist on disk.
fn build_router(sources: &[anamnesis_store::SourceRow]) -> PathRouter {
    let mut router = PathRouter::default();
    for s in sources {
        if !is_fs_watchable(&s.adapter) {
            continue;
        }
        let Some(loc) = s.location.as_deref() else {
            continue;
        };
        let path = PathBuf::from(loc);
        if !path.exists() {
            continue;
        }
        router.insert(
            watch_root_for(&path),
            SourceKey {
                adapter: s.adapter.clone(),
                instance: s.instance.clone(),
            },
        );
    }
    router
}

/// Run an incremental import for one source, reusing the full
/// normalize → chunk → index → embed pipeline via `ImportService`.
///
/// Mirrors `cmd_import`'s adapter construction (kept in sync by hand;
/// PR 2+ can extract a shared `build_adapter` helper). URL adapters are
/// rejected — they never reach here (filtered by [`is_fs_watchable`]).
async fn import_one_source(
    data_dir: &Path,
    key: &SourceKey,
    location: &Path,
    since: Option<DateTime<Utc>>,
) -> Result<anamnesis_importer::ImportSummary> {
    use anamnesis_importer::{ImportOptions, ImportService};

    let store = Store::open(super::db_path(data_dir))?;
    let service = ImportService::new(&store, super::audit(data_dir));
    let scan_opts = ScanOpts { since, full: false };
    let instance = if key.instance.is_empty() {
        None
    } else {
        Some(key.instance.as_str())
    };
    let opts = ImportOptions {
        dry_run: false,
        canonical_location: Some(location.display().to_string()),
        source_was_explicit: true,
        scan_opts,
    };

    macro_rules! run {
        ($adapter:expr) => {{
            service
                .import(&$adapter, opts)
                .await
                .map_err(|e| anyhow!("watch import {}: {e}", key.label()))
        }};
    }

    let loc = location.to_path_buf();
    match key.adapter.as_str() {
        anamnesis_adapter_claude_code::ADAPTER_ID => {
            use anamnesis_adapter_claude_code::{ClaudeCodeAdapter, ClaudeCodeConfig};
            run!(ClaudeCodeAdapter::new(ClaudeCodeConfig {
                projects_root: loc,
                instance: instance.map(str::to_owned),
            }))
        }
        anamnesis_adapter_mem0::ADAPTER_ID => {
            run!(anamnesis_adapter_mem0::sqlite_adapter(loc, instance))
        }
        anamnesis_adapter_codex::ADAPTER_ID => {
            run!(anamnesis_adapter_codex::codex_adapter(loc, instance))
        }
        anamnesis_adapter_letta::ADAPTER_ID => {
            run!(anamnesis_adapter_letta::letta_adapter(loc, instance))
        }
        anamnesis_adapter_hermes::ADAPTER_ID => {
            run!(anamnesis_adapter_hermes::hermes_adapter(loc, instance))
        }
        anamnesis_adapter_openclaw::ADAPTER_ID => {
            run!(anamnesis_adapter_openclaw::openclaw_adapter(loc, instance))
        }
        anamnesis_adapter_tdai::ADAPTER_ID => {
            run!(anamnesis_adapter_tdai::tdai_adapter(loc, instance))
        }
        anamnesis_adapter_openviking::ADAPTER_ID => {
            run!(anamnesis_adapter_openviking::openviking_adapter(
                loc, instance
            ))
        }
        anamnesis_adapter_mempalace::ADAPTER_ID => {
            run!(anamnesis_adapter_mempalace::mempalace_adapter(
                loc, instance
            ))
        }
        anamnesis_adapter_memori::ADAPTER_ID => {
            run!(anamnesis_adapter_memori::memori_adapter(loc, instance))
        }
        anamnesis_adapter_memos::ADAPTER_ID => {
            run!(anamnesis_adapter_memos::memos_adapter(loc, instance))
        }
        anamnesis_adapter_memary::ADAPTER_ID => {
            run!(anamnesis_adapter_memary::memary_adapter(loc, instance))
        }
        other => Err(anyhow!("watch: adapter {other:?} is not fs-watchable")),
    }
}

/// `anamnesis watch` entry point. Enumerates registered sources, runs a
/// one-shot catch-up import, then watches their filesystem roots and
/// re-imports on debounced change until Ctrl-C.
pub async fn run_watch(data_dir: &Path, no_embed: bool) -> Result<()> {
    let _ = no_embed; // PR 1: embedding worker scheduling deferred to PR 2.

    let sources = {
        let store = Store::open(super::db_path(data_dir))?;
        store.list_sources_full()?
    };
    let router = build_router(&sources);
    if router.is_empty() {
        return Err(anyhow!(
            "watch: no fs-watchable sources registered. Run `anamnesis source add <adapter> \
             --path <location>` (and at least one `anamnesis import`) first."
        ));
    }

    // last_import_at lookup keyed by source for the incremental `since`.
    let last_import: HashMap<SourceKey, Option<i64>> = sources
        .iter()
        .map(|s| {
            (
                SourceKey {
                    adapter: s.adapter.clone(),
                    instance: s.instance.clone(),
                },
                s.last_import_at,
            )
        })
        .collect();
    let location_of: HashMap<SourceKey, PathBuf> = sources
        .iter()
        .filter(|s| is_fs_watchable(&s.adapter))
        .filter_map(|s| {
            s.location.as_deref().map(|loc| {
                (
                    SourceKey {
                        adapter: s.adapter.clone(),
                        instance: s.instance.clone(),
                    },
                    PathBuf::from(loc),
                )
            })
        })
        .collect();

    let watched: Vec<String> = router.roots().map(|p| p.display().to_string()).collect();
    println!(
        "anamnesis watch: monitoring {} source root(s):",
        watched.len()
    );
    for (key, loc) in &location_of {
        println!("  {} → {}", key.label(), loc.display());
    }

    // 1. Catch-up sweep: import anything that changed while watch was down.
    for (key, loc) in &location_of {
        let since = incremental_since(*last_import.get(key).unwrap_or(&None));
        match import_one_source(data_dir, key, loc, since).await {
            Ok(s) => println!(
                "  catch-up {} — {} raw, {} upserted",
                key.label(),
                s.raw_seen,
                s.records_upserted
            ),
            Err(e) => eprintln!("  catch-up {} failed: {e}", key.label()),
        }
    }

    // 2. Arm the fs watcher. notify's callback runs on its own thread; we
    //    forward each event into an unbounded tokio channel (send is
    //    sync + non-blocking, safe to call from the callback).
    use notify::{RecursiveMode, Watcher};
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<notify::Event>();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(ev) = res {
            let _ = tx.send(ev);
        }
    })
    .map_err(|e| anyhow!("watch: init fs watcher: {e}"))?;
    for root in router.roots() {
        watcher
            .watch(root, RecursiveMode::Recursive)
            .map_err(|e| anyhow!("watch: arm {}: {e}", root.display()))?;
    }

    println!("anamnesis watch: live. Ctrl-C to stop.");

    // 3. Event loop: debounce changes → incremental import. Sleep until
    //    either the next deadline or a new event, whichever comes first.
    let mut planner = DebouncePlanner::new(DEBOUNCE_WINDOW);
    loop {
        let sleep = async {
            match planner.next_deadline() {
                Some(deadline) => {
                    let now = Instant::now();
                    let dur = deadline.saturating_duration_since(now);
                    tokio::time::sleep(dur).await;
                }
                // Nothing pending → park until an event or Ctrl-C wakes us.
                None => std::future::pending::<()>().await,
            }
        };

        tokio::select! {
            maybe_ev = rx.recv() => {
                match maybe_ev {
                    Some(ev) => {
                        let now = Instant::now();
                        for path in &ev.paths {
                            if let Some(key) = router.route(path) {
                                planner.observe(key.clone(), now);
                            }
                        }
                    }
                    None => break, // watcher dropped — shut down.
                }
            }
            _ = sleep => {
                for key in planner.take_due(Instant::now()) {
                    let Some(loc) = location_of.get(&key) else { continue };
                    let since = incremental_since(*last_import.get(&key).unwrap_or(&None));
                    match import_one_source(data_dir, &key, loc, since).await {
                        Ok(s) => println!(
                            "auto-sync {} — {} raw, {} upserted, {} chunks",
                            key.label(), s.raw_seen, s.records_upserted, s.chunks_written
                        ),
                        Err(e) => eprintln!("auto-sync {} failed: {e}", key.label()),
                    }
                }
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\nanamnesis watch: stopping.");
                break;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(adapter: &str, instance: &str) -> SourceKey {
        SourceKey {
            adapter: adapter.into(),
            instance: instance.into(),
        }
    }

    #[test]
    fn debounce_collapses_burst_into_single_due() {
        let mut p = DebouncePlanner::new(Duration::from_secs(2));
        let t0 = Instant::now();
        let k = key("mem0", "");
        // Three events 100ms apart — each pushes the deadline out.
        p.observe(k.clone(), t0);
        p.observe(k.clone(), t0 + Duration::from_millis(100));
        p.observe(k.clone(), t0 + Duration::from_millis(200));
        // Before the last deadline: nothing due.
        assert!(p.take_due(t0 + Duration::from_millis(2100)).is_empty());
        // After 200ms + 2s window: exactly one due, not three.
        let due = p.take_due(t0 + Duration::from_millis(2201));
        assert_eq!(due, vec![k]);
        // Drained — second poll is empty.
        assert!(p.take_due(t0 + Duration::from_secs(10)).is_empty());
    }

    #[test]
    fn debounce_tracks_multiple_sources_independently() {
        let mut p = DebouncePlanner::new(Duration::from_secs(2));
        let t0 = Instant::now();
        let mem0 = key("mem0", "");
        let letta = key("letta", "prod");
        p.observe(mem0.clone(), t0);
        p.observe(letta.clone(), t0 + Duration::from_secs(1));
        // At t0+2.5s: mem0's window (t0+2s) elapsed, letta's (t0+3s) not.
        let due = p.take_due(t0 + Duration::from_millis(2500));
        assert_eq!(due, vec![mem0]);
        assert_eq!(p.pending_len(), 1);
        // At t0+3.1s: letta now due.
        assert_eq!(p.take_due(t0 + Duration::from_millis(3100)), vec![letta]);
    }

    #[test]
    fn debounce_next_deadline_is_earliest_pending() {
        let mut p = DebouncePlanner::new(Duration::from_secs(2));
        let t0 = Instant::now();
        assert!(p.next_deadline().is_none());
        p.observe(key("a", ""), t0 + Duration::from_secs(5));
        p.observe(key("b", ""), t0 + Duration::from_secs(1));
        // Earliest deadline = b's = (t0+1s)+2s = t0+3s.
        assert_eq!(p.next_deadline(), Some(t0 + Duration::from_secs(3)));
    }

    #[test]
    fn path_router_longest_prefix_wins() {
        let mut r = PathRouter::default();
        r.insert(PathBuf::from("/home/u/.config"), key("broad", ""));
        r.insert(PathBuf::from("/home/u/.config/mem0"), key("mem0", ""));
        // A path under the nested root routes to the more specific source.
        assert_eq!(
            r.route(Path::new("/home/u/.config/mem0/history.db")),
            Some(&key("mem0", ""))
        );
        // A path only under the broad root routes there.
        assert_eq!(
            r.route(Path::new("/home/u/.config/other.json")),
            Some(&key("broad", ""))
        );
        // Outside any root → no route.
        assert!(r.route(Path::new("/tmp/elsewhere")).is_none());
    }

    #[test]
    fn path_router_matches_exact_root() {
        let mut r = PathRouter::default();
        r.insert(PathBuf::from("/data/letta.db"), key("letta", ""));
        assert_eq!(
            r.route(Path::new("/data/letta.db")),
            Some(&key("letta", ""))
        );
    }

    #[test]
    fn incremental_since_rewinds_by_overlap() {
        // 1_000_000 → 1_000_000 - 5 = 999_995.
        let since = incremental_since(Some(1_000_000)).unwrap();
        assert_eq!(since.timestamp(), 999_995);
        // Never imported → full catch-up (None).
        assert!(incremental_since(None).is_none());
        // Clamp at epoch — no negative timestamps.
        assert_eq!(incremental_since(Some(2)).unwrap().timestamp(), 0);
    }

    #[test]
    fn generic_mcp_is_not_fs_watchable() {
        assert!(!is_fs_watchable(anamnesis_adapter_generic_mcp::ADAPTER_ID));
        assert!(is_fs_watchable("mem0"));
        assert!(is_fs_watchable("claude-code"));
    }

    #[test]
    fn build_router_skips_url_and_missing_locations() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("mem0.db");
        std::fs::write(&real, b"x").unwrap();
        let rows = vec![
            anamnesis_store::SourceRow {
                adapter: "mem0".into(),
                instance: String::new(),
                location: Some(real.to_str().unwrap().to_string()),
                config_json: None,
                added_at: 0,
                last_import_at: None,
            },
            // URL adapter — skipped.
            anamnesis_store::SourceRow {
                adapter: "generic-mcp".into(),
                instance: String::new(),
                location: Some("http://127.0.0.1:7878".into()),
                config_json: None,
                added_at: 0,
                last_import_at: None,
            },
            // Missing path — skipped.
            anamnesis_store::SourceRow {
                adapter: "letta".into(),
                instance: String::new(),
                location: Some("/does/not/exist.db".into()),
                config_json: None,
                added_at: 0,
                last_import_at: None,
            },
        ];
        let router = build_router(&rows);
        // Only the real mem0 source got a root (watched at its parent dir).
        let roots: Vec<_> = router.roots().collect();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], dir.path());
    }
}
