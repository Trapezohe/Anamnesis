//! Shared watch-daemon heartbeat — pure types + helpers, no IO.
//!
//! The `cli` watch daemon writes [`heartbeat_path`] periodically; both
//! `anamnesis watch status` (CLI) and the MCP `watch_status` tool read it
//! to report whether auto-sync is alive. Keeping the struct + decisions
//! here lets both readers share one definition without depending on each
//! other. File read/write lives in those IO crates (core has none).

use std::path::{Path, PathBuf};

/// A heartbeat older than this many seconds reads as STALE — the daemon
/// likely died without removing its file. Three missed 15s beats.
pub const HEARTBEAT_STALE_SECS: i64 = 45;

/// On-disk liveness record the watch daemon refreshes periodically. A
/// detached launchd/systemd daemon is otherwise invisible.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WatchHeartbeat {
    /// PID of the daemon process (for the operator to `kill` if needed).
    pub pid: u32,
    /// Unix epoch (secs) when this daemon started.
    pub started_at: i64,
    /// Unix epoch (secs) of the most recent heartbeat.
    pub last_beat: i64,
    /// Number of watched sources (fs roots + polled URL sources).
    pub roots: usize,
}

/// Location of the heartbeat file under a data dir. Pure path math, no IO.
pub fn heartbeat_path(data_dir: &Path) -> PathBuf {
    data_dir.join("watch.state.json")
}

/// Whether a heartbeat at `last_beat` is still live as of `now` (both Unix
/// epoch secs). A `last_beat` in the future (clock skew / DST) counts as
/// live, never stale.
pub fn is_heartbeat_live(now: i64, last_beat: i64, stale_after_secs: i64) -> bool {
    now - last_beat <= stale_after_secs
}

/// Render an age in seconds as a compact `s`/`m`/`h`/`d` string. Negative
/// inputs (clock skew) clamp to `0s`.
pub fn humanize_age(secs: i64) -> String {
    let s = secs.max(0);
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m", s / 60)
    } else if s < 86_400 {
        format!("{}h", s / 3600)
    } else {
        format!("{}d", s / 86_400)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn liveness_window() {
        assert!(is_heartbeat_live(1000, 1000, 45));
        assert!(is_heartbeat_live(1045, 1000, 45)); // boundary
        assert!(!is_heartbeat_live(1046, 1000, 45)); // one past
        assert!(is_heartbeat_live(1000, 1010, 45)); // future beat → live
    }

    #[test]
    fn age_units() {
        assert_eq!(humanize_age(-5), "0s");
        assert_eq!(humanize_age(0), "0s");
        assert_eq!(humanize_age(59), "59s");
        assert_eq!(humanize_age(60), "1m");
        assert_eq!(humanize_age(3599), "59m");
        assert_eq!(humanize_age(3600), "1h");
        assert_eq!(humanize_age(86_399), "23h");
        assert_eq!(humanize_age(86_400), "1d");
    }

    #[test]
    fn heartbeat_serde_round_trips() {
        let hb = WatchHeartbeat {
            pid: 4242,
            started_at: 1_700_000_000,
            last_beat: 1_700_000_030,
            roots: 3,
        };
        let json = serde_json::to_string(&hb).unwrap();
        assert_eq!(serde_json::from_str::<WatchHeartbeat>(&json).unwrap(), hb);
    }
}
