//! `anamnesis watch {install,uninstall,status}` — auto-start the R151
//! watch daemon at login (R152, PR 2 of N). Completes "装上就自动持续
//! 整理": after `watch install`, the OS launches `anamnesis watch` on
//! every login and restarts it on crash, so the store stays in sync
//! with zero operator action.
//!
//! Platforms:
//!   - macOS  → launchd user agent (`~/Library/LaunchAgents/*.plist`)
//!   - Linux  → systemd user service (`~/.config/systemd/user/*.service`)
//!   - Windows → no native unit this PR; prints Task Scheduler guidance.
//!
//! ## Layering (same discipline as watch.rs)
//!
//! Unit-file TEXT generation is pure + testable ([`launchd_plist`],
//! [`systemd_unit`], the `*_path` helpers). The IO layer (write file,
//! shell out to `launchctl` / `systemctl`) is covered by cross-platform
//! compilation + manual verification, never by unit tests — CI must not
//! mutate the runner's service registry.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

/// launchd agent label (also the plist filename stem).
const LAUNCHD_LABEL: &str = "com.anamnesis.watch";
/// systemd user unit name.
const SYSTEMD_UNIT: &str = "anamnesis-watch.service";

/// XML-escape a path for safe embedding in a launchd plist `<string>`.
/// Paths can legally contain `&` / `<` / `>` which would break the XML.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Render the launchd user-agent plist that runs `anamnesis watch`
/// against `data_dir` at login, keeps it alive on crash, and logs to
/// `<data_dir>/watch.{log,err.log}`. `exe` must be the absolute path
/// to the `anamnesis` binary.
pub fn launchd_plist(exe: &Path, data_dir: &Path) -> String {
    let exe = xml_escape(&exe.display().to_string());
    let dir = xml_escape(&data_dir.display().to_string());
    let out = xml_escape(&data_dir.join("watch.log").display().to_string());
    let err = xml_escape(&data_dir.join("watch.err.log").display().to_string());
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe}</string>
        <string>watch</string>
        <string>--data-dir</string>
        <string>{dir}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{out}</string>
    <key>StandardErrorPath</key>
    <string>{err}</string>
</dict>
</plist>
"#
    )
}

/// Render the systemd user unit. `exe` must be absolute. The Linux
/// default `data_dir` (`~/.local/share/anamnesis`) has no spaces; if a
/// caller points `--data-dir` at a path with spaces, systemd ExecStart
/// would need quoting — out of scope for this PR (documented).
pub fn systemd_unit(exe: &Path, data_dir: &Path) -> String {
    let exe = exe.display();
    let dir = data_dir.display();
    format!(
        "[Unit]\n\
         Description=Anamnesis watch — auto-sync agent memory frameworks\n\
         After=default.target\n\
         \n\
         [Service]\n\
         Type=simple\n\
         ExecStart={exe} watch --data-dir {dir}\n\
         Restart=on-failure\n\
         RestartSec=5\n\
         \n\
         [Install]\n\
         WantedBy=default.target\n"
    )
}

/// Absolute plist path for the current user's home.
pub fn launchd_plist_path(home: &Path) -> PathBuf {
    home.join("Library")
        .join("LaunchAgents")
        .join(format!("{LAUNCHD_LABEL}.plist"))
}

/// Absolute systemd user-unit path for the current user's home.
pub fn systemd_unit_path(home: &Path) -> PathBuf {
    home.join(".config")
        .join("systemd")
        .join("user")
        .join(SYSTEMD_UNIT)
}

/// Install + enable an OS service that runs `anamnesis watch` at login.
pub fn install(data_dir: &Path) -> Result<()> {
    let exe = std::env::current_exe().context("resolve anamnesis binary path")?;
    let home = super::dirs_home()?;
    let data_dir = data_dir.to_path_buf();

    if cfg!(target_os = "macos") {
        let path = launchd_plist_path(&home);
        write_unit(&path, &launchd_plist(&exe, &data_dir))?;
        // Reload: unload first (ignore "not loaded" error), then load -w
        // so RunAtLoad fires now and the agent persists across logins.
        let _ = run("launchctl", &["unload", path_str(&path)?]);
        run("launchctl", &["load", "-w", path_str(&path)?]).context("launchctl load failed")?;
        println!("anamnesis watch: installed launchd agent {LAUNCHD_LABEL}");
        println!("  plist: {}", path.display());
        println!("  it now runs at every login and restarts on crash.");
        println!("  logs:  {}", data_dir.join("watch.log").display());
        Ok(())
    } else if cfg!(target_os = "linux") {
        let path = systemd_unit_path(&home);
        write_unit(&path, &systemd_unit(&exe, &data_dir))?;
        run("systemctl", &["--user", "daemon-reload"]).context("systemctl daemon-reload failed")?;
        run("systemctl", &["--user", "enable", "--now", SYSTEMD_UNIT])
            .context("systemctl enable --now failed")?;
        println!("anamnesis watch: installed systemd user service {SYSTEMD_UNIT}");
        println!("  unit: {}", path.display());
        println!("  started now and enabled at login. Logs: journalctl --user -u {SYSTEMD_UNIT}");
        println!("  (a login session must persist — `loginctl enable-linger $USER` for headless.)");
        Ok(())
    } else {
        // Windows + everything else: no native unit this PR.
        println!(
            "anamnesis watch: no native auto-start on this platform yet. Register a \
             Task Scheduler job that runs at logon:"
        );
        println!(
            "  schtasks /Create /SC ONLOGON /TN AnamnesisWatch /TR \"{} watch --data-dir {}\"",
            exe.display(),
            data_dir.display()
        );
        Ok(())
    }
}

/// Stop + remove the auto-start service.
pub fn uninstall() -> Result<()> {
    let home = super::dirs_home()?;
    if cfg!(target_os = "macos") {
        let path = launchd_plist_path(&home);
        let _ = run("launchctl", &["unload", path_str(&path)?]);
        if path.exists() {
            std::fs::remove_file(&path).context("remove launchd plist")?;
        }
        println!("anamnesis watch: removed launchd agent {LAUNCHD_LABEL}");
        Ok(())
    } else if cfg!(target_os = "linux") {
        let path = systemd_unit_path(&home);
        let _ = run("systemctl", &["--user", "disable", "--now", SYSTEMD_UNIT]);
        if path.exists() {
            std::fs::remove_file(&path).context("remove systemd unit")?;
        }
        let _ = run("systemctl", &["--user", "daemon-reload"]);
        println!("anamnesis watch: removed systemd user service {SYSTEMD_UNIT}");
        Ok(())
    } else {
        println!(
            "anamnesis watch: no native auto-start to remove on this platform. If you \
             created a Task Scheduler job: schtasks /Delete /TN AnamnesisWatch /F"
        );
        Ok(())
    }
}

/// Report whether the auto-start service is installed.
pub fn status() -> Result<()> {
    let home = super::dirs_home()?;
    let (label, path) = if cfg!(target_os = "macos") {
        (LAUNCHD_LABEL, launchd_plist_path(&home))
    } else if cfg!(target_os = "linux") {
        (SYSTEMD_UNIT, systemd_unit_path(&home))
    } else {
        println!("anamnesis watch: auto-start status unavailable on this platform.");
        return Ok(());
    };
    if path.exists() {
        println!("anamnesis watch: auto-start INSTALLED ({label})");
        println!("  unit: {}", path.display());
    } else {
        println!("anamnesis watch: auto-start NOT installed.");
        println!("  run `anamnesis watch install` to enable it.");
    }
    Ok(())
}

/// Write a unit file, creating parent dirs. Overwrites an existing unit
/// (re-install should pick up a new binary path / data dir).
fn write_unit(path: &Path, contents: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::write(path, contents).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// `&str` view of a path or a clear error (non-UTF-8 paths can't be
/// passed to the service loaders we shell out to).
fn path_str(p: &Path) -> Result<&str> {
    p.to_str()
        .ok_or_else(|| anyhow!("path is not valid UTF-8: {}", p.display()))
}

/// Run a loader command, surfacing a clear error on non-zero exit.
fn run(cmd: &str, args: &[&str]) -> Result<()> {
    let status = std::process::Command::new(cmd)
        .args(args)
        .status()
        .with_context(|| format!("spawn `{cmd}` (is it installed / on PATH?)"))?;
    if !status.success() {
        return Err(anyhow!("`{cmd} {}` exited with {status}", args.join(" ")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launchd_plist_embeds_exe_args_and_logs() {
        let plist = launchd_plist(
            Path::new("/usr/local/bin/anamnesis"),
            Path::new("/Users/u/Library/Application Support/anamnesis"),
        );
        assert!(plist.contains("<string>com.anamnesis.watch</string>"));
        assert!(plist.contains("<string>/usr/local/bin/anamnesis</string>"));
        assert!(plist.contains("<string>watch</string>"));
        assert!(plist.contains("<string>--data-dir</string>"));
        // The macOS default data dir has a space — launchd's array form
        // keeps it as a single arg (no quoting needed).
        assert!(plist.contains("Library/Application Support/anamnesis</string>"));
        assert!(plist.contains("<key>RunAtLoad</key>\n    <true/>"));
        assert!(plist.contains("<key>KeepAlive</key>\n    <true/>"));
        assert!(plist.contains("watch.log</string>"));
        assert!(plist.contains("watch.err.log</string>"));
    }

    #[test]
    fn launchd_plist_xml_escapes_paths() {
        let plist = launchd_plist(Path::new("/opt/a&b/anamnesis"), Path::new("/data/x<y"));
        assert!(plist.contains("/opt/a&amp;b/anamnesis"));
        assert!(plist.contains("/data/x&lt;y"));
        // The raw, unescaped forms must NOT leak into the XML.
        assert!(!plist.contains("/opt/a&b/"));
        assert!(!plist.contains("x<y"));
    }

    #[test]
    fn systemd_unit_has_execstart_restart_and_install_section() {
        let unit = systemd_unit(
            Path::new("/usr/bin/anamnesis"),
            Path::new("/home/u/.local/share/anamnesis"),
        );
        assert!(unit.contains(
            "ExecStart=/usr/bin/anamnesis watch --data-dir /home/u/.local/share/anamnesis"
        ));
        assert!(unit.contains("Restart=on-failure"));
        assert!(unit.contains("WantedBy=default.target"));
        assert!(unit.contains("[Service]"));
        assert!(unit.contains("[Install]"));
    }

    #[test]
    fn unit_paths_are_under_home() {
        let home = Path::new("/home/u");
        assert_eq!(
            launchd_plist_path(home),
            Path::new("/home/u/Library/LaunchAgents/com.anamnesis.watch.plist")
        );
        assert_eq!(
            systemd_unit_path(home),
            Path::new("/home/u/.config/systemd/user/anamnesis-watch.service")
        );
    }

    #[test]
    fn write_unit_creates_parents_and_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/deep/unit.service");
        write_unit(&path, "first").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "first");
        // Re-install overwrites (new binary path / data dir).
        write_unit(&path, "second").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
    }
}
