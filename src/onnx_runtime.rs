use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use flate2::read::GzDecoder;
use reqwest::blocking::Client;
use serde::Deserialize;
use tar::Archive;
use zip::ZipArchive;

const GITHUB_RELEASES_API: &str = "https://api.github.com/repos/microsoft/onnxruntime/releases";
const ONNX_RUNTIME_MINOR: &str = "1.24";
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(600);
const HTTP_USER_AGENT: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    draft: bool,
    prerelease: bool,
    assets: Vec<GitHubAsset>,
}

#[derive(Debug, Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
}

pub fn configure_onnx_runtime() -> Result<()> {
    let dylib_path = match find_bundled_onnx_runtime() {
        Some(path) => path,
        None => download_onnx_runtime()?,
    };

    tracing::info!(dylib = %dylib_path.display(), "initializing ONNX Runtime from dylib");
    let committed = ort::init_from(&dylib_path)
        .with_context(|| format!("failed to load ONNX Runtime dylib {}", dylib_path.display()))?
        .commit();
    tracing::info!(committed, "ONNX Runtime dylib initialized");
    Ok(())
}

fn find_bundled_onnx_runtime() -> Option<PathBuf> {
    let exe = env::current_exe().ok()?;
    let exe_dir = exe.parent()?;
    onnx_runtime_candidates(exe_dir)
        .into_iter()
        .find(|path| path.is_file())
}

fn onnx_runtime_candidates(exe_dir: &Path) -> Vec<PathBuf> {
    let file_name = runtime_library_file_name();
    vec![exe_dir.join(file_name), exe_dir.join("lib").join(file_name)]
}

fn download_onnx_runtime() -> Result<PathBuf> {
    let exe = env::current_exe().context("failed to get current executable path")?;
    let exe_dir = exe
        .parent()
        .ok_or_else(|| anyhow!("current executable has no parent: {}", exe.display()))?;
    let lib_dir = exe_dir.join("lib");
    let client = build_http_client()?;
    let release = latest_1_24_release(&client)?;
    let asset = runtime_asset(&release)?;
    let archive_path = exe_dir.join(format!(".{}", asset.name));

    tracing::info!(
        version = %release.version(),
        asset = %asset.name,
        url = %asset.browser_download_url,
        target = %lib_dir.display(),
        "downloading ONNX Runtime"
    );
    download_asset(&client, asset, &archive_path)?;
    let extracted = extract_runtime_libraries(&archive_path, &lib_dir, &asset.name)?;
    let _ = fs::remove_file(&archive_path);

    if extracted == 0 {
        return Err(anyhow!(
            "no ONNX Runtime runtime library files were found in downloaded archive {}",
            asset.name
        ));
    }

    find_bundled_onnx_runtime().ok_or_else(|| {
        anyhow!(
            "ONNX Runtime was downloaded but {} was not found under {} or {}",
            runtime_library_file_name(),
            exe_dir.display(),
            lib_dir.display()
        )
    })
}

fn build_http_client() -> Result<Client> {
    Client::builder()
        .timeout(DOWNLOAD_TIMEOUT)
        .user_agent(HTTP_USER_AGENT)
        .build()
        .context("failed to build ONNX Runtime download HTTP client")
}

fn latest_1_24_release(client: &Client) -> Result<GitHubRelease> {
    let mut candidates = Vec::new();

    for page in 1..=5 {
        let url = format!("{GITHUB_RELEASES_API}?per_page=100&page={page}");
        let releases: Vec<GitHubRelease> = client
            .get(&url)
            .send()
            .with_context(|| format!("failed to query ONNX Runtime releases from {url}"))?
            .error_for_status()
            .with_context(|| format!("failed to query ONNX Runtime releases from {url}"))?
            .json()
            .context("failed to parse ONNX Runtime releases response")?;

        if releases.is_empty() {
            break;
        }

        candidates.extend(releases.into_iter().filter_map(|release| {
            let patch = parse_1_24_patch(&release.tag_name)?;
            if release.draft || release.prerelease {
                return None;
            }
            Some((patch, release))
        }));
    }

    candidates.sort_by_key(|(patch, _)| *patch);
    candidates
        .pop()
        .map(|(_, release)| release)
        .ok_or_else(|| anyhow!("no ONNX Runtime {ONNX_RUNTIME_MINOR}.x release found"))
}

fn parse_1_24_patch(tag_name: &str) -> Option<u32> {
    let prefix = format!("v{ONNX_RUNTIME_MINOR}.");
    let patch = tag_name.strip_prefix(&prefix)?;
    patch
        .split(['-', '+'])
        .next()
        .and_then(|patch| patch.parse::<u32>().ok())
}

fn runtime_asset(release: &GitHubRelease) -> Result<&GitHubAsset> {
    let expected = runtime_asset_name(&release.version())?;
    release
        .assets
        .iter()
        .find(|asset| asset.name == expected)
        .or_else(|| {
            let platform = runtime_platform()?;
            let extension = runtime_archive_extension();
            release.assets.iter().find(|asset| {
                asset.name.contains(platform)
                    && asset.name.contains(&release.version())
                    && asset.name.ends_with(extension)
            })
        })
        .ok_or_else(|| {
            anyhow!(
                "no current-platform ONNX Runtime asset found in release {}; expected {}",
                release.tag_name,
                expected
            )
        })
}

fn runtime_asset_name(version: &str) -> Result<String> {
    Ok(format!(
        "onnxruntime-{}-{}.{}",
        runtime_platform()
            .ok_or_else(|| anyhow!("unsupported platform for ONNX Runtime download"))?,
        version,
        runtime_archive_extension()
    ))
}

impl GitHubRelease {
    fn version(&self) -> String {
        self.tag_name.trim_start_matches('v').to_string()
    }
}

fn download_asset(client: &Client, asset: &GitHubAsset, archive_path: &Path) -> Result<()> {
    let mut response = client
        .get(&asset.browser_download_url)
        .send()
        .with_context(|| format!("failed to download {}", asset.browser_download_url))?
        .error_for_status()
        .with_context(|| format!("failed to download {}", asset.browser_download_url))?;
    let mut output = File::create(archive_path)
        .with_context(|| format!("failed to create {}", archive_path.display()))?;
    io::copy(&mut response, &mut output)
        .with_context(|| format!("failed to write {}", archive_path.display()))?;
    output
        .flush()
        .with_context(|| format!("failed to flush {}", archive_path.display()))?;
    Ok(())
}

fn extract_runtime_libraries(
    archive_path: &Path,
    exe_dir: &Path,
    asset_name: &str,
) -> Result<usize> {
    if asset_name.ends_with(".zip") {
        extract_zip_runtime_libraries(archive_path, exe_dir)
    } else if asset_name.ends_with(".tgz") || asset_name.ends_with(".tar.gz") {
        extract_tgz_runtime_libraries(archive_path, exe_dir)
    } else {
        Err(anyhow!(
            "unsupported ONNX Runtime archive format: {asset_name}"
        ))
    }
}

fn extract_zip_runtime_libraries(archive_path: &Path, lib_dir: &Path) -> Result<usize> {
    let file = File::open(archive_path)
        .with_context(|| format!("failed to open {}", archive_path.display()))?;
    let mut archive = ZipArchive::new(file)
        .with_context(|| format!("failed to read zip archive {}", archive_path.display()))?;
    let mut extracted = 0usize;

    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        if entry.is_dir() {
            continue;
        }
        let Some(file_name) = Path::new(entry.name())
            .file_name()
            .and_then(|name| name.to_str())
        else {
            continue;
        };
        if !is_runtime_library_file(file_name) {
            continue;
        }

        fs::create_dir_all(lib_dir)
            .with_context(|| format!("failed to create {}", lib_dir.display()))?;
        let target = lib_dir.join(file_name);
        let mut output = File::create(&target)
            .with_context(|| format!("failed to create {}", target.display()))?;
        io::copy(&mut entry, &mut output)
            .with_context(|| format!("failed to extract {}", target.display()))?;
        extracted += 1;
        tracing::info!(file = %target.display(), "extracted ONNX Runtime runtime file");
    }

    Ok(extracted)
}

fn extract_tgz_runtime_libraries(archive_path: &Path, lib_dir: &Path) -> Result<usize> {
    let file = File::open(archive_path)
        .with_context(|| format!("failed to open {}", archive_path.display()))?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    let mut extracted = 0usize;

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !is_runtime_library_file(file_name) {
            continue;
        }

        fs::create_dir_all(lib_dir)
            .with_context(|| format!("failed to create {}", lib_dir.display()))?;
        let target = lib_dir.join(file_name);
        let _ = fs::remove_file(&target);
        entry
            .unpack(&target)
            .with_context(|| format!("failed to extract {}", target.display()))?;
        extracted += 1;
        tracing::info!(file = %target.display(), "extracted ONNX Runtime runtime file");
    }

    Ok(extracted)
}

fn runtime_platform() -> Option<&'static str> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        return Some("linux-x64");
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        return Some("linux-aarch64");
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        return Some("win-x64");
    }
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        return Some("win-arm64");
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        return Some("osx-x86_64");
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        return Some("osx-arm64");
    }
    #[allow(unreachable_code)]
    None
}

fn runtime_archive_extension() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "zip"
    }
    #[cfg(not(target_os = "windows"))]
    {
        "tgz"
    }
}

fn runtime_library_file_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "onnxruntime.dll"
    }
    #[cfg(target_os = "linux")]
    {
        "libonnxruntime.so"
    }
    #[cfg(target_os = "macos")]
    {
        "libonnxruntime.dylib"
    }
}

fn is_runtime_library_file(file_name: &str) -> bool {
    #[cfg(target_os = "windows")]
    {
        let lower = file_name.to_ascii_lowercase();
        lower.starts_with("onnxruntime") && lower.ends_with(".dll")
    }
    #[cfg(target_os = "linux")]
    {
        file_name.starts_with("libonnxruntime") && file_name.contains(".so")
    }
    #[cfg(target_os = "macos")]
    {
        file_name.starts_with("libonnxruntime") && file_name.contains(".dylib")
    }
}
