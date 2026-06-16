use std::fs;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use reqwest::header::{CONTENT_LENGTH, RANGE};

const TOKENIZER_FILE: &str = "tokenizer.json";
const HUGGING_FACE_BASE_URL: &str = "https://huggingface.co/openai/privacy-filter/resolve/main";
const HUGGING_FACE_ONNX_TREE_URL: &str =
    "https://huggingface.co/openai/privacy-filter/tree/main/onnx";
const MODELSCOPE_BASE_URL: &str =
    "https://modelscope.cn/models/openai-mirror/privacy-filter/resolve/master";
const MODELSCOPE_ONNX_TREE_URL: &str =
    "https://modelscope.cn/models/openai-mirror/privacy-filter/tree/master/onnx";
const CNB_RAW_BASE_URL: &str = "https://cnb.cool/ai-models/openai/privacy-filter/-/git/raw/main";
const CNB_LFS_BASE_URL: &str = "https://cnb.cool/ai-models/openai/privacy-filter/-/lfs";
const CNB_ONNX_TREE_URL: &str = "https://cnb.cool/ai-models/openai/privacy-filter/-/tree/main/onnx";
const PROBE_TIMEOUT: Duration = Duration::from_secs(8);
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(600);
const PROBE_RANGE: &str = "bytes=0-1023";
const HTTP_USER_AGENT: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFiles {
    pub root: PathBuf,
    pub tokenizer_path: PathBuf,
    pub onnx_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelSourceKind {
    Direct,
    CnbLfs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelSource {
    name: &'static str,
    base_url: &'static str,
    onnx_tree_url: &'static str,
    kind: ModelSourceKind,
}

#[derive(Debug)]
struct ProbeResult {
    source: ModelSource,
    elapsed: Duration,
    bytes_read: usize,
    content_length: Option<u64>,
}

#[derive(Debug)]
struct SourceDownload {
    url: String,
}

#[derive(Debug)]
struct CnbLfsPointer {
    oid: String,
    size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OnnxModelSpec {
    variant: &'static str,
    onnx_file: &'static str,
    files: &'static [&'static str],
}

const FP32_FILES: &[&str] = &[
    "onnx/model.onnx",
    "onnx/model.onnx_data",
    "onnx/model.onnx_data_1",
    "onnx/model.onnx_data_2",
];
const FP16_FILES: &[&str] = &[
    "onnx/model_fp16.onnx",
    "onnx/model_fp16.onnx_data",
    "onnx/model_fp16.onnx_data_1",
];
const Q4_FILES: &[&str] = &["onnx/model_q4.onnx", "onnx/model_q4.onnx_data"];
const Q4F16_FILES: &[&str] = &["onnx/model_q4f16.onnx", "onnx/model_q4f16.onnx_data"];
const QUANTIZED_FILES: &[&str] = &[
    "onnx/model_quantized.onnx",
    "onnx/model_quantized.onnx_data",
];

const ONNX_MODEL_SPECS: &[OnnxModelSpec] = &[
    OnnxModelSpec {
        variant: "fp32",
        onnx_file: "onnx/model.onnx",
        files: FP32_FILES,
    },
    OnnxModelSpec {
        variant: "fp16",
        onnx_file: "onnx/model_fp16.onnx",
        files: FP16_FILES,
    },
    OnnxModelSpec {
        variant: "q4",
        onnx_file: "onnx/model_q4.onnx",
        files: Q4_FILES,
    },
    OnnxModelSpec {
        variant: "q4f16",
        onnx_file: "onnx/model_q4f16.onnx",
        files: Q4F16_FILES,
    },
    OnnxModelSpec {
        variant: "quantized",
        onnx_file: "onnx/model_quantized.onnx",
        files: QUANTIZED_FILES,
    },
];

const MODEL_SOURCES: [ModelSource; 3] = [
    ModelSource {
        name: "Hugging Face",
        base_url: HUGGING_FACE_BASE_URL,
        onnx_tree_url: HUGGING_FACE_ONNX_TREE_URL,
        kind: ModelSourceKind::Direct,
    },
    ModelSource {
        name: "ModelScope",
        base_url: MODELSCOPE_BASE_URL,
        onnx_tree_url: MODELSCOPE_ONNX_TREE_URL,
        kind: ModelSourceKind::Direct,
    },
    ModelSource {
        name: "CNB",
        base_url: CNB_RAW_BASE_URL,
        onnx_tree_url: CNB_ONNX_TREE_URL,
        kind: ModelSourceKind::CnbLfs,
    },
];

pub fn resolve_model_files(
    model_id: &str,
    onnx_variant: &str,
    model_dir: &Path,
    model_check: bool,
) -> Result<ModelFiles> {
    let spec = find_onnx_model_spec(onnx_variant)?;
    let model_root = model_root_dir(model_dir, model_id)?;

    if let Some(files) = find_model_files_in_root(spec, &model_root) {
        return Ok(files);
    }

    if !model_check {
        return Err(anyhow!(
            "model files not found under {}; set MODEL_CHECK=true to automatically download MODEL_ID={} ONNX_VARIANT={}",
            model_root.display(),
            model_id,
            onnx_variant
        ));
    }

    download_model_files(spec, &model_root)
}

fn model_root_dir(model_dir: &Path, model_id: &str) -> Result<PathBuf> {
    let model_id = model_id.trim();
    if model_id.is_empty() {
        return Err(anyhow!("invalid MODEL_ID: model id must not be empty"));
    }

    let model_id_path = Path::new(model_id);
    if !model_id_path
        .components()
        .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(anyhow!(
            "invalid MODEL_ID: {model_id}; expected a relative model id like openai/privacy-filter"
        ));
    }

    Ok(forward_slash_join(model_dir, model_id))
}

fn forward_slash_join(base: &Path, relative: &str) -> PathBuf {
    let mut base = base.to_string_lossy().replace('\\', "/");
    while base.len() > 1 && base.ends_with('/') {
        base.pop();
    }

    let relative = relative.replace('\\', "/");
    let relative = relative.trim_start_matches('/');

    if base.is_empty() {
        PathBuf::from(relative)
    } else if relative.is_empty() {
        PathBuf::from(base)
    } else if base == "/" {
        PathBuf::from(format!("/{relative}"))
    } else {
        PathBuf::from(format!("{base}/{relative}"))
    }
}

fn find_onnx_model_spec(onnx_variant: &str) -> Result<OnnxModelSpec> {
    ONNX_MODEL_SPECS
        .iter()
        .copied()
        .find(|spec| spec.variant == onnx_variant)
        .ok_or_else(|| {
            anyhow!(
                "invalid ONNX_VARIANT: {onnx_variant}; expected one of: {}",
                ONNX_MODEL_SPECS
                    .iter()
                    .map(|spec| spec.variant)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

pub fn find_local_model_files(
    model_id: &str,
    onnx_variant: &str,
    model_dir: &Path,
) -> Option<ModelFiles> {
    let spec = find_onnx_model_spec(onnx_variant).ok()?;
    let model_root = model_root_dir(model_dir, model_id).ok()?;
    find_model_files_in_root(spec, &model_root)
}

fn find_model_files_in_root(spec: OnnxModelSpec, model_root: &Path) -> Option<ModelFiles> {
    let tokenizer_path = forward_slash_join(model_root, TOKENIZER_FILE);
    let onnx_path = forward_slash_join(model_root, spec.onnx_file);

    if tokenizer_path.is_file()
        && spec
            .files
            .iter()
            .all(|file| forward_slash_join(model_root, file).is_file())
    {
        Some(ModelFiles {
            root: model_root.to_path_buf(),
            tokenizer_path,
            onnx_path,
        })
    } else {
        None
    }
}

fn download_model_files(spec: OnnxModelSpec, model_root: &Path) -> Result<ModelFiles> {
    fs::create_dir_all(model_root)
        .with_context(|| format!("failed to create model dir {}", model_root.display()))?;

    let source = select_fastest_source()?;
    tracing::info!(source = source.name, "selected model download source");

    let mut required_files = Vec::with_capacity(spec.files.len() + 1);
    required_files.push(TOKENIZER_FILE);
    required_files.extend_from_slice(spec.files);

    for filename in required_files {
        let target_path = forward_slash_join(model_root, filename);
        if target_path.is_file() {
            continue;
        }
        download_file(source, filename, &target_path).with_context(|| {
            format!(
                "available ONNX files on {}: {}",
                source.name,
                list_available_onnx_files(source)
                    .map(|files| files.join(", "))
                    .unwrap_or_else(|err| format!("failed to query model list: {err}"))
            )
        })?;
    }

    find_model_files_in_root(spec, model_root).ok_or_else(|| {
        anyhow!(
            "model files are incomplete after download; expected {} and [{}] under {}",
            TOKENIZER_FILE,
            spec.files.join(", "),
            model_root.display()
        )
    })
}

fn select_fastest_source() -> Result<ModelSource> {
    let handles = MODEL_SOURCES.map(|source| thread::spawn(move || probe_source(source)));
    let mut probes = Vec::new();
    let mut errors = Vec::new();

    for handle in handles {
        match handle
            .join()
            .map_err(|_| anyhow!("model source probe thread panicked"))?
        {
            Ok(probe) => probes.push(probe),
            Err(err) => errors.push(err.to_string()),
        }
    }

    for probe in &probes {
        let speed = format_probe_speed(probe.bytes_read, probe.elapsed);
        eprintln!(
            "模型源测速 {}: {} ms, {}, {}",
            probe.source.name,
            probe.elapsed.as_millis(),
            format_bytes(probe.bytes_read as u64),
            speed
        );
        tracing::info!(
            source = probe.source.name,
            elapsed_ms = probe.elapsed.as_millis(),
            bytes_read = probe.bytes_read,
            speed = %speed,
            content_length = probe.content_length,
            "model source probe result"
        );
    }
    for error in &errors {
        eprintln!("模型源测速失败: {error}");
        tracing::warn!(error = %error, "model source probe failed");
    }

    probes.sort_by_key(|probe| probe.elapsed);
    let selected = probes.into_iter().next().ok_or_else(|| {
        anyhow!(
            "no available model download source; probe errors: {}",
            errors.join("; ")
        )
    })?;

    eprintln!(
        "选择模型下载源: {} (测速耗时最短: {} ms)",
        selected.source.name,
        selected.elapsed.as_millis()
    );
    tracing::info!(
        source = selected.source.name,
        elapsed_ms = selected.elapsed.as_millis(),
        content_length = selected.content_length,
        "selected model download source by fastest probe"
    );

    Ok(selected.source)
}

fn probe_source(source: ModelSource) -> Result<ProbeResult> {
    let client = build_http_client(PROBE_TIMEOUT).context("failed to build probe HTTP client")?;
    let start = Instant::now();
    let download = resolve_source_download(&client, source, TOKENIZER_FILE)?;
    let response = source_download_request(&client, &download)
        .header(RANGE, PROBE_RANGE)
        .send()
        .with_context(|| format!("failed to probe {} at {}", source.name, download.url))?;
    let status = response.status();
    if !status.is_success() {
        return Err(anyhow!(
            "failed to probe {} at {}: HTTP {}",
            source.name,
            download.url,
            status
        ));
    }

    let content_length = response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let mut reader = response.take(1024);
    let mut buffer = Vec::new();
    reader
        .read_to_end(&mut buffer)
        .with_context(|| format!("failed to read probe response from {}", source.name))?;

    if buffer.is_empty() {
        return Err(anyhow!("empty probe response from {}", source.name));
    }

    Ok(ProbeResult {
        source,
        elapsed: start.elapsed(),
        bytes_read: buffer.len(),
        content_length,
    })
}

fn build_http_client(timeout: Duration) -> reqwest::Result<Client> {
    Client::builder()
        .timeout(timeout)
        .user_agent(HTTP_USER_AGENT)
        .build()
}

fn resolve_source_download(
    client: &Client,
    source: ModelSource,
    filename: &str,
) -> Result<SourceDownload> {
    match source.kind {
        ModelSourceKind::Direct => Ok(SourceDownload {
            url: file_url(source, filename),
        }),
        ModelSourceKind::CnbLfs => resolve_cnb_lfs_download(client, filename),
    }
}

fn source_download_request<'a>(
    client: &'a Client,
    download: &'a SourceDownload,
) -> reqwest::blocking::RequestBuilder {
    client.get(&download.url)
}

fn resolve_cnb_lfs_download(client: &Client, filename: &str) -> Result<SourceDownload> {
    let raw_url = format!("{CNB_RAW_BASE_URL}/{filename}");
    let pointer_text = client
        .get(&raw_url)
        .send()
        .with_context(|| format!("failed to fetch CNB raw pointer for {filename}"))?
        .error_for_status()
        .with_context(|| format!("failed to fetch CNB raw pointer from {raw_url}"))?
        .text()
        .with_context(|| format!("failed to read CNB raw pointer for {filename}"))?;

    if !pointer_text
        .lines()
        .any(|line| line.trim() == "version https://git-lfs.github.com/spec/v1")
    {
        return Ok(SourceDownload { url: raw_url });
    }

    let pointer = parse_cnb_lfs_pointer(&pointer_text)
        .with_context(|| format!("failed to parse CNB LFS pointer for {filename}"))?;
    tracing::debug!(
        filename,
        oid = pointer.oid,
        size = pointer.size,
        "resolved CNB LFS pointer"
    );

    Ok(SourceDownload {
        url: cnb_lfs_download_url(&pointer.oid, filename),
    })
}

fn cnb_lfs_download_url(oid: &str, filename: &str) -> String {
    let display_name = filename.rsplit('/').next().unwrap_or(filename);
    format!("{CNB_LFS_BASE_URL}/{oid}?name={display_name}")
}

fn parse_cnb_lfs_pointer(pointer_text: &str) -> Result<CnbLfsPointer> {
    let mut oid = None;
    let mut size = None;

    for line in pointer_text.lines() {
        if let Some(value) = line.strip_prefix("oid sha256:") {
            oid = Some(value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("size ") {
            size = Some(
                value
                    .trim()
                    .parse::<u64>()
                    .with_context(|| format!("invalid CNB LFS pointer size: {value}"))?,
            );
        }
    }

    Ok(CnbLfsPointer {
        oid: oid.ok_or_else(|| anyhow!("missing oid in CNB LFS pointer"))?,
        size: size.ok_or_else(|| anyhow!("missing size in CNB LFS pointer"))?,
    })
}

fn format_probe_speed(bytes_read: usize, elapsed: Duration) -> String {
    let seconds = elapsed.as_secs_f64();
    if seconds <= f64::EPSILON {
        return "unknown/s".to_string();
    }

    format!("{}/s", format_bytes((bytes_read as f64 / seconds) as u64))
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut size = bytes as f64;
    let mut unit = UNITS[0];

    for next_unit in UNITS.iter().skip(1) {
        if size < 1024.0 {
            break;
        }
        size /= 1024.0;
        unit = next_unit;
    }

    if unit == "B" {
        format!("{bytes} {unit}")
    } else {
        format!("{size:.1} {unit}")
    }
}

fn download_file(source: ModelSource, filename: &str, target_path: &Path) -> Result<()> {
    let parent = target_path
        .parent()
        .ok_or_else(|| anyhow!("target file has no parent: {}", target_path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create model file dir {}", parent.display()))?;

    let client =
        build_http_client(DOWNLOAD_TIMEOUT).context("failed to build download HTTP client")?;
    let download = resolve_source_download(&client, source, filename)?;
    let tmp_path = target_path.with_extension(format!(
        "{}.tmp",
        target_path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("download")
    ));

    tracing::info!(
        source = source.name,
        url = %download.url,
        target = %target_path.display(),
        "downloading model file"
    );

    let mut response = source_download_request(&client, &download)
        .send()
        .with_context(|| format!("failed to download {filename} from {}", source.name))?
        .error_for_status()
        .with_context(|| format!("failed to download {filename} from {}", download.url))?;
    let total_size = response.content_length();
    let progress = download_progress_bar(filename, total_size);
    let mut tmp_file = fs::File::create(&tmp_path)
        .with_context(|| format!("failed to create temp file {}", tmp_path.display()))?;

    let mut downloaded = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let bytes_read = response
            .read(&mut buffer)
            .with_context(|| format!("failed to read download stream for {filename}"))?;
        if bytes_read == 0 {
            break;
        }

        tmp_file
            .write_all(&buffer[..bytes_read])
            .with_context(|| format!("failed to write temp file {}", tmp_path.display()))?;
        downloaded += bytes_read as u64;
        progress.set_position(downloaded);
    }

    progress.finish_with_message(format!("{filename} done"));
    tmp_file
        .flush()
        .with_context(|| format!("failed to flush temp file {}", tmp_path.display()))?;

    fs::rename(&tmp_path, target_path).with_context(|| {
        format!(
            "failed to move temp file {} to {}",
            tmp_path.display(),
            target_path.display()
        )
    })?;

    Ok(())
}

fn download_progress_bar(filename: &str, total_size: Option<u64>) -> ProgressBar {
    let progress = match total_size {
        Some(size) => ProgressBar::new(size),
        None => ProgressBar::new_spinner(),
    };

    let style = match total_size {
        Some(_) => ProgressStyle::with_template(
            "{msg} {wide_bar:.cyan/blue} {bytes}/{total_bytes} {bytes_per_sec}",
        ),
        None => ProgressStyle::with_template("{msg} {spinner} {bytes} {bytes_per_sec}"),
    }
    .unwrap_or_else(|_| ProgressStyle::default_bar());

    progress.set_style(style);
    progress.set_message(filename.to_string());
    progress
}
fn list_available_onnx_files(source: ModelSource) -> Result<Vec<String>> {
    let client =
        build_http_client(PROBE_TIMEOUT).context("failed to build model list HTTP client")?;
    let html = client
        .get(source.onnx_tree_url)
        .send()
        .with_context(|| format!("failed to query ONNX file list from {}", source.name))?
        .error_for_status()
        .with_context(|| {
            format!(
                "failed to query ONNX file list from {}",
                source.onnx_tree_url
            )
        })?
        .text()
        .with_context(|| format!("failed to read ONNX file list from {}", source.name))?;

    let mut files = Vec::new();
    for token in html.split(['\"', '\'', '<', '>', ' ', '\n', '\r', '\t']) {
        if let Some(start) = token.find("onnx/") {
            let candidate = token[start..]
                .split(['?', '#'])
                .next()
                .unwrap_or_default()
                .trim_matches('/');
            if candidate.ends_with(".onnx") || candidate.contains(".onnx_data") {
                let candidate = candidate.to_string();
                if !files.contains(&candidate) {
                    files.push(candidate);
                }
            }
        }
    }

    files.sort();
    if files.is_empty() {
        Err(anyhow!(
            "no ONNX files found in model list page {}",
            source.onnx_tree_url
        ))
    } else {
        Ok(files)
    }
}

fn file_url(source: ModelSource, filename: &str) -> String {
    format!("{}/{}", source.base_url, filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_ID: &str = "openai/privacy-filter";

    #[test]
    fn finds_model_id_directory_layout() {
        let temp = tempfile::tempdir().unwrap();
        let model_root = forward_slash_join(temp.path(), MODEL_ID);
        fs::create_dir_all(forward_slash_join(&model_root, "onnx")).unwrap();
        fs::write(forward_slash_join(&model_root, TOKENIZER_FILE), "{}").unwrap();
        fs::write(
            forward_slash_join(&model_root, "onnx/model_quantized.onnx"),
            "onnx",
        )
        .unwrap();
        fs::write(
            forward_slash_join(&model_root, "onnx/model_quantized.onnx_data"),
            "data",
        )
        .unwrap();

        let files = find_local_model_files(MODEL_ID, "quantized", temp.path()).unwrap();
        assert_eq!(files.root, model_root);
        assert_eq!(
            files.tokenizer_path,
            forward_slash_join(&model_root, TOKENIZER_FILE)
        );
        assert_eq!(
            files.onnx_path,
            forward_slash_join(&model_root, "onnx/model_quantized.onnx")
        );
    }

    #[test]
    fn finds_fp32_directory_layout_with_multiple_external_data_files() {
        let temp = tempfile::tempdir().unwrap();
        let model_root = forward_slash_join(temp.path(), MODEL_ID);
        fs::create_dir_all(forward_slash_join(&model_root, "onnx")).unwrap();
        fs::write(forward_slash_join(&model_root, TOKENIZER_FILE), "{}").unwrap();
        for file in FP32_FILES {
            fs::write(forward_slash_join(&model_root, file), "data").unwrap();
        }

        let files = find_local_model_files(MODEL_ID, "fp32", temp.path()).unwrap();
        assert_eq!(
            files.onnx_path,
            forward_slash_join(&model_root, "onnx/model.onnx")
        );
    }

    #[test]
    fn ignores_incomplete_model_id_directory_layout() {
        let temp = tempfile::tempdir().unwrap();
        let model_root = forward_slash_join(temp.path(), MODEL_ID);
        fs::create_dir_all(forward_slash_join(&model_root, "onnx")).unwrap();
        fs::write(forward_slash_join(&model_root, TOKENIZER_FILE), "{}").unwrap();
        fs::write(
            forward_slash_join(&model_root, "onnx/model_quantized.onnx"),
            "onnx",
        )
        .unwrap();

        assert!(find_local_model_files(MODEL_ID, "quantized", temp.path()).is_none());
    }

    #[test]
    fn builds_forward_slash_local_model_paths() {
        let root = model_root_dir(Path::new(r".\models"), MODEL_ID).unwrap();
        let onnx_path = forward_slash_join(&root, "onnx/model_quantized.onnx");

        assert_eq!(root.to_string_lossy(), "./models/openai/privacy-filter");
        assert_eq!(
            onnx_path.to_string_lossy(),
            "./models/openai/privacy-filter/onnx/model_quantized.onnx"
        );
    }

    #[test]
    fn rejects_invalid_model_id_paths() {
        let temp = tempfile::tempdir().unwrap();
        let err = resolve_model_files("../privacy-filter", "quantized", temp.path(), false)
            .expect_err("path traversal model id should be rejected");

        assert!(err.to_string().contains("invalid MODEL_ID"));
    }

    #[test]
    fn builds_source_file_urls() {
        let url = file_url(MODEL_SOURCES[1], "onnx/model_quantized.onnx");
        assert_eq!(
            url,
            "https://modelscope.cn/models/openai-mirror/privacy-filter/resolve/master/onnx/model_quantized.onnx"
        );

        let cnb_url = file_url(MODEL_SOURCES[2], TOKENIZER_FILE);
        assert_eq!(
            cnb_url,
            "https://cnb.cool/ai-models/openai/privacy-filter/-/git/raw/main/tokenizer.json"
        );
    }

    #[test]
    fn parses_cnb_lfs_pointer() {
        let pointer = parse_cnb_lfs_pointer(
            "version https://git-lfs.github.com/spec/v1\noid sha256:0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3\nsize 27868174\n",
        )
        .unwrap();

        assert_eq!(
            pointer.oid,
            "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3"
        );
        assert_eq!(pointer.size, 27_868_174);
    }

    #[test]
    fn builds_cnb_lfs_download_urls_from_pointer_oid_and_filename() {
        let url = cnb_lfs_download_url(
            "50f4c8c7f3c27fbc1fe16d4f74f6f7c3b74ba8f18a262e8b6911854c64c33a6d",
            "onnx/model_quantized.onnx_data",
        );

        assert_eq!(
            url,
            "https://cnb.cool/ai-models/openai/privacy-filter/-/lfs/50f4c8c7f3c27fbc1fe16d4f74f6f7c3b74ba8f18a262e8b6911854c64c33a6d?name=model_quantized.onnx_data"
        );
    }
}
