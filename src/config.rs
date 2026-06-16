use std::env;
use std::net::IpAddr;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use clap::Parser;

pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 4175;
pub const DEFAULT_MODEL_ID: &str = "openai/privacy-filter";
pub const DEFAULT_ONNX_VARIANT: &str = "quantized";
pub const DEFAULT_MODEL_DIR: &str = "./models";
pub const DEFAULT_MODEL_CHECK: bool = true;
pub const DEFAULT_MAX_TOKENS: usize = 128000;

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct CliArgs {
    /// 服务监听 IP；对应 .env 配置 HOST
    #[arg(long)]
    host: Option<String>,

    /// 服务监听端口；对应 .env 配置 PORT
    #[arg(long)]
    port: Option<u16>,

    /// API Key；非空时 /detect 和 /mask 需要 x-api-key；对应 .env 配置 API_KEY
    #[arg(long)]
    api_key: Option<String>,

    /// 模型仓库 ID；对应 .env 配置 MODEL_ID
    #[arg(long)]
    model_id: Option<String>,

    /// ONNX 型号，可选 fp32、fp16、q4、q4f16、quantized；对应 .env 配置 ONNX_VARIANT
    #[arg(long)]
    onnx_variant: Option<String>,

    /// 本地模型目录；对应 .env 配置 MODEL_DIR
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// 是否启动时检测本地模型并在缺失时自动下载；对应 .env 配置 MODEL_CHECK
    #[arg(long)]
    model_check: Option<bool>,

    /// 单次推理最大 token 数；对应 .env 配置 MAX_TOKENS
    #[arg(long)]
    max_tokens: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub api_key: String,
    pub model_id: String,
    pub onnx_variant: String,
    pub model_dir: PathBuf,
    pub model_check: bool,
    pub max_tokens: usize,
}

impl Config {
    pub fn from_args() -> Result<Self> {
        Self::from_cli_args(CliArgs::parse())
    }

    fn from_cli_args(args: CliArgs) -> Result<Self> {
        let host = args
            .host
            .unwrap_or_else(|| env_string("HOST", DEFAULT_HOST));
        let port = match args.port {
            Some(port) => port,
            None => parse_env("PORT", DEFAULT_PORT)?,
        };
        let api_key = args.api_key.unwrap_or_else(|| env_string("API_KEY", ""));
        let model_id = args
            .model_id
            .unwrap_or_else(|| env_string("MODEL_ID", DEFAULT_MODEL_ID));
        let onnx_variant = args
            .onnx_variant
            .unwrap_or_else(|| env_string("ONNX_VARIANT", DEFAULT_ONNX_VARIANT));
        let model_dir = args.model_dir.unwrap_or_else(|| {
            env::var("MODEL_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from(DEFAULT_MODEL_DIR))
        });
        let model_check = match args.model_check {
            Some(model_check) => model_check,
            None => parse_env("MODEL_CHECK", DEFAULT_MODEL_CHECK)?,
        };
        let max_tokens = match args.max_tokens {
            Some(max_tokens) => max_tokens,
            None => parse_env("MAX_TOKENS", DEFAULT_MAX_TOKENS)?,
        };

        Ok(Self {
            host,
            port,
            api_key,
            model_id,
            onnx_variant,
            model_dir,
            model_check,
            max_tokens,
        })
    }

    pub fn bind_addr(&self) -> Result<std::net::SocketAddr> {
        let ip: IpAddr = self
            .host
            .parse()
            .with_context(|| format!("invalid HOST: {}", self.host))?;
        Ok(std::net::SocketAddr::new(ip, self.port))
    }

    pub fn auth_enabled(&self) -> bool {
        !self.api_key.is_empty()
    }
}

pub fn load_dotenv() {
    match dotenvy::dotenv() {
        Ok(path) => tracing::debug!(path = %path.display(), "loaded .env file"),
        Err(dotenvy::Error::Io(err)) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => tracing::warn!(error = %err, "failed to load .env file"),
    }
}

fn env_string(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn parse_env<T>(name: &str, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match env::var(name) {
        Ok(value) => value
            .parse::<T>()
            .map_err(|err| anyhow!("invalid {name}: {err}")),
        Err(_) => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clear_env() -> [(String, Option<String>); 8] {
        [
            ("HOST".to_string(), None),
            ("PORT".to_string(), None),
            ("API_KEY".to_string(), None),
            ("MODEL_ID".to_string(), None),
            ("ONNX_VARIANT".to_string(), None),
            ("MODEL_DIR".to_string(), None),
            ("MODEL_CHECK".to_string(), None),
            ("MAX_TOKENS".to_string(), None),
        ]
    }

    #[test]
    fn default_config_uses_unprefixed_defaults() {
        temp_env::with_vars(clear_env(), || {
            let config =
                Config::from_cli_args(CliArgs::parse_from(["privacy-filter-api"])).unwrap();
            assert_eq!(config.host, DEFAULT_HOST);
            assert_eq!(config.port, DEFAULT_PORT);
            assert_eq!(config.api_key, "");
            assert_eq!(config.model_id, DEFAULT_MODEL_ID);
            assert_eq!(config.onnx_variant, DEFAULT_ONNX_VARIANT);
            assert_eq!(config.model_dir, PathBuf::from(DEFAULT_MODEL_DIR));
            assert_eq!(config.model_check, DEFAULT_MODEL_CHECK);
            assert_eq!(config.max_tokens, DEFAULT_MAX_TOKENS);
            assert!(!config.auth_enabled());
        });
    }

    #[test]
    fn cli_overrides_env() {
        temp_env::with_vars(
            [
                ("HOST", Some("0.0.0.0")),
                ("PORT", Some("8080")),
                ("API_KEY", Some("from-env")),
                ("MODEL_ID", Some("env/model")),
                ("ONNX_VARIANT", Some("fp16")),
                ("MODEL_DIR", Some("./env-models")),
                ("MODEL_CHECK", Some("false")),
                ("MAX_TOKENS", Some("1024")),
            ],
            || {
                let config = Config::from_cli_args(CliArgs::parse_from([
                    "privacy-filter-api",
                    "--host",
                    "127.0.0.2",
                    "--port",
                    "9090",
                    "--api-key",
                    "from-cli",
                    "--model-id",
                    "cli/model",
                    "--onnx-variant",
                    "q4",
                    "--model-dir",
                    "./cli-models",
                    "--model-check",
                    "true",
                    "--max-tokens",
                    "2048",
                ]))
                .unwrap();

                assert_eq!(config.host, "127.0.0.2");
                assert_eq!(config.port, 9090);
                assert_eq!(config.api_key, "from-cli");
                assert_eq!(config.model_id, "cli/model");
                assert_eq!(config.onnx_variant, "q4");
                assert_eq!(config.model_dir, PathBuf::from("./cli-models"));
                assert!(config.model_check);
                assert_eq!(config.max_tokens, 2048);
            },
        );
    }
}
