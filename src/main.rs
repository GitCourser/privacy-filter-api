use anyhow::{Context, Result};
use privacy_filter_api::{
    api,
    config::{self, Config},
    model::PrivacyFilterModel,
    model_download::resolve_model_files,
    onnx_runtime::configure_onnx_runtime,
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing_subscriber::{fmt::time::ChronoLocal, layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> Result<()> {
    config::load_dotenv();
    let config = Arc::new(Config::from_args()?);

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer().with_timer(ChronoLocal::rfc_3339()))
        .init();

    configure_onnx_runtime()?;
    ensure_model_files(config.as_ref())?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build Tokio runtime")?;
    runtime.block_on(run(config))
}

fn ensure_model_files(config: &Config) -> Result<()> {
    if !config.model_check {
        tracing::info!("startup model check skipped because MODEL_CHECK=false");
        return Ok(());
    }

    tracing::info!(
        model_id = %config.model_id,
        onnx_variant = %config.onnx_variant,
        model_dir = %config.model_dir.display(),
        "startup model check started"
    );
    let files = resolve_model_files(
        &config.model_id,
        &config.onnx_variant,
        &config.model_dir,
        true,
    )?;
    tracing::info!(
        root = %files.root.display(),
        tokenizer = %files.tokenizer_path.display(),
        onnx = %files.onnx_path.display(),
        "startup model check completed"
    );
    Ok(())
}

async fn run(config: Arc<Config>) -> Result<()> {
    let addr = config.bind_addr()?;
    let model = Arc::new(PrivacyFilterModel::new((*config).clone()));
    let app = api::router(config, model);

    let listener = TcpListener::bind(addr).await?;
    tracing::info!("listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}
