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
    install_panic_hook();

    config::load_dotenv();
    let config = Arc::new(Config::from_args()?);

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer().with_timer(ChronoLocal::rfc_3339()))
        .init();

    configure_onnx_runtime(config.omp_num_threads)?;
    ensure_model_files(config.as_ref())?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build Tokio runtime")?;
    runtime.block_on(run(config))
}

fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let location = info.location();
        let payload = info.payload();
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "Box<dyn Any>".to_string()
        };

        let bt = std::backtrace::Backtrace::force_capture();
        let location_str = location
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "<unknown>".to_string());

        eprintln!(
            "\n========================================\n\
             PANIC: process will abort (panic=abort)\n\
             message: {msg}\n\
             location: {location_str}\n\
             backtrace:\n{bt}\n\
             ========================================"
        );
    }));
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

    let server = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal());

    match server.await {
        Ok(()) => {
            tracing::info!("server stopped");
            Ok(())
        }
        Err(err) => {
            let err = anyhow::Error::from(err).context("server exited with error");
            tracing::error!(error = %err, "server exited with error");
            Err(err)
        }
    }
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut sigint = signal(SignalKind::interrupt()).expect("failed to install SIGINT handler");
        let mut sigterm =
            signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
        let mut sighup = signal(SignalKind::hangup()).expect("failed to install SIGHUP handler");
        let mut sigquit = signal(SignalKind::quit()).expect("failed to install SIGQUIT handler");

        tokio::select! {
            _ = sigint.recv() => {
                tracing::info!("received SIGINT (Ctrl+C), starting graceful shutdown");
            }
            _ = sigterm.recv() => {
                tracing::warn!(
                    "received SIGTERM, starting graceful shutdown (non-manual termination)"
                );
            }
            _ = sighup.recv() => {
                tracing::warn!(
                    "received SIGHUP, starting graceful shutdown (non-manual termination)"
                );
            }
            _ = sigquit.recv() => {
                tracing::warn!(
                    "received SIGQUIT, starting graceful shutdown (non-manual termination)"
                );
            }
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
        tracing::info!("received Ctrl+C, starting graceful shutdown");
    }
}
