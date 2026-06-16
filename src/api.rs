use std::sync::Arc;

use axum::Router;
use axum::extract::{Json, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::Config;
use crate::entity::Entity;
use crate::model::{IoInfo, PrivacyFilterModel};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub model: Arc<PrivacyFilterModel>,
}

#[derive(Debug, Deserialize)]
pub struct TextRequest {
    pub text: Value,
    #[serde(default)]
    pub aggregation_strategy: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MaskRequest {
    pub text: Value,
    #[serde(default)]
    pub mask_token: Option<Value>,
    #[serde(default)]
    pub aggregation_strategy: Option<String>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    ok: bool,
    model: String,
    loaded: bool,
    auth: &'static str,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

#[derive(Debug, Serialize)]
struct InspectResponse {
    model: String,
    inputs: Vec<IoInfo>,
    outputs: Vec<IoInfo>,
}

#[derive(Debug, Serialize)]
struct DetectResponse {
    model: String,
    entities: Vec<Entity>,
}

#[derive(Debug, Serialize)]
struct MaskResponse {
    model: String,
    masked_text: String,
    entities: Vec<Entity>,
}

pub fn router(config: Arc<Config>, model: Arc<PrivacyFilterModel>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/inspect", get(inspect))
        .route("/detect", post(detect))
        .route("/mask", post(mask))
        .with_state(AppState { config, model })
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        ok: true,
        model: state.model.model_id().to_string(),
        loaded: state.model.loaded(),
        auth: if state.config.auth_enabled() {
            "enabled"
        } else {
            "disabled"
        },
    })
}

async fn inspect(State(state): State<AppState>) -> Response {
    let model = Arc::clone(&state.model);
    match tokio::task::spawn_blocking(move || model.inspect_io()).await {
        Ok(Ok(info)) => Json(InspectResponse {
            model: state.model.model_id().to_string(),
            inputs: info.inputs,
            outputs: info.outputs,
        })
        .into_response(),
        Ok(Err(err)) => internal_error(err),
        Err(err) => internal_error(anyhow::anyhow!("model inspect task failed: {err}")),
    }
}

async fn detect(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<TextRequest>,
) -> Response {
    if let Some(response) = check_auth(&state.config, &headers) {
        return response;
    }

    let Some(text) = payload.text.as_str() else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "`text` must be a non-empty string.",
        );
    };

    if text.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "`text` must be a non-empty string.",
        );
    }

    // aggregation_strategy is accepted but ignored (HuggingFace Transformers.js default is "simple")
    // The Rust ONNX model always uses greedy decoding which is equivalent to "simple"

    let model = Arc::clone(&state.model);
    let text = text.to_string();
    match tokio::task::spawn_blocking(move || model.detect(&text)).await {
        Ok(Ok(entities)) => Json(DetectResponse {
            model: state.model.model_id().to_string(),
            entities,
        })
        .into_response(),
        Ok(Err(err)) => internal_error(err),
        Err(err) => internal_error(anyhow::anyhow!("model detect task failed: {err}")),
    }
}

async fn mask(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<MaskRequest>,
) -> Response {
    if let Some(response) = check_auth(&state.config, &headers) {
        return response;
    }

    let Some(text) = payload.text.as_str() else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "`text` must be a non-empty string.",
        );
    };

    if text.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "`text` must be a non-empty string.",
        );
    }

    // aggregation_strategy is accepted but ignored (same as /detect)

    let mask_token = match payload.mask_token {
        Some(value) => match value.as_str() {
            Some(mask_token) => mask_token.to_string(),
            None => return json_error(StatusCode::BAD_REQUEST, "`mask_token` must be a string."),
        },
        None => "[{label}]".to_string(),
    };

    let model = Arc::clone(&state.model);
    let text = text.to_string();
    match tokio::task::spawn_blocking(move || model.mask(&text, &mask_token)).await {
        Ok(Ok((masked_text, entities))) => Json(MaskResponse {
            model: state.model.model_id().to_string(),
            masked_text,
            entities,
        })
        .into_response(),
        Ok(Err(err)) => internal_error(err),
        Err(err) => internal_error(anyhow::anyhow!("model mask task failed: {err}")),
    }
}

fn check_auth(config: &Config, headers: &HeaderMap) -> Option<Response> {
    if !config.auth_enabled() {
        return None;
    }

    // Check x-api-key header
    let api_key_header = headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok());
    if api_key_header == Some(config.api_key.as_str()) {
        return None;
    }

    // Check Bearer token (Authorization: Bearer <key>)
    let bearer_token = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    if bearer_token == Some(config.api_key.as_str()) {
        return None;
    }

    Some(json_error(StatusCode::UNAUTHORIZED, "Unauthorized."))
}

fn json_error(status: StatusCode, detail: &str) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: detail.to_string(),
            message: None,
        }),
    )
        .into_response()
}

fn internal_error(err: anyhow::Error) -> Response {
    tracing::error!(error = %err, "request failed");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: "Inference failed.".to_string(),
            message: Some(err.to_string()),
        }),
    )
        .into_response()
}
