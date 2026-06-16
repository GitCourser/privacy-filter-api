use std::sync::{Mutex, MutexGuard};

use anyhow::{Context, Result, anyhow};
use ndarray::{Array2, ArrayView2, Axis, Ix2, Ix3};
use ort::{session::Session, value::TensorRef};
use tokenizers::Tokenizer;

use crate::config::Config;
use crate::entity::Entity;
use crate::model_download::resolve_model_files;

pub const ID2LABEL: [&str; 33] = [
    "O",
    "B-account_number",
    "I-account_number",
    "E-account_number",
    "S-account_number",
    "B-private_address",
    "I-private_address",
    "E-private_address",
    "S-private_address",
    "B-private_date",
    "I-private_date",
    "E-private_date",
    "S-private_date",
    "B-private_email",
    "I-private_email",
    "E-private_email",
    "S-private_email",
    "B-private_person",
    "I-private_person",
    "E-private_person",
    "S-private_person",
    "B-private_phone",
    "I-private_phone",
    "E-private_phone",
    "S-private_phone",
    "B-private_url",
    "I-private_url",
    "E-private_url",
    "S-private_url",
    "B-secret",
    "I-secret",
    "E-secret",
    "S-secret",
];

#[derive(Debug, Clone, serde::Serialize)]
pub struct IoInfo {
    pub name: String,
    pub shape: Vec<String>,
    #[serde(rename = "type")]
    pub ty: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct InspectInfo {
    pub inputs: Vec<IoInfo>,
    pub outputs: Vec<IoInfo>,
}

pub struct PrivacyFilterModel {
    config: Config,
    state: Mutex<ModelState>,
}

struct ModelState {
    loaded: bool,
    tokenizer: Option<Tokenizer>,
    session: Option<Session>,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl PrivacyFilterModel {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            state: Mutex::new(ModelState {
                loaded: false,
                tokenizer: None,
                session: None,
                input_names: Vec::new(),
                output_names: Vec::new(),
            }),
        }
    }

    pub fn model_id(&self) -> &str {
        &self.config.model_id
    }

    pub fn loaded(&self) -> bool {
        self.state.lock().map(|state| state.loaded).unwrap_or(false)
    }

    pub fn inspect_io(&self) -> Result<InspectInfo> {
        let mut state = self.load_state()?;
        let session = state
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("model session is not loaded"))?;
        Ok(InspectInfo {
            inputs: session
                .inputs()
                .iter()
                .map(|input| IoInfo {
                    name: input.name().to_string(),
                    shape: input
                        .dtype()
                        .tensor_shape()
                        .map(|shape| shape.iter().map(|dim| dim.to_string()).collect())
                        .unwrap_or_default(),
                    ty: input
                        .dtype()
                        .tensor_type()
                        .map(|ty| format!("{:?}", ty))
                        .unwrap_or_else(|| format!("{:?}", input.dtype())),
                })
                .collect(),
            outputs: session
                .outputs()
                .iter()
                .map(|output| IoInfo {
                    name: output.name().to_string(),
                    shape: output
                        .dtype()
                        .tensor_shape()
                        .map(|shape| shape.iter().map(|dim| dim.to_string()).collect())
                        .unwrap_or_default(),
                    ty: output
                        .dtype()
                        .tensor_type()
                        .map(|ty| format!("{:?}", ty))
                        .unwrap_or_else(|| format!("{:?}", output.dtype())),
                })
                .collect(),
        })
    }

    pub fn detect(&self, text: &str) -> Result<Vec<Entity>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let mut state = self.load_state()?;
        let tokenizer = state
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("tokenizer is not loaded"))?;
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|err| anyhow!("tokenize failed: {err}"))?;
        let token_ids: Vec<i64> = encoding
            .get_ids()
            .iter()
            .take(self.config.max_tokens)
            .map(|id| i64::from(*id))
            .collect();
        let offsets: Vec<(usize, usize)> = encoding
            .get_offsets()
            .iter()
            .take(self.config.max_tokens)
            .map(|&(start, end)| (start, end))
            .collect();

        if token_ids.is_empty() {
            return Ok(Vec::new());
        }

        let input_ids = Array2::from_shape_vec((1, token_ids.len()), token_ids)
            .context("failed to build input_ids tensor")?;
        let attention_mask = Array2::<i64>::ones((1, input_ids.len_of(Axis(1))));

        let input_ids_tensor = TensorRef::from_array_view(input_ids.view())?;
        let attention_mask_tensor = TensorRef::from_array_view(attention_mask.view())?;

        let session = state
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("model session is not loaded"))?;
        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ])?;
        let first = outputs
            .values()
            .next()
            .ok_or_else(|| anyhow!("model returned no outputs"))?;
        let (shape, logits_data) = first.try_extract_tensor::<f32>()?;
        let logits = ndarray::ArrayViewD::from_shape(shape.to_ixdyn(), logits_data)
            .context("failed to view logits tensor")?;

        match logits.ndim() {
            2 => {
                let token_logits = logits
                    .into_dimensionality::<Ix2>()
                    .context("failed to view logits as [tokens, labels]")?;
                Ok(decode_entities(text, token_logits, &offsets))
            }
            3 => {
                let batch_logits = logits
                    .into_dimensionality::<Ix3>()
                    .context("failed to view logits as [batch, tokens, labels]")?;
                let token_logits = batch_logits.index_axis(Axis(0), 0);
                Ok(decode_entities(text, token_logits, &offsets))
            }
            ndim => Err(anyhow!("unsupported logits dimension: {ndim}")),
        }
    }

    pub fn mask(&self, text: &str, mask_token: &str) -> Result<(String, Vec<Entity>)> {
        let entities = self.detect(text)?;
        Ok((mask_entities(text, mask_token, &entities), entities))
    }

    fn load_state(&self) -> Result<MutexGuard<'_, ModelState>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("model state lock poisoned"))?;
        if state.loaded {
            return Ok(state);
        }

        tracing::info!(
            model_id = %self.config.model_id,
            onnx_variant = %self.config.onnx_variant,
            model_dir = %self.config.model_dir.display(),
            model_check = self.config.model_check,
            "model load started"
        );
        tracing::info!("resolving model files");
        let files = resolve_model_files(
            &self.config.model_id,
            &self.config.onnx_variant,
            &self.config.model_dir,
            self.config.model_check,
        )?;
        tracing::info!(
            root = %files.root.display(),
            tokenizer = %files.tokenizer_path.display(),
            onnx = %files.onnx_path.display(),
            "model files resolved"
        );

        tracing::info!(tokenizer = %files.tokenizer_path.display(), "loading tokenizer");
        let tokenizer = Tokenizer::from_file(&files.tokenizer_path).map_err(|err| {
            anyhow!(
                "failed to load tokenizer {}: {err}",
                files.tokenizer_path.display()
            )
        })?;
        tracing::info!("tokenizer loaded");

        tracing::info!("creating ONNX Runtime session builder");
        let mut builder = Session::builder()?;
        tracing::info!(onnx = %files.onnx_path.display(), "loading ONNX model");
        let session = builder
            .commit_from_file(&files.onnx_path)
            .with_context(|| format!("failed to load ONNX model {}", files.onnx_path.display()))?;
        tracing::info!("ONNX model loaded");

        let input_names = session
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect();
        let output_names = session
            .outputs()
            .iter()
            .map(|output| output.name().to_string())
            .collect();

        state.tokenizer = Some(tokenizer);
        state.session = Some(session);
        state.input_names = input_names;
        state.output_names = output_names;
        state.loaded = true;
        tracing::info!("model load completed");
        Ok(state)
    }
}

pub fn decode_entities(
    text: &str,
    logits: ArrayView2<'_, f32>,
    offsets: &[(usize, usize)],
) -> Vec<Entity> {
    let probs = softmax(logits);
    let mut entities = Vec::new();
    let mut active_label: Option<String> = None;
    let mut active_start: Option<usize> = None;
    let mut active_end: Option<usize> = None;
    let mut active_scores: Vec<f32> = Vec::new();

    let close_active = |entities: &mut Vec<Entity>,
                        active_label: &mut Option<String>,
                        active_start: &mut Option<usize>,
                        active_end: &mut Option<usize>,
                        active_scores: &mut Vec<f32>| {
        if let (Some(label), Some(start), Some(end)) =
            (active_label.take(), active_start.take(), active_end.take())
        {
            if end > start {
                let score = mean(active_scores);
                entities.push(Entity::new(
                    label,
                    score,
                    text[start..end].to_string(),
                    None,
                    None,
                ));
            }
            active_scores.clear();
        }
    };

    let token_count = probs.len_of(Axis(0)).min(offsets.len());
    for (index, &(start, end)) in offsets.iter().enumerate().take(token_count) {
        if start == end {
            continue;
        }

        let row = probs.index_axis(Axis(0), index);
        let (label_id, score) = argmax(row.as_slice().unwrap_or(&[]));
        let raw_label = ID2LABEL.get(label_id).copied().unwrap_or("O");
        if raw_label == "O" {
            close_active(
                &mut entities,
                &mut active_label,
                &mut active_start,
                &mut active_end,
                &mut active_scores,
            );
            continue;
        }

        let Some((prefix, label)) = raw_label.split_once('-') else {
            continue;
        };

        if prefix == "S" {
            close_active(
                &mut entities,
                &mut active_label,
                &mut active_start,
                &mut active_end,
                &mut active_scores,
            );
            entities.push(Entity::new(
                label,
                score,
                text[start..end].to_string(),
                None,
                None,
            ));
            continue;
        }

        if prefix == "B" || active_label.as_deref() != Some(label) {
            close_active(
                &mut entities,
                &mut active_label,
                &mut active_start,
                &mut active_end,
                &mut active_scores,
            );
            active_label = Some(label.to_string());
            active_start = Some(start);
            active_end = Some(end);
            active_scores = vec![score];
            if prefix == "E" {
                close_active(
                    &mut entities,
                    &mut active_label,
                    &mut active_start,
                    &mut active_end,
                    &mut active_scores,
                );
            }
            continue;
        }

        active_end = Some(end);
        active_scores.push(score);
        if prefix == "E" {
            close_active(
                &mut entities,
                &mut active_label,
                &mut active_start,
                &mut active_end,
                &mut active_scores,
            );
        }
    }

    close_active(
        &mut entities,
        &mut active_label,
        &mut active_start,
        &mut active_end,
        &mut active_scores,
    );
    entities
}

pub fn softmax(values: ArrayView2<'_, f32>) -> Array2<f32> {
    let mut result = values.to_owned();
    for mut row in result.axis_iter_mut(Axis(0)) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for value in row.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        if sum != 0.0 {
            for value in row.iter_mut() {
                *value /= sum;
            }
        }
    }
    result
}

pub fn mask_entities(text: &str, mask_token: &str, entities: &[Entity]) -> String {
    // Try offset-based masking first (matches JS maskByOffsets)
    let mut sorted: Vec<&Entity> = entities
        .iter()
        .filter(|e| {
            if let (Some(start), Some(end)) = (e.start, e.end) {
                start < end
            } else {
                false
            }
        })
        .collect();
    sorted.sort_by_key(|e| e.start);

    if !sorted.is_empty() {
        let mut parts = Vec::new();
        let mut cursor = 0usize;
        for entity in sorted {
            let Some(start) = entity.start else { continue };
            let Some(end) = entity.end else { continue };
            if start < cursor {
                continue;
            }
            parts.push(text[cursor..start].to_string());
            parts.push(mask_token.replace("{label}", &entity.label));
            cursor = end;
        }
        parts.push(text[cursor..].to_string());
        return parts.concat();
    }

    // Fallback: text-based replacement (matches JS maskByText)
    let mut masked = text.to_string();
    for entity in entities {
        let value = entity.text.trim().to_string();
        if value.is_empty() {
            continue;
        }
        masked = masked.replace(&value, &mask_token.replace("{label}", &entity.label));
    }
    masked
}

fn argmax(values: &[f32]) -> (usize, f32) {
    values
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .unwrap_or((0, 0.0))
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn mask_replaces_label_placeholder_via_offset() {
        let text = "My name is Harry Potter.";
        let entities = vec![Entity::new(
            "private_person",
            0.9,
            " Harry Potter",
            Some(10),
            Some(23),
        )];

        let masked = mask_entities(text, "<{label}>", &entities);

        // JS maskByOffsets: text.slice(0, 10) = "My name is", no trailing space
        assert_eq!(masked, "My name is<private_person>.");
    }

    #[test]
    fn mask_fallback_to_text_replacement_when_no_offsets() {
        let text = "My name is Harry Potter.";
        // entity with None offsets => triggers text-based fallback
        let entities = vec![Entity::new(
            "private_person",
            0.9,
            "Harry Potter",
            None,
            None,
        )];

        let masked = mask_entities(text, "[{label}]", &entities);

        assert_eq!(masked, "My name is [private_person].");
    }

    #[test]
    fn mask_empty_entities_returns_original_text() {
        let entities = vec![];
        let masked = mask_entities("hello world", "[{label}]", &entities);
        assert_eq!(masked, "hello world");
    }

    #[test]
    fn decode_bioes_sequence() {
        let text = "Harry";
        let mut logits = array![[0.0; 33], [0.0; 33]];
        logits[[0, 17]] = 9.0;
        logits[[1, 19]] = 9.0;
        let offsets = vec![(0, 2), (2, 5)];

        let entities = decode_entities(text, logits.view(), &offsets);

        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].label, "private_person");
        assert_eq!(entities[0].text, "Harry");
        assert_eq!(entities[0].start, None);
        assert_eq!(entities[0].end, None);
    }
}
