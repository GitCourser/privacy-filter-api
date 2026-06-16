use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Entity {
    pub label: String,
    pub score: f32,
    pub text: String,
    pub start: Option<usize>,
    pub end: Option<usize>,
}

impl Entity {
    pub fn new(
        label: impl Into<String>,
        score: f32,
        text: impl Into<String>,
        start: Option<usize>,
        end: Option<usize>,
    ) -> Self {
        Self {
            label: label.into(),
            score,
            text: text.into(),
            start,
            end,
        }
    }
}
