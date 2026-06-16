## Project Name
privacy-filter-api

Brief one-line description: 基于 Rust、Axum 和 ONNX Runtime 的隐私实体检测与脱敏 HTTP API。

## Overview
privacy-filter-api 是一个 Rust 后端服务，用于加载 openai/privacy-filter 模型，对输入文本进行隐私实体识别，并提供脱敏结果。服务暴露健康检查、模型检查、实体检测和文本脱敏等 HTTP 接口。

项目在启动阶段初始化 ONNX Runtime，并可按配置检查或自动下载模型文件。推理逻辑使用 tokenizers 编码文本，通过 ort 调用 ONNX 模型，并将 BIOES 标签解码为实体列表。

配置支持命令行参数、环境变量和 .env 文件，优先级为命令行参数高于环境变量，高于默认值。API Key 为空时不启用鉴权，非空时 /detect 与 /mask 支持 x-api-key 或 Bearer token 鉴权。

## Technology Stack
- Language/Runtime: Rust 2024 edition
- Framework(s): Axum 0.8, Tokio 1
- Key Dependencies: ort, tokenizers, ndarray, reqwest, serde, clap, anyhow, thiserror, tracing, tower-http
- Build Tools: Cargo
- Quality Tools: rustfmt, clippy, cargo test

## Project Structure
```text
.
├── Cargo.toml          # Rust 包配置、依赖与版本信息
├── Cargo.lock          # 锁定依赖版本
├── .env.example        # 服务配置示例
├── src/
│   ├── main.rs         # 程序入口，加载配置、初始化日志/ONNX Runtime、启动 HTTP 服务
│   ├── lib.rs          # 模块导出
│   ├── api.rs          # Axum 路由、请求/响应结构、鉴权与错误响应
│   ├── config.rs       # CLI 与环境变量配置解析
│   ├── entity.rs       # 隐私实体数据结构
│   ├── model.rs        # 模型加载、推理、标签解码和脱敏逻辑
│   ├── model_download.rs # 模型文件查找、下载、多源测速与完整性检查
│   └── onnx_runtime.rs # ONNX Runtime 动态库查找、下载与初始化
└── .snow/              # Snow CLI 项目级配置目录
```

## Key Features
- 提供 `/health` 健康检查接口。
- 提供 `/inspect` 查看 ONNX 模型输入输出信息。
- 提供 `/detect` 检测隐私实体。
- 提供 `/mask` 按实体类型替换敏感文本。
- 支持 API Key 鉴权，兼容 `x-api-key` 与 `Authorization: Bearer <key>`。
- 支持 `fp32`、`fp16`、`q4`、`q4f16`、`quantized` ONNX 变体。
- 启动或首次推理时可自动解析并下载模型文件。
- 支持 Hugging Face、ModelScope、CNB 多源模型下载测速与选择。
- 支持自动下载并加载当前平台的 ONNX Runtime 动态库。

## Getting Started
### Prerequisites
- Rust toolchain，建议安装 stable 版本并包含 rustfmt、clippy。
- 可访问模型源与 ONNX Runtime 发布资产的网络环境。
- Linux、macOS 或 Windows 中受 ONNX Runtime 运行库支持的平台。

### Installation
```bash
cargo fetch
cp .env.example .env
```

如缺少 rustfmt 或 clippy：
```bash
rustup component add rustfmt clippy
```

### Usage
开发运行：
```bash
cargo run
```

指定配置运行：
```bash
cargo run -- --host 0.0.0.0 --port 4175 --model-check true --onnx-variant quantized
```

请求示例：
```bash
curl http://127.0.0.1:4175/health
curl -X POST http://127.0.0.1:4175/detect \
  -H 'content-type: application/json' \
  -d '{"text":"My name is Harry Potter."}'
curl -X POST http://127.0.0.1:4175/mask \
  -H 'content-type: application/json' \
  -d '{"text":"My name is Harry Potter.","mask_token":"[{label}]"}'
```

## Development

### Available Scripts
本项目使用 Cargo 命令：
```bash
cargo fmt
cargo clippy --all-targets --all-features
cargo test
cargo run
cargo build
```

### Development Workflow
1. 修改代码前先定位相关模块，优先阅读完整函数或结构体边界。
2. Rust 代码修改后运行 `cargo fmt`。
3. 运行 `cargo clippy --all-targets --all-features` 检查 lint。
4. 运行 `cargo test` 验证单元测试。
5. 涉及接口行为时，用 curl 或集成测试验证响应结构和状态码。

## Configuration
配置来源优先级：命令行参数 > `.env`/环境变量 > 默认值。

常用配置：
- `HOST`: 服务监听 IP，默认 `127.0.0.1`。
- `PORT`: 服务监听端口，默认 `4175`。
- `API_KEY`: API Key，空字符串表示不启用鉴权。
- `MODEL_ID`: 模型仓库 ID，默认 `openai/privacy-filter`。
- `ONNX_VARIANT`: ONNX 型号，默认 `quantized`，可选 `fp32`、`fp16`、`q4`、`q4f16`、`quantized`。
- `MODEL_DIR`: 本地模型根目录，默认 `./models`。
- `MODEL_CHECK`: 是否启动时检查并自动下载缺失模型，默认 `true`。
- `MAX_TOKENS`: 单次推理最大 token 数，默认 `128000`。

## Architecture
入口 `main.rs` 负责加载 `.env`、解析配置、初始化 tracing、配置 ONNX Runtime、按需检查模型文件，并启动 Axum 服务。

HTTP 层在 `api.rs` 中定义路由和请求处理。耗时的模型检查、加载和推理操作通过 `tokio::task::spawn_blocking` 执行，避免阻塞异步 worker。错误统一转为 JSON 响应。

模型层在 `model.rs` 中维护 `PrivacyFilterModel`，内部用 `Mutex<ModelState>` 延迟加载 tokenizer 与 ONNX session。检测时将文本编码为 input_ids 与 attention_mask，运行 ONNX session 后对 logits 做 softmax 和 BIOES 解码。脱敏优先使用 offset 替换，缺少 offset 时回退到文本替换。

模型下载层在 `model_download.rs` 中根据 `MODEL_ID` 和 `ONNX_VARIANT` 解析本地模型路径；缺失且 `MODEL_CHECK=true` 时，对 Hugging Face、ModelScope、CNB 进行探测测速并选择最快源下载所需文件。

ONNX Runtime 层在 `onnx_runtime.rs` 中优先查找可执行文件同目录或 `lib/` 子目录中的动态库；缺失时自动从 Microsoft ONNX Runtime GitHub Releases 下载适配当前平台的 1.24.x 运行库并初始化 `ort`。

## Contributing
- 保持 Rust 代码通过 `cargo fmt` 格式化。
- 提交前运行 `cargo clippy --all-targets --all-features` 与 `cargo test`。
- 新增配置时同步更新 `.env.example` 与本文件。
- 新增接口时保持响应结构稳定，并补充必要测试。
- 注意不要在异步 handler 中直接执行阻塞模型加载或推理。
