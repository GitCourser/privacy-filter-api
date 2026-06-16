# privacy-filter-api

隐私信息检测与脱敏 HTTP API，基于 ONNX Runtime 运行自部署 [openai/privacy-filter](https://github.com/openai/privacy-filter) 模型。

推荐 Agents 用 [snow-cli](https://github.com/MayDay-wpf/snow-cli) 可直接支持 [隐私设置指南](https://github.com/MayDay-wpf/snow-cli/blob/main/docs/usage/zh/24.%E9%9A%90%E7%A7%81%E8%AE%BE%E7%BD%AE%E6%8C%87%E5%8D%97.md)  
`snow-cli` 状态栏插件 `snow_statusline/privacy.js`

其他开源 Agents 需自行适配，如：CodexCli, OpenCode

## 功能

- 本地/远程/Docker 运行 `openai/privacy-filter` 模型
- 自动从最快的镜像站下载模型文件
- 自动下载 ONNX Runtime 动态库（终端连不上 github 时可[手动下载](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.4)）
- 接口
  - GET /health：健康检查
  - POST /detect：检测 PII 实体
  - POST /mask：检测并脱敏文本
- 设置 `API_KEY` 后，`/detect` 与 `/mask` 需要鉴权

## 快速开始

```bash
cd privacy-filter-api
cp .env.example .env
cargo run --release   # 源码
./privacy-filter-api  # 二进制
```

默认监听：

```text
http://127.0.0.1:4175
```

也可以通过命令行覆盖配置：

```bash
cargo run --release -- --host 0.0.0.0 --port 8080 --onnx-variant fp32  # 源码
./privacy-filter-api --host 0.0.0.0 --port 8080 --onnx-variant fp32  # 二进制
```

## Docker

```bash
# 本地构建镜像
docker build -t privacy-filter-api .

# 拉取镜像
docker pull ghcr.io/GitCourser/privacy-filter-api

# 启动
docker run -d \
  -p 4175:4175 \
  -v $PWD/models:/app/models \
  -e API_KEY=your-secret-key \
  -e ONNX_VARIANT=fp32 \
  --restart unless-stopped \
  --name privacy-filter-api \
  ghcr.io/GitCourser/privacy-filter-api
```

## 配置

配置优先级：

```text
命令行参数 > .env/环境变量 > 默认值
```

| .env / 环境变量 | 命令行参数       | 默认值                  | 说明                                            |
| --------------- | ---------------- | ----------------------- | ----------------------------------------------- |
| `HOST`          | `--host`         | `127.0.0.1`             | 服务监听 IP                                     |
| `PORT`          | `--port`         | `4175`                  | 服务监听端口                                    |
| `API_KEY`       | `--api-key`      | 空                      | 非空时启用鉴权                                  |
| `MODEL_ID`      | `--model-id`     | `openai/privacy-filter` | 模型仓库 ID                                     |
| `ONNX_VARIANT`  | `--onnx-variant` | `quantized`             | 可选 `fp32`、`fp16`、`q4`、`q4f16`、`quantized` |
| `MODEL_DIR`     | `--model-dir`    | `./models`              | 本地模型根目录                                  |
| `MODEL_CHECK`   | `--model-check`  | `true`                  | 启动时检查并自动下载缺失模型                    |
| `MAX_TOKENS`    | `--max-tokens`   | `128000`                | 单次推理最大 token 数                           |

## 模型和运行库

默认模型目录为：

```text
./models/openai/privacy-filter/
```

`MODEL_CHECK=true` 时，启动阶段会检查必需模型文件；缺失时会从可用镜像源自动下载。默认 ONNX 型号为 `quantized`，对应 `onnx/model_quantized.onnx`。

ONNX Runtime 动态库会优先从可执行文件同目录或 `lib/` 子目录查找；缺失时会自动下载当前平台可用的 CPU 运行库。

## API 示例

如果设置了 `API_KEY`，请在受保护接口中加入以下任一请求头：

```text
x-api-key: your-secret-key
Authorization: Bearer your-secret-key
```

### GET /health

```bash
curl http://127.0.0.1:4175/health
```

```json
{
  "ok": true,
  "model": "openai/privacy-filter",
  "loaded": false,
  "auth": "disabled"
}
```

### GET /inspect

触发模型加载，并返回 ONNX 输入输出信息。

```bash
curl http://127.0.0.1:4175/inspect
```

### POST /detect

```bash
curl -X POST http://127.0.0.1:4175/detect \
  -H 'content-type: application/json' \
  -H 'x-api-key: your-secret-key' \
  -d '{"text":"My name is Harry Potter and my email is harry.potter@hogwarts.edu."}'
```

```json
{
  "model": "openai/privacy-filter",
  "entities": [
    {
      "label": "private_person",
      "score": 0.9999,
      "text": " Harry Potter",
      "start": null,
      "end": null
    }
  ]
}
```

### POST /mask

`mask_token` 可省略，默认值为 `[{label}]`。

```bash
curl -X POST http://127.0.0.1:4175/mask \
  -H 'content-type: application/json' \
  -H 'x-api-key: your-secret-key' \
  -d '{"text":"My name is Harry Potter and my email is harry.potter@hogwarts.edu.","mask_token":"[{label}]"}'
```

```json
{
  "model": "openai/privacy-filter",
  "masked_text": "My name is [private_person] and my email is [private_email].",
  "entities": []
}
```

实际 `entities` 会返回模型识别出的实体列表。

## 常见错误

```json
{ "error": "Unauthorized." }
{ "error": "`text` must be a non-empty string." }
{ "error": "`mask_token` must be a string." }
{ "error": "Inference failed.", "message": "..." }
```

## 许可证

本项目使用 MIT License。

本项目使用了 [openai/privacy-filter](https://github.com/openai/privacy-filter)，其许可证为 Apache-2.0。
