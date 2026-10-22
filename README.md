# ondevice-quant-lab

端侧模型量化实验台：把 HuggingFace 模型导出成 **LiteRT-LM `.litertlm`**，并做 **全量 int4** 量化。

> 只包含量化流水线本身（workflow + 导出脚本），不包含任何业务代码。
> 主工程保持私有。

## 为什么在 GitHub Actions 里跑

macOS 上 TensorFlow 与 PyTorch 各自静态链接 LLVM，同进程会：

```
LLVM ERROR: inconsistency in registered CommandLine options
```

Linux x86_64 没有这个问题，所以转换放在 `ubuntu-latest`。

## 用法

1. Actions → **Quantize Qwen3-0.6B (LiteRT-LM int4)** → **Run workflow**
2. 参数：`block_size=64`、`quantize_embedding=true`
3. 产物：Releases 里的 `*.litertlm`（约 300 MB）

不需要上传模型：workflow 会用 `hf download` 直接拉 `Qwen/Qwen3-0.6B`。

## 当前目标

| 指标 | 基线 | 目标 |
| --- | --- | --- |
| 权重体积 | 495 MB（0.83 B/参数） | **~300 MB（0.5 B/参数）** |
| 中文工具调用 | 5/5（temperature=0） | 保持 5/5 |

## 经验

- 先 `python scripts/export_int4.py --inspect` 看当前版本的真实 API，再改调用；
- 体积判断用 `字节数 ÷ 参数量`：0.5=int4 / 1.0=int8 / 2.0=fp16；
- 产物用 `litert-lm-peek` 校验 section 大小与模板。
