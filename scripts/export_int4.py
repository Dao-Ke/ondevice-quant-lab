#!/usr/bin/env python3
"""Export a HF model to LiteRT-LM with full int4 quantization.

Why this script is defensive:
    `litert-torch`'s export API is not stable across versions, and its CLI
    imports every model extension (one of which needs TensorFlow). Instead of
    hard-coding a call that may not exist, this script:
      1. `--inspect` prints the actual API surface of the installed version
         (run this first; it is how we discover the correct entrypoint);
      2. otherwise it locates a supported export entrypoint and calls it with
         only the kwargs its signature accepts.

Usage (in CI):
    python scripts/export_int4.py --inspect
    python scripts/export_int4.py --model model --output out \
        --block-size 64 --quantize-embedding true
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import sys
import traceback

MODULES = [
    "litert_torch.generative.export_hf.export",
    "litert_torch.generative.export_hf.core.export_lib",
    "litert_torch.generative.export_hf",
]
ENTRYPOINTS = ["export_model", "export", "export_hf", "convert", "main"]


def _load():
    loaded = []
    for name in MODULES:
        try:
            loaded.append((name, importlib.import_module(name)))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {name}: {type(exc).__name__}: {exc}")
    return loaded


def inspect_api() -> int:
    mods = _load()
    if not mods:
        print("No litert_torch export module could be imported.")
        return 1
    for name, mod in mods:
        print(f"\n=== {name} ===")
        for attr in sorted(a for a in dir(mod) if not a.startswith("_")):
            obj = getattr(mod, attr)
            if callable(obj):
                try:
                    sig = str(inspect.signature(obj))
                except (TypeError, ValueError):
                    sig = "(?)"
                print(f"  {attr}{sig}")
    print("\n把以上输出贴回项目文档，即可确定本版本的正确调用方式。")
    return 0


def call_export(model: str, output: str, block_size: int, quantize_embedding: bool, recipe: str) -> int:
    mods = _load()
    fn = None
    for _, mod in mods:
        for name in ENTRYPOINTS:
            candidate = getattr(mod, name, None)
            if callable(candidate):
                fn = candidate
                break
        if fn:
            break
    if fn is None:
        print("找不到导出入口。请先运行 --inspect，并按官方文档实现调用。", file=sys.stderr)
        return 2

    sig = inspect.signature(fn)
    kwargs = {}
    mapping = {
        # 依据 2026-10-22 CI 日志中的真实签名：
        # export(model, output_dir, quantization_recipe=..., bundle_litert_lm=..., use_jinja_template=...)
        "model": model,
        "model_path": model,
        "model_name_or_path": model,
        "output_dir": output,
        "output_path": output,
        "quantization_recipe": recipe,
        "bundle_litert_lm": True,
        "use_jinja_template": True,
        "block_size": block_size,
        "quantize_embedding": quantize_embedding,
        "quantize_embeddings": quantize_embedding,
    }
    for key, value in mapping.items():
        if key in sig.parameters:
            kwargs[key] = value

    print(f"Calling {fn.__module__}.{fn.__name__}{sig}")
    print(f"kwargs={kwargs}")
    try:
        result = fn(**kwargs)
        print(f"OK result={result}")
        return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("\n调用失败。请用 --inspect 的输出对照官方示例修正参数。", file=sys.stderr)
        return 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model")
    parser.add_argument("--output", default="out")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--quantize-embedding", default="true")
    parser.add_argument("--recipe", default="dynamic_wi4c_afp32",
                        help="int4 权重配方；int8 用 dynamic_wi8_afp32")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    if args.inspect:
        return inspect_api()

    quant = str(args.quantize_embedding).lower() in ("1", "true", "yes")
    return call_export(args.model, args.output, args.block_size, quant, args.recipe)


if __name__ == "__main__":
    raise SystemExit(main())
