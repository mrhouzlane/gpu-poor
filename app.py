"""Local-first Gradio demo. No weights, no GPU compute, no deployment side effects."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from huggingface_hub.errors import HfHubHTTPError

from gpu_poor import GPUS, estimate, load_model, load_prices
from gpu_poor.cli import POLICY, batch_list, format_table


def estimate_ui(
    model_id: str,
    gpu_key: str,
    dtype: str,
    batches: str,
    context: float,
    memory_fraction: float,
    price_file: str | None,
    include_prices: bool,
) -> tuple[str, dict[str, object]]:
    """The same library powers the app and CLI."""
    try:
        model = load_model(model_id.strip())
        gpu = GPUS[gpu_key]
        price = load_prices(price_file, gpu_key) if include_prices else None
        rows = [
            asdict(
                estimate(
                    model,
                    gpu,
                    dtype=dtype,
                    batch=b,
                    context=int(context),
                    memory_fraction=memory_fraction,
                    prices=price,
                )
            )
            for b in batch_list(batches)
        ]
        title = f"### {model.model_id} on {gpu.name}\n"
        title += (
            f"{model.parameters:,} parameters, "
            f"{model.kv_bytes_per_token():,} BF16 KV bytes/token.\n"
        )
        if price:
            label = "EXAMPLE" if price.is_example else "Archived research data, not a live index"
            title += (
                f"\n**{label}**: {price.date}, {price.kind}, USD/GPU hour "
                f"{price.low:g} / {price.median:g} / {price.high:g}.\n"
            )
        if model.sliding_window:
            title += "\nFull-context KV used; sliding attention eviction is not modeled.\n"
        return title + "\n```text\n" + format_table(rows) + "\n```", {
            "notice": POLICY,
            "model": asdict(model),
            "gpu": asdict(gpu),
            "prices": asdict(price) if price else None,
            "estimates": rows,
        }
    except (ValueError, OSError, KeyError, argparse.ArgumentTypeError, HfHubHTTPError) as exc:
        raise gr.Error(str(exc)) from exc


def build_app() -> gr.Blocks:
    with gr.Blocks(title="gpu-poor") as demo:
        gr.Markdown("# gpu-poor\nA model config, a GPU, a dated price range. No weight download.")
        gr.Markdown("**" + POLICY + "**")
        with gr.Row():
            model_id = gr.Textbox(value="Qwen/Qwen2.5-7B-Instruct", label="Hugging Face model id")
            gpu = gr.Dropdown(choices=list(GPUS), value="H100", label="Single GPU")
            dtype = gr.Dropdown(
                choices=["bf16", "fp8", "int4"],
                value="bf16",
                label="Weight storage dtype (compute stays BF16)",
            )
        with gr.Row():
            batches = gr.Textbox(value="1,8,32", label="Batch sizes")
            context = gr.Number(
                value=4096, minimum=1, precision=0, label="Resident context tokens per sequence"
            )
            memory = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="GPU memory fraction for weights + KV",
            )
        prices = gr.File(
            label="Optional price CSV or Parquet", file_types=[".csv", ".parquet"], type="filepath"
        )
        include_prices = gr.Checkbox(value=True, label="Include dated rental cost range")
        gr.Markdown(
            "Bundled prices are historical research inputs, **not** the held Hourly index. "
            "If no exact SKU has on demand prices, turn off costs or supply your own file."
        )
        run = gr.Button("Estimate", variant="primary")
        output = gr.Markdown()
        details = gr.JSON(label="Assumptions and full results")
        run.click(
            estimate_ui,
            [model_id, gpu, dtype, batches, context, memory, prices, include_prices],
            [output, details],
            api_name="estimate",
        )
        gr.Markdown(
            "Dense Llama, Mistral and Qwen2 only in v0.1. Unsupported configs fail clearly. "
            "MIT code; price data: GPU rental price data by gpurentalprices.com, CC BY 4.0."
        )
    return demo


if __name__ == "__main__":
    build_app().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
        allowed_paths=[str(Path(__file__).parent / "examples")],
    )
