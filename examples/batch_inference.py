"""
DeepSeek-OCR-2 — Batch Inference Example (vLLM)

Process multiple document images and convert to markdown.

Requirements:
    pip install vllm flash-attn==2.7.3 transformers==4.46.3
"""

from vllm import LLM, SamplingParams


def main():
    # Initialize model
    llm = LLM(
        model="deepseek-ai/DeepSeek-OCR-2",
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        enforce_eager=False,
        gpu_memory_utilization=0.75,
        enable_prefix_caching=False,
    )

    # Sampling params with NGram repetition prevention
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        skip_special_tokens=False,
        extra_args={
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": {128821, 128822},  # <td>, </td>
        },
    )

    # Process a batch of images
    image_paths = [
        "samples/invoice.png",
        "samples/research_paper.png",
        "samples/table_document.png",
    ]

    prompt = "<image>\n<|grounding|>Convert the document to markdown."

    inputs = [
        {"prompt": prompt, "multi_modal_data": {"image": path}}
        for path in image_paths
    ]

    results = llm.generate(inputs, sampling_params=sampling_params)

    for path, result in zip(image_paths, results):
        output_text = result.outputs[0].text
        print(f"\n{'='*60}")
        print(f"File: {path}")
        print(f"{'='*60}")
        print(output_text[:500])  # Print first 500 chars
        print("...")


if __name__ == "__main__":
    main()
