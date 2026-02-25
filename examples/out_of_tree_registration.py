"""
DeepSeek-OCR-2 — Out-of-Tree Model Registration Example

Use this if your vLLM version doesn't natively include DeepseekOCR2ForCausalLM.
Registers the model at runtime before loading.

This approach works on vLLM 0.11+ without needing to rebuild from source.

Prerequisites:
    1. Clone the DeepSeek-OCR-2 repo:
       git clone https://github.com/deepseek-ai/DeepSeek-OCR-2.git

    2. Add the vLLM integration files to your Python path:
       export PYTHONPATH="${PYTHONPATH}:DeepSeek-OCR-2/DeepSeek-OCR2-vllm"
"""

from vllm import LLM, ModelRegistry, SamplingParams


def register_deepseek_ocr2():
    """Register DeepSeek-OCR-2 if not already in the model registry."""
    supported = ModelRegistry.get_supported_archs()
    if "DeepseekOCR2ForCausalLM" in supported:
        print("DeepseekOCR2ForCausalLM already registered in vLLM")
        return

    # Option A: Register from DeepSeek's bundled vLLM scripts
    # Requires DeepSeek-OCR2-vllm/ on your PYTHONPATH
    ModelRegistry.register_model(
        "DeepseekOCR2ForCausalLM",
        "deepseek_ocr2:DeepseekOCR2ForCausalLM",
    )
    print("Registered DeepseekOCR2ForCausalLM from bundled scripts")


def main():
    # Register model architecture
    register_deepseek_ocr2()

    # Now load normally
    llm = LLM(
        model="deepseek-ai/DeepSeek-OCR-2",
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.75,
        enable_prefix_caching=False,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        skip_special_tokens=False,
        extra_args={
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": {128821, 128822},
        },
    )

    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    result = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"image": "test_document.png"}},
        sampling_params=sampling_params,
    )

    print(result[0].outputs[0].text)


if __name__ == "__main__":
    main()
