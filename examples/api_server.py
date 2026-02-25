"""
DeepSeek-OCR-2 — API Client Example

Connects to a running vLLM server and sends OCR requests.

Start the server first:
    vllm serve deepseek-ai/DeepSeek-OCR-2 \
        --trust-remote-code \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --no-enable-prefix-caching \
        --logits_processors vllm.model_executor.models.deepseek_ocr2:NGramPerReqLogitsProcessor

Requirements:
    pip install openai
"""

import base64
import sys
from pathlib import Path

from openai import OpenAI


def ocr_image(client: OpenAI, image_path: str, prompt: str = None) -> str:
    """Send an image to the OCR API and return the extracted text."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    ext = Path(image_path).suffix.lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
        ext, "image/png"
    )

    if prompt is None:
        prompt = "<|grounding|>Convert the document to markdown."

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-OCR-2",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=8192,
        temperature=0.0,
        extra_body={
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": [128821, 128822],
        },
    )

    return response.choices[0].message.content


def main():
    if len(sys.argv) < 2:
        print("Usage: python api_server.py <image_path> [prompt]")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else None

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy",
    )

    result = ocr_image(client, image_path, prompt)
    print(result)


if __name__ == "__main__":
    main()
