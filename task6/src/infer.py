import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import requests
import torch
from PIL import Image
from transformers import ViltForQuestionAnswering, ViltProcessor


DEFAULT_REMOTE_MODEL = "dandelin/vilt-b32-finetuned-vqa"
DEFAULT_LOCAL_MODEL = "task6/model/vilt-vqa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ViLT for VQA inference.")
    parser.add_argument(
        "--image",
        required=True,
        help="Local image path or image URL.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question about the image.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_LOCAL_MODEL,
        help=(
            "Local model directory or Hugging Face model name. "
            f"Default: {DEFAULT_LOCAL_MODEL}"
        ),
    )
    parser.add_argument(
        "--output",
        default="task6/outputs/answers/result.json",
        help="Path to save the inference result JSON.",
    )
    return parser.parse_args()


def load_image(image_source: str) -> Image.Image:
    if image_source.startswith(("http://", "https://")):
        response = requests.get(image_source, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_source).convert("RGB")


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], file_path: str) -> None:
    ensure_parent_dir(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_source = args.model_name
    if args.model_name == DEFAULT_LOCAL_MODEL and not Path(DEFAULT_LOCAL_MODEL).exists():
        model_source = DEFAULT_REMOTE_MODEL

    print(f"Loading model: {model_source}")
    processor = ViltProcessor.from_pretrained(model_source)
    # Prefer safetensors to avoid the torch.load restriction on older torch versions.
    model = ViltForQuestionAnswering.from_pretrained(
        model_source,
        use_safetensors=True,
    ).to(device)
    model.eval()

    image = load_image(args.image)
    encoding = processor(image, args.question, return_tensors="pt")
    encoding = {key: value.to(device) for key, value in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        predicted_idx = int(logits.argmax(-1).item())
        answer = model.config.id2label[predicted_idx]

    result = {
        "model_name": model_source,
        "image": args.image,
        "question": args.question,
        "answer": answer,
        "device": str(device),
    }
    save_json(result, args.output)

    print("Question:", args.question)
    print("Answer:", answer)
    print(f"Saved result to: {args.output}")


if __name__ == "__main__":
    main()
