import json
import textwrap
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .augmentations import CardOCRAugmenter
from .config import CardOCRTrainingConfig


def normalize_target_text(text: str) -> str:
    dedented = textwrap.dedent(text).strip()
    lines = [line.rstrip() for line in dedented.splitlines()]
    return "\n".join(lines).strip()


def load_manifest_records(
    manifest_path: str | Path,
    split: str | None = None,
    skip_missing_images: bool = True,
) -> list[dict]:
    path = Path(manifest_path)
    records: list[dict] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if split is not None and record.get("split") != split:
                continue
            if skip_missing_images and (not record.get("image_exists", True)):
                continue

            image_path = _resolve_manifest_relative_path(path, record["image_path"])
            if skip_missing_images and (not image_path.exists()):
                continue

            record["image_path"] = str(image_path)
            record["target_text"] = normalize_target_text(record["target_text"])
            records.append(record)

    return records


def _resolve_manifest_relative_path(manifest_path: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate

    search_roots = [
        manifest_path.parent,
        manifest_path.parent.parent,
    ]

    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved

    return (manifest_path.parent.parent / candidate).resolve()


def summarize_manifest(records: list[dict]) -> dict:
    split_counts = Counter(record.get("split", "unknown") for record in records)
    return {
        "num_records": len(records),
        "split_counts": dict(split_counts),
    }


class CardImageTextDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        processor,
        config: CardOCRTrainingConfig,
        training: bool = False,
    ):
        self.records = records
        self.processor = processor
        self.config = config
        self.training = training
        self.augmenter = CardOCRAugmenter(config.augmentations) if training else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]

        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
            image = image.resize(
                (self.config.image_width, self.config.image_height),
                resample=Image.Resampling.BICUBIC,
            )
            if self.augmenter is not None:
                image = self.augmenter(image)

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            record["target_text"],
            truncation=True,
            max_length=self.config.max_target_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


class CardOCRDataCollator:
    def __init__(self, processor, interpolate_pos_encoding: bool = True):
        self.processor = processor
        self.interpolate_pos_encoding = interpolate_pos_encoding

    def __call__(self, features: list[dict]) -> dict:
        pixel_values = torch.stack([feature["pixel_values"] for feature in features])

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        padded_labels = self.processor.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        labels = padded_labels.masked_fill(
            padded_labels == self.processor.tokenizer.pad_token_id,
            -100,
        )

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "interpolate_pos_encoding": self.interpolate_pos_encoding,
        }
