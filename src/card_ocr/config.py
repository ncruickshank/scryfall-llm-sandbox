from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class OCRAugmentationConfig:
    rotation_degrees: float = 10.0
    rotation_probability: float = 0.7
    blur_probability: float = 0.3
    blur_radius_min: float = 0.2
    blur_radius_max: float = 1.4
    dilation_probability: float = 0.2
    erosion_probability: float = 0.2
    downscale_probability: float = 0.3
    downscale_min_scale: float = 0.55
    downscale_max_scale: float = 0.9
    underline_probability: float = 0.25
    underline_count_min: int = 1
    underline_count_max: int = 3


@dataclass
class OCRLoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    bias: str = "none"
    target_module_suffixes: tuple[str, ...] = (
        "query",
        "key",
        "value",
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
    )


@dataclass
class CardOCRTrainingConfig:
    manifest_path: str = "data/card_image_text_manifest.jsonl"
    base_model_name: str = "microsoft/trocr-base-printed"
    output_dir: str = "models/scryfall-card-ocr"
    image_height: int = 576
    image_width: int = 800
    interpolate_pos_encoding: bool = True
    max_target_length: int = 512
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 8.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_total_limit: int = 2
    generation_max_length: int = 512
    generation_num_beams: int = 1
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0
    early_stopping_patience: int = 3
    report_to: str | list[str] = "none"
    seed: int = 42
    use_fp16: bool = True
    use_bf16: bool = False
    skip_missing_images: bool = True
    train_split_name: str = "train"
    eval_split_name: str = "val"
    test_split_name: str = "test"
    augmentations: OCRAugmentationConfig = field(default_factory=OCRAugmentationConfig)
    lora: OCRLoRAConfig = field(default_factory=OCRLoRAConfig)

    def resolved_manifest_path(self) -> Path:
        return Path(self.manifest_path)

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self) -> dict:
        return asdict(self)
