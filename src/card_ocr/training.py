from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    set_seed,
)
from transformers.models.vision_encoder_decoder.configuration_vision_encoder_decoder import (
    VisionEncoderDecoderConfig,
)

from .config import CardOCRTrainingConfig
from .dataset import CardImageTextDataset, CardOCRDataCollator, load_manifest_records
from .metrics import build_ocr_metrics


def _ensure_vision_encoder_decoder_vocab_size_property():
    """
    PEFT save logic expects a top-level `config.vocab_size` on the config class when
    it reloads the base config from the pretrained model id. VisionEncoderDecoderConfig
    stores text vocab under `config.decoder.vocab_size`, so we bridge that gap here.
    """
    if isinstance(getattr(VisionEncoderDecoderConfig, "vocab_size", None), property):
        return

    def _get_vocab_size(self):
        return self.decoder.vocab_size

    def _set_vocab_size(self, value):
        self.decoder.vocab_size = value

    VisionEncoderDecoderConfig.vocab_size = property(_get_vocab_size, _set_vocab_size)


_ensure_vision_encoder_decoder_vocab_size_property()


class CardOCRFineTuner:
    def __init__(self, config: CardOCRTrainingConfig):
        self.config = config
        set_seed(config.seed)

        self.processor = TrOCRProcessor.from_pretrained(config.base_model_name)
        self.processor.image_processor.do_resize = False

        self.model = VisionEncoderDecoderModel.from_pretrained(config.base_model_name)
        self._configure_model()
        self._attach_lora()
        self._sync_text_config_metadata()

        self.train_records = load_manifest_records(
            config.manifest_path,
            split=config.train_split_name,
            skip_missing_images=config.skip_missing_images,
        )
        self.eval_records = load_manifest_records(
            config.manifest_path,
            split=config.eval_split_name,
            skip_missing_images=config.skip_missing_images,
        )
        self.test_records = load_manifest_records(
            config.manifest_path,
            split=config.test_split_name,
            skip_missing_images=config.skip_missing_images,
        )

        self.train_dataset = CardImageTextDataset(
            self.train_records,
            processor=self.processor,
            config=config,
            training=True,
        )
        self.eval_dataset = CardImageTextDataset(
            self.eval_records,
            processor=self.processor,
            config=config,
            training=False,
        )
        self.test_dataset = CardImageTextDataset(
            self.test_records,
            processor=self.processor,
            config=config,
            training=False,
        )

        self.data_collator = CardOCRDataCollator(
            processor=self.processor,
            interpolate_pos_encoding=config.interpolate_pos_encoding,
        )
        self.compute_metrics = build_ocr_metrics(self.processor)
        self.trainer = self._build_trainer()

    def train(self, resume_from_checkpoint: str | None = None):
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.trainer.save_model()
        self.processor.save_pretrained(self.config.output_dir)
        return train_result

    def evaluate(self) -> dict:
        return self.trainer.evaluate(self.eval_dataset)

    def predict_test(self):
        return self.trainer.predict(self.test_dataset)

    def generate_from_image(self, image):
        self.model.eval()
        device = next(self.model.parameters()).device

        image = image.convert("RGB")
        image = image.resize(
            (self.config.image_width, self.config.image_height),
            resample=Image.Resampling.BICUBIC,
        )
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = self.model.generate(
            pixel_values=pixel_values,
            max_length=self.config.generation_max_length,
            num_beams=self.config.generation_num_beams,
            interpolate_pos_encoding=self.config.interpolate_pos_encoding,
        )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def _configure_model(self):
        tokenizer = self.processor.tokenizer
        decoder_start_token_id = tokenizer.eos_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.bos_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.cls_token_id

        self.model.config.decoder_start_token_id = decoder_start_token_id
        self.model.config.pad_token_id = tokenizer.pad_token_id
        self.model.config.eos_token_id = tokenizer.eos_token_id
        self.model.generation_config.max_length = self.config.generation_max_length
        self.model.generation_config.num_beams = self.config.generation_num_beams
        self._sync_text_config_metadata()

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

    def _attach_lora(self):
        target_modules = self._discover_lora_target_modules()
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            bias=self.config.lora.bias,
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, lora_config)
        self._sync_text_config_metadata()
        self.model.print_trainable_parameters()

    def _sync_text_config_metadata(self):
        """
        Keep a top-level vocab size synchronized for wrapped configs that expect it.
        """
        config_candidates = []

        if hasattr(self.model, "config"):
            config_candidates.append(self.model.config)
        if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "config"):
            config_candidates.append(self.model.base_model.config)
        if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model"):
            base_wrapped_model = self.model.base_model.model
            if hasattr(base_wrapped_model, "config"):
                config_candidates.append(base_wrapped_model.config)

        seen_config_ids = set()
        for model_config in config_candidates:
            config_id = id(model_config)
            if config_id in seen_config_ids:
                continue
            seen_config_ids.add(config_id)

            decoder_config = model_config.decoder
            model_config.vocab_size = decoder_config.vocab_size

    def _discover_lora_target_modules(self) -> list[str]:
        suffixes = self.config.lora.target_module_suffixes
        matches: set[str] = set()

        for module_name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue

            for suffix in suffixes:
                if module_name.endswith(suffix):
                    matches.add(module_name.split(".")[-1])

        if not matches:
            raise ValueError("No LoRA target modules were discovered for the selected TrOCR model.")

        return sorted(matches)

    def _build_training_arguments(self) -> Seq2SeqTrainingArguments:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fp16 = self.config.use_fp16 and torch.cuda.is_available()
        bf16 = self.config.use_bf16 and torch.cuda.is_available()

        return Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            do_train=True,
            do_eval=True,
            predict_with_generate=True,
            generation_max_length=self.config.generation_max_length,
            generation_num_beams=self.config.generation_num_beams,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            gradient_checkpointing=self.config.gradient_checkpointing,
            fp16=fp16,
            bf16=bf16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            report_to=self.config.report_to,
            seed=self.config.seed,
            remove_unused_columns=False,
            label_names=["labels"],
        )

    def _build_trainer(self) -> Seq2SeqTrainer:
        training_args = self._build_training_arguments()
        callbacks = [EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]

        return Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            processing_class=self.processor,
            callbacks=callbacks,
        )
