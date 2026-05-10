from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class ScryfallCardTextRecognizer:
    def __init__(
        self,
        adapter_dir: str,
        base_model_name: str = "microsoft/trocr-base-printed",
        image_height: int = 576,
        image_width: int = 800,
        interpolate_pos_encoding: bool = True,
    ):
        self.adapter_dir = Path(adapter_dir)
        self.image_height = image_height
        self.image_width = image_width
        self.interpolate_pos_encoding = interpolate_pos_encoding

        self.processor = TrOCRProcessor.from_pretrained(self.adapter_dir)
        self.processor.image_processor.do_resize = False

        base_model = VisionEncoderDecoderModel.from_pretrained(base_model_name)
        self.model = PeftModel.from_pretrained(base_model, self.adapter_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path: str | Path, max_length: int = 512, num_beams: int = 1) -> str:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((self.image_width, self.image_height), resample=Image.Resampling.BICUBIC)

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                interpolate_pos_encoding=self.interpolate_pos_encoding,
            )

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
