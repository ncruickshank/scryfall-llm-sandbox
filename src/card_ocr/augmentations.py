import random

from PIL import Image, ImageDraw, ImageFilter

from .config import OCRAugmentationConfig


class CardOCRAugmenter:
    """
    Lightweight PIL-based augmentations modeled after the TrOCR paper setup,
    with a few implementation choices adapted for full-card MTG images.
    """

    def __init__(self, config: OCRAugmentationConfig):
        self.config = config

    def __call__(self, image: Image.Image, rng: random.Random | None = None) -> Image.Image:
        rng = rng or random
        augmented = image.convert("RGB")

        if rng.random() < self.config.rotation_probability:
            angle = rng.uniform(-self.config.rotation_degrees, self.config.rotation_degrees)
            augmented = augmented.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=(255, 255, 255),
            )

        if rng.random() < self.config.blur_probability:
            blur_radius = rng.uniform(self.config.blur_radius_min, self.config.blur_radius_max)
            augmented = augmented.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        if rng.random() < self.config.dilation_probability:
            augmented = self._morphological_filter(augmented, ImageFilter.MaxFilter)

        if rng.random() < self.config.erosion_probability:
            augmented = self._morphological_filter(augmented, ImageFilter.MinFilter)

        if rng.random() < self.config.downscale_probability:
            augmented = self._downscale_and_restore(augmented, rng)

        if rng.random() < self.config.underline_probability:
            augmented = self._draw_underlines(augmented, rng)

        return augmented

    def _morphological_filter(self, image: Image.Image, filter_cls: type[ImageFilter.Filter]) -> Image.Image:
        grayscale = image.convert("L")
        filtered = grayscale.filter(filter_cls(size=3))
        return filtered.convert("RGB")

    def _downscale_and_restore(self, image: Image.Image, rng: random.Random) -> Image.Image:
        width, height = image.size
        scale = rng.uniform(self.config.downscale_min_scale, self.config.downscale_max_scale)
        scaled_width = max(32, int(width * scale))
        scaled_height = max(32, int(height * scale))

        downscaled = image.resize((scaled_width, scaled_height), resample=Image.Resampling.BILINEAR)
        return downscaled.resize((width, height), resample=Image.Resampling.BICUBIC)

    def _draw_underlines(self, image: Image.Image, rng: random.Random) -> Image.Image:
        draw = ImageDraw.Draw(image)
        width, height = image.size
        min_lines = self.config.underline_count_min
        max_lines = self.config.underline_count_max
        line_count = rng.randint(min_lines, max_lines)

        for _ in range(line_count):
            start_x = int(rng.uniform(0.1, 0.4) * width)
            end_x = int(rng.uniform(0.6, 0.92) * width)
            y = int(rng.uniform(0.28, 0.86) * height)
            thickness = rng.randint(1, 3)
            draw.line((start_x, y, end_x, y), fill=(0, 0, 0), width=thickness)

        return image
