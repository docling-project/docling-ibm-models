from PIL import Image
from torchvision.transforms import functional as F
from transformers import AutoImageProcessor, BaseImageProcessor


class SamOptImageProcessor(BaseImageProcessor):

    def __init__(self, size=(1024, 1024), mean=None, std=None, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.mean = mean
        self.std = std

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image")

        image = F.resize(image, self.size)
        image = F.to_tensor(image)

        image = F.normalize(image, mean=self.mean, std=self.std)

        return image


AutoImageProcessor.register(
    config_class="SamOptImageProcessor",
    slow_image_processor_class=SamOptImageProcessor,
)
