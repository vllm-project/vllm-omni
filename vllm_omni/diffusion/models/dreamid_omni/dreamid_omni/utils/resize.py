import math
from typing import List, Union
import torch
from PIL import Image
from torchvision.transforms import functional as TVF
from torchvision.transforms.functional import InterpolationMode, to_tensor
from typing import Literal
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize


class AreaResize:
    def __init__(
        self,
        max_area: float,
        downsample_only: bool = False,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        self.max_area = max_area
        self.downsample_only = downsample_only
        self.interpolation = interpolation

    def __call__(self, image: Union[torch.Tensor, Image.Image, List[Image.Image]]):

        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, list) and isinstance(image[0], Image.Image):
            width, height = image[0].size
        else:
            raise NotImplementedError

        scale = math.sqrt(self.max_area / (height * width))

        # keep original height and width for small pictures.
        scale = 1 if scale >= 1 and self.downsample_only else scale

        resized_height, resized_width = round(height * scale), round(width * scale)

        if isinstance(image, list) and isinstance(image[0], Image.Image):
            image = torch.stack(
                [
                    to_tensor(
                        TVF.resize(
                            _image,
                            size=(resized_height, resized_width),
                            interpolation=self.interpolation,
                        )
                    )
                    for _image in image
                ]
            )
        else:
            image = TVF.resize(
                image,
                size=(resized_height, resized_width),
                interpolation=self.interpolation,
            )
            if isinstance(image, Image.Image):
                image = to_tensor(image)
        return image


class FixResize:
    def __init__(
        self,
        size: list,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image: Union[torch.Tensor, Image.Image, List[Image.Image]]):

        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, list) and isinstance(image[0], Image.Image):
            width, height = image[0].size
        else:
            raise NotImplementedError

        # 计算resize的比�?        scale_width = self.size[1] / width
        scale_height = self.size[0] / height
        scale = min(scale_width, scale_height)
        
        # 计算resize后的尺寸
        resized_width = int(width * scale)
        resized_height = int(height * scale)

        if isinstance(image, list) and isinstance(image[0], Image.Image):
            final_frames = []
            for _image in image:
                resized_frame = TVF.resize(
                            _image,
                            size=(resized_height, resized_width),
                            interpolation=self.interpolation,
                        )
                
                if resized_width == self.size[1] and resized_height == self.size[0]:
                    # 如果已经是目标尺寸，则无需裁剪
                    final_frame = resized_frame
                else:
                    # 计算crop的起始点
                    start_x = (resized_width - self.size[1]) // 2
                    start_y = (resized_height - self.size[0]) // 2
                    
                    # Crop 图像
                    final_frame = resized_frame.crop((start_x, start_y, start_x + self.size[1], start_y + self.size[0]))
                
                final_frames.append(to_tensor(final_frame))
            
            image = torch.stack(final_frames)
            
        return image
    
    
    
def NaResize(
    resolution, # int or list
    mode: Literal["square"],
    downsample_only: bool,
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
):
    if mode == "area":
        return AreaResize(
            max_area=resolution**2,
            downsample_only=downsample_only,
            interpolation=interpolation,
        )
    if mode == "fix":
        return FixResize(
            size=resolution,
            interpolation=interpolation,
        )
    if mode == "square":
        return Compose(
            [
                Resize(
                    size=resolution,
                    interpolation=interpolation,
                ),
                CenterCrop(resolution),
            ]
        )
    raise ValueError(f"Unknown resize mode: {mode}")
