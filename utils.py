"""Utility functions for working with PyTorch models and images."""

from PIL import Image
from torchvision.transforms.functional import to_pil_image


def get_parameters_stats(parameters):
    """Get the number of trainable and all parameters in a model."""
    alll = trainablee = 0
    for p in parameters:
        numel = p.numel()
        alll += numel
        trainablee += numel if p.requires_grad else 0

    return {"all": alll, "trainable": trainablee}


def tensor_to_pil(tensor):
    """Convert a tensor to a PIL image."""
    # min-max normalize tensor to [0, 1]
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()

    # if batched, apply to_pil_image for each one
    if tensor.dim() == 4:
        return [to_pil_image(img_tensor) for img_tensor in tensor]

    return to_pil_image(tensor)


def image_grid(imgs, rows, cols):
    """Create a grid of images from a list of images."""
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid
