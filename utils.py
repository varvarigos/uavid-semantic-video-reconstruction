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


def extract_images_from_grid(
    image_path, img_width=512, img_height=512, num_columns=2
):
    """
    Extracts images from the first two columns of a composite image.

    Args:
    image_path (Path): Path to the composite image.
    img_width (int): Width of each sub-image.
    img_height (int): Height of each sub-image.
    num_columns (int): Number of columns to process (default is 2).

    Returns:
    list: A list containing two lists of PIL Image objects, each corresponding to one of the first two columns.
    """
    # Open the full image
    full_image = Image.open(image_path)

    # Initialize lists to store images from each column
    column_images = [[] for _ in range(num_columns)]

    # Calculate the number of rows by dividing the full image height by the height of each small image
    num_rows = full_image.height // img_height

    # Crop and store images from the specified columns
    for col in range(num_columns):
        for row in range(num_rows):
            left = col * img_width
            upper = row * img_height
            right = left + img_width
            lower = upper + img_height

            # Crop the image
            cropped_img = full_image.crop((left, upper, right, lower))

            # Store the cropped image in the corresponding list
            column_images[col].append(cropped_img)

    return column_images
