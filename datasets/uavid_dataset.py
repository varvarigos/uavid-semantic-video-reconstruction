import warnings
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset, default_collate
from torchvision import transforms as T


class UavidDatasetWithTransform(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        path: Path,
        size=512,
        center_crop=False,
        transform=None,
        indices=None,
        max_previous_frames=None,
        oracle=False,
        prediction_steps=1,
        shift_indices=0,
    ):
        assert not (
            oracle and shift_indices != 0
        ), "Oracle mode is not supported with `shift_indices != 0`"

        if shift_indices != 0:
            warnings.warn(
                "`shift_indices` is not equal to 0. Do not use this if you are not sure."
            )

        # self.image_processor = AutoProcessor.from_pretrained(
        #     "geolocal/StreetCLIP"
        # )

        self.UAVID_SHAPE = (3840, 2160)

        self.path = path
        self.oracle = oracle
        self.prediction_steps = prediction_steps
        self.shift_indices = shift_indices

        self.all_images = []
        for dirr in self.path.rglob("*"):
            if (
                dirr.is_file()
                and ("Images" in dirr.as_posix())
                and ("ADE20K_Labels" not in dirr.as_posix())
                and ("seq" in dirr.as_posix())
            ):
                # if dirr.stem[-3] != "0":
                self.all_images.append(dirr)

        self.indices = indices

        self.max_previous_frames = (
            max_previous_frames if max_previous_frames is not None else 0
        )
        self.size = size
        self.center_crop = center_crop
        self.transform = transform

        self.vae_transforms = T.Compose(
            [
                T.Resize(
                    (size, size), interpolation=T.InterpolationMode.BILINEAR
                ),
                T.CenterCrop(
                    size
                ),  # (T.CenterCrop(size) if center_crop else T.RandomCrop(size)),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )

        self.input_img_encdr_trnsfrms = T.Compose(
            [
                T.Resize(
                    (224, 224),
                    interpolation=T.InterpolationMode.BICUBIC,
                    # antialias=False,
                ),
                T.CenterCrop(224),
                lambda image: image.convert("RGB"),
                T.ToTensor(),
                T.Normalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        self.conditioning_image_transforms = T.Compose(
            [
                T.Resize(
                    (size, size), interpolation=T.InterpolationMode.BILINEAR
                ),
                T.CenterCrop(size),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.all_images) - (
            (len(self.all_images) // 10) * self.prediction_steps
        )

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx % len(self.indices)]

        imgs_per_sequence = 10
        usable_imgs_per_sequence = imgs_per_sequence - self.prediction_steps
        if self.oracle:
            current_frame = self.all_images[idx]
            previous_frames = [current_frame]
        else:
            # for usable_imgs_per_sequence = 9
            # idx = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...
            # index = 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, ...
            index = (
                (idx // usable_imgs_per_sequence) * 10
                + (idx % usable_imgs_per_sequence)
                + 1
                + self.shift_indices
            )
            current_frame = self.all_images[index]

            # for usable_imgs_per_sequence = 9
            # idx = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...
            # 10 * (idx // 9) = 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, ...
            previous_frames = self.all_images[
                10 * (idx // usable_imgs_per_sequence) : index
            ][-self.max_previous_frames :]

        current_segmentation_map = Path(
            str(current_frame).replace("Images", "ADE20K_Labels")
        )

        previous_frames = [
            Image.open(previous_frame) for previous_frame in previous_frames
        ]
        current_frame = Image.open(current_frame)
        current_segmentation_map = Image.open(current_segmentation_map)

        previous_frames = [
            exif_transpose(previous_frame) for previous_frame in previous_frames
        ]
        current_frame = exif_transpose(current_frame)
        current_segmentation_map = exif_transpose(current_segmentation_map)

        example = {}

        if not current_frame.mode == "RGB":
            current_frame = current_frame.convert("RGB")
        if not previous_frames[0].mode == "RGB":
            previous_frame = [
                previous_frame.convert("RGB")
                for previous_frame in previous_frames
            ]
        if not current_segmentation_map.mode == "RGB":
            current_segmentation_map = current_segmentation_map.convert("RGB")

        if self.transform is not None:
            transformed = self.transform(
                image=np.asarray(current_frame),
                seg_map=np.asarray(current_segmentation_map),
            )

            used_transform = transformed["replay"]
            previous_frames = [
                A.ReplayCompose.replay(
                    used_transform, image=np.asarray(previous_frame)
                )
                for previous_frame in previous_frames
            ]

            current_frame = Image.fromarray(transformed["image"])
            previous_frames = [
                Image.fromarray(previous_frame["image"])
                for previous_frame in previous_frames
            ]
            current_segmentation_map = Image.fromarray(transformed["seg_map"])

        example["pixel_values"] = self.vae_transforms(current_frame)
        example["pixel_values_clip"] = [
            self.input_img_encdr_trnsfrms(previous_frame)
            for previous_frame in previous_frames
        ]
        # example["pixel_values_clip"] = [
        #     torch.from_numpy(
        #         self.image_processor(images=previous_frame)["pixel_values"][0]
        #     )
        #     for previous_frame in previous_frames
        # ]
        example["segmentation_mask"] = self.conditioning_image_transforms(
            current_segmentation_map
        )
        example["pixel_values_prev"] = self.vae_transforms(previous_frames[-1])
        return example


def uavid_collate_fn(batch: list[dict]):
    """batch = [
        {"pixel_values": tensor, "pixel_values_clip": list[tensors] (minibatch of images), "instance_attention_mask": tensor},
        ...,
        {"pixel_values": tensor, "pixel_values_clip": list[tensors] (minibatch of images), "instance_attention_mask": tensor},
    ]
    """
    pixel_values_clip = [
        torch.stack(sample.pop("pixel_values_clip")) for sample in batch
    ]  # list[tensors[number_of_previous_frames x frame_related_dimension]]
    # maybe remove "pixel_values_clip" from each dict of the list of samples

    # for i in range(0, len(pixel_values_clip)):
    #     pixel_values_clip[i].requires_grad = True

    # for i in range(0, len(batch)):
    #     batch[i].pop("pixel_values_clip")

    out_dict = default_collate(batch)
    out_dict["pixel_values_clip"] = pixel_values_clip

    return out_dict


class PredictionsUavidDataset(Dataset):
    def __init__(
        self,
        orig_dataset: Dataset,
        predictions: list[Image],
    ):
        assert len(predictions) > 0

        self.orig_dataset = orig_dataset
        self.predictions = predictions

        self.prediction_steps = self.orig_dataset.prediction_steps
        self.UAVID_SHAPE = self.orig_dataset.UAVID_SHAPE
        self.path = self.orig_dataset.path
        self.oracle = self.orig_dataset.oracle
        self.all_images = self.orig_dataset.all_images
        self.indices = self.orig_dataset.indices
        self.max_previous_frames = self.orig_dataset.max_previous_frames
        self.size = self.orig_dataset.size
        self.center_crop = self.orig_dataset.center_crop
        self.transform = self.orig_dataset.transform
        self.vae_transforms = self.orig_dataset.vae_transforms
        self.input_img_encdr_trnsfrms = (
            self.orig_dataset.input_img_encdr_trnsfrms
        )
        self.conditioning_image_transforms = (
            self.orig_dataset.conditioning_image_transforms
        )

        self.predicted_imgs = [
            []
            for _ in range(
                len(self.indices)
                if self.indices is not None
                else len(self.all_images)
                - ((len(self.all_images) // 10) * self.prediction_steps)
            )
        ]
        for prediction in predictions:
            img_size = prediction.size[0] // 3
            num_images = prediction.size[1] // img_size
            # cut each image from the first column of the prediction
            for i in range(num_images):
                self.predicted_imgs[i].append(
                    prediction.crop(
                        (0, i * img_size, img_size, (i + 1) * img_size)
                    )
                )

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.all_images) - (
            (len(self.all_images) // 10) * self.prediction_steps
        )

    def __getitem__(self, idx):
        orig_idx = idx
        predicitons_len = len(self.predicted_imgs[orig_idx])

        if self.indices is not None:
            idx = self.indices[idx % len(self.indices)]

        imgs_per_sequence = 10
        usable_imgs_per_sequence = imgs_per_sequence - self.prediction_steps

        # for usable_imgs_per_sequence = 9
        # idx = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...
        # index = 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, ...
        index = (
            (idx // usable_imgs_per_sequence) * 10
            + (idx % usable_imgs_per_sequence)
            + 1
        )

        current_frame = self.all_images[index + predicitons_len]

        # for usable_imgs_per_sequence = 9
        # idx = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...
        # 10 * (idx // 9) = 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, ...
        previous_frames = self.all_images[
            10 * (idx // usable_imgs_per_sequence) : index
        ][-self.max_previous_frames :]

        current_segmentation_map = Path(
            str(current_frame).replace("Images", "ADE20K_Labels")
        )

        previous_frames = (
            [Image.open(previous_frame) for previous_frame in previous_frames]
            + self.predicted_imgs[orig_idx]
        )[-self.max_previous_frames :]

        current_frame = Image.open(current_frame)
        if self.oracle:
            previous_frames = [current_frame]

        current_segmentation_map = Image.open(current_segmentation_map)

        previous_frames = [
            exif_transpose(previous_frame) for previous_frame in previous_frames
        ]
        current_frame = exif_transpose(current_frame)
        current_segmentation_map = exif_transpose(current_segmentation_map)

        example = {}

        if not current_frame.mode == "RGB":
            current_frame = current_frame.convert("RGB")
        if not previous_frames[0].mode == "RGB":
            previous_frame = [
                previous_frame.convert("RGB")
                for previous_frame in previous_frames
            ]
        if not current_segmentation_map.mode == "RGB":
            current_segmentation_map = current_segmentation_map.convert("RGB")

        if self.transform is not None:
            transformed = self.transform(
                image=np.asarray(current_frame),
                seg_map=np.asarray(current_segmentation_map),
            )

            used_transform = transformed["replay"]
            previous_frames = [
                A.ReplayCompose.replay(
                    used_transform, image=np.asarray(previous_frame)
                )
                for previous_frame in previous_frames
            ]

            current_frame = Image.fromarray(transformed["image"])
            previous_frames = [
                Image.fromarray(previous_frame["image"])
                for previous_frame in previous_frames
            ]
            current_segmentation_map = Image.fromarray(transformed["seg_map"])

        example["pixel_values"] = self.vae_transforms(current_frame)
        example["pixel_values_clip"] = [
            self.input_img_encdr_trnsfrms(previous_frame)
            for previous_frame in previous_frames
        ]
        example["segmentation_mask"] = self.conditioning_image_transforms(
            current_segmentation_map
        )
        example["pixel_values_prev"] = self.vae_transforms(previous_frames[-1])

        return example
