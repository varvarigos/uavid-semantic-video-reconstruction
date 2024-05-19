import torch
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from tqdm import tqdm

from datasets import UavidDatasetWithTransform, uavid_collate_fn
from utils import tensor_to_pil

# fid = FrechetInceptionDistance(feature=64).to(device='cuda')
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device="cuda")
psnr = PeakSignalNoiseRatio().to(device="cuda")

# val_dataloader ....
# val_dataset = UavidDatasetWithTransform(
#     path=cfg.dataset.dataset_path / "uavid_val",
#     size=cfg.dataset.resolution,
#     center_crop=cfg.dataset.center_crop,
#     indices=[i in range(0, 61)],
#     max_previous_frames=cfg.dataset.max_previous_frames,
# )

# val_dataloader = torch.utils.data.DataLoader(
#     val_dataset,
#     batch_size=cfg.dataloader.val_batch_size,
#     shuffle=False,
#     num_workers=cfg.dataloader.num_workers,
#     collate_fn=uavid_collate_fn,
# )

# images = []
# gt_images = []
# seg_maps = []

# for batch in tqdm(val_dataloader):
#     current_ssim = 0
#     current_psnr = 0
#     with torch.no_grad():
#         # with accelerator.autocast():
#         for k, v in batch.items():
#             if k == "pixel_values_clip":
#                 # v is a list of tensors
#                 batch[k] = [vi.to(device=cfg.device) for vi in v]
#                 if torch.is_floating_point(v[0]) and cfg.device != torch.device(
#                     "cpu"
#                 ):
#                     batch[k] = [vi.to(dtype=cfg.dtype) for vi in batch[k]]
#             else:
#                 batch[k] = v.to(device=cfg.device)
#                 if torch.is_floating_point(v) and cfg.device != torch.device(
#                     "cpu"
#                 ):
#                     batch[k] = batch[k].to(dtype=cfg.dtype)

#         encoder_hidden_states = [
#             image_encoder(previous_frames_i).image_embeds.unsqueeze(
#                 1
#             )  # --> tensor[Ni x 1 x F]
#             for previous_frames_i in batch["pixel_values_clip"]
#         ]

#         ## IF AVERAGE (not LSTM)
#         encoder_hidden_states = torch.stack(
#             [i.mean(dim=0) for i in encoder_hidden_states]  # tensor[F]
#         )  # tensor[batch_size x 1 x F]

#         # encoder_hidden_states = model.image_encoder(
#         #     batch["pixel_values_clip"]
#         # ).image_embeds.unsqueeze(1)

#         seg_maps.extend(tensor_to_pil(batch["segmentation_mask"]))
#         gt_images.extend(tensor_to_pil(batch["pixel_values"]))
#         pipe.set_progress_bar_config(disable=True)
#         with self.accelerator.autocast():
#             for (
#                 encoder_hidden_state,
#                 segmentation_mask,
#                 gt_image,
#             ) in zip(
#                 encoder_hidden_states,
#                 batch["segmentation_mask"],
#                 batch["pixel_values"],
#             ):
#                 images.extend(
#                     pipe(
#                         prompt_embeds=encoder_hidden_state.unsqueeze(0),
#                         negative_prompt_embeds=torch.zeros_like(
#                             encoder_hidden_state
#                         ).unsqueeze(0),
#                         image=tensor_to_pil(segmentation_mask),
#                         ip_adapter_image=(
#                             # TODO: this is wrong, we must use the
#                             # previous image here or some average
#                             # embeding of the previous images directly
#                             # into `ip_adapter_image_embeds``
#                             tensor_to_pil(gt_image)
#                             if self.cfg.model.use_ip_adapter
#                             else None
#                         ),
#                         num_inference_steps=100,
#                         guidance_scale=5,
#                         generator=generator,
#                     ).images
#                 )
#         for image in images:
#             current_ssim += ssim(image, gt_image)
#             current_psnr += psnr(image, gt_image)

#         total_ssim += current_ssim.item()
#         total_psnr += current_psnr.item()

# average_ssim = total_ssim / len(val_dataset)
# average_psnr = total_psnr / len(val_dataset)

################################################################################
# predictions_dir = # take this from cmd line or something

# for each image or folder/img in dir
#  load image
#  split image in each "frame"
#  for each frame caklcualte the ssim and psnr adn append to a list

# average of list
# save this number togheter with the predicitons so you do not have to run this again
