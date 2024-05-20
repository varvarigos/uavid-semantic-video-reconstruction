import os

from PIL import Image
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchvision import transforms

ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device="cuda")
psnr = PeakSignalNoiseRatio().to(device="cuda")
# fid = FrechetInceptionDistance(feature=64).to(device='cuda')

predictions_dir = "/teamspace/studios/this_studio/outputs/2024-05-15__01-05-46/single_frame_exp_2_350epochs/predictions"
gt_dir = "/teamspace/studios/this_studio/outputs/2024-05-15__01-05-46/single_frame_exp_2_350epochs/predictions"

ssim_list = []
psnr_list = []

convert_tensor = transforms.ToTensor()

for img in os.listdir(predictions_dir):
    pred = Image.open(os.path.join(predictions_dir, img))
    gt = Image.open(os.path.join(gt_dir, img))

    pred = pred.split()
    gt = gt.split()

    for i, pred_i in enumerate(pred):
        gt_i = gt[i]
        ssim_list.append(
            ssim(
                convert_tensor(pred_i).unsqueeze(0),
                convert_tensor(gt_i).unsqueeze(0),
            )
        )
        psnr_list.append(psnr(convert_tensor(pred_i), convert_tensor(gt_i)))

ssim_avg = sum(ssim_list) / len(ssim_list)
psnr_avg = sum(psnr_list) / len(psnr_list)

print(f"SSIM: {ssim_avg}")
print(f"PSNR: {psnr_avg}")
