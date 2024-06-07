import hydra
from hydra.utils import instantiate
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision import transforms

from config import EvalConfig
from utils import extract_images_from_grid


# take the config from the config.yaml file and not from the python file
@hydra.main(config_path="conf", config_name="train_config")
def main(cfg: EvalConfig) -> None:
    cfg = instantiate(cfg)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device="cuda")
    psnr = PeakSignalNoiseRatio().to(device="cuda")

    images_path = cfg.eval_script.images_path

    all_images = extract_images_from_grid(
        images_path,  # img_height=256, img_width=256
    )
    pred_imgs = all_images[0]
    gt_imgs = all_images[1]

    ssim_list = []
    psnr_list = []

    convert_tensor = transforms.ToTensor()

    for _, (pred, gt) in enumerate(zip(pred_imgs, gt_imgs)):
        ssim_list.append(
            ssim(
                convert_tensor(pred).unsqueeze(0),
                convert_tensor(gt).unsqueeze(0),
            )
        )
        psnr_list.append(psnr(convert_tensor(pred), convert_tensor(gt)))

    ssim_avg = sum(ssim_list) / len(ssim_list)
    psnr_avg = sum(psnr_list) / len(psnr_list)

    print(f"SSIM: {ssim_avg}")
    print(f"PSNR: {psnr_avg}")

    with open(images_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(f"SSIM: {ssim_avg}\n")
        f.write(f"PSNR: {psnr_avg}\n")


if __name__ == "__main__":
    main()
