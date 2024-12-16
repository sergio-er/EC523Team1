#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-11 17:17:41

import os
import argparse
from pathlib import Path
import shutil
from omegaconf import OmegaConf
from sampler import ResShiftSampler

from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url

_STEP = {
    'v1': 15,
    'v2': 15,
    'v3': 4,
    'bicsr': 4,
    'inpaint_imagenet': 4,
    'inpaint_face': 4,
    'faceir': 4,
    'deblur': 4,
}
_LINK = {
    'vqgan': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth',
    'vqgan_face256': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/celeba256_vq_f4_dim3_face.pth',
    'vqgan_face512': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/ffhq512_vq_f8_dim8_face.pth',
    'v1': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v1.pth',
    'v2': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v2.pth',
    'v3': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s4_v3.pth',
    'bicsr': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_bicsrx4_s4.pth',
    'inpaint_imagenet': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_inpainting_imagenet_s4.pth',
    'inpaint_face': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_inpainting_face_s4.pth',
    'faceir': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_faceir_s4.pth',
    'deblur': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_deblur_s4.pth',
}

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-i",
        "--in_path",
        type=str,
        default="/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/Sergio/processed_images/high_resolution",
        help="Input path.",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/Sergio/processed_images/resshift_out",
        help="Output path.",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="",
        help="Mask path for inpainting.",
    )
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--bs", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="v3",
        choices=["v1", "v2", "v3"],
        help="Checkpoint version.",
    )
    parser.add_argument(
        "--chop_size",
        type=int,
        default=512,
        choices=[512, 256, 64],
        help="Chopping forward.",
    )
    parser.add_argument(
        "--chop_stride",
        type=int,
        default=-1,
        help="Chopping stride.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="realsr",
        choices=["realsr", "bicsr", "inpaint_imagenet", "inpaint_face", "faceir", "deblur"],
        help="Processing task.",
    )
    args = parser.parse_args()

    return args

def get_configs(args):
    ckpt_dir = Path("/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/Sergio/ResShift/logs/data_50k/2024-11-28-14-47/ema_ckpts")
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    ckpt_path = ckpt_dir / "ema_model_350000.pth"
    vqgan_path = Path("/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/Sergio/ResShift/weights/autoencoder_vq_f4.pth")
    if args.task == "realsr":
        configs = OmegaConf.load("./configs/realsr_swinunet_realesrgan256_journal.yaml")
    else:
        raise ValueError(f"Unsupported task type: {args.task}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"EMA checkpoint file not found: {ckpt_path}")
    if not vqgan_path.exists():
        raise FileNotFoundError(f"VQGAN file not found: {vqgan_path}")
    configs.model.ckpt_path = str(ckpt_path)
    configs.autoencoder.ckpt_path = str(vqgan_path)
    configs.diffusion.params.sf = args.scale
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)
    return configs

def main():
    args = get_parser()

    configs = get_configs(args)

    resshift_sampler = ResShiftSampler(
        configs,
        sf=args.scale,
        chop_size=args.chop_size,
        chop_stride=args.chop_stride,
        chop_bs=1,
        use_amp=True,
        seed=args.seed,
        padding_offset=configs.model.params.get("lq_size", 64),
    )

    mask_path = args.mask_path if args.task.startswith("inpaint") else None

    # Create directories for fake and original images
    fake_dir = Path(args.out_path) / "fake_images"
    original_dir = Path(args.out_path) / "original_images"
    fake_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)

    # Get list of input HR images
    hr_images = list(Path(args.in_path).glob("*.png"))

    for hr_image_path in hr_images:
        base_name = hr_image_path.stem
        fake_B_filename = f"{base_name}_fake.png"
        original_filename = f"{base_name}_original.png"

        fake_B_path = fake_dir / fake_B_filename
        original_path = original_dir / original_filename

        shutil.copy(hr_image_path, original_path)

        resshift_sampler.inference(
            str(hr_image_path),
            str(fake_B_path.parent),
            mask_path=mask_path,
            bs=args.bs,
            noise_repeat=False,
        )

if __name__ == "__main__":
    main()
