import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

from libs import LIB_DeepFace

def compare_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        return None

    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    mse = np.mean((img1 - img2) ** 2)

    psnr = cv2.PSNR(img1, img2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ssim_score = ssim(gray1, gray2)

    return {
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim_score
    }

def generate_magick_png(input_path, output_path, magick_cmd):
    os.system(f"{magick_cmd} {input_path} -quality 100 {output_path}")


def compare_demographics(d1, d2):
    diff = {}

    for key in d1.keys():
        if key == "File":
            continue

        diff[key] = abs(d1[key] - d2[key])

    return diff


def run_test(image_path, Options):
    temp_png = "temp_test.png"

    # 1. Generar imagen con magick
    generate_magick_png(image_path, temp_png, Options.os_png_tool)

    # 2. Comparar imágenes
    quality_metrics = compare_images(image_path, temp_png)

    # 3. Obtener demographics
    demo_magick = LIB_DeepFace.SingleSampleDemographic(image_path, Options)
    demo_cv2 = LIB_DeepFace.SingleSampleDemographic_cv2(image_path, Options)

    if demo_magick is None or demo_cv2 is None:
        return None

    # 4. Comparar resultados
    demo_diff = compare_demographics(demo_magick, demo_cv2)

    return {
        "image": image_path,
        "quality": quality_metrics,
        "demographic_diff": demo_diff
    }

run_test()