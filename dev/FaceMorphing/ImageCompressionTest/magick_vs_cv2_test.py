import cv2, os
import numpy as np
from types import SimpleNamespace
from skimage.metrics import structural_similarity as ssim

from libs import LIB_DeepFace


# Compute similarity metrics (MSE, PSNR, SSIM)
def compare_images(img1_path, img2_path):

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("Could not load one of the images.")
        return None

    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    mse = np.mean((img1 - img2) ** 2)
    psnr = cv2.PSNR(img1, img2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ssim_score = ssim(gray1, gray2)

    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim_score}


# Compare demographics
def compare_demographics(d1, d2):

    diff = {k: abs(d1[k] - d2[k]) for k in d1 if k != "File"}

    avg_diff = np.mean(list(diff.values()))

    print("\n--- Demographic Differences ---")
    for k, v in diff.items():
        print(f"{k:20s}: {v:.4f}")

    print(f"\nAverage demographic shift: {avg_diff:.4f}")

    return avg_diff


# Helper: print metrics
def print_metrics(name, m):

    print(f"\n[{name}]")
    print(f"MSE (Mean Squared Error)           | Expected: close to 0 |  : {m['MSE']:.4f} ")
    print(f"PSNR (Peak Signal-to-Noise Ratio)  | Expected: > 40 dB    |  : {m['PSNR']:.2f} dB ")
    print(f"SSIM (Structural Similarity Index) | Expected: close to 1 |  : {m['SSIM']:.4f} ")


# MAIN
def run_test(image_path):

    magick_img = "temp_magick.png"
    cv2_img = "temp_cv2.png"

    # Generate images
    LIB_DeepFace.generate_magick_png(image_path, magick_img, "magick")

    img = cv2.imread(image_path)
    cv2.imwrite(cv2_img, img)

    # ---------------- QUALITY ----------------
    metrics_magick = compare_images(image_path, magick_img)
    metrics_cv2 = compare_images(image_path, cv2_img)

    # ---------------- DEMOGRAPHICS ----------------
    demo_magick = LIB_DeepFace.SingleSampleDemographic(
        Options=SimpleNamespace(
            input_file=image_path,
            temp_output_file=magick_img,
            os_png_tool="magick",
            remove_temp_file=False
        )
    )

    demo_cv2 = LIB_DeepFace.SingleSampleDemographic(
        Options=SimpleNamespace(
            input_file=image_path,
            temp_output_file=cv2_img,
            os_png_tool="cv2",
            remove_temp_file=False
        )
    )

    if demo_magick is None or demo_cv2 is None:
        print("Demographic extraction failed.")
        return

    # OUTPUT
    print("\n==============================")
    print("IMAGE:", image_path)
    print("==============================")
    

    print("\n--- IMAGE QUALITY ---")
    print_metrics("MAGICK", metrics_magick)

    size_in_bytes = os.path.getsize(magick_img)
    print(f"File size: {size_in_bytes} bytes")


    print_metrics("CV2", metrics_cv2)
    size_in_bytes = os.path.getsize(cv2_img)
    print(f"File size: {size_in_bytes} bytes")

    # DEMOGRAPHICS
    avg_diff = compare_demographics(demo_magick, demo_cv2)

    try:
        os.remove(magick_img)
        os.remove(cv2_img)
    except:
        pass

    print("\nComparison completed.")


if __name__ == "__main__":
    run_test("./DATA/img5.jpg")