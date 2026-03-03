from PIL import Image
import numpy as np


def abs_pixel_diff(img_path1, img_path2, out_path="diff.png"):
    img1 = np.array(Image.open(img_path1))
    img2 = np.array(Image.open(img_path2))

    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions and channels")

    # avoid uint8 underflow
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))

    # scalar metric
    mean_abs_diff = diff.mean()

    # convert diff back to uint8 for visualization
    diff_img = np.clip(diff, 0, 255).astype(np.uint8)
    Image.fromarray(diff_img).save(out_path)

    return mean_abs_diff, diff

print(abs_pixel_diff("out-peristalsis-1img-10k/000000.png", "datasets/one-image-10k-src/000000.png"))