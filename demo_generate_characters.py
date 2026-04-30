
"""
Minimal demo script.

Expected use:
1. Prepare a list of PIL images.
2. Train row residual basis.
3. Mix parents and generate new samples.
"""

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from rowvae_core import train_row_residual_basis, mix_generate, to_image

def draw_simple_character(seed, size=128):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), (250, 250, 250))
    d = ImageDraw.Draw(img)

    colors = [(255,190,120), (170,220,255), (220,175,255), (170,235,170), (255,170,195)]
    body = colors[seed % len(colors)]
    cx, cy = size // 2, size // 2
    rx, ry = 40, 42

    # ears
    if seed % 2 == 0:
        d.polygon([(cx-rx+8,cy-ry+20),(cx-rx+28,cy-ry-18),(cx-rx+48,cy-ry+24)], fill=body)
        d.polygon([(cx+rx-48,cy-ry+24),(cx+rx-28,cy-ry-18),(cx+rx-8,cy-ry+20)], fill=body)
    else:
        d.ellipse((cx-rx-12,cy-ry+15,cx-rx+20,cy-ry+48), fill=body)
        d.ellipse((cx+rx-20,cy-ry+15,cx+rx+12,cy-ry+48), fill=body)

    d.ellipse((cx-rx, cy-ry, cx+rx, cy+ry), fill=body)
    d.ellipse((cx-rx//2, cy+ry-5, cx+rx//2, cy+ry+36), fill=tuple(int(v*0.9) for v in body))

    eye = (20, 24, 38)
    d.ellipse((cx-28, cy-20, cx-12, cy+2), fill=eye)
    d.ellipse((cx+12, cy-20, cx+28, cy+2), fill=eye)
    d.ellipse((cx-23, cy-16, cx-19, cy-12), fill=(255,255,255))
    d.ellipse((cx+17, cy-16, cx+21, cy-12), fill=(255,255,255))

    d.ellipse((cx-7, cy+5, cx+7, cy+17), fill=(95,55,45))
    d.arc((cx-22, cy+15, cx, cy+36), 15, 165, fill=(95,55,45), width=2)
    d.arc((cx, cy+15, cx+22, cy+36), 15, 165, fill=(95,55,45), width=2)

    return img.filter(ImageFilter.GaussianBlur(radius=0.25))

if __name__ == "__main__":
    images = [draw_simple_character(i) for i in range(12)]
    model = train_row_residual_basis(images, latent_size=32, out_size=128, k=24)

    generated = mix_generate(
        model,
        parent_ids=[0, 3, 7],
        weights=[0.3, 0.4, 0.3],
        noise_pair=(1, 5),
        noise_strength=0.25,
    )

    to_image(generated).save("generated_sample.png")
    print("saved generated_sample.png")
