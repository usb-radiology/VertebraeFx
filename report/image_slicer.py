import io

import matplotlib.pyplot as plt
import numpy as np
from classification_2d.lit_classifier import LitClassifier
from monai.transforms import (AddChannel, Compose, ResizeWithPadOrCrop,
                              ScaleIntensity, ToTensor)
from PIL import Image
from rich.console import Console
from scipy.ndimage import find_objects

Z_SLICE_OFFSET = 5

console = Console()


def transform():
    return Compose(
        [
            ScaleIntensity(),
            ResizeWithPadOrCrop((256, 256)),
            ToTensor(),
            AddChannel(),
        ]
    )


def load_sample(image_path):
    image = np.asarray(Image.open(image_path).convert("RGB"))
    image = np.transpose(image, (2, 0, 1))
    transforms = transform()
    t = transforms(image)
    return t.float()


def load_model(model_path):
    model = LitClassifier.load_from_checkpoint(model_path)
    model.eval()
    return model


def predict(model, v_mappings):
    results = {}
    for k, images in v_mappings.items():
        single_predictions = []
        for i in images:
            s = load_sample(i)
            y = model(s)
            predicted_class = y.argmax()
            single_predictions.append(predicted_class.item())
        status = 1 if np.sum(single_predictions)>0 else 0
        results[k] = {"single_predictions": single_predictions,
                      "status": status}
    return results


def generate_images(full_image, full_mask, vertebrae):
    pixel_margin = 20
    mask = full_mask[:, :, :] == vertebrae.value
    slices = find_objects(mask)

    if len(slices) > 0:
        y1 = slices[0][0].start
        y2 = slices[0][0].stop
        x1 = slices[0][1].start
        x2 = slices[0][1].stop
        z1 = slices[0][2].start
        z2 = slices[0][2].stop
        y, x, _ = mask.shape
        # with margin it could be that bbox is outside of image therefore
        # these checks
        y1 = 0 if y1 - pixel_margin < 0 else y1 - pixel_margin
        y2 = y2 if y2 + pixel_margin > y else y2 + pixel_margin
        x1 = 0 if x1 - pixel_margin < 0 else x1 - pixel_margin
        x2 = x2 if x2 + pixel_margin > x else x2 + pixel_margin

        if z1 + Z_SLICE_OFFSET > z2 + 1 - Z_SLICE_OFFSET:
            console.log(
                f"Skipping {vertebrae} because start+{Z_SLICE_OFFSET} > end + 1 - {Z_SLICE_OFFSET} start:{z1}, end:{z2}"
            )
            return

        images = []
        for i in range(z1 + Z_SLICE_OFFSET, z2 - Z_SLICE_OFFSET):
            # T for test images
            buf = io.BytesIO()
            plt.imsave(buf, full_image[y1:y2, x1:x2, i], cmap="gray")
            buf.seek(0)
            images.append(buf)
        return images
