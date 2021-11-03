import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from common.colors import filter_colors
from common.vertebrae import Vertebrae

plt.style.use("dark_background")

def calculate_density(image, mask):
    image_data = np.rot90(np.fliplr(image.get_fdata()))
    mask_data = np.rot90(np.fliplr(mask.get_fdata())).astype(int)
    print(image.header["pixdim"])
    voxel_dims = (image.header["pixdim"])[1:4]
    print("Voxel dimensions:")
    print("  x = {} mm".format(voxel_dims[0]))
    print("  y = {} mm".format(voxel_dims[1]))
    print("  z = {} mm".format(voxel_dims[2]))

    a = time.perf_counter()
    image_hu_data = (image_data - image.dataobj.inter) / image.dataobj.slope
    print(f"hu quant took {time.perf_counter()-a:.3f}s")

    a = time.perf_counter()
    # exclude 0 as it is background
    labels = (lambda x: x[x > 0])(np.unique(mask_data))
    print(labels)
    mapped_labels = [Vertebrae(i) for i in labels]
    print(f"np unique took {time.perf_counter()-a:.3f}s")
    print(f"labels found: {labels}")

    result = deque()
    for vertebrae in mapped_labels:
        a = time.perf_counter()
        label_mask = (mask_data == vertebrae).astype(int)
        masked_image = image_hu_data * label_mask
        # don't need to include volume because it is canceled out
        density = int(np.sum(masked_image) / np.sum(label_mask))
        result.append({"Vertebrae": vertebrae.name, "Density": density})
        print(
            f"Calculating density for label_nr: {vertebrae} took {time.perf_counter()-a:.2f}s"
        )
        print(f"Label: {vertebrae}, Density: {density}")
    return result, image_data, mask_data


def mask_slice(mask):
    """ Gets the slice where the maximum of segmentation is visible. """
    slice_values = [np.count_nonzero(mask[:, :, i]) for i in range(mask.shape[-1])]
    return np.argmax(slice_values)


def create_overview_image(image_data, mask_data, all_vertebraes, output):
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8,6))
    slice_nr = mask_slice(mask_data)
    c = filter_colors(all_vertebraes)
    c.reverse()

    new_mask = np.zeros((mask_data.shape[0], mask_data.shape[1], 3))
    for i, v in enumerate(all_vertebraes):
        new_mask[mask_data[:,:,slice_nr] == v.value] = c[i]
    fig.suptitle(f"Showing slice {slice_nr}")
    ax[0].axis("off")
    ax[0].imshow(image_data[:, :, slice_nr], cmap="gray")
    ax[1].axis("off")
    ax[1].imshow(new_mask/255)

    fname = f"{output}/visualize_prediction.png"
    plt.savefig(fname, dpi=300)
    
    return fname

