# This simple script is used to show the visulization result of RuralUse data

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# color patterns
palette = [
    ['background', [255, 255, 255]],
    ['vegetation', [0, 255, 0]],
    ['buildings', [128, 0, 128]],
    ['road', [0, 0, 255]],
    ['water', [255, 255, 0]],
    ['farmland', [0, 255, 255]]
]

# dic
palette_dict = {i: color for i, (name, color) in enumerate(palette)}

# root
img_dir = 'RuralUse/img_dir/Images'
mask_dir = 'RuralUse/ann_dir/mask'

# read
img_files = sorted(os.listdir(img_dir))
mask_files = sorted(os.listdir(mask_dir))

# alphas set
alphas = [0, 0.2, 0.4, 0.6, 0.8, 1]

# for
for img_file, mask_file in zip(img_files, mask_files):
    # read
    img_path = os.path.join(img_dir, img_file)
    mask_path = os.path.join(mask_dir, mask_file)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # color
    colored_mask = np.zeros_like(img)
    for i, color in palette_dict.items():
        colored_mask[mask == i] = color

    # create one line to show the result
    fig, axes = plt.subplots(1, len(alphas), figsize=(15, 5))

    for i, alpha in enumerate(alphas):
        # overlap
        overlay = cv2.addWeighted(img, 1 - alpha, colored_mask, alpha, 0)

        # vis
        axes[i].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f'Alpha: {alpha}')
        axes[i].axis('off')

    # plt.suptitle(f'Image: {img_file}, Mask: {mask_file}')

    plt.savefig(f'save/{img_file}.png', bbox_inches='tight', pad_inches=0)
    plt.show()
