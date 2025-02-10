# This script is used to check the category distribution

import os
import numpy as np
from collections import defaultdict
import cv2
# Define class names and their corresponding indices
palette = [
    ['background', 0],
    ['vegetation', 1],
    ['buildings', 2],
    ['road', 3],
    ['water', 4],
    ['farmland', 5]
]

# Convert class names and indices into dictionaries
class_indices = {name: idx for name, idx in palette}
index_names = {idx: name for name, idx in palette}

# Path to the mask directory
mask_dir = 'RuralUse/ann_dir/mask'

# Get the list of mask files
mask_files = sorted(os.listdir(mask_dir))

# Initialize statistics dictionaries
class_pixel_counts = defaultdict(int)  # Total pixel count for each class
class_image_counts = defaultdict(int)  # Number of images where each class appears

# Iterate through all mask files
for mask_file in mask_files:
    # Read the mask file
    mask_path = os.path.join(mask_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Count the number of pixels for each class in the current mask
    unique_classes, counts = np.unique(mask, return_counts=True)

    # Update the statistics
    for cls, count in zip(unique_classes, counts):
        class_name = index_names.get(cls, 'unknown')
        class_pixel_counts[class_name] += count
        if count > 0:  # If the class appears in the current image
            class_image_counts[class_name] += 1

# Output the statistics
print("Class Pixel Counts:")
for class_name, pixel_count in class_pixel_counts.items():
    print(f"{class_name}: {pixel_count} pixels")

print("\nClass Occurrence Counts:")
for class_name, image_count in class_image_counts.items():
    print(f"{class_name}: appears in {image_count} images")