import cv2
import numpy as np
import os
import random


# Define augmentation functions
def add_random_black_spots(image):
    img_with_spots = image.copy()
    h, w = image.shape[:2]

    num_spots = random.randint(20, 50)
    for _ in range(num_spots):
        spot_size = random.randint(1, 5)
        x = random.randint(0, w - spot_size)
        y = random.randint(0, h - spot_size)
        img_with_spots[y:y + spot_size, x:x + spot_size] = 0

    return img_with_spots


def flip_image(image):
    return cv2.flip(image, 1)  # Horizontal flip


def adjust_brightness(image, factor=1.1):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    bright_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bright_img


# Apply augmentations to a dataset
def augment_dataset(input_folder, output_folder, num_augmentations=5):
    os.makedirs(output_folder, exist_ok=True)
    k = 0
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            continue

        cv2.imwrite(os.path.join(output_folder, str(k) + ".png"), image)
        k += 1

        for i in range(num_augmentations):
            aug_image = image
            if random.choice([True, False]):
                aug_image = add_random_black_spots(aug_image)
            if random.choice([True, False]):
                aug_image = flip_image(aug_image)
            if random.choice([True, False]):
                aug_image = adjust_brightness(aug_image, factor=random.uniform(0.7, 1.3))

            # Save augmented image
            aug_filename = str(k) + ".png"
            k += 1
            cv2.imwrite(os.path.join(output_folder, aug_filename), aug_image)


# Example usage
input_folder = './dataset/mydrilling'
output_folder = './dataset/mydrilling_aug'
augment_dataset(input_folder, output_folder)
