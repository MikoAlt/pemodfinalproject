import cv2
import numpy as np

def preprocess_image(image_path, target_size=(40, 40)):
    """
    Load an image, convert to grayscale, resize, and normalize.
    Returns: (flattened_vector, resized_grayscale_image)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, target_size)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_flattened = img_normalized.flatten()
    return img_flattened, img_resized

def save_image(img, save_path):
    """
    Save an image to the specified path.
    """
    cv2.imwrite(str(save_path), img)

def rotate_image(img, angle):
    """
    Rotate image by a given angle.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def flip_image(img, flip_code):
    """
    Flip image: 0 for vertical, 1 for horizontal, -1 for both.
    """
    return cv2.flip(img, flip_code)

def add_noise(img, sigma=0.05):
    """
    Add random Gaussian noise to the image.
    """
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    img_noisy = (img.astype(np.float32) / 255.0) + noise
    img_noisy = np.clip(img_noisy * 255.0, 0, 255).astype(np.uint8)
    return img_noisy
