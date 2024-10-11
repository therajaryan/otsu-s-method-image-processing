import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu

def multi_threshold_image(im, thresholds):
    """
    Apply multi-thresholding to the input image based on the provided threshold values.

    Parameters:
    im (numpy.ndarray): Grayscale input image.
    thresholds (list or array): List of threshold values.

    Returns:
    numpy.ndarray: Multi-segmented image after applying the thresholds.
    """
    segmented_im = np.digitize(im, bins=thresholds)
    return segmented_im

def plot_histogram(im, thresholds):
    """
    Plot the histogram of pixel intensities with the thresholds indicated.

    Parameters:
    im (numpy.ndarray): Grayscale input image.
    thresholds (list or array): Threshold values to display on the histogram.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(im.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    for th in thresholds:
        plt.axvline(th, color='red', linestyle='dashed', linewidth=2, label=f'Threshold = {th}')
    plt.title('Histogram of Pixel Intensities with Multi-Otsu Thresholds')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Load the image and apply Multi-Otsu's method
path_image = 'dp.jpg'  # Replace with your image path
im = np.asarray(Image.open(path_image).convert('L'))

# Find the optimal thresholds using Multi-Otsu's method
n_classes = 3  # Number of classes (regions) you want to segment
thresholds = threshold_multiotsu(im, classes=n_classes)
print(f"Optimal thresholds determined by Multi-Otsu's method: {thresholds}")

# Apply the optimal thresholds to segment the image
im_multi_otsu = multi_threshold_image(im, thresholds)

# Plot the histogram with the thresholds
plot_histogram(im, thresholds)

# Display the original grayscale image and the multi-segmented image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(im, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(1, 2, 2)
plt.imshow(im_multi_otsu, cmap='gray')
plt.title('Segmented Image Using Multi-Otsu\'s Method')

plt.tight_layout()
plt.show()
