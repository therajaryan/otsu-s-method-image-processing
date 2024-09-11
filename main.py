import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def threshold_image(im, th):
    """
    Apply thresholding to the input image based on the provided threshold value.

    Parameters:
    im (numpy.ndarray): Grayscale input image.
    th (int): Threshold value.

    Returns:
    numpy.ndarray: Binarized image after applying the threshold.
    """
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1
    return thresholded_im

def compute_otsu_criteria(im, th):
    """
    Compute Otsu's thresholding criteria for a given threshold value.

    Parameters:
    im (numpy.ndarray): Grayscale input image.
    th (int): Threshold value.

    Returns:
    float: Weighted within-class variance for the given threshold.
    """
    thresholded_im = threshold_image(im, th)
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    if weight1 == 0 or weight0 == 0:
        return np.inf

    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    return weight0 * var0 + weight1 * var1

def find_best_threshold(im):
    """
    Find the optimal threshold value using Otsu's method.

    Parameters:
    im (numpy.ndarray): Grayscale input image.

    Returns:
    int: Optimal threshold value.
    """
    threshold_range = range(np.max(im) + 1)
    criterias = [compute_otsu_criteria(im, th) for th in threshold_range]
    best_threshold = threshold_range[np.argmin(criterias)]
    print(f"The optimal threshold value determined by Otsu's method is: {best_threshold}")
    return best_threshold

def plot_histogram(im, threshold):
    """
    Plot the histogram of pixel intensities with the threshold indicated.

    Parameters:
    im (numpy.ndarray): Grayscale input image.
    threshold (int): Threshold value to display on the histogram.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(im.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold = {threshold}')
    plt.title('Histogram of Pixel Intensities with Otsu\'s Threshold')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Load the image and apply Otsu's method
path_image = 'dp.jpg'  # Replace with your image path
im = np.asarray(Image.open(path_image).convert('L'))

# Find the optimal threshold
optimal_threshold = find_best_threshold(im)

# Apply the optimal threshold to segment the image
im_otsu = threshold_image(im, optimal_threshold)

# Plot the histogram with the threshold
plot_histogram(im, optimal_threshold)

# Display the original grayscale image and the segmented image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(im, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(1, 2, 2)
plt.imshow(im_otsu, cmap='gray')
plt.title('Segmented Image Using Otsu\'s Method')

plt.tight_layout()
plt.show()

