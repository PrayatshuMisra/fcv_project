import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def apply_contrast_stretching(image):
    """
    Apply contrast stretching to an RGB image.
    Enhances contrast by stretching the intensity values to the full 0-255 range.
    """
    # Convert to LAB color space to separate lightness from color
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Min-Max contrast stretching on L channel
    min_val = np.min(l)
    max_val = np.max(l)
    if max_val > min_val:
        stretched = ((l - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        stretched = l
        
    lab_stretched = cv2.merge((stretched, a, b))
    return cv2.cvtColor(lab_stretched, cv2.COLOR_LAB2RGB)

def apply_histogram_equalization(image):
    """
    Apply histogram equalization to an RGB image.
    Improves contrast by flattening the histogram of the lightness channel.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Histogram Equalization on L channel
    l_eq = cv2.equalizeHist(l)
    
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """
    Apply Gaussian Blur for noise reduction.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_canny_edge(image, t1=100, t2=200):
    """
    Apply Canny Edge Detection to find structural boundaries.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, t1, t2)
    return edges

def apply_kmeans(image, k=3):
    """
    Apply K-Means clustering for color-based segmentation.
    Returns the segmented image and the labels.
    """
    # Reshape image into a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Define criteria and apply kmeans
    # Criteria: Stop when either max_iter is reached or epsilon (accuracy) is reached
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers back to 8-bit values
    centers = np.uint8(centers)
    
    # Map labels back to the original image dimensions
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    
    return segmented_image, labels.reshape(image.shape[:2])

def extract_lbp(image, radius=3, n_points=24):
    """
    Extract Local Binary Pattern (LBP) features.
    Returns the LBP image and the normalized histogram of LBP codes.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    
    # Compute histogram
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return lbp, hist

def calculate_abcd_metrics(image, labels, k_clusters):
    """
    Simulates the ABCD diagnostic rule (Asymmetry, Border, Color, Diameter).
    We assume the cluster with the darkest average color or lowest area (but not the background) is the lesion.
    For simplicity, we find the cluster in the center of the image.
    """
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    
    # Guess which cluster is the lesion (the one in the center of the image)
    lesion_cluster_idx = labels[center_y, center_x]
    
    # Create binary mask for the lesion
    mask = (labels == lesion_cluster_idx).astype(np.uint8) * 255
    
    # Morphological cleanup
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {"error": "Lesion not found"}
        
    # Get the largest contour (assuming it's the lesion)
    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    
    if area == 0:
        return {"error": "Invalid lesion area"}
        
    # A - Asymmetry: Calculate bounding box, split in half, check overlap
    x, y, bw, bh = cv2.boundingRect(main_contour)
    # Simple metric: Ratio of bounding box dimensions (perfect circle = 1.0)
    asymmetry_score = max(bw/bh, bh/bw) 
    
    # B - Border Irregularity: Compactness metric (Perimeter^2 / (4 * pi * Area))
    # A perfect circle has compactness of 1.0. Irregular borders have higher values.
    compactness = (perimeter ** 2) / (4 * np.pi * area)
    
    # C - Color Variance: Standard deviation of RGB channels inside the mask
    mean_val, std_val = cv2.meanStdDev(image, mask=mask)
    color_variance = float(np.mean(std_val))
    
    # Scoring system
    risk_points = 0
    
    # Thresholds (Arbitrary for demonstration)
    if asymmetry_score > 1.3:
        risk_points += 1
    if compactness > 1.5:
        risk_points += 1
    if color_variance > 35.0:
        risk_points += 1
        
    if risk_points >= 2:
        risk_level = "High Risk (Consult Dermatologist)"
    elif risk_points == 1:
        risk_level = "Moderate Risk (Monitor Closely)"
    else:
        risk_level = "Low Risk (Likely Benign)"
        
    return {
        "asymmetry": round(asymmetry_score, 2),
        "border_irregularity": round(compactness, 2),
        "color_variance": round(color_variance, 2),
        "risk_level": risk_level,
        "risk_score": risk_points
    }
