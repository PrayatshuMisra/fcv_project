import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def apply_contrast_stretching(image):
    """
    Min-Max contrast stretching on the Luminance (L) channel in LAB color space.
    Enhances contrast while preserving natural chromaticity.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    min_val = float(np.min(l))
    max_val = float(np.max(l))
    if max_val > min_val:
        stretched = ((l - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        stretched = l
        
    lab_stretched = cv2.merge((stretched, a, b))
    return cv2.cvtColor(lab_stretched, cv2.COLOR_LAB2RGB)

def apply_histogram_equalization(image):
    """
    Global Histogram Equalization on the Luminance (L) channel in LAB color space.
    Flattens the intensity histogram to maximize global dynamic range.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel.
    Standard clinical preprocessing for dermoscopy: enhances local contrast without
    over-amplifying background skin noise.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

def apply_gamma_correction(image, gamma=1.0):
    """
    Power-Law (Gamma) Transformation: I_out = 255 * (I_in / 255)^gamma
    Normalizes exposure for underexposed or overexposed clinical photographs.
    """
    inv_gamma = 1.0 / max(gamma, 1e-4)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)

def apply_hair_removal(image):
    """
    DullRazor-inspired morphological hair artifact suppression.
    Uses Black Top-Hat filtering to identify dark, thin hair fibers
    followed by Telea inpainting to reconstruct underlying skin.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Structural element for hair extraction (cross-like / elongated lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold to create binary hair mask
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint hair regions using fast marching method
    inpainted = cv2.inpaint(image, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted, hair_mask

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Gaussian low-pass filtering for high-frequency noise attenuation."""
    k = max(3, kernel_size[0] if kernel_size[0] % 2 == 1 else kernel_size[0] + 1)
    return cv2.GaussianBlur(image, (k, k), 0)

def apply_median_blur(image, kernel_size=5):
    """Non-linear median filtering to remove impulsive noise and fine skin hair."""
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.medianBlur(image, k)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Bilateral filter for edge-preserving smoothing.
    Smooths skin surface variations while maintaining sharp lesion boundaries.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_canny_edge(image, t1=100, t2=200):
    """Canny Edge Detection with multi-stage gradient & hysteresis thresholding."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return cv2.Canny(gray, t1, t2)

def apply_sobel_edge(image, ksize=3):
    """
    Sobel gradient magnitude operator: sqrt(Gx^2 + Gy^2).
    Evaluates first-order spatial derivatives along horizontal and vertical axes.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_mag = np.clip(grad_mag / (grad_mag.max() + 1e-7) * 255.0, 0, 255).astype(np.uint8)
    return grad_mag

def get_color_space_maps(image):
    """
    Decomposes the image into HSV and CIE LAB color spaces.
    Returns RGB visualizations of each channel for clinical inspection.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    
    # Map single channels to RGB colormaps for intuitive inspection
    h_col = cv2.applyColorMap(h, cv2.COLORMAP_HSV)
    h_col = cv2.cvtColor(h_col, cv2.COLOR_BGR2RGB)
    
    a_col = cv2.applyColorMap(a, cv2.COLORMAP_COOL)
    a_col = cv2.cvtColor(a_col, cv2.COLOR_BGR2RGB)
    
    b_col = cv2.applyColorMap(b, cv2.COLORMAP_VIRIDIS)
    b_col = cv2.cvtColor(b_col, cv2.COLOR_BGR2RGB)
    
    return {
        "hsv_hue": h_col,
        "hsv_saturation": cv2.cvtColor(s, cv2.COLOR_GRAY2RGB),
        "lab_lightness": cv2.cvtColor(l, cv2.COLOR_GRAY2RGB),
        "lab_a": a_col,
        "lab_b": b_col
    }

def apply_kmeans(image, k=3):
    """
    K-Means clustering in 3D color space.
    Partitions the image into k clusters based on Euclidean color distance.
    """
    pixel_values = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(image.shape)
    labels_2d = labels.reshape(image.shape[:2])
    return segmented_image, labels_2d, centers

def apply_otsu_thresholding(image):
    """
    Otsu's global binarization: minimizes intra-class intensity variance.
    Serves as an automated threshold benchmark against K-Means segmentation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Lesions are darker than healthy skin -> invert binary mask
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return otsu_mask

def identify_lesion_cluster(image, labels, k_clusters):
    """
    Robust automated lesion cluster localization for real dermoscopy images.
    Identifies the background skin cluster as the dominant area cluster in the field.
    The lesion mask is formed by non-background clusters, with morphological refinement.
    """
    h, w = labels.shape
    counts = np.bincount(labels.flatten(), minlength=k_clusters)
    bg_cluster = int(np.argmax(counts))
    
    # Identify the most prominent non-background lesion cluster
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    best_lesion_cluster = None
    min_mean_intensity = float('inf')
    
    for c in range(k_clusters):
        if c == bg_cluster:
            continue
        mask_c = (labels == c)
        if np.sum(mask_c) == 0:
            continue
        mean_intensity = float(np.mean(gray[mask_c]))
        if mean_intensity < min_mean_intensity:
            min_mean_intensity = mean_intensity
            best_lesion_cluster = c
            
    if best_lesion_cluster is None:
        best_lesion_cluster = (bg_cluster + 1) % k_clusters
        
    # Lesion mask includes non-background tissue
    binary_mask = (labels != bg_cluster).astype(np.uint8) * 255
    
    # Morphological cleanup (Opening to remove specks, Closing to seal holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    return best_lesion_cluster, binary_mask

def extract_lbp(image, radius=3, n_points=24):
    """
    Computes rotation-invariant Uniform Local Binary Pattern (LBP) texture descriptor.
    Generates normalized 1D histogram representing micro-surface roughness.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    
    bins = np.arange(0, n_points + 3)
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return lbp, hist

def extract_glcm_features(image):
    """
    Gray-Level Co-occurrence Matrix (GLCM) statistical texture analysis.
    Extracts Haralick texture descriptors across multiple spatial offsets and angles.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Downscale intensity levels to 32 bins for fast, robust GLCM computation
    gray_quantized = (gray // 8).astype(np.uint8)
    
    glcm = graycomatrix(
        gray_quantized,
        distances=[1, 3],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=32,
        symmetric=True,
        normed=True
    )
    
    contrast = float(np.mean(graycoprops(glcm, 'contrast')))
    dissimilarity = float(np.mean(graycoprops(glcm, 'dissimilarity')))
    homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
    energy = float(np.mean(graycoprops(glcm, 'energy')))
    correlation = float(np.mean(graycoprops(glcm, 'correlation')))
    
    return {
        "contrast": round(contrast, 3),
        "dissimilarity": round(dissimilarity, 3),
        "homogeneity": round(homogeneity, 3),
        "energy": round(energy, 3),
        "correlation": round(correlation, 3)
    }

def generate_lesion_contour_overlay(image, binary_mask):
    """
    Draws diagnostic geometric primitives on the original image:
    - Primary lesion boundary contour (Red)
    - Convex Hull (Cyan)
    - Minimum Area Rotated Bounding Box (Green)
    - Lesion Centroid (Yellow Crosshair)
    """
    overlay = image.copy()
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return overlay, None
        
    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    if area < 50:
        return overlay, None
        
    # Draw primary lesion boundary (Red)
    cv2.drawContours(overlay, [main_contour], -1, (255, 30, 30), 2)
    
    # Convex Hull (Cyan)
    hull = cv2.convexHull(main_contour)
    cv2.drawContours(overlay, [hull], -1, (0, 220, 255), 2)
    
    # Minimum area bounding box (Green)
    rect = cv2.minAreaRect(main_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(overlay, [box], 0, (40, 255, 40), 1)
    
    # Centroid (Yellow)
    M = cv2.moments(main_contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.drawMarker(overlay, (cx, cy), (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
        
    return overlay, main_contour

def calculate_abcd_metrics(image, binary_mask):
    """
    Comprehensive Stolz ABCD Rule & Total Dermoscopy Score (TDS) Calculation:
    - A (Asymmetry, 0-2): Calculated via principal axes alignment and 2-axis fold-over mask overlap
    - B (Border Irregularity, 0-8): Evaluated by radial 8-octant contour indentation & compactness
    - C (Color Variegation, 1-6): Detection of 6 clinical dermatological color bands
    - D (Diameter, 1-5): Calibrated millimeter lesion diameter
    - TDS = 1.3 * A + 0.1 * B + 0.5 * C + 0.5 * D
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"error": "Lesion boundary could not be resolved"}
        
    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    
    if area < 50 or perimeter < 10:
        return {"error": "Inadequate lesion area detected"}
        
    # --- A: Asymmetry Assessment (0 - 2) ---
    M = cv2.moments(main_contour)
    asymmetry_score = 0
    asymmetry_detail = "Symmetric along both axes"
    
    if M["m00"] > 0 and len(main_contour) >= 5:
        # Orient mask along principal inertia axis
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        mu20 = M["mu20"] / M["m00"]
        mu02 = M["mu02"] / M["m00"]
        mu11 = M["mu11"] / M["m00"]
        
        theta = 0.5 * np.arctan2(2 * mu11, (mu20 - mu02))
        angle_deg = np.degrees(theta)
        
        # Center and rotate binary mask
        h, w = binary_mask.shape
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rotated_mask = cv2.warpAffine(binary_mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST)
        
        # Crop to bounding box of rotated mask
        r_contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if r_contours:
            r_main = max(r_contours, key=cv2.contourArea)
            rx, ry, rw, rh = cv2.boundingRect(r_main)
            cropped = rotated_mask[ry:ry+rh, rx:rx+rw]
            
            # Horizontal fold asymmetry (flip vertical axis)
            ch, cw = cropped.shape
            half_w = cw // 2
            left_half = cropped[:, :half_w]
            right_half = cv2.flip(cropped[:, half_w:half_w*2], 1)
            min_w = min(left_half.shape[1], right_half.shape[1])
            diff_h = np.bitwise_xor(left_half[:, :min_w], right_half[:, :min_w])
            asym_h = np.sum(diff_h > 0) / (np.sum(cropped > 0) + 1e-7)
            
            # Vertical fold asymmetry (flip horizontal axis)
            half_h = ch // 2
            top_half = cropped[:half_h, :]
            bottom_half = cv2.flip(cropped[half_h:half_h*2, :], 0)
            min_h = min(top_half.shape[0], bottom_half.shape[0])
            diff_v = np.bitwise_xor(top_half[:min_h, :], bottom_half[:min_h, :])
            asym_v = np.sum(diff_v > 0) / (np.sum(cropped > 0) + 1e-7)
            
            if asym_h > 0.08:
                asymmetry_score += 1
            if asym_v > 0.08:
                asymmetry_score += 1
                
            if asymmetry_score == 2:
                asymmetry_detail = "Asymmetric along both major and minor axes"
            elif asymmetry_score == 1:
                asymmetry_detail = "Asymmetric along one axis"
        else:
            asymmetry_score = 1
    else:
        asymmetry_score = 1
        
    # --- B: Border Irregularity (0 - 8) ---
    # Compactness metric: P^2 / (4 * pi * Area)
    compactness = (perimeter ** 2) / (4 * np.pi * area)
    
    # Evaluate 8 radial octants (every 45 degrees) around the centroid
    M_cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else w // 2
    M_cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else h // 2
    
    # Calculate radial distances to contour points
    pts = main_contour.reshape(-1, 2)
    angles = np.arctan2(pts[:, 1] - M_cy, pts[:, 0] - M_cx)
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    distances = np.sqrt((pts[:, 1] - M_cy)**2 + (pts[:, 0] - M_cx)**2)
    
    octant_irregular_count = 0
    for octant in range(8):
        start_angle = octant * (np.pi / 4)
        end_angle = (octant + 1) * (np.pi / 4)
        oct_mask = (angles >= start_angle) & (angles < end_angle)
        if np.sum(oct_mask) > 3:
            oct_dists = distances[oct_mask]
            # Standard deviation of distance in octant indicates border raggedness
            oct_std = np.std(oct_dists) / (np.mean(oct_dists) + 1e-7)
            if oct_std > 0.14 or compactness > 1.6:
                octant_irregular_count += 1
                
    border_score = min(8, max(0, octant_irregular_count))
    
    # --- C: Color Variegation (1 - 6) ---
    # Detect 6 clinically established dermatological colors inside lesion mask:
    # 1. White, 2. Red, 3. Light Brown, 4. Dark Brown, 5. Slate Blue/Gray, 6. Black
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lesion_pixels_rgb = image[binary_mask > 0]
    lesion_pixels_hsv = hsv[binary_mask > 0]
    
    total_lesion_px = len(lesion_pixels_rgb)
    colors_detected = []
    
    if total_lesion_px > 0:
        r, g, b = lesion_pixels_rgb[:, 0], lesion_pixels_rgb[:, 1], lesion_pixels_rgb[:, 2]
        h_ch, s_ch, v_ch = lesion_pixels_hsv[:, 0], lesion_pixels_hsv[:, 1], lesion_pixels_hsv[:, 2]
        
        # 1. Black: Low Value / intensity
        black_px = np.sum(v_ch < 45)
        if (black_px / total_lesion_px) > 0.03:
            colors_detected.append("Black")
            
        # 2. White (Hypopigmentation): High brightness, low saturation
        white_px = np.sum((v_ch > 200) & (s_ch < 40))
        if (white_px / total_lesion_px) > 0.03:
            colors_detected.append("White")
            
        # 3. Red / Erythema: Hue in red ranges (0-10 or 170-180), high saturation
        red_px = np.sum(((h_ch < 10) | (h_ch > 170)) & (s_ch > 60) & (v_ch > 80))
        if (red_px / total_lesion_px) > 0.03:
            colors_detected.append("Red")
            
        # 4. Dark Brown: Low-to-medium value, warm hue
        dark_brown_px = np.sum((h_ch >= 10) & (h_ch <= 30) & (s_ch > 70) & (v_ch >= 45) & (v_ch <= 120))
        if (dark_brown_px / total_lesion_px) > 0.04:
            colors_detected.append("Dark Brown")
            
        # 5. Light Brown: Medium-to-high value, warm hue
        light_brown_px = np.sum((h_ch >= 10) & (h_ch <= 35) & (s_ch >= 40) & (v_ch > 120) & (v_ch <= 200))
        if (light_brown_px / total_lesion_px) > 0.04:
            colors_detected.append("Light Brown")
            
        # 6. Slate Blue / Gray: Moderate-to-low saturation, bluish hue, moderate value
        blue_gray_px = np.sum((h_ch >= 95) & (h_ch <= 135) & (s_ch > 25) & (v_ch > 50) & (v_ch < 180))
        if (blue_gray_px / total_lesion_px) > 0.02:
            colors_detected.append("Slate Blue/Gray")
            
    color_score = max(1, min(6, len(colors_detected)))
    if color_score == 0:
        color_score = 1
        colors_detected = ["Monochromatic Brown"]
        
    # --- D: Diameter Score (1 - 5) ---
    # Approximate maximum feret diameter in pixels
    _, _, bw, bh = cv2.boundingRect(main_contour)
    max_dim_px = max(bw, bh)
    
    # Calibration assumption: typical macro dermoscopy image has ~25 px/mm
    est_diameter_mm = max_dim_px / 25.0
    if est_diameter_mm < 3.0:
        diameter_score = 1
    elif est_diameter_mm < 6.0:
        diameter_score = 2
    elif est_diameter_mm < 8.0:
        diameter_score = 3
    elif est_diameter_mm < 10.0:
        diameter_score = 4
    else:
        diameter_score = 5
        
    # --- Stolz Total Dermoscopy Score (TDS) ---
    # Formula: TDS = 1.3 * A + 0.1 * B + 0.5 * C + 0.5 * D
    tds = (1.3 * asymmetry_score) + (0.1 * border_score) + (0.5 * color_score) + (0.5 * diameter_score)
    tds = round(float(tds), 2)
    
    # Clinical categorization based on Stolz standard thresholds
    if tds < 4.75:
        risk_level = "Low Risk / Likely Benign"
        clinical_action = "Routine annual dermatological surveillance recommended."
        badge_class = "risk-benign"
    elif tds < 5.45:
        risk_level = "Moderate Risk / Suspicious Lesion"
        clinical_action = "Short-term digital dermoscopy follow-up (3 months) or specialist evaluation."
        badge_class = "risk-moderate"
    else:
        risk_level = "High Risk / Suggestive of Malignancy"
        clinical_action = "Urgent dermatological referral and histopathological biopsy strongly indicated."
        badge_class = "risk-high"
        
    return {
        "asymmetry_score": asymmetry_score,
        "asymmetry_detail": asymmetry_detail,
        "border_score": border_score,
        "border_compactness": round(float(compactness), 2),
        "color_score": color_score,
        "colors_detected": colors_detected,
        "diameter_score": diameter_score,
        "diameter_mm": round(float(est_diameter_mm), 1),
        "diameter_px": int(max_dim_px),
        "tds": tds,
        "risk_level": risk_level,
        "clinical_action": clinical_action,
        "badge_class": badge_class,
        "area_px": int(area),
        "perimeter_px": int(perimeter)
    }
