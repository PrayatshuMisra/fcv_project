from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
import cv2
import numpy as np
import base64
import os
import utils

app = FastAPI(title="ClinicalVision - Skin Conditions Triage System", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static frontend assets
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def redirect_to_ui():
    return RedirectResponse(url="/static/index.html")

def img_to_base64(img):
    """Convert RGB image array to base64 JPEG string"""
    if img is None:
        return ""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')

def single_channel_to_base64(img):
    """Convert Grayscale or 1-channel binary mask to base64 JPEG string"""
    if img is None:
        return ""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/samples")
def get_sample_images():
    """Returns catalog of pre-loaded clinical dermoscopy test cases"""
    samples = [
        {
            "id": "melanoma",
            "name": "Superficial Spreading Melanoma (ISIC_0000031)",
            "diagnosis": "Malignant Melanoma (Histopathology Confirmed)",
            "description": "Validated ISIC Archive case ISIC_0000031: Marked structural asymmetry, ragged notched borders, and deep melanin pigmentation.",
            "filename": "melanoma.jpg"
        },
        {
            "id": "benign_nevus",
            "name": "Benign Melanocytic Nevus (ISIC_0000001)",
            "diagnosis": "Benign Melanocytic Nevus (Histopathology Confirmed)",
            "description": "Validated ISIC Archive case ISIC_0000001: Uniform benign melanocytic proliferation with regular reticular network.",
            "filename": "benign_nevus.jpg"
        },
        {
            "id": "basal_cell",
            "name": "Nodular Basal Cell Carcinoma",
            "diagnosis": "Basal Cell Carcinoma (BCC)",
            "description": "Real clinical dermoscopy: Pearly translucent nodule with prominent arborizing telangiectasia and focal blue-gray nests.",
            "filename": "basal_cell.jpg"
        },
        {
            "id": "seborrheic_keratosis",
            "name": "Pigmented Benign Keratosis (ISIC_0024312)",
            "diagnosis": "Seborrheic Keratosis (Histopathology Confirmed)",
            "description": "Validated ISIC Archive case ISIC_0024312: Benign epidermal proliferation exhibiting characteristic rough micro-texture and pseudofollicular cysts.",
            "filename": "seborrheic_keratosis.jpg"
        }
    ]
    return JSONResponse(content={"samples": samples})

@app.post("/process")
async def process_image(
    file: UploadFile = File(None),
    image_url: str = Form(None),
    enhancement_type: str = Form("CLAHE"),
    gamma_val: float = Form(1.0),
    filter_type: str = Form("Bilateral"),
    blur_kernel: int = Form(5),
    apply_hair_removal: bool = Form(False),
    edge_operator: str = Form("Canny"),
    t1: int = Form(100),
    t2: int = Form(200),
    k_clusters: int = Form(3),
    lbp_radius: int = Form(3),
    lbp_points: int = Form(24)
):
    try:
        # Fallback for direct non-FastAPI invocations
        if hasattr(enhancement_type, 'default'): enhancement_type = enhancement_type.default
        if hasattr(gamma_val, 'default'): gamma_val = float(gamma_val.default)
        if hasattr(filter_type, 'default'): filter_type = filter_type.default
        if hasattr(blur_kernel, 'default'): blur_kernel = int(blur_kernel.default)
        if hasattr(apply_hair_removal, 'default'): apply_hair_removal = bool(apply_hair_removal.default)
        if hasattr(edge_operator, 'default'): edge_operator = edge_operator.default
        if hasattr(t1, 'default'): t1 = int(t1.default)
        if hasattr(t2, 'default'): t2 = int(t2.default)
        if hasattr(k_clusters, 'default'): k_clusters = int(k_clusters.default)
        if hasattr(lbp_radius, 'default'): lbp_radius = int(lbp_radius.default)
        if hasattr(lbp_points, 'default'): lbp_points = int(lbp_points.default)

        # 1. Read input image from multipart upload, direct URL, or sample selector
        contents = None
        if hasattr(file, 'filename') and file.filename:
            contents = await file.read()
        elif image_url:
            # Handle local sample reference
            if image_url.startswith("/static/samples/") or image_url.startswith("static/samples/"):
                clean_path = image_url.lstrip("/")
                if os.path.exists(clean_path):
                    with open(clean_path, "rb") as f:
                        contents = f.read()
            else:
                import urllib.request
                from urllib.parse import urlparse, parse_qs
                
                # Extract image URL if pasted from Google image search
                if "google.com/imgres" in image_url:
                    parsed_url = urlparse(image_url)
                    qs = parse_qs(parsed_url.query)
                    if 'imgurl' in qs:
                        image_url = qs['imgurl'][0]
                        
                req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    contents = response.read()
                    
        if contents is None:
            return JSONResponse(status_code=400, content={"error": "No image provided. Please upload a file or choose a sample."})
            
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return JSONResponse(status_code=400, content={"error": "Failed to decode image. Please ensure the file is a valid JPG or PNG."})
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Phase 0: Optional Morphological Hair Removal (DullRazor Pre-Filter)
        hair_mask_b64 = None
        if apply_hair_removal:
            img_processed, hair_mask = utils.apply_hair_removal(img_rgb)
            hair_mask_b64 = single_channel_to_base64(hair_mask)
        else:
            img_processed = img_rgb.copy()
            
        # Phase 1: Spatial Domain Enhancement (Lab 2)
        if enhancement_type == "CLAHE":
            enhanced_img = utils.apply_clahe(img_processed, clip_limit=2.0)
        elif enhancement_type == "Contrast Stretching":
            enhanced_img = utils.apply_contrast_stretching(img_processed)
        elif enhancement_type == "Histogram Equalization":
            enhanced_img = utils.apply_histogram_equalization(img_processed)
        elif enhancement_type == "Gamma Correction":
            enhanced_img = utils.apply_gamma_correction(img_processed, gamma=gamma_val)
        else:
            enhanced_img = img_processed.copy()
            
        # Phase 2: Spatial Filtering & Artifact Suppression (Lab 3)
        if filter_type == "Bilateral":
            filtered_img = utils.apply_bilateral_filter(enhanced_img, d=9, sigma_color=75, sigma_space=75)
        elif filter_type == "Median":
            filtered_img = utils.apply_median_blur(enhanced_img, kernel_size=blur_kernel)
        else:
            filtered_img = utils.apply_gaussian_blur(enhanced_img, kernel_size=(blur_kernel, blur_kernel))
            
        # Phase 3: Edge Detection (Lab 4)
        if edge_operator == "Sobel":
            edges_img = utils.apply_sobel_edge(filtered_img, ksize=3)
        else:
            edges_img = utils.apply_canny_edge(filtered_img, t1=t1, t2=t2)
            
        # Phase 4: Segmentation & Cluster Analysis (Lab 5)
        segmented_img, labels_2d, centers = utils.apply_kmeans(filtered_img, k=k_clusters)
        otsu_mask = utils.apply_otsu_thresholding(filtered_img)
        
        # Robust automated lesion localization
        lesion_cluster_idx, lesion_binary_mask = utils.identify_lesion_cluster(filtered_img, labels_2d, k_clusters)
        
        # Overlay lesion boundary, convex hull, and centroid on original image
        contour_overlay_img, main_contour = utils.generate_lesion_contour_overlay(img_rgb, lesion_binary_mask)
        
        # Multi-Color Space decomposition (Lab 1)
        color_maps = utils.get_color_space_maps(enhanced_img)
        
        # Phase 5: Quantitative Feature Extraction (Lab 5)
        # Local Binary Pattern
        lbp_img, lbp_hist = utils.extract_lbp(enhanced_img, radius=lbp_radius, n_points=lbp_points)
        lbp_vis = cv2.normalize(lbp_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # GLCM Texture Haralick Descriptors
        glcm_metrics = utils.extract_glcm_features(enhanced_img)
        
        # Phase 6: Dermatological ABCD Assessment & Stolz TDS
        abcd_metrics = utils.calculate_abcd_metrics(img_rgb, lesion_binary_mask)
        
        # Extract individual K-Means cluster isolation cards
        clusters = []
        for i in range(k_clusters):
            mask_i = (labels_2d == i).astype(np.uint8) * 255
            cluster_isolated = cv2.bitwise_and(filtered_img, filtered_img, mask=mask_i)
            clusters.append({
                "cluster_index": i,
                "is_lesion": (i == lesion_cluster_idx),
                "image": img_to_base64(cluster_isolated)
            })
            
        return JSONResponse(content={
            "original": img_to_base64(img_rgb),
            "hair_removed": img_to_base64(img_processed) if apply_hair_removal else None,
            "hair_mask": hair_mask_b64,
            "enhanced": img_to_base64(enhanced_img),
            "filtered": img_to_base64(filtered_img),
            "edges": single_channel_to_base64(edges_img),
            "segmented": img_to_base64(segmented_img),
            "otsu_mask": single_channel_to_base64(otsu_mask),
            "contour_overlay": img_to_base64(contour_overlay_img),
            "lesion_mask": single_channel_to_base64(lesion_binary_mask),
            "clusters": clusters,
            "color_maps": {
                "hsv_hue": img_to_base64(color_maps["hsv_hue"]),
                "hsv_saturation": img_to_base64(color_maps["hsv_saturation"]),
                "lab_lightness": img_to_base64(color_maps["lab_lightness"]),
                "lab_a": img_to_base64(color_maps["lab_a"]),
                "lab_b": img_to_base64(color_maps["lab_b"])
            },
            "lbp_image": single_channel_to_base64(lbp_vis),
            "lbp_hist": lbp_hist.tolist(),
            "glcm_metrics": glcm_metrics,
            "abcd_metrics": abcd_metrics,
            "dimensions": f"{img_rgb.shape[1]}x{img_rgb.shape[0]} px",
            "pipeline_summary": {
                "enhancement": enhancement_type,
                "filter": filter_type,
                "hair_removal": apply_hair_removal,
                "edge_operator": edge_operator,
                "k_clusters": k_clusters
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Pipeline processing failure: {str(e)}"})
