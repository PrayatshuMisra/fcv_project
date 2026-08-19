from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
import cv2
import numpy as np
import base64
import utils

app = FastAPI(title="Skin Conditions Triage API")

# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def redirect_to_ui():
    return RedirectResponse(url="/static/index.html")

def img_to_base64(img):
    """Convert RGB image to Base64 string for HTML display"""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def single_channel_to_base64(img):
    """Convert Grayscale/1-Channel image to Base64 string"""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    enhancement_type: str = Form("Contrast Stretching"),
    blur_kernel: int = Form(5),
    t1: int = Form(100),
    t2: int = Form(200),
    k_clusters: int = Form(3),
    lbp_radius: int = Form(3),
    lbp_points: int = Form(24)
):
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file format."})
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Phase 1: Enhancement
        if enhancement_type == "Contrast Stretching":
            enhanced_img = utils.apply_contrast_stretching(img_rgb)
        else:
            enhanced_img = utils.apply_histogram_equalization(img_rgb)
            
        # Phase 2: Blur and Edges
        blurred_img = utils.apply_gaussian_blur(enhanced_img, kernel_size=(blur_kernel, blur_kernel))
        edges_img = utils.apply_canny_edge(blurred_img, t1, t2)
        
        # Phase 3: Segmentation
        segmented_img, labels = utils.apply_kmeans(blurred_img, k=k_clusters)
        
        # Phase 4: Feature Extraction
        lbp_img, lbp_hist = utils.extract_lbp(enhanced_img, radius=lbp_radius, n_points=lbp_points)
        
        # Phase 5: ABCD Risk Assessment
        abcd_metrics = utils.calculate_abcd_metrics(img_rgb, labels, k_clusters)
        
        # Normalize LBP image for visualization
        lbp_vis = cv2.normalize(lbp_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Extract individual clusters
        clusters = []
        for i in range(k_clusters):
            mask = (labels == i).astype(np.uint8) * 255
            cluster_isolated = cv2.bitwise_and(blurred_img, blurred_img, mask=mask)
            clusters.append(img_to_base64(cluster_isolated))
            
        return JSONResponse(content={
            "original": img_to_base64(img_rgb),
            "enhanced": img_to_base64(enhanced_img),
            "blurred": img_to_base64(blurred_img),
            "edges": single_channel_to_base64(edges_img),
            "segmented": img_to_base64(segmented_img),
            "clusters": clusters,
            "lbp_image": single_channel_to_base64(lbp_vis),
            "lbp_hist": lbp_hist.tolist(),
            "dimensions": f"{img_rgb.shape[1]}x{img_rgb.shape[0]} pixels",
            "abcd_metrics": abcd_metrics
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
