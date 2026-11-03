/**
 * ClinicalVision PRO — Dermoscopy Visual Triage Workstation Controller
 * Engineered for CSE 3176: Foundations of Computer Vision Lab
 * Department of Computer Science & Engineering, MIT Manipal
 */

const fileInput = document.getElementById('file-input');
const uploadZone = document.getElementById('upload-zone');
const loader = document.getElementById('loader');
const results = document.getElementById('results');
const form = document.getElementById('pipeline-form');
const exportBtn = document.getElementById('export-report-btn');
const systemStatus = document.getElementById('system-status');

let lbpChartInstance = null;
let currentProcessedData = null;
let activeCaseId = null;

// Dynamic Gamma control visibility
const enhancementSelect = document.getElementById('enhancement_type');
const gammaGroup = document.getElementById('gamma-group');
if (enhancementSelect && gammaGroup) {
    enhancementSelect.addEventListener('change', () => {
        gammaGroup.style.display = (enhancementSelect.value === 'Gamma Correction') ? 'block' : 'none';
    });
}

// Mobile Sidebar Toggle
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (sidebar) {
        sidebar.classList.toggle('open');
    }
}

// Reset Pipeline Parameters to Standard Clinical Defaults
function resetPipelineDefaults() {
    form.reset();
    document.getElementById('enhancement_type').value = 'CLAHE';
    if (gammaGroup) gammaGroup.style.display = 'none';
    document.getElementById('gamma_val').value = '1.0';
    document.getElementById('gamma-val').innerText = '1.0';
    document.getElementById('apply_hair_removal').checked = false;
    document.getElementById('filter_type').value = 'Bilateral';
    document.getElementById('blur_kernel').value = '5';
    document.getElementById('blur-val').innerText = '5';
    document.getElementById('edge_operator').value = 'Canny';
    document.getElementById('t1').value = '100';
    document.getElementById('t1-val').innerText = '100';
    document.getElementById('t2').value = '200';
    document.getElementById('t2-val').innerText = '200';
    document.getElementById('k_clusters').value = '3';
    document.getElementById('k-val').innerText = '3';
    document.getElementById('lbp_radius').value = '3';
    document.getElementById('r-val').innerText = '3';
    document.getElementById('lbp_points').value = '24';
    document.getElementById('p-val').innerText = '24';

    // If a case or image is loaded, re-evaluate
    triggerReanalysis();
}

// Input Method Tab Switching
function switchTab(tabId) {
    document.querySelectorAll('.tab-pane').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.tab-button').forEach(el => el.classList.remove('active'));
    
    const target = document.getElementById('tab-' + tabId);
    if (target) target.style.display = 'block';
    
    if (window.event && window.event.currentTarget) {
        window.event.currentTarget.classList.add('active');
    }
}

// Global Clipboard Paste Handler (Ctrl+V)
window.addEventListener('paste', e => {
    if (e.clipboardData && e.clipboardData.files.length > 0) {
        const file = e.clipboardData.files[0];
        if (file.type.startsWith('image/')) {
            clearActiveCaseSelection();
            processImage(file);
        }
    }
});

// File Selection Listener
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        clearActiveCaseSelection();
        processImage(e.target.files[0]);
    }
});

// Drag and drop events
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--clinical-blue)';
    uploadZone.style.backgroundColor = 'rgba(2, 132, 199, 0.05)';
});

uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--border-strong)';
    uploadZone.style.backgroundColor = 'var(--surface-canvas)';
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--border-strong)';
    uploadZone.style.backgroundColor = 'var(--surface-canvas)';
    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        clearActiveCaseSelection();
        processImage(e.dataTransfer.files[0]);
    }
});

// Re-process when parameters change
form.addEventListener('change', () => {
    triggerReanalysis();
});

function triggerReanalysis() {
    if (activeCaseId) {
        loadSampleCase(activeCaseId);
    } else if (fileInput.files.length > 0) {
        processImage(fileInput.files[0]);
    } else if (document.getElementById('url-input') && document.getElementById('url-input').value) {
        processUrl();
    }
}

function clearActiveCaseSelection() {
    activeCaseId = null;
    document.querySelectorAll('.pacs-case-card').forEach(c => c.classList.remove('active-case'));
}

// 1-Click Clinical Reference Case Loader
async function loadSampleCase(caseId) {
    activeCaseId = caseId;
    document.querySelectorAll('.pacs-case-card').forEach(c => c.classList.remove('active-case'));
    const targetCard = document.getElementById('card-' + caseId);
    if (targetCard) targetCard.classList.add('active-case');

    const formData = new FormData(form);
    formData.append('image_url', `/static/samples/${caseId}.jpg`);
    submitFormData(formData);
}

// URL Processing
function processUrl() {
    const url = document.getElementById('url-input').value.trim();
    if (!url) return;
    clearActiveCaseSelection();
    const formData = new FormData(form);
    formData.append('image_url', url);
    submitFormData(formData);
}

// Camera Capture Handling
let cameraStream = null;
async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        const video = document.getElementById('camera-stream');
        video.srcObject = cameraStream;
        video.style.display = 'block';
        document.getElementById('camera-placeholder').style.display = 'none';
        document.getElementById('start-camera-btn').style.display = 'none';
        document.getElementById('capture-btn').style.display = 'inline-flex';
    } catch (err) {
        alert("Camera sensor access denied or unavailable: " + err.message);
    }
}

function captureCamera() {
    const video = document.getElementById('camera-stream');
    const canvas = document.getElementById('camera-canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    canvas.toBlob(blob => {
        const file = new File([blob], "dermoscopy_capture.jpg", { type: "image/jpeg" });
        clearActiveCaseSelection();
        processImage(file);
        
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        video.style.display = 'none';
        document.getElementById('camera-placeholder').style.display = 'flex';
        document.getElementById('start-camera-btn').style.display = 'inline-flex';
        document.getElementById('capture-btn').style.display = 'none';
    }, 'image/jpeg');
}

function processImage(file) {
    if (!file) return;
    const formData = new FormData(form);
    formData.append('file', file);
    submitFormData(formData);
}

// Master Submission & API Client
async function submitFormData(formData) {
    loader.classList.remove('hidden');
    results.classList.add('hidden');
    if (exportBtn) exportBtn.style.display = 'none';
    
    if (systemStatus) {
        systemStatus.innerHTML = `
            <span class="status-indicator-dot" style="background-color: var(--clinical-blue);"></span>
            <span class="status-text">Processing Frame...</span>
        `;
    }

    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Pipeline Error: ${response.statusText}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        currentProcessedData = data;

        // 1. Image Dimensions & Raw Input
        document.getElementById('img-original').src = `data:image/jpeg;base64,${data.original}`;
        document.getElementById('img-dim').innerText = data.dimensions || '-- px';

        // 2. Hair Removal Section
        const hairSec = document.getElementById('section-hair-removal');
        if (data.hair_mask && data.hair_removed) {
            hairSec.style.display = 'block';
            document.getElementById('img-hair-mask').src = `data:image/jpeg;base64,${data.hair_mask}`;
            document.getElementById('img-hair-inpainted').src = `data:image/jpeg;base64,${data.hair_removed}`;
        } else {
            hairSec.style.display = 'none';
        }

        // 3. Spatial Enhancement (Lab 2)
        document.getElementById('img-enhanced').src = `data:image/jpeg;base64,${data.enhanced}`;
        document.getElementById('enhanced-title').innerText = `Enhanced Image (${data.pipeline_summary.enhancement})`;

        // 4. Clinical Multi-Color Spaces (Lab 1)
        if (data.color_maps) {
            document.getElementById('img-hsv-hue').src = `data:image/jpeg;base64,${data.color_maps.hsv_hue}`;
            document.getElementById('img-hsv-sat').src = `data:image/jpeg;base64,${data.color_maps.hsv_saturation}`;
            document.getElementById('img-lab-l').src = `data:image/jpeg;base64,${data.color_maps.lab_lightness}`;
            document.getElementById('img-lab-a').src = `data:image/jpeg;base64,${data.color_maps.lab_a}`;
            document.getElementById('img-lab-b').src = `data:image/jpeg;base64,${data.color_maps.lab_b}`;
        }

        // 5. Spatial Filtering & Boundary Edges (Labs 3 & 4)
        document.getElementById('img-blurred').src = `data:image/jpeg;base64,${data.filtered}`;
        document.getElementById('filtered-title').innerText = `Filtered (${data.pipeline_summary.filter})`;
        document.getElementById('img-edges').src = `data:image/jpeg;base64,${data.edges}`;
        document.getElementById('edges-title').innerText = `Boundary Map (${data.pipeline_summary.edge_operator})`;

        // 6. Master Segmentation & Geometric Overlays (Labs 4 & 5)
        document.getElementById('img-segmented').src = `data:image/jpeg;base64,${data.segmented}`;
        document.getElementById('img-overlay').src = `data:image/jpeg;base64,${data.contour_overlay}`;
        document.getElementById('img-overlay-2').src = `data:image/jpeg;base64,${data.contour_overlay}`;

        // 7. Render Isolated Tissue Clusters
        const clusterContainer = document.getElementById('cluster-container');
        clusterContainer.innerHTML = '';
        if (data.clusters) {
            data.clusters.forEach((cl) => {
                const isLesion = cl.is_lesion;
                const badge = isLesion 
                    ? `<span class="footer-tag" style="background:#fee2e2; color:#b91c1c; font-weight:700;">Target Lesion</span>` 
                    : `<span class="footer-tag">Background / Skin</span>`;
                clusterContainer.innerHTML += `
                    <div class="image-frame-card" style="${isLesion ? 'border: 2px solid #ef4444;' : ''}" onclick="openLightboxDirect('data:image/jpeg;base64,${cl.image}', 'Isolated Tissue Cluster ${cl.cluster_index + 1}')">
                        <div class="frame-media">
                            <img src="data:image/jpeg;base64,${cl.image}" alt="Cluster ${cl.cluster_index + 1}">
                        </div>
                        <div class="frame-footer">
                            <span class="footer-title">Cluster ${cl.cluster_index + 1}</span>
                            ${badge}
                        </div>
                    </div>
                `;
            });
        }

        // 8. LBP Texture Map & Histogram
        document.getElementById('img-lbp').src = `data:image/jpeg;base64,${data.lbp_image}`;
        renderLbpChart(data.lbp_hist);

        // 9. GLCM Haralick Second-Order Statistics
        if (data.glcm_metrics) {
            document.getElementById('val-glcm-contrast').innerText = Number(data.glcm_metrics.contrast).toFixed(3);
            document.getElementById('val-glcm-dissimilarity').innerText = Number(data.glcm_metrics.dissimilarity).toFixed(3);
            document.getElementById('val-glcm-homogeneity').innerText = Number(data.glcm_metrics.homogeneity).toFixed(3);
            document.getElementById('val-glcm-energy').innerText = Number(data.glcm_metrics.energy).toFixed(4);
            document.getElementById('val-glcm-correlation').innerText = Number(data.glcm_metrics.correlation).toFixed(3);
        }

        // 10. Stolz ABCD & Total Dermoscopy Score (TDS) HUD
        if (data.abcd_metrics && !data.abcd_metrics.error) {
            const m = data.abcd_metrics;
            document.getElementById('val-asymmetry').innerText = m.asymmetry_score;
            document.getElementById('desc-asymmetry').innerText = m.asymmetry_detail;

            document.getElementById('val-border').innerText = m.border_score;
            document.getElementById('val-compactness').innerText = m.border_compactness;

            document.getElementById('val-color').innerText = m.color_score;
            document.getElementById('colors-list').innerText = m.colors_detected.join(', ');

            document.getElementById('val-diameter').innerText = m.diameter_score;
            document.getElementById('val-diameter-mm').innerText = m.diameter_mm;

            // TDS Value and Clinical Stratification
            const tds = Number(m.tds);
            document.getElementById('tds-val').innerText = tds.toFixed(2);
            document.getElementById('needle-score').innerText = tds.toFixed(2);
            document.getElementById('risk-level').innerText = m.risk_level;
            document.getElementById('clinical-action').innerText = m.clinical_action;

            const badgeElem = document.getElementById('risk-badge');
            badgeElem.className = 'risk-badge-large';
            if (tds < 4.75) {
                badgeElem.classList.add('risk-badge-benign');
                badgeElem.innerText = 'Benign / Routine Surveillance';
            } else if (tds < 5.45) {
                badgeElem.classList.add('risk-badge-suspicious');
                badgeElem.innerText = 'Suspicious / Biopsy Indicated';
            } else {
                badgeElem.classList.add('risk-badge-malignant');
                badgeElem.innerText = 'High Risk / Malignant';
            }

            // Calibrated Needle Marker Position Calculation:
            // 0.0 to 4.75 maps to 0% to 33.3%
            // 4.75 to 5.45 maps to 33.3% to 66.6%
            // 5.45 to 8.90 maps to 66.6% to 100%
            let pct = 0;
            if (tds <= 0) {
                pct = 2;
            } else if (tds < 4.75) {
                pct = (tds / 4.75) * 33.3;
            } else if (tds < 5.45) {
                pct = 33.3 + ((tds - 4.75) / (5.45 - 4.75)) * 33.3;
            } else {
                const span = Math.min(tds - 5.45, 8.90 - 5.45);
                pct = 66.6 + (span / (8.90 - 5.45)) * 33.3;
            }
            pct = Math.max(3, Math.min(97, pct));
            document.getElementById('meter-marker').style.left = `${pct}%`;

        } else {
            document.getElementById('tds-val').innerText = '--';
            document.getElementById('needle-score').innerText = '0.0';
            document.getElementById('risk-level').innerText = 'Lesion Segmentation Inconclusive';
            document.getElementById('clinical-action').innerText = 'Please adjust K-Means clusters or edge thresholds in the sidebar.';
            const badgeElem = document.getElementById('risk-badge');
            badgeElem.className = 'risk-badge-large risk-badge-suspicious';
            badgeElem.innerText = 'Requires Refinement';
            document.getElementById('meter-marker').style.left = '50%';
        }

        // Finalize state
        loader.classList.add('hidden');
        results.classList.remove('hidden');
        if (exportBtn) exportBtn.style.display = 'inline-flex';

        if (systemStatus) {
            systemStatus.innerHTML = `
                <span class="status-indicator-dot" style="background-color: var(--triage-benign);"></span>
                <span class="status-text">Diagnostic Analysis Complete</span>
            `;
        }

        // Smooth scroll to summary HUD
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
        alert(err.message);
        loader.classList.add('hidden');
        if (systemStatus) {
            systemStatus.innerHTML = `
                <span class="status-indicator-dot" style="background-color: var(--triage-malignant);"></span>
                <span class="status-text">Pipeline Fault</span>
            `;
        }
    }
}

// Chart.js Micro-Texture LBP Histogram Renderer
function renderLbpChart(histData) {
    const ctx = document.getElementById('lbpChart').getContext('2d');
    
    if (lbpChartInstance) {
        lbpChartInstance.destroy();
    }

    const labels = Array.from({ length: histData.length }, (_, i) => i.toString());

    lbpChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: histData,
                backgroundColor: '#0284c7',
                hoverBackgroundColor: '#0369a1',
                borderRadius: 3,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#0f172a',
                    titleFont: { family: 'Inter', size: 12, weight: '600' },
                    bodyFont: { family: 'JetBrains Mono', size: 12 },
                    padding: 8,
                    displayColors: false,
                    callbacks: {
                        label: (ctx) => `Frequency: ${ctx.parsed.y.toFixed(5)}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: {
                        font: { family: 'JetBrains Mono', size: 10 },
                        color: '#64748b',
                        maxTicksLimit: 14
                    }
                },
                y: {
                    grid: { color: '#f1f5f9' },
                    ticks: {
                        font: { family: 'JetBrains Mono', size: 10 },
                        color: '#64748b'
                    }
                }
            }
        }
    });
}

// Image Lightbox Modal Inspection
function openLightbox(imgElementId, title) {
    const imgEl = document.getElementById(imgElementId);
    if (!imgEl || !imgEl.src) return;
    openLightboxDirect(imgEl.src, title);
}

function openLightboxDirect(src, title) {
    const modal = document.getElementById('lightbox-modal');
    const modalImg = document.getElementById('lightbox-img');
    const modalTitle = document.getElementById('lightbox-title');
    
    modalImg.src = src;
    modalTitle.innerText = title || 'High-Resolution Diagnostic Inspection';
    modal.style.display = 'flex';
}

function closeLightbox() {
    const modal = document.getElementById('lightbox-modal');
    modal.style.display = 'none';
}

// Keyboard ESC listener for lightbox
window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeLightbox();
    }
});

// Printable Clinical Consultation Dossier Generator
function exportClinicalReport() {
    if (!currentProcessedData) {
        alert("Please run an analysis first before generating the clinical dossier.");
        return;
    }
    window.print();
}

// Auto-run first sample on startup for immediate clinical demonstration
window.addEventListener('DOMContentLoaded', () => {
    // Automatically load the landmark Melanoma ISIC case on load
    loadSampleCase('melanoma');
});


// ==========================================================================
// Interactive Concept & Parameter Manual (Modal Controller)
// ==========================================================================

function openConceptGuide() {
    const modal = document.getElementById('concept-guide-modal');
    if (modal) {
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }
}

function closeConceptGuide() {
    const modal = document.getElementById('concept-guide-modal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = '';
    }
}

function switchGuideTab(categoryId) {
    document.querySelectorAll('.guide-nav-chip').forEach(chip => chip.classList.remove('active'));
    if (window.event && window.event.currentTarget) {
        window.event.currentTarget.classList.add('active');
    }

    const cards = document.querySelectorAll('.guide-card, .guide-intro-callout');
    cards.forEach(card => {
        const cat = card.getAttribute('data-category');
        if (categoryId === 'all' || cat === categoryId) {
            card.style.display = '';
        } else {
            card.style.display = 'none';
        }
    });

    const scrollContainer = document.getElementById('guide-modal-scroll');
    if (scrollContainer) {
        scrollContainer.scrollTop = 0;
    }
}

// Global Escape Key Listener for Modals
window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeConceptGuide();
        closeLightbox();
    }
});
