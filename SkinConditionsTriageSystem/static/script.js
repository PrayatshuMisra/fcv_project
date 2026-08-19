const fileInput = document.getElementById('file-input');
const uploadZone = document.getElementById('upload-zone');
const loader = document.getElementById('loader');
const results = document.getElementById('results');
const form = document.getElementById('pipeline-form');

let lbpChartInstance = null;

// Handle file selection
fileInput.addEventListener('change', (e) => {
    if(e.target.files.length > 0) {
        processImage(e.target.files[0]);
    }
});

// Drag and drop handlers (Updated to match Clinical Light Theme)
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--accent-blue)';
    uploadZone.style.backgroundColor = 'rgba(2, 132, 199, 0.04)'; 
});
uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--accent-blue)'; 
    uploadZone.style.backgroundColor = 'transparent';
});
uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--accent-blue)';
    uploadZone.style.backgroundColor = 'transparent';
    if(e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        processImage(e.dataTransfer.files[0]);
    }
});

// Trigger re-process when parameters change (if an image is loaded)
form.addEventListener('change', () => {
    if(fileInput.files.length > 0) {
        processImage(fileInput.files[0]);
    }
});

async function processImage(file) {
    // Show loader, hide results
    uploadZone.style.display = 'none';
    loader.classList.remove('hidden');
    results.classList.add('hidden');

    const formData = new FormData(form);
    formData.append('file', file);

    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });

        if(!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }

        const data = await response.json();
        
        if(data.error) {
            throw new Error(data.error);
        }

        // Update Images (Base64)
        document.getElementById('img-original').src = `data:image/jpeg;base64,${data.original}`;
        document.getElementById('img-dim').innerText = data.dimensions;
        
        document.getElementById('img-enhanced').src = `data:image/jpeg;base64,${data.enhanced}`;
        document.getElementById('img-blurred').src = `data:image/jpeg;base64,${data.blurred}`;
        document.getElementById('img-edges').src = `data:image/jpeg;base64,${data.edges}`;
        document.getElementById('img-segmented').src = `data:image/jpeg;base64,${data.segmented}`;
        document.getElementById('img-lbp').src = `data:image/jpeg;base64,${data.lbp_image}`;

        // Update Clusters
        const clusterContainer = document.getElementById('cluster-container');
        clusterContainer.innerHTML = '';
        data.clusters.forEach((clusterB64, idx) => {
            clusterContainer.innerHTML += `
                <div class="image-card">
                    <img src="data:image/jpeg;base64,${clusterB64}" alt="Cluster ${idx+1}">
                    <div class="card-footer">Region ${idx+1}</div>
                </div>
            `;
        });

        // Update ABCD Metrics
        if (data.abcd_metrics && !data.abcd_metrics.error) {
            document.getElementById('val-asymmetry').innerText = data.abcd_metrics.asymmetry;
            document.getElementById('val-border').innerText = data.abcd_metrics.border_irregularity;
            document.getElementById('val-color').innerText = data.abcd_metrics.color_variance;
            
            const riskBanner = document.getElementById('risk-banner');
            const riskLevel = document.getElementById('risk-level');
            
            riskBanner.className = 'risk-banner'; // reset
            if (data.abcd_metrics.risk_score >= 2) {
                riskBanner.classList.add('risk-high');
            } else if (data.abcd_metrics.risk_score === 1) {
                riskBanner.classList.add('risk-moderate');
            } else {
                riskBanner.classList.add('risk-low');
            }
            riskLevel.innerText = data.abcd_metrics.risk_level;
        } else {
            document.getElementById('val-asymmetry').innerText = 'N/A';
            document.getElementById('val-border').innerText = 'N/A';
            document.getElementById('val-color').innerText = 'N/A';
            document.getElementById('risk-banner').className = 'risk-banner';
            document.getElementById('risk-level').innerText = 'Lesion isolation failed. Adjust parameters.';
        }

        // Update Chart
        renderChart(data.lbp_hist);

        // Hide loader, show results
        loader.classList.add('hidden');
        results.classList.remove('hidden');
        uploadZone.style.display = 'block';

    } catch (error) {
        alert(error.message);
        loader.classList.add('hidden');
        uploadZone.style.display = 'block';
    }
}

function renderChart(histData) {
    const ctx = document.getElementById('lbpChart').getContext('2d');
    
    if(lbpChartInstance) {
        lbpChartInstance.destroy();
    }

    const labels = Array.from({length: histData.length}, (_, i) => i.toString());

    lbpChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Normalized Texture Frequency',
                data: histData,
                backgroundColor: '#0284c7', // Updated to match the new Sky 600 clinical blue
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: 'LBP Micro-Texture Signature',
                    color: '#0f172a', // Updated to Slate 900 for Light Mode readability
                    font: { family: 'Inter', size: 14, weight: '600' }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#64748b' } // Slate 500
                },
                y: {
                    grid: { color: '#e2e8f0' }, // Slate 200 grid lines
                    ticks: { color: '#64748b' } // Slate 500
                }
            }
        }
    });
}