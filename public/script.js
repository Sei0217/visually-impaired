// public/script.js (with debug logging)

console.log('🔍 script.js loaded');

// Grab existing elements by ID
const fileInput = document.getElementById('imageInput');
const analyzeBtn = document.getElementById('analyzeImageBtn');
const videoEl = document.getElementById('videoPreview') || document.getElementById('video');
const imageEl = document.getElementById('imagePreview');
const startCamBtn = document.getElementById('startCameraBtn');
const captureBtn = document.getElementById('captureBtn');
const resultImg = document.getElementById('result') || document.getElementById('annotatedImage');
const resultList = document.getElementById('resultList') || document.getElementById('resultsSection');

console.log('📋 Elements found:', {
  fileInput: !!fileInput,
  analyzeBtn: !!analyzeBtn,
  videoEl: !!videoEl,
  imageEl: !!imageEl,
  startCamBtn: !!startCamBtn,
  captureBtn: !!captureBtn,
  resultImg: !!resultImg,
  resultList: !!resultList
});

const hiddenCanvas = document.createElement('canvas');

// ---------- helpers ----------
async function postToDetectFromBlob(blob) {
  console.log('📤 Posting to /api/detect, blob size:', blob.size);
  const fd = new FormData();
  fd.append('image', blob, 'frame.jpg');
  const res = await fetch('/api/detect', { method: 'POST', body: fd });

  const text = await res.text();
  console.log('📥 Response status:', res.status);
  
  if (!res.ok) throw new Error(`HTTP ${res.status} ${text}`);

  let data;
  try { data = JSON.parse(text); } catch { throw new Error(text); }

  if (data.error) throw new Error(data.error);
  console.log('✅ Detection results:', data.detections?.length || 0, 'objects found');
  renderResults(data);
  return data;
}

function renderResults(data) {
  console.log('🎨 Rendering results...');
  if (resultImg && data.image) {
    resultImg.src = `data:image/jpeg;base64,${data.image}`;
    resultImg.style.display = 'block';
    console.log('✅ Annotated image displayed');
  }
  if (resultList) {
    const dets = data.detections || [];
    resultList.innerHTML = dets.length
      ? dets.map(d => `<li>${d.label} — ${(d.conf * 100).toFixed(1)}%</li>`).join('')
      : '<li>No detections</li>';
    resultList.parentElement.style.display = 'block';
    console.log('✅ Results list updated');
  }
}

async function dataURLtoBlob(dataURL) {
  const r = await fetch(dataURL);
  return await r.blob();
}

// ---------- Upload flow ----------
// Upload button click handler
const uploadBtn = document.getElementById('uploadBtn');
let activeStream = null; // Track active camera stream

if (uploadBtn && fileInput) {
  console.log('📤 Upload button listener attached');
  uploadBtn.addEventListener('click', () => {
    console.log('🖱️ Upload button clicked!');
    // Stop camera when switching to upload
    stopCamera();
    fileInput.click();
  });
}

// Helper function to stop camera
function stopCamera() {
  if (activeStream) {
    console.log('🛑 Stopping camera stream');
    activeStream.getTracks().forEach(track => track.stop());
    activeStream = null;
  }
  if (videoEl) {
    videoEl.srcObject = null;
    videoEl.style.display = 'none';
  }
  if (captureBtn) {
    captureBtn.style.display = 'none';
  }
}

if (fileInput) {
  console.log('📁 File input listener attached');
  fileInput.addEventListener('change', async () => {
    const f = fileInput.files?.[0];
    if (!f) return;
    console.log('📂 File selected:', f.name, f.size, 'bytes');
    // show preview
    if (imageEl) {
      imageEl.src = URL.createObjectURL(f);
      imageEl.style.display = 'block';
    }
    // hide video if running
    if (videoEl) videoEl.style.display = 'none';
    // hide placeholder
    const placeholder = document.getElementById('previewPlaceholder');
    if (placeholder) placeholder.style.display = 'none';
    // enable analyze button
    if (analyzeBtn) analyzeBtn.disabled = false;
    
    try { await postToDetectFromBlob(f); }
    catch (e) { 
      console.error('❌ Analysis failed:', e);
      alert('Failed to analyze image.\n' + (e.message || e)); 
    }
  });
}

// ---------- Camera start ----------
async function startCamera() {
  console.log('📷 Starting camera...');
  // Stop any existing stream first
  stopCamera();
  
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    activeStream = stream; // Store the stream so we can stop it later
    console.log('✅ Camera stream obtained');
    if (videoEl) {
      videoEl.srcObject = stream;
      console.log('✅ Video element srcObject set');
      // Make sure video is visible
      videoEl.style.display = 'block';
      console.log('✅ Video display set to block');
      // Hide image preview
      if (imageEl) imageEl.style.display = 'none';
      if (resultImg) resultImg.style.display = 'none';
      // Show capture button
      if (captureBtn) {
        captureBtn.style.display = 'inline-flex';
        console.log('✅ Capture button shown');
      }
      // Hide placeholder
      const placeholder = document.getElementById('previewPlaceholder');
      if (placeholder) {
        placeholder.style.display = 'none';
        console.log('✅ Placeholder hidden');
      }
    }
  } catch (e) {
    console.error('❌ Camera error:', e);
    alert('Cannot access camera.\n' + (e.message || e));
  }
}

if (startCamBtn) {
  console.log('🎥 Camera button listener attached');
  startCamBtn.addEventListener('click', () => {
    console.log('🖱️ Camera button clicked!');
    startCamera();
  });
} else {
  console.warn('⚠️ startCameraBtn not found');
}

if (videoEl && !startCamBtn) {
  console.log('🎥 Auto-starting camera (no button found)');
  startCamera();
}

// ---------- Capture from camera ----------
if (captureBtn && videoEl) {
  console.log('📸 Capture button listener attached');
  captureBtn.addEventListener('click', async () => {
    console.log('🖱️ Capture button clicked!');
    if (!videoEl.videoWidth) {
      console.warn('⚠️ Video not ready, width:', videoEl.videoWidth);
      return alert('Camera not ready yet.');
    }
    console.log('📸 Capturing frame, size:', videoEl.videoWidth, 'x', videoEl.videoHeight);
    hiddenCanvas.width = videoEl.videoWidth;
    hiddenCanvas.height = videoEl.videoHeight;
    hiddenCanvas.getContext('2d').drawImage(videoEl, 0, 0);
    const blob = await new Promise(r => hiddenCanvas.toBlob(r, 'image/jpeg', 0.92));
    console.log('✅ Frame captured, blob size:', blob.size);
    
    // Show preview
    if (imageEl) {
      imageEl.src = hiddenCanvas.toDataURL('image/jpeg', 0.92);
      imageEl.style.display = 'block';
    }
    
    // Stop camera after capture
    stopCamera();
    
    // Show start camera button again
    if (startCamBtn) startCamBtn.style.display = 'inline-flex';
    
    try { await postToDetectFromBlob(blob); }
    catch (e) { 
      console.error('❌ Analysis failed:', e);
      alert('Failed to analyze image.\n' + (e.message || e)); 
    }
  });
}

// ---------- Explicit Analyze button ----------
if (analyzeBtn) {
  console.log('🔍 Analyze button listener attached');
  analyzeBtn.addEventListener('click', async () => {
    console.log('🖱️ Analyze button clicked!');
    try {
      // 1) prefer file input
      if (fileInput?.files?.[0]) {
        console.log('📂 Using file input');
        await postToDetectFromBlob(fileInput.files[0]);
        return;
      }
      // 2) else use preview image (dataURL) if available
      if (imageEl?.src?.startsWith('data:')) {
        console.log('🖼️ Using preview image data URL');
        const blob = await dataURLtoBlob(imageEl.src);
        await postToDetectFromBlob(blob);
        return;
      }
      // 3) else capture current video frame
      if (videoEl?.videoWidth) {
        console.log('📹 Capturing current video frame');
        hiddenCanvas.width = videoEl.videoWidth;
        hiddenCanvas.height = videoEl.videoHeight;
        hiddenCanvas.getContext('2d').drawImage(videoEl, 0, 0);
        const blob = await new Promise(r => hiddenCanvas.toBlob(r, 'image/jpeg', 0.92));
        await postToDetectFromBlob(blob);
        return;
      }
      console.warn('⚠️ No image source available');
      alert('No image to analyze. Upload a photo or start the camera.');
    } catch (e) {
      console.error('❌ Analysis failed:', e);
      alert('Failed to analyze image.\n' + (e.message || e));
    }
  });
}

console.log('✅ script.js initialization complete');