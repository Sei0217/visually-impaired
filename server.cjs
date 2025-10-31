// server.cjs
const path = require('path');
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = process.env.PORT || 3000;

// Targets
const API_TARGET = process.env.API_TARGET || 'http://127.0.0.1:8000'; // FastAPI
const PI_HOST   = process.env.PI_HOST   || '192.168.100.50';          // Raspberry Pi
const PI_TARGET = `http://${PI_HOST}:3001`;

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve /public
app.use(express.static(path.join(__dirname, 'public')));

// (optional) uploads
const upload = multer({ dest: path.join(__dirname, 'uploads') });

// ---- Proxy to FastAPI (fixes Cannot POST /api/detect) ----
app.use('/api', createProxyMiddleware({
  target: API_TARGET,
  changeOrigin: true,              // <-- correct spelling
  // no pathRewrite needed; FastAPI already exposes /api/detect
}));

// ---- Proxy to Pi camera (your existing one, with typo fixed) ----
app.use('/pi', createProxyMiddleware({
  target: PI_TARGET,
  changeOrigin: true,              // <-- fixed
  pathRewrite: { '^/pi': '' }      // /pi/video -> /video
}));

// Simple health for Node server
app.get('/health', (_req, res) => res.json({ ok: true }));

app.listen(PORT, () => {
  console.log(`✅ Web server: http://localhost:${PORT}`);
  console.log(`↪ Proxy /api -> ${API_TARGET}`);
  console.log(`↪ Proxy /pi  -> ${PI_TARGET}`);
});
