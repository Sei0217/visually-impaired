// api/detect.js — Axios proxy with health preflight & long timeout
const { IncomingForm } = require("formidable");
const fs = require("fs");
const FormData = require("form-data");
const axios = require("axios");

// Wake Render free instance by polling /health
async function waitForHealth(base, totalMs = 120000) {
  const healthURL = `${base.replace(/\/+$/, "")}/health`;
  const t0 = Date.now();
  while (Date.now() - t0 < totalMs) {
    try {
      const r = await axios.get(healthURL, { timeout: 5000, validateStatus: () => true });
      if (r.status === 200) return true;
    } catch {}
    await new Promise(r => setTimeout(r, 1200));
  }
  return false;
}

module.exports = async (req, res) => {
  if (req.method !== "POST") return res.status(405).json({ error: "Only POST allowed" });

  const form = new IncomingForm({
    keepExtensions: true,
    uploadDir: "/tmp",
    maxFileSize: 25 * 1024 * 1024,
    multiples: true,
  });

  form.parse(req, async (err, fields, files) => {
    try {
      if (err) return res.status(400).json({ error: "Bad multipart" });

      // pick first file (handles arrays/any key)
      const pickFirstFile = (obj) => {
        if (!obj) return null;
        const pref = ["image", "file", "photo", "picture", "img", "upload"];
        for (const k of pref) {
          const v = obj[k];
          if (!v) continue;
          return Array.isArray(v) ? v[0] : v;
        }
        for (const v of Object.values(obj)) return Array.isArray(v) ? v[0] : v;
      };
      const f = pickFirstFile(files);
      if (!f?.filepath) return res.status(400).json({ error: "No image file" });

      const base = (process.env.ML_SERVER_URL || "https://visually-impaired.onrender.com").trim();

      // wake Render before sending large multipart
      const healthy = await waitForHealth(base, 120000);
      if (!healthy) return res.status(503).json({ error: "ML server unavailable (health check failed)" });

      const fd = new FormData();
      fd.append("image", fs.createReadStream(f.filepath), f.originalFilename || "upload.jpg");

      const qs = new URLSearchParams({ return_image: "true" });
      if (fields?.conf) qs.set("conf", String(fields.conf));
      if (fields?.imgsz) qs.set("imgsz", String(fields.imgsz));

      const url = `${base.replace(/\/+$/, "")}/api/detect?${qs.toString()}`;

      const upstream = await axios.post(url, fd, {
        headers: fd.getHeaders(),
        timeout: 120000, // up to 2 minutes for free spin-up
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        validateStatus: () => true,
      });

      return res.status(upstream.status).send(upstream.data);
    } catch (e) {
      const code = e?.response?.status || 500;
      const data = e?.response?.data || { error: e.message || "Proxy error" };
      console.error("[detect] proxy error:", code, e?.message);
      return res.status(code >= 400 ? code : 500).json(typeof data === "object" ? data : { error: String(data) });
    } finally {
      try {
        const arr = files ? Object.values(files).flat() : [];
        for (const it of arr) if (it?.filepath) fs.unlinkSync(it.filepath);
      } catch {}
    }
  });
};
