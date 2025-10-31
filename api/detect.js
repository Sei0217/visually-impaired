// api/detect.js  (CommonJS, robust file pick)
const { IncomingForm } = require("formidable");
const fs = require("fs");
const FormData = require("form-data");

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Only POST allowed" });
    return;
  }

  const form = new IncomingForm({
    keepExtensions: true,
    uploadDir: "/tmp",
    maxFileSize: 15 * 1024 * 1024,
    multiples: true, // allow arrays; we’ll just pick the first
  });

  // helper to pick first valid file (handles arrays and any key)
  const pickFirstFile = (files) => {
    if (!files) return null;
    const preferred = ["image", "file", "photo", "picture", "img", "upload"];
    // 1) try preferred keys
    for (const k of preferred) {
      const v = files[k];
      if (!v) continue;
      if (Array.isArray(v)) {
        if (v[0]?.filepath) return v[0];
      } else if (v?.filepath) {
        return v;
      }
    }
    // 2) try any key
    for (const v of Object.values(files)) {
      if (Array.isArray(v)) {
        if (v[0]?.filepath) return v[0];
      } else if (v?.filepath) {
        return v;
      }
    }
    return null;
  };

  form.parse(req, async (err, fields, files) => {
    try {
      if (err) throw err;

      const f = pickFirstFile(files);

      // Helpful log to Vercel -> Deployments -> Functions -> api/detect
      try { console.log("detect files keys:", Object.keys(files || {})); } catch {}

      if (!f?.filepath) {
        return res.status(400).json({
          error: "No image file",
          receivedKeys: Object.keys(files || {}),
        });
      }

      const fd = new FormData();
      fd.append("image", fs.createReadStream(f.filepath), f.originalFilename || "upload.jpg");

      const qs = new URLSearchParams({ return_image: "true" });
      if (fields?.conf) qs.set("conf", String(fields.conf));
      if (fields?.imgsz) qs.set("imgsz", String(fields.imgsz));

      const base = process.env.ML_SERVER_URL || "https://visually-impaired.onrender.com";
      const url = `${base}/api/detect?${qs.toString()}`;

      const upstream = await fetch(url, { method: "POST", body: fd, headers: fd.getHeaders() });
      const text = await upstream.text();
      let data; try { data = JSON.parse(text); } catch { data = { raw: text }; }

      return res.status(upstream.status).json(data);
    } catch (e) {
      console.error("detect proxy error:", e);
      return res.status(500).json({ error: "Failed to reach ML server" });
    } finally {
      // cleanup temp files if any
      try {
        for (const v of Object.values(files || {})) {
          const arr = Array.isArray(v) ? v : [v];
          for (const it of arr) if (it?.filepath) fs.unlinkSync(it.filepath);
        }
      } catch {}
    }
  });
};
