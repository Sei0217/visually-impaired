// api/detect.js  — Axios-based proxy (fixes multipart boundary issue)
const { IncomingForm } = require("formidable");
const fs = require("fs");
const FormData = require("form-data");
const axios = require("axios");

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Only POST allowed" });
  }

  const form = new IncomingForm({
    keepExtensions: true,
    uploadDir: "/tmp",
    maxFileSize: 25 * 1024 * 1024, // 25MB
    multiples: true,
  });

  form.parse(req, async (err, fields, files) => {
    try {
      if (err) {
        console.error("[detect] parse error:", err);
        return res.status(400).json({ error: "Bad multipart" });
      }

      // robust picker (accepts arrays / any key)
      const pickFirstFile = (obj) => {
        if (!obj) return null;
        const preferred = ["image", "file", "photo", "picture", "img", "upload"];
        for (const k of preferred) {
          const v = obj[k];
          if (!v) continue;
          if (Array.isArray(v)) return v[0];
          return v;
        }
        for (const v of Object.values(obj)) {
          if (Array.isArray(v)) return v[0];
          if (v) return v;
        }
        return null;
      };

      const f = pickFirstFile(files);
      if (!f?.filepath) {
        console.warn("[detect] no file. keys:", Object.keys(files || {}));
        return res.status(400).json({ error: "No image file" });
      }

      const fd = new FormData();
      fd.append("image", fs.createReadStream(f.filepath), f.originalFilename || "upload.jpg");

      const qs = new URLSearchParams({ return_image: "true" });
      if (fields?.conf) qs.set("conf", String(fields.conf));
      if (fields?.imgsz) qs.set("imgsz", String(fields.imgsz));

      const base = process.env.ML_SERVER_URL || "https://visually-impaired.onrender.com";
      const url = `${base}/api/detect?${qs.toString()}`;

      // Axios handles streaming multipart perfectly; set size limits + timeout
      const upstream = await axios.post(url, fd, {
        headers: fd.getHeaders(),
        timeout: 45000,
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        validateStatus: () => true, // we'll forward status as-is
      });

      return res.status(upstream.status).json(upstream.data);
    } catch (e) {
      console.error("[detect] proxy error:", e?.response?.status, e?.message);
      return res.status(500).json({ error: "Failed to reach ML server" });
    } finally {
      // cleanup temp files
      try {
        const arr = files ? Object.values(files).flat() : [];
        for (const it of arr) if (it?.filepath) fs.unlinkSync(it.filepath);
      } catch {}
    }
  });
};
