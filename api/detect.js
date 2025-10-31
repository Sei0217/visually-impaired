// api/detect.cjs
const { IncomingForm } = require("formidable");
const fs = require("fs");
const FormData = require("form-data");

// Vercel serverless config: disable default body parser
module.exports.config = { api: { bodyParser: false } };

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Only POST allowed" });
    return;
  }

  const form = new IncomingForm({
    keepExtensions: true,
    uploadDir: "/tmp",
    maxFileSize: 15 * 1024 * 1024, // 15MB
    multiples: false,
  });

  form.parse(req, async (err, fields, files) => {
    try {
      if (err) throw err;

      const f = files.image || files.file || (files && Object.values(files)[0]);
      if (!f || !f.filepath) {
        res.status(400).json({ error: "No image file" });
        return;
      }

      // Build multipart for the upstream ML server
      const fd = new FormData();
      fd.append("image", fs.createReadStream(f.filepath), f.originalFilename || "upload.jpg");

      // Optional query params
      const qs = new URLSearchParams({ return_image: "true" });
      if (fields.conf) qs.set("conf", String(fields.conf));
      if (fields.imgsz) qs.set("imgsz", String(fields.imgsz));

      const base = process.env.ML_SERVER_URL || "https://visually-impaired.onrender.com";
      const url = `${base}/api/detect?${qs.toString()}`;

      // Node 18 has global fetch
      const upstream = await fetch(url, {
        method: "POST",
        body: fd,
        headers: fd.getHeaders(),
      });

      const text = await upstream.text();
      let data;
      try { data = JSON.parse(text); } catch { data = { raw: text }; }

      res.status(upstream.status).json(data);
    } catch (e) {
      console.error("detect proxy error:", e);
      res.status(500).json({ error: "Failed to reach ML server" });
    } finally {
      // cleanup temp file(s)
      try { if (files?.image?.filepath) fs.unlinkSync(files.image.filepath); } catch {}
      try { if (files?.file?.filepath) fs.unlinkSync(files.file.filepath); } catch {}
      try {
        const any = files && Object.values(files)[0];
        if (any?.filepath) fs.unlinkSync(any.filepath);
      } catch {}
    }
  });
};
