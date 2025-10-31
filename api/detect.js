// api/detect.js
import { IncomingForm } from "formidable";
import fs from "fs";
import FormData from "form-data";

export const config = { api: { bodyParser: false } };

export default function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Only POST allowed" });

  const form = new IncomingForm({ keepExtensions: true, uploadDir: "/tmp", maxFileSize: 15 * 1024 * 1024 });

  form.parse(req, async (err, fields, files) => {
    try {
      if (err) throw err;
      const f = files.image || files.file || Object.values(files)[0];
      if (!f?.filepath) return res.status(400).json({ error: "No image file" });

      const fd = new FormData();
      fd.append("image", fs.createReadStream(f.filepath), f.originalFilename || "upload.jpg");
      const qs = new URLSearchParams({ return_image: "true" });
      if (fields.conf) qs.set("conf", fields.conf);
      if (fields.imgsz) qs.set("imgsz", fields.imgsz);

      const url = `${process.env.ML_SERVER_URL}/api/detect?${qs.toString()}`;
      const r = await fetch(url, { method: "POST", body: fd, headers: fd.getHeaders() });
      const text = await r.text();
      let data; try { data = JSON.parse(text); } catch { data = { raw: text }; }
      return res.status(r.status).json(data);
    } catch (e) {
      console.error("detect proxy error:", e);
      return res.status(500).json({ error: "Failed to reach ML server" });
    } finally {
      try { if (files?.image?.filepath) fs.unlinkSync(files.image.filepath); } catch {}
      try { if (files?.file?.filepath) fs.unlinkSync(files.file.filepath); } catch {}
    }
  });
}
