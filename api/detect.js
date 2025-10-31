// api/detect.js — 307 redirect to the ML server (keeps POST + multipart body)

module.exports = (req, res) => {
  // (optional) allow OPTIONS so forms/fetch preflight won't complain
  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "POST,OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    return res.end();
  }

  if (req.method !== "POST") {
    res.statusCode = 405;
    res.setHeader("Content-Type", "application/json");
    return res.end(JSON.stringify({ error: "Only POST allowed" }));
  }

  const base = (process.env.ML_SERVER_URL || "https://visually-impaired.onrender.com").replace(/\/+$/, "");

  // keep any incoming query string (e.g., from your UI), then set fast defaults
  const qidx = req.url.indexOf("?");
  const search = new URLSearchParams(qidx >= 0 ? req.url.slice(qidx + 1) : "");
  if (!search.has("imgsz")) search.set("imgsz", "320");      // faster default
  if (!search.has("conf"))  search.set("conf", "0.25");
  search.set("return_image", "true");

  const target = `${base}/api/detect?${search.toString()}`;

  // 307 preserves method + body; browser will POST directly to Render
  res.statusCode = 307;
  res.setHeader("Location", target);
  return res.end();
};
