// super simple test
module.exports = (req, res) => {
  res.status(200).json({ ok: true, when: new Date().toISOString() });
};
