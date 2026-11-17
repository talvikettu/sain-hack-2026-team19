import http from 'node:http';
import { readFile } from 'node:fs/promises';
const backend = new URL(process.env.BACKEND_URL || 'http://127.0.0.1:8000');
const files = { '/': ['index.html', 'text/html'], '/styles.css': ['styles.css', 'text/css'], '/app.js': ['app.js', 'text/javascript'], '/api.js': ['api.js', 'text/javascript'] };
http.createServer(async (req, res) => {
  const path = new URL(req.url, 'http://localhost').pathname;
  if (/^\/(experts|chat|health|source)(\/|$)/.test(path)) {
    const upstream = http.request(new URL(req.url, backend), { method: req.method, headers: { ...req.headers, host: backend.host } }, response => {
      res.writeHead(response.statusCode, response.headers);
      response.pipe(res);
    });
    upstream.setTimeout(310000, () => upstream.destroy());
    upstream.on('error', () => { if (!res.headersSent) res.writeHead(502, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ detail: 'Backend недоступен. Запустите сервер на порту 8000.' })); });
    req.pipe(upstream);
    return;
  }
  if (!files[path]) { res.writeHead(404); res.end(); return; }
  try {
    const [name, type] = files[path];
    const body = await readFile(new URL(`../src/${name}`, import.meta.url));
    res.writeHead(200, { 'Content-Type': `${type}; charset=utf-8`, 'Cache-Control': 'no-store' });
    res.end(body);
  } catch { res.writeHead(500); res.end('Не удалось прочитать файл интерфейса'); }
}).listen(5173, '127.0.0.1', () => console.log('Local: http://127.0.0.1:5173'));
