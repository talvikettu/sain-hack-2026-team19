import { mkdir, copyFile } from 'node:fs/promises';
const root = new URL('../', import.meta.url);
await mkdir(new URL('dist/', root), { recursive: true });
for (const file of ['index.html', 'styles.css', 'app.js', 'api.js']) {
  await copyFile(new URL(`src/${file}`, root), new URL(`dist/${file}`, root));
}
console.log('Frontend built: web/dist');
