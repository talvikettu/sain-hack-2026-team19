import { test } from 'node:test';
import assert from 'node:assert/strict';
import { api, request } from '../src/api.js';

test('chat preserves expert and session and sends JSON to backend', async () => {
  const original = globalThis.fetch;
  const payload = { expert_id: 'clinician', message: 'Вопрос', session_id: 'session-1' };
  globalThis.fetch = async (path, options) => {
    assert.equal(path, '/chat');
    assert.equal(options.method, 'POST');
    assert.equal(options.headers['Content-Type'], 'application/json');
    assert.deepEqual(JSON.parse(options.body), payload);
    return Response.json({ session_id: 'session-1', text: 'Ответ', citations: [] });
  };
  try { assert.equal((await api.chat(payload)).text, 'Ответ'); }
  finally { globalThis.fetch = original; }
});
test('source IDs are encoded rather than interpreted as URL paths', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async path => { assert.equal(path, '/source/art%2F1/s%3F2'); return Response.json({ text: 'Источник' }); };
  try { await api.source('art/1', 's?2'); } finally { globalThis.fetch = original; }
});
test('server, non-JSON, network and timeout failures become readable errors', async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => Response.json({ detail: 'Dataset is not loaded yet' }, { status: 503 });
    await assert.rejects(request('/experts'), /Данные ещё не загружены/);
    globalThis.fetch = async () => new Response('<html>Error</html>');
    await assert.rejects(request('/chat'), /некорректный ответ/);
    globalThis.fetch = async () => { throw new TypeError('Failed to fetch'); };
    await assert.rejects(request('/chat'), /Нет соединения/);
    globalThis.fetch = async (_path, { signal }) => new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError'))));
    await assert.rejects(request('/chat', { method: 'POST', timeout: 5 }), /Время ожидания истекло/);
    await assert.rejects(request('/experts', { timeout: 5 }), /Сервер не ответил вовремя/);
    globalThis.fetch = async () => Response.json({ detail: 'Upstream connection refused' }, { status: 502 });
    await assert.rejects(request('/experts'), /Нет подключения к backend/);
  } finally { globalThis.fetch = original; }
});
