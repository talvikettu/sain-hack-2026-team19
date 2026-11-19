export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}
export async function request(path, { timeout = 310000, ...options } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  const headers = { Accept: 'application/json', ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers };
  try {
    const response = await fetch(path, { ...options, signal: controller.signal, headers });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      const message = response.status === 401 ? 'Сессия истекла. Обновите страницу и попробуйте снова.' : response.status === 502 ? 'Нет подключения к backend. Запустите сервер на порту 8000 и проверьте подключение снова.' : response.status === 503 ? 'Эксперт пока недоступен. Данные ещё не загружены или сервер занят. Попробуйте позже.' : typeof body?.detail === 'string' ? body.detail : `Ошибка сервера (${response.status}). Попробуйте ещё раз.`;
      throw new ApiError(message, response.status);
    }
    if (!body) throw new ApiError('Сервер вернул некорректный ответ.');
    return body;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    const timeoutMessage = options.method === 'POST' ? 'Время ожидания истекло. Ответ мог сохраниться на сервере; повторная отправка создаст новый запрос.' : 'Сервер не ответил вовремя. Если данные ещё загружаются, проверьте подключение чуть позже.';
    throw new ApiError(error.name === 'AbortError' ? timeoutMessage : 'Нет соединения с сервером. Проверьте подключение и попробуйте ещё раз.');
  } finally { clearTimeout(timer); }
}
export const api = {
  experts: () => request('/experts', { timeout: 15000 }),
  session: () => request('/session', { method: 'POST', timeout: 15000 }),
  chat: (payload, token) => request('/chat', { method: 'POST', body: JSON.stringify(payload), headers: { Authorization: `Bearer ${token}` } }),
  source: (article, section, generation) => request(`/source/${encodeURIComponent(article)}/${encodeURIComponent(section)}${generation ? `?generation=${encodeURIComponent(generation)}` : ''}`),
};
