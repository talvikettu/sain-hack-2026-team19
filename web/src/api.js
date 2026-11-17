export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}
export async function request(path, { timeout = 310000, ...options } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal, headers: { Accept: 'application/json', ...(options.body ? { 'Content-Type': 'application/json' } : {}) } });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      const message = response.status === 503 ? 'Эксперт пока недоступен. Данные ещё не загружены или сервер занят. Попробуйте позже.' : typeof body?.detail === 'string' ? body.detail : `Ошибка сервера (${response.status}). Попробуйте ещё раз.`;
      throw new ApiError(message, response.status);
    }
    if (!body) throw new ApiError('Сервер вернул некорректный ответ.');
    return body;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(error.name === 'AbortError' ? 'Время ожидания истекло. Ответ мог сохраниться на сервере; повторная отправка создаст новый запрос.' : 'Нет соединения с сервером. Проверьте подключение и попробуйте ещё раз.');
  } finally { clearTimeout(timer); }
}
export const api = {
  experts: () => request('/experts', { timeout: 310000 }),
  chat: payload => request('/chat', { method: 'POST', body: JSON.stringify(payload) }),
  source: (article, section) => request(`/source/${encodeURIComponent(article)}/${encodeURIComponent(section)}`),
};
