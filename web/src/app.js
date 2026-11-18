import { api } from './api.js';

const $ = id => document.getElementById(id);
const profiles = {
  clinician: { name: 'Сомнолог', description: 'Читает доказательства как клиницист: механизм, сила доказательств и границы применимости выводов.', note: 'Крепкий сон. Лёгкое пробуждение.' },
  biohacker: { name: 'Биохакер', description: 'Превращает вывод исследования в одну переменную, которую можно проверить на себе, — с метрикой и сроком.', note: 'Энергия в течение дня.' },
  nutritionist: { name: 'Нутрициолог', description: 'Находит одну небольшую замену, которую можно привязать к уже имеющейся привычке, — без запретов и десятишаговых планов.', note: 'Доказательно. Проверяемо. Практично.' },
};
let available = [];
let selected;
let busy = false;
let sourceVersion = 0;
const chats = new Map();
const suggestions = ['Как влияет чашка кофе вечером на мой сон?', 'Почему я просыпаюсь ночью?', 'Как наладить режим сна?'];

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function profile(id) { return profiles[id] || { name: id, description: available.find(e => e.expert_id === id)?.role || '', note: 'Ответы на основе исследований.' }; }
function expertIcon(id) {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('class', 'icon');
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS(ns, 'use');
  use.setAttribute('href', '#icon-' + (Object.hasOwn(profiles, id) ? id : 'expert'));
  svg.append(use);
  return svg;
}
function badge(id) {
  const node = el('span', 'expert-symbol');
  node.append(expertIcon(id));
  return node;
}
function chatState() {
  if (!chats.has(selected)) chats.set(selected, { messages: [], session: undefined, draft: '' });
  return chats.get(selected);
}
function renderExperts() {
  $('experts').replaceChildren();
  const ids = available.length ? available.map(e => e.expert_id) : Object.keys(profiles);
  for (const id of ids) {
    const p = profile(id);
    const card = el('article', `expert-card ${id}`);
    const heading = el('div', 'card-heading');
    heading.append(el('h2', '', p.name), badge(id));
    card.append(heading, el('p', 'description', p.description));
    const bottom = el('div', 'card-bottom');
    const button = el('button', 'select-button', 'Выбрать →');
    button.dataset.expert = id;
    button.setAttribute('aria-label', `Выбрать эксперта: ${p.name}`);
    button.addEventListener('click', () => openChat(id));
    bottom.append(el('span', 'card-note', p.note), button);
    card.append(bottom);
    $('experts').append(card);
  }
}
async function loadExperts() {
  $('connection').textContent = 'Подключаем экспертов…';
  $('reload').hidden = true;
  renderExperts();
  try {
    const data = await api.experts();
    if (!Array.isArray(data.experts) || data.experts.some(e => typeof e.expert_id !== 'string')) throw new Error('Некорректный список экспертов.');
    available = data.experts;
    $('connection').textContent = available.length ? '' : 'Список экспертов пока пуст.';
    $('reload').hidden = available.length > 0;
  } catch (error) {
    available = [];
    $('connection').textContent = error.message;
    $('reload').hidden = false;
  }
  renderExperts();
}
function openChat(id) {
  selected = id;
  document.body.dataset.expert = id;
  $('selection').hidden = true;
  $('conversation').hidden = false;
  $('chat-title').textContent = profile(id).name;
  $('back').replaceChildren(expertIcon(id));
  $('question').value = chatState().draft;
  const connected = available.some(e => e.expert_id === id);
  $('chat-error').textContent = connected ? '' : 'Нет подключения к backend. Интерфейс доступен для просмотра, но для ответов запустите сервер на порту 8000. После запуска можно отправить вопрос.';
  $('chat-error').hidden = connected;
  renderMessages();
  $('question').focus();
}
function back() {
  chatState().draft = $('question').value;
  $('conversation').hidden = true;
  $('selection').hidden = false;
  stopSpeech();
  const returnButton = [...$('experts').querySelectorAll('button')].find(button => button.dataset.expert === selected);
  (returnButton || $('selection-title')).focus();
}
let speechButton;
function stopSpeech() {
  window.speechSynthesis?.cancel();
  if (speechButton) { speechButton.textContent = '▷ Прослушать ответ'; speechButton.setAttribute('aria-pressed', 'false'); speechButton = undefined; }
}
function renderMessages() {
  const state = chatState();
  const log = $('messages');
  log.replaceChildren();
  if (!state.messages.length) {
    const welcome = el('div', 'welcome');
    welcome.append(badge(selected), el('h2', '', 'О чём поговорим?'), el('p', '', 'Задайте вопрос о сне. Эксперт ответит на основе исследований и покажет источники.'));
    log.append(welcome);
  }
  for (const message of state.messages) {
    const bubble = el('article', message.role === 'user' ? 'message user-message' : 'message answer-message');
    if (message.role === 'assistant') bubble.append(el('span', 'eyebrow', profile(selected).name));
    bubble.append(el('p', 'message-text', message.text));
    if (message.refused) bubble.append(el('p', 'boundary', 'Для этого вопроса нужен специалист. Ответ ограничен данными исследований.'));
    if (message.citations?.length) {
      const citations = el('div', 'citations');
      citations.append(el('span', 'eyebrow', 'ИСТОЧНИКИ'));
      message.citations.forEach((citation, i) => {
        const button = el('button', 'source-link', `${i + 1}. ${citation.article_title || citation.article_id}${citation.year ? ` · ${citation.year}` : ''} ↗`);
        button.addEventListener('click', () => showSource(citation));
        citations.append(button);
      });
      bubble.append(citations);
    }
    if (message.role === 'assistant' && 'speechSynthesis' in window && typeof SpeechSynthesisUtterance !== 'undefined') {
      const button = el('button', 'speech-button', '▷ Прослушать ответ');
      button.setAttribute('aria-pressed', 'false');
      button.addEventListener('click', () => {
        const wasSpeaking = speechButton === button;
        stopSpeech();
        if (wasSpeaking) return;
        const speech = new SpeechSynthesisUtterance(message.text);
        speech.lang = 'ru-RU';
        speechButton = button;
        button.textContent = 'Ⅱ Остановить';
        button.setAttribute('aria-pressed', 'true');
        speech.onend = speech.onerror = () => { if (speechButton === button) stopSpeech(); };
        window.speechSynthesis.speak(speech);
      });
      bubble.append(button);
    }
    log.append(bubble);
  }
  if (busy) { const loading = el('p', 'loading', 'Эксперт изучает источники…'); loading.setAttribute('role', 'status'); log.append(loading); }
  $('suggestions').replaceChildren();
  if (!state.messages.length) suggestions.forEach(text => {
    const button = el('button', 'suggestion', `${text} ↗`);
    button.disabled = busy;
    button.addEventListener('click', () => { $('question').value = text; $('question').focus(); });
    $('suggestions').append(button);
  });
  const scrollArea = log.parentElement;
  scrollArea.scrollTop = state.messages.length ? scrollArea.scrollHeight : 0;
}
function setBusy(value) {
  busy = value;
  for (const id of ['send', 'question', 'back', 'choose', 'new-chat']) $(id).disabled = value;
  $('composer').setAttribute('aria-busy', String(value));
}
$('composer').addEventListener('submit', async event => {
  event.preventDefault();
  const text = $('question').value.trim();
  if (busy || !text) return;
  stopSpeech();
  const state = chatState();
  state.messages.push({ role: 'user', text });
  state.draft = '';
  $('question').value = '';
  $('chat-error').hidden = true;
  setBusy(true);
  renderMessages();
  try {
    const answer = await api.chat({ expert_id: selected, message: text, ...(state.session ? { session_id: state.session } : {}) });
    if (typeof answer.text !== 'string' || typeof answer.session_id !== 'string' || !Array.isArray(answer.citations)) throw new Error('Сервер вернул некорректный ответ.');
    state.session = answer.session_id;
    state.messages.push({ role: 'assistant', ...answer });
  } catch (error) {
    state.messages.pop();
    $('question').value = text;
    state.draft = text;
    $('chat-error').textContent = error.message;
    $('chat-error').hidden = false;
  } finally { setBusy(false); renderMessages(); $('question').focus(); }
});
$('question').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('composer').requestSubmit(); }
});
$('back').addEventListener('click', back);
$('choose').addEventListener('click', back);
$('new-chat').addEventListener('click', () => {
  stopSpeech(); chats.delete(selected); $('question').value = ''; $('chat-error').hidden = true; renderMessages(); $('question').focus();
});
$('reload').addEventListener('click', loadExperts);
async function showSource(citation) {
  const version = ++sourceVersion;
  $('source-title').textContent = citation.article_title || citation.article_id;
  $('source-body').replaceChildren(el('p', '', 'Загружаем раздел исследования…'));
  if (!$('source-dialog').open) $('source-dialog').showModal();
  try {
    const source = await api.source(citation.article_id, citation.section_id);
    if (version !== sourceVersion || !$('source-dialog').open) return;
    $('source-title').textContent = source.article_title;
    const authors = Array.isArray(source.authors) ? source.authors.join(', ') : source.authors || '';
    $('source-body').replaceChildren(el('p', 'source-meta', `${authors}${source.year ? ` · ${source.year}` : ''}`), el('h3', '', source.section_title), el('p', 'source-text', source.text), el('p', 'source-meta', `${source.article_id} / ${source.section_id}`));
    if (source.doi) {
      const doi = el('a', 'doi', `DOI: ${source.doi}`);
      doi.href = `https://doi.org/${encodeURIComponent(source.doi)}`;
      doi.target = '_blank'; doi.rel = 'noopener noreferrer';
      $('source-body').append(doi);
    }
    if (source.license) $('source-body').append(el('p', 'source-meta', `Лицензия: ${source.license}`));
  } catch (error) { if (version === sourceVersion) $('source-body').replaceChildren(el('p', 'error', error.message)); }
}
$('close-source').addEventListener('click', () => $('source-dialog').close());
$('source-dialog').addEventListener('close', () => { sourceVersion++; });
window.addEventListener('pagehide', stopSpeech);
loadExperts();
