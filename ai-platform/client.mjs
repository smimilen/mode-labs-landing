export const MODEL_ID = 'modegen-fashn-v11-fp32';

export function endpoint(value, local = false) {
  const url = new URL(value.trim());
  const dev = local && url.protocol === 'http:' && ['127.0.0.1', 'localhost'].includes(url.hostname);
  if ((!dev && !(url.protocol === 'https:' && /^[a-z0-9-]+\.gradio\.live$/.test(url.hostname))) ||
      url.username || url.password || url.search || url.hash || url.pathname !== '/' || (!dev && url.port)) {
    throw new Error('Вставьте HTTPS-адрес ….gradio.live из последней ячейки Kaggle.');
  }
  return url.origin;
}

export function validateStatus(info) {
  const s = info?.settings;
  if (info?.protocol !== 1 || s?.model_id !== MODEL_ID || s.seed !== 42 || s.steps !== 30 ||
      s.guidance_scale !== 1.5 || s.segmentation_free !== true || s.dtype !== 'float32' ||
      s.garment_photo_type !== 'flat-lay' || s.category !== 'tops' || s.training_steps !== 0 ||
      s.attention_query_chunk !== 512 || s.weights_revision !== '7720683168567eb5a2a4c67f15116c6e29c83ded' ||
      !Number.isInteger(info.remaining) || info.remaining < 0) {
    throw new Error('Это другая версия сервера. Запустите notebook по ссылке на этой странице.');
  }
  return info;
}

export function parseEvent(block) {
  let event = '', data = [];
  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
  }
  return {event, data: data.length ? JSON.parse(data.join('\n')) : null};
}

async function checked(response) {
  if (response.ok) return response;
  if (response.status === 401) throw new Error('Неверный ключ подключения. Скопируйте его из текущей сессии Kaggle.');
  if (response.status === 413) throw new Error('Слишком большой файл. Максимум 8 МБ на фото.');
  if (response.status === 503) throw new Error('Очередь заполнена. Дождитесь текущего кадра.');
  throw new Error('Сессия Kaggle недоступна. Проверьте, что последняя ячейка ещё работает.');
}

export async function call(url, key, name, data, {timeoutMs = 600000, fetcher = fetch} = {}) {
  if (!['status','tryon'].includes(name)) throw new Error('Неизвестная операция.');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'};
  let reader;
  try {
    const response = await checked(await fetcher(`${url}/gradio_api/call/${name}`, {
      method: 'POST', headers, body: JSON.stringify({data}), signal: controller.signal,
      credentials: 'omit', cache: 'no-store', redirect: 'error', referrerPolicy: 'no-referrer'}));
    const job = await response.json();
    if (!/^[a-f0-9]{16,64}$/.test(job.event_id || '')) throw new Error('Сервер не вернул номер задания.');
    const stream = await checked(await fetcher(`${url}/gradio_api/call/${name}/${job.event_id}`, {
      headers, signal: controller.signal, credentials: 'omit', cache: 'no-store', redirect: 'error',
      referrerPolicy: 'no-referrer'}));
    reader = stream.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value, {stream: !done}).replaceAll('\r', '');
      if (buffer.length > 24000000) throw new Error('Ответ сервера слишком большой.');
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const event = parseEvent(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        if (event.event === 'error') throw new Error('Генерация не завершилась. Проверьте журнал Kaggle.');
        if (event.event === 'complete') {
          if (!Array.isArray(event.data) || event.data.length !== 1) throw new Error('Пустой ответ сервера.');
          return event.data[0];
        }
      }
      if (done) throw new Error('Соединение оборвалось до получения результата.');
    }
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('Время ожидания истекло. Задание может ещё работать в Kaggle; проверьте его перед повтором.');
    if (error instanceof TypeError) throw new Error('Нет связи с Kaggle. Проверьте адрес и работающую сессию; повтор автоматически не запускался.');
    throw error;
  } finally {
    clearTimeout(timer);
    if (reader) { try { await reader.cancel(); } catch {} }
    controller.abort();
  }
}
