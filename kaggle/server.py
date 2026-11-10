"""Private, temporary website bridge. Model inference stays inside Kaggle."""
import base64
import binascii
import hmac
import io
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time

os.environ.update(USE_TORCH='1', USE_TF='0', USE_FLAX='0', FORCE_TF_AVAILABLE='0',
                  GRADIO_ANALYTICS_ENABLED='False', HF_HUB_DISABLE_TELEMETRY='1')

MODEL_ID = 'modegen-fashn-v11-fp32'
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_BODY = 24 * 1024 * 1024
MAX_PIXELS = 16_000_000
SETTINGS = dict(model_id=MODEL_ID, seed=42, steps=30, guidance_scale=1.5,
                segmentation_free=True, category='tops', garment_photo_type='flat-lay',
                dtype='float32', attention_query_chunk=512, training_steps=0,
                weights_revision='7720683168567eb5a2a4c67f15116c6e29c83ded')
ORIGINS = {'https://modegen.ru', 'https://www.modegen.ru'}


class AccessGuard:
    """Check the key before Gradio parses/enqueues any payload; never use URL tokens."""
    def __init__(self, app, key, origins=ORIGINS):
        self.app, self.key, self.origins = app, key, set(origins)

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        from starlette.responses import JSONResponse
        headers = dict(scope['headers'])
        origin = headers.get(b'origin', b'').decode()
        path = scope['path']
        async def reject(code, message):
            await JSONResponse({'error': message}, status_code=code)(scope, receive, send)
        if origin and origin not in self.origins:
            return await reject(403, 'Origin not allowed')
        if scope['method'] == 'OPTIONS':
            if not origin:
                return await reject(403, 'Origin required')
            response = JSONResponse({}, headers={
                'Access-Control-Allow-Origin': origin,
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Authorization, Content-Type',
                'Vary': 'Origin'})
            return await response(scope, receive, send)
        # Gradio's launch() performs this local startup request before creating its tunnel.
        local = (scope.get('client') or ('',))[0] in ('127.0.0.1', '::1')
        startup = path == '/gradio_api/startup-events' and local and not origin
        if path == '/' and scope['method'] in ('GET', 'HEAD'):
            return await JSONResponse({'service': 'Modegen private GPU session'})(scope, receive, send)
        if not startup:
            supplied = headers.get(b'authorization', b'')
            if not hmac.compare_digest(supplied, ('Bearer ' + self.key).encode()):
                return await reject(401, 'Invalid session key')
            if not re.fullmatch(r'/gradio_api/call/(status|tryon)(/[a-f0-9]+)?', path):
                return await reject(404, 'Not found')
        try:
            if int(headers.get(b'content-length', b'0')) > MAX_BODY:
                return await reject(413, 'Request too large')
        except ValueError:
            return await reject(400, 'Invalid length')
        # Count actual bytes as well (including chunked requests) before forwarding.
        if scope['method'] == 'POST':
            chunks, total = [], 0
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    return
                total += len(message.get('body', b''))
                if total > MAX_BODY:
                    return await reject(413, 'Request too large')
                chunks.append(message)
                if not message.get('more_body'):
                    break
            original_receive = receive
            async def replay():
                return chunks.pop(0) if chunks else await original_receive()
            receive = replay
        async def cors_send(message):
            if message['type'] == 'http.response.start':
                extra = [(b'cache-control', b'no-store')]
                if origin:
                    extra += [(b'access-control-allow-origin', origin.encode()), (b'vary', b'Origin')]
                message['headers'] = message.get('headers', []) + extra
            await send(message)
        await self.app(scope, receive, cors_send)


def decode_image(value):
    from PIL import Image, ImageOps
    if not isinstance(value, str) or len(value) > MAX_IMAGE_BYTES * 4 // 3 + 100:
        raise ValueError('Фото должно быть не больше 8 МБ.')
    match = re.fullmatch(r'data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=]+)', value)
    if not match:
        raise ValueError('Нужен JPG, PNG или WebP.')
    try:
        data = base64.b64decode(match[2], validate=True)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError('Фото больше 8 МБ.')
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ('JPEG', 'PNG', 'WEBP'):
                raise ValueError('Неподдерживаемый формат фото.')
            if min(image.size) < 64 or image.width * image.height > MAX_PIXELS:
                raise ValueError('Фото: от 64 пикселей по стороне до 16 мегапикселей.')
            if getattr(image, 'n_frames', 1) != 1:
                raise ValueError('Нужно неподвижное фото.')
            return ImageOps.exif_transpose(image).convert('RGB')
    except (binascii.Error, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Не удалось прочитать фото.') from exc


class Session:
    def __init__(self, pipe, max_generations=6):
        self.pipe = pipe
        self.maximum, self.attempts = max_generations, 0
        self.lock = threading.Lock()
        self.started = self.last_used = time.monotonic()

    def status(self):
        return dict(protocol=1, settings=SETTINGS, busy=self.lock.locked(),
                    remaining=max(0, self.maximum-self.attempts), quality_verified=False)

    def generate(self, payload):
        import gradio as gr
        if not isinstance(payload, dict) or set(payload) != {'person', 'garment'}:
            raise gr.Error('Нужны фото человека и одежды.', print_exception=False)
        try:
            person, garment = decode_image(payload['person']), decode_image(payload['garment'])
        except ValueError as exc:
            raise gr.Error(str(exc), print_exception=False) from None
        if not self.lock.acquire(blocking=False):
            raise gr.Error('Уже создаётся кадр. Дождитесь его окончания.', print_exception=False)
        try:
            if self.attempts >= self.maximum:
                raise gr.Error('Лимит этой тестовой сессии исчерпан.', print_exception=False)
            self.attempts += 1
            self.last_used = time.monotonic()
            result = self.pipe(person_image=person, garment_image=garment,
                category='tops', garment_photo_type='flat-lay', num_samples=1,
                num_timesteps=30, guidance_scale=1.5, seed=42,
                segmentation_free=True).images[0]
            buf = io.BytesIO()
            result.save(buf, format='PNG')
            return dict(image='data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode(),
                        settings=SETTINGS, width=result.width, height=result.height,
                        elapsed_seconds=round(time.monotonic()-self.last_used, 2))
        except gr.Error:
            raise
        except Exception as exc:
            print('Inference failed:', type(exc).__name__, flush=True)
            raise gr.Error('Генерация не завершилась. Проверьте журнал Kaggle; повторного запуска не было.',
                           print_exception=False) from None
        finally:
            self.last_used = time.monotonic()
            self.lock.release()


def build_app(session):
    import gradio as gr
    with gr.Blocks(analytics_enabled=False) as app:
        data = gr.JSON(visible=False)
        output = gr.JSON(visible=False)
        gr.Button(visible=False).click(session.generate, data, output,
            api_name='tryon', concurrency_limit=1, concurrency_id='gpu')
        gr.Button(visible=False).click(session.status, None, output,
            api_name='status', concurrency_limit=None, queue=False)
    app.queue(max_size=2, default_concurrency_limit=1)
    return app


def main():
    from engine import download_weights, make_pipeline
    import torch
    from starlette.middleware import Middleware
    if not torch.cuda.is_available():
        raise RuntimeError('Включите GPU в Kaggle. На CPU этот сервер не запускается.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path('/kaggle/working/modegen_web')
    root.mkdir(parents=True, exist_ok=True)
    print('Загружаю FASHN v1.1. Фото для запуска не требуются.', flush=True)
    pipe = make_pipeline(download_weights(root), 'cpu', 512)
    def finite_output(module, inputs, output):
        if not torch.isfinite(output).all().item():
            raise RuntimeError('Non-finite model output')
    pipe.tryon_model.final_layer.register_forward_hook(finite_output)
    session = Session(pipe)
    key = secrets.token_urlsafe(32)
    app = build_app(session)
    try:
        _, _, share_url = app.launch(share=True, server_name='127.0.0.1', server_port=7860,
            prevent_thread_lock=True, show_error=False, quiet=True, max_threads=2,
            app_kwargs={'middleware': [Middleware(AccessGuard, key=key)]})
        if not share_url or not share_url.startswith('https://'):
            raise RuntimeError('Не удалось открыть HTTPS-подключение. Сервер остановлен.')
        print('\nОткройте https://modegen.ru/ai-platform/ → Подключение Kaggle', flush=True)
        print('Адрес подключения:', share_url, flush=True)
        print('Ключ подключения:', key, flush=True)
        print('Ключ только для вашей вкладки сайта. Не публикуйте вывод notebook.', flush=True)
        print('Сессия: до 6 кадров; остановка через 30 минут простоя или 2 часа работы.\n'
              'Держите эту ячейку запущенной. По окончании остановите сессию Kaggle.', flush=True)
        while True:
            time.sleep(5)
            now = time.monotonic()
            if not session.lock.locked() and (now-session.last_used > 1800 or
                    now-session.started > 7200):
                print('Тестовый сервер остановлен по времени. Остановите GPU-сессию Kaggle.', flush=True)
                break
    finally:
        app.close()


if __name__ == '__main__':
    main()
