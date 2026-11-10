"""Build a code-only manual serving notebook. No credentials, photos or model weights."""
import ast
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
INTRO='''# Modegen — подключение Kaggle к сайту

1. Создайте **новый приватный notebook** и импортируйте этот файл.
2. Включите **GPU T4** (T4 ×2 тоже подходит) и **Internet On**. Input добавлять не нужно.
3. Нажмите **Run All в редакторе**, не Save & Run All. Последняя ячейка должна оставаться работающей, пока вы пользуетесь сайтом.
4. Дождитесь строк «Адрес подключения» и «Ключ подключения». На https://modegen.ru/ai-platform/ откройте студию и вставьте их в «Подключение Kaggle».
5. Загрузите фото человека или выберите модель; загрузите разложенную футболку/рубашку/лонгслив; нажмите «Создать кадр».
6. После теста остановите сессию Kaggle, чтобы не расходовать GPU-квоту.

Это **последний завершённый тест FASHN v1.1**: закреплённые исходники и веса,
FP32, seed42, 30 шагов, CFG1.5, segmentation_free=True, верх, flat-lay,
чанк attention512, препроцессингCPU. Нового обучения и LoRA нет.
Настройки теста v1.2 с маской сюда не перенесены: его GPU-результат не получен.
Результаты v1.1 пока теряли принт; подключение к сайту не улучшает модель.

Подключение временное: HTTPS-туннель Gradio ведёт в эту GPU-сессию. Любой запрос
с фото и любой результат требуют ключа; ключ не вшит в сайт и не передаётся в URL.
Фото идут только в вашу сессию через туннель, сервер не пишет их в файлы.
**Не публикуйте вывод notebook с ключом.** Новый запуск создаёт новый ключ.
Лимит пилота — 6 попыток, 30 минут простоя / 2 часа работы; автоповторов нет.
Остановка веб-сервера не заменяет остановку самой GPU-сессии Kaggle.

Сервер и браузерный протокол проверены локально на CPU с тестовой заглушкой.
Реальная модель проверялась отдельно в v1.1. Новый полный путь сайт → Kaggle →
изображение требует первого ручного GPU-запуска этого notebook.
Никакой оплаты не подключается. Не запускайте параллельно с другим GPU-notebook.
'''

def build():
    scripts={p.stem:p.read_text() for p in (ROOT/'engine.py',ROOT/'setup_runtime.py',ROOT/'server.py')}
    for name,source in scripts.items():ast.parse(source,filename=name)
    setup='''from pathlib import Path
import subprocess, sys, json
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Сначала включите GPU в настройках Kaggle.')
ROOT=Path('/kaggle/working/modegen_web')
ROOT.mkdir(parents=True,exist_ok=True)
'''+ 'SCRIPTS = '+repr(scripts)+'''
for name,source in SCRIPTS.items(): (ROOT/(name+'.py')).write_text(source)
subprocess.run([sys.executable,'-u',str(ROOT/'setup_runtime.py'),'--output',str(ROOT)],check=True)
'''
    serve="subprocess.run([sys.executable,'-u',str(ROOT/'server.py')],check=True)\n"
    cells=[dict(id='instructions',cell_type='markdown',metadata={},source=INTRO.splitlines(True))]
    for name,source in [('setup',setup),('serve',serve)]:
        ast.parse(source)
        cells.append(dict(id=name,cell_type='code',metadata={},source=source.splitlines(True),execution_count=None,outputs=[]))
    nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'},'kaggle':{'isInternetEnabled':True,'language':'python','sourceType':'notebook'}},cells=cells)
    target=ROOT.parent/'ai-platform/modegen_kaggle_web.ipynb'
    target.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
    print(target.name,target.stat().st_size,'bytes; GPU not run')
if __name__=='__main__':build()
