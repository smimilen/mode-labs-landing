"""LOCAL TEST ONLY: CPU placeholder, never a generated fashion image."""
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import sys, threading
from types import SimpleNamespace
from PIL import Image, ImageDraw
from starlette.middleware import Middleware
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'kaggle'))
from server import AccessGuard,Session,build_app
class Stub:
 def __call__(self,**kwargs):
  im=Image.new('RGB',(576,768),'#18202d');d=ImageDraw.Draw(im)
  d.text((40,320),'TRANSPORT TEST ONLY',fill='white',font_size=34)
  d.text((40,370),'NOT A MODEL GENERATION',fill='#ff8a3d',font_size=28)
  return SimpleNamespace(images=[im])
class PageHandler(SimpleHTTPRequestHandler):
 def do_GET(self):
  if self.path.split('?')[0].startswith(('/ai-platform/','/img/','/logo.svg')):
   return super().do_GET()
  self.send_error(404)
server=ThreadingHTTPServer(('127.0.0.1',8766),partial(PageHandler,directory=str(ROOT)))
threading.Thread(target=server.serve_forever,daemon=True).start()
app=build_app(Session(Stub()))
app.launch(server_name='127.0.0.1',server_port=7872,share=False,quiet=True,
 app_kwargs={'middleware':[Middleware(AccessGuard,key='local-test-key-not-a-real-secret-1234567890',origins={'http://127.0.0.1:8766'})]})
