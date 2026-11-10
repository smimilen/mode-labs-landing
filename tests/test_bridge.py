"""CPU transport tests. The stub is deliberately not a neural-network result."""
import base64
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
from PIL import Image
import pytest
from starlette.middleware import Middleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kaggle'))
from server import AccessGuard, Session, SETTINGS, build_app, decode_image

KEY = 'local-test-key-not-a-real-secret-1234567890'
ORIGIN = 'http://127.0.0.1:8766'


def photo(size=(80, 96)):
    buf=io.BytesIO();Image.new('RGB',size,'orange').save(buf,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()


class Stub:
    def __init__(self): self.calls=[]
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[Image.new('RGB',(80,96),'purple')])


@pytest.fixture(scope='module')
def bridge():
    stub=Stub();session=Session(stub,max_generations=2);app=build_app(session)
    app.launch(server_name='127.0.0.1',server_port=7871,share=False,
        prevent_thread_lock=True,quiet=True,show_error=False,
        app_kwargs={'middleware':[Middleware(AccessGuard,key=KEY,origins={ORIGIN})]})
    yield session,stub
    app.close()


def request(name, data, key=KEY, origin=ORIGIN):
    headers={'Authorization':'Bearer '+key,'Origin':origin}
    with httpx.Client(base_url='http://127.0.0.1:7871',timeout=10) as client:
        response=client.post('/gradio_api/call/'+name,json={'data':data},headers=headers)
        if response.status_code!=200:return response,None
        result=client.get('/gradio_api/call/'+name+'/'+response.json()['event_id'],headers=headers)
        blocks=result.text.strip().split('\n\n')
        completed=[b for b in blocks if b.startswith('event: complete')]
        return response,json.loads(completed[-1].split('data: ',1)[1])[0] if completed else result.text


def test_auth_and_origin_before_queue(bridge):
    assert request('status',[],key='wrong')[0].status_code==401
    assert request('status',[],origin='https://untrusted.example')[0].status_code==403
    assert not bridge[1].calls
    r=httpx.get('http://127.0.0.1:7871/gradio_api/file=/etc/passwd',headers={'Authorization':'Bearer '+KEY})
    assert r.status_code==404


def test_preflight_and_status(bridge):
    response=httpx.options('http://127.0.0.1:7871/gradio_api/call/status',headers={
        'Origin':ORIGIN,'Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'authorization,content-type'})
    assert response.status_code==200
    assert response.headers['access-control-allow-origin']==ORIGIN
    response,result=request('status',[])
    assert response.status_code==200 and result['settings']==SETTINGS and result['remaining']==2


def test_invalid_inputs_do_not_invoke_model(bridge):
    _,result=request('tryon',[{'person':photo(),'garment':'https://private.example/photo.png'}])
    assert isinstance(result,str) and 'event: error' in result
    assert bridge[0].attempts==0
    for value in ['data:image/svg+xml;base64,PHN2Zz4=',photo((10,10)),photo((4096,4096))]:
        with pytest.raises(ValueError):decode_image(value)


def test_real_http_roundtrip_and_limits(bridge):
    for _ in range(2):
        response,result=request('tryon',[{'person':photo(),'garment':photo()}])
        assert response.status_code==200 and result['settings']==SETTINGS
        assert decode_image(result['image']).size==(80,96)
    kwargs=bridge[1].calls[0]
    assert kwargs['segmentation_free'] is True and kwargs['seed']==42
    assert kwargs['guidance_scale']==1.5 and kwargs['num_timesteps']==30
    assert kwargs['category']=='tops' and kwargs['garment_photo_type']=='flat-lay'
    _,result=request('tryon',[{'person':photo(),'garment':photo()}])
    assert 'event: error' in result and len(bridge[1].calls)==2
    assert request('status',[])[1]['remaining']==0


def test_notebook_exact_source_no_credentials():
    root=Path(__file__).resolve().parents[1]
    import ast
    nb=json.loads((root/'ai-platform/modegen_kaggle_web.ipynb').read_text())
    source=''.join(nb['cells'][1]['source']);tree=ast.parse(source)
    embedded=next(n.value for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='SCRIPTS')
    scripts=ast.literal_eval(embedded)
    for name,code in scripts.items():
        assert code==(root/'kaggle'/f'{name}.py').read_text()
        ast.parse(code)
    assert all(not cell.get('outputs') for cell in nb['cells'])
