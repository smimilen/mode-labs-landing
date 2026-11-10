import test from 'node:test';
import assert from 'node:assert/strict';
import {endpoint, parseEvent, call, validateStatus} from '../ai-platform/client.mjs';

test('endpoint never accepts tokens, alternate hosts or insecure production URLs',()=>{
  assert.equal(endpoint('https://abc-123.gradio.live/'),'https://abc-123.gradio.live');
  assert.equal(endpoint('http://127.0.0.1:7871',true),'http://127.0.0.1:7871');
  for(const s of ['http://abc.gradio.live','https://evil.example','https://abc.gradio.live.evil.example',
      'https://user:secret@abc.gradio.live','https://abc.gradio.live/?token=secret','http://127.0.0.1:7871'])
    assert.throws(()=>endpoint(s));
});
test('SSE parsing handles heartbeats and split data',()=>{
  assert.deepEqual(parseEvent('event: heartbeat\ndata: null'),{event:'heartbeat',data:null});
  assert.deepEqual(parseEvent('event: complete\ndata: [{"a":\ndata: 1}]'),{event:'complete',data:[{a:1}]});
});
test('transport reads chunked SSE; secret stays in header',async()=>{
  const requests=[];const key='PRIVATE_TEST_VALUE';
  const fetcher=async(url,options)=>{
    requests.push([url,options]);assert.equal(options.headers.Authorization,'Bearer '+key);
    assert.ok(!url.includes(key));assert.ok(!(options.body||'').includes(key));
    if(options.method==='POST')return Response.json({event_id:'a'.repeat(32)});
    const chunks=['event: heart','beat\ndata: null\n\nevent: comp','lete\ndata: [{"image":"ok"}]\n','\n'];
    return new Response(new ReadableStream({start(controller){chunks.forEach(c=>controller.enqueue(new TextEncoder().encode(c)));controller.close();}}));
  };
  assert.deepEqual(await call('https://test.gradio.live',key,'tryon',[{}],{fetcher}),{image:'ok'});
  assert.equal(requests.length,2);
});
test('failure never retries a generation',async()=>{
  let count=0;await assert.rejects(call('https://test.gradio.live','secret','tryon',[],{
    fetcher:async()=>{count++;return new Response('',{status:401});}}),/ключ/);
  assert.equal(count,1);
  assert.throws(()=>validateStatus({protocol:1,settings:{model_id:'another-model'},remaining:6}),/версия/);
});
