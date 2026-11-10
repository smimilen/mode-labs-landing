import {endpoint, validateStatus, call, MODEL_ID} from './client.mjs';

const $ = id => document.getElementById(id);
function read(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) || fallback; } catch { return fallback; } }
let user = read('mg_user', null), register = true, busy = false, connecting = false;
let connection = null, garment = null, person = null, model = null, currentURL = null;
let remaining = 0;
const local = ['127.0.0.1', 'localhost'].includes(location.hostname);
const models = [[5,'Нова'],[6,'Сэй'],[7,'Рия'],[9,'Лина'],[10,'Каора'],[11,'Диана'],
  [15,'Алина'],[16,'Кай'],[8,'Эмре'],[12,'Оливер'],[13,'Марко'],[14,'Дино']];

function note(text, type='') { $('toast').textContent=text; $('toast').className='toast show '+type;
  clearTimeout(note.timer); note.timer=setTimeout(()=>$('toast').classList.remove('show'),6000); }
function status(text, type='') { $('connectionStatus').textContent=text; $('connectionStatus').className='status '+type; }
function controls() {
  $('genBtn').disabled = !(user && connection && garment && (person || model) && remaining>0) || busy || connecting;
  $('connectBtn').disabled = busy || connecting;
  $('disconnectBtn').disabled = !connection || busy || connecting;
  ['file','personFile','serverUrl','serverKey'].forEach(id=>$(id).disabled=busy || connecting);
  document.querySelectorAll('.mopt, #logoutBtn').forEach(el=>el.disabled=busy || connecting);
}
function disconnect() {
  connection=null; remaining=0; $('serverKey').value='';
  status('Сессия не подключена.'); controls();
}
function render() {
  $('viewAuth').classList.toggle('hide',!!user); $('viewApp').classList.toggle('hide',!user);
  $('navRight').replaceChildren();
  if(user) { const tag=document.createElement('span');tag.className='studio-tag';tag.textContent='ТЕСТ БЕЗ ОПЛАТЫ';
    const out=document.createElement('button');out.className='btn btn-ghost';out.id='logoutBtn';out.textContent='Выйти';
    out.onclick=()=>{ if(busy) return; user=null;localStorage.removeItem('mg_user');disconnect();render(); };
    $('navRight').append(tag,out); renderGallery(); }
  controls();
}

async function passwordHash(password, salt) {
  const material=await crypto.subtle.importKey('raw',new TextEncoder().encode(password),'PBKDF2',false,['deriveBits']);
  const result=await crypto.subtle.deriveBits({name:'PBKDF2',salt:new TextEncoder().encode(salt),iterations:210000,hash:'SHA-256'},material,256);
  return btoa(String.fromCharCode(...new Uint8Array(result)));
}
$('switchAuth').onclick=()=>{register=!register; $('authTitle').textContent=register?'Создать аккаунт':'Вход';
  $('aBtn').textContent=register?'Зарегистрироваться':'Войти'; $('switchText').textContent=register?'Уже есть аккаунт?':'Нет аккаунта?';
  $('switchAuth').textContent=register?'Войти':'Регистрация'; $('aPass').minLength=register?8:4;
  $('aPass').autocomplete=register?'new-password':'current-password'; $('aErr').textContent='';};
$('authForm').onsubmit=async event=>{
  event.preventDefault(); $('aErr').textContent=''; $('aBtn').disabled=true;
  try {
    if (!window.isSecureContext) throw new Error('Откройте страницу по HTTPS.');
    const email=$('aEmail').value.trim().toLowerCase(), pass=$('aPass').value;
    const users=read('mg_users',{}), old=Object.hasOwn(users,email)?users[email]:null;
    if(register && old) throw new Error('Аккаунт уже существует — войдите.');
    if(!register && (!old || (old.passwordHash ? await passwordHash(pass,old.salt)!==old.passwordHash : old.pass!==pass)))
      throw new Error('Неверный email или пароль.');
    const record=old || {credits:3,gallery:[]};
    // Migrate a legacy local plaintext password only after a successful login.
    if(!record.passwordHash) {record.salt=crypto.randomUUID();record.passwordHash=await passwordHash(pass,record.salt);delete record.pass;}
    users[email]=record;localStorage.setItem('mg_users',JSON.stringify(users));
    user={email};localStorage.setItem('mg_user',JSON.stringify(user));$('aPass').value='';render();
  } catch(error) {$('aErr').textContent=error.message;} finally {$('aBtn').disabled=false;}
};
// Guest access does not create a server account; useful for this private pilot.
const guest=document.createElement('button');guest.type='button';guest.className='btn btn-ghost';
guest.style.marginTop='14px';guest.textContent='Попробовать без регистрации';
guest.onclick=()=>{user={email:'guest'};render();};$('authForm').append(guest);

for(const [id,name] of models) {
  const button=document.createElement('button');button.type='button';button.className='mopt';button.dataset.model=id;
  button.setAttribute('aria-label','Модель '+name);
  const img=document.createElement('img');img.src='/img/model'+id+'.jpg';img.alt=name;img.loading='lazy';
  const caption=document.createElement('span');caption.className='mname';caption.textContent=name;button.append(img,caption);
  button.onclick=()=>{if(busy) return;model=id;person=null;$('personFile').value='';$('personPreview').classList.add('hide');
    document.querySelectorAll('.mopt').forEach(el=>el.classList.toggle('sel',el===button));controls();};
  $('mgrid').append(button);
}

async function readPhoto(file) {
  if(!file || !['image/jpeg','image/png','image/webp'].includes(file.type)) throw new Error('Выберите JPG, PNG или WebP. HEIC сначала экспортируйте в JPG.');
  if(file.size>8*1024*1024) throw new Error('Максимум 8 МБ на фото.');
  const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;reader.readAsDataURL(file);});
  const img=new Image();img.src=data;await img.decode();
  if(Math.min(img.naturalWidth,img.naturalHeight)<64 || img.naturalWidth*img.naturalHeight>16000000)
    throw new Error('Размер фото: от 64 пикселей по стороне до 16 мегапикселей.');
  return data;
}
async function selectFile(file, kind) {
  if(busy) return;
  try {const value=await readPhoto(file);if(busy) return;
    if(kind==='garment') {garment=value;const preview=new Image();preview.src=value;preview.alt='Фото одежды';preview.className='preview';$('dropInner').replaceChildren(preview);}
    else {person=value;model=null;$('personPreview').src=value;$('personPreview').classList.remove('hide');document.querySelectorAll('.mopt').forEach(el=>el.classList.remove('sel'));}
    $('resultMessage').textContent='';controls();
  } catch(error) {note(error.message,'err');}
}
$('file').onchange=e=>selectFile(e.target.files[0],'garment');
$('personFile').onchange=e=>selectFile(e.target.files[0],'person');
$('drop').ondragover=e=>{e.preventDefault();$('drop').classList.add('over');};
$('drop').ondragleave=()=>$('drop').classList.remove('over');
$('drop').ondrop=e=>{e.preventDefault();$('drop').classList.remove('over');selectFile(e.dataTransfer.files[0],'garment');};
$('connectBtn').onclick=async()=>{
  connecting=true;connection=null;remaining=0;controls();status('Проверяю сессию…');
  try {
    if(!window.isSecureContext) throw new Error('Для ключа и фото нужен HTTPS. Исправьте сертификат сайта.');
    const url=endpoint($('serverUrl').value,local), key=$('serverKey').value.trim();
    if(!/^[A-Za-z0-9_-]{32,128}$/.test(key)) throw new Error('Вставьте полный ключ из последней ячейки Kaggle.');
    const info=validateStatus(await call(url,key,'status',[],{timeoutMs:20000}));
    connection={url,key};remaining=info.remaining;$('serverKey').value='';
    status(`Подключено: FASHN v1.1 · осталось ${remaining} запусков в этой сессии.`,'ok');
  } catch(error){status(error.message,'err');} finally{connecting=false;controls();}
};
$('disconnectBtn').onclick=disconnect;
$('serverUrl').oninput=()=>{if(connection) disconnect();};

function resultBlob(data) {
  if(typeof data!=='string' || data.length>24000000 || !/^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(data))
    throw new Error('Сервер вернул некорректное изображение.');
  const raw=atob(data.split(',')[1]), bytes=Uint8Array.from(raw,c=>c.charCodeAt(0));
  return new Blob([bytes],{type:'image/png'});
}
$('genBtn').onclick=async()=>{
  if($('genBtn').disabled || busy) return;
  busy=true;controls();$('resultMessage').textContent='';$('resWrap').classList.remove('show');
  $('resultEmpty').classList.add('hide');$('progWrap').classList.add('show');$('ptext').textContent='Проверяю связь с Kaggle…';
  try {
    const active=connection, owner=user.email;
    const info=validateStatus(await call(active.url,active.key,'status',[],{timeoutMs:20000}));remaining=info.remaining;
    if(info.busy || !remaining) throw new Error(info.busy?'В Kaggle уже создаётся кадр. Дождитесь окончания.':'Лимит этой сессии исчерпан.');
    let human=person;
    if(!human){const response=await fetch('/img/model'+model+'.jpg');if(!response.ok) throw new Error('Фото выбранной модели не загрузилось.');human=await readPhoto(new File([await response.blob()],'person.jpg',{type:'image/jpeg'}));}
    $('ptext').textContent='Создаём кадр в Kaggle. Обычно около 4–5 минут; не запускайте второй запрос.';
    const result=await call(active.url,active.key,'tryon',[{person:human,garment}]);
    validateStatus({protocol:1,settings:result.settings,remaining:Math.max(0,remaining-1)});
    const blob=resultBlob(result.image), nextURL=URL.createObjectURL(blob);
    const image=new Image();image.src=nextURL;
    try{await image.decode();}catch{URL.revokeObjectURL(nextURL);throw new Error('Полученный кадр не открывается.');}
    if(currentURL) URL.revokeObjectURL(currentURL);currentURL=nextURL;
    $('resImg').src=nextURL;$('dlBtn').href=nextURL;$('resWrap').classList.add('show');
    $('resultMeta').textContent=`FASHN v1.1 · ${image.naturalWidth} × ${image.naturalHeight} · seed 42`;
    remaining=Math.max(0,remaining-1);status(`Подключено: FASHN v1.1 · осталось ${remaining} запусков.`,'ok');
    try {await saveImage(owner,blob);note('Кадр готов и добавлен в галерею.','ok');}
    catch {note('Кадр готов. Галерея недоступна — сохраните PNG кнопкой «Скачать».','err');}
  } catch(error){$('resultMessage').textContent=error.message;$('resultEmpty').classList.remove('hide');}
  finally {busy=false;$('progWrap').classList.remove('show');controls();}
};
$('resetBtn').onclick=()=>{$('resWrap').classList.remove('show');$('resultEmpty').classList.remove('hide');};
document.querySelectorAll('[data-tab]').forEach(button=>button.onclick=()=>{
  const tab=button.dataset.tab;document.querySelectorAll('[data-tab]').forEach(el=>el.classList.toggle('active',el===button));
  $('tabStudio').classList.toggle('hide',tab!=='studio');$('tabGallery').classList.toggle('hide',tab!=='gallery');if(tab==='gallery')renderGallery();
});

function database() {return new Promise((resolve,reject)=>{const request=indexedDB.open('modegen-results',1);
  request.onupgradeneeded=()=>request.result.createObjectStore('images',{keyPath:'id'});
  request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);});}
async function saveImage(owner,blob) {const db=await database();try{await new Promise((resolve,reject)=>{
  const tx=db.transaction('images','readwrite');tx.objectStore('images').put({id:crypto.randomUUID(),owner,blob,date:Date.now(),model:MODEL_ID});
  tx.oncomplete=resolve;tx.onerror=tx.onabort=()=>reject(tx.error);});} finally{db.close();}}
let galleryURLs=[];
async function renderGallery() {
  if(!user)return;const owner=user.email;let records=[];
  try{const db=await database();try{records=await new Promise((resolve,reject)=>{const r=db.transaction('images').objectStore('images').getAll();r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});}finally{db.close();}}
  catch{ $('gEmpty').textContent='Галерея недоступна в этом браузере.'; }
  if(!user || owner!==user.email)return;
  galleryURLs.forEach(url=>URL.revokeObjectURL(url));galleryURLs=[];$('ggrid').replaceChildren();
  const own=records.filter(r=>r.owner===owner).sort((a,b)=>b.date-a.date);$('gEmpty').classList.toggle('hide',!!own.length);
  for(const r of own){const url=URL.createObjectURL(r.blob);galleryURLs.push(url);const card=document.createElement('article');card.className='gitem';
    const img=new Image();img.src=url;img.alt='Сохранённый результат примерки';img.loading='lazy';
    const meta=document.createElement('div');meta.className='gmeta';meta.textContent=new Date(r.date).toLocaleString('ru');
    const link=document.createElement('a');link.href=url;link.download='modegen-result.png';link.className='gdl';link.textContent='Скачать PNG';card.append(img,meta,link);$('ggrid').append(card);}
}
render();
