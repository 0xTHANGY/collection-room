'use strict';
const $ = (id) => document.getElementById(id);
const dialog = $('item-dialog');
let opener = null;
let imageRequest = 0;
const roomImage = $('room-image');
function roomReady() {
  $('room').setAttribute('aria-busy', 'false');
  $('room-status').hidden = roomImage.naturalWidth > 0;
  if (!roomImage.naturalWidth) $('room-status').textContent = '房间图片暂时无法显示，请使用下方收藏清单。';
}
roomImage.addEventListener('load', roomReady);
roomImage.addEventListener('error', roomReady);
if (roomImage.complete) roomReady();
function openItem(item, trigger) {
  opener = trigger;
  $('detail-id').textContent = item.id;
  $('detail-name').textContent = item.display_name || item.id;
  $('detail-category').textContent = item.category;
  $('detail-story').textContent = typeof item.story === 'string' ? item.story : '故事待补充';
  const photo = $('detail-image');
  const status = $('photo-status');
  const request = ++imageRequest;
  photo.hidden = true;
  photo.alt = item.display_name || item.id;
  status.hidden = false;
  status.textContent = '照片加载中…';
  const loader = new Image();
  loader.onload = () => {
    if (request !== imageRequest) return;
    photo.src = item.image;
    photo.hidden = false;
    status.hidden = true;
  };
  loader.onerror = () => {
    if (request !== imageRequest) return;
    status.textContent = `照片暂时无法显示 · ${item.display_name || item.id}`;
  };
  loader.src = item.image;
  dialog.showModal();
  document.body.classList.add('modal-open');
}
$('close-dialog').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', (event) => { if (event.target === dialog) {
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
}});
dialog.addEventListener('close', () => {
  imageRequest++;
  document.body.classList.remove('modal-open');
  opener?.focus({preventScroll:true});
});
$('reveal').addEventListener('click', () => {
  const active = $('room').classList.toggle('reveal');
  $('reveal').setAttribute('aria-pressed', String(active));
});
async function init() {
  try {
    const response = await fetch('data/items.json');
    if (!response.ok) throw new Error('Could not load collection');
    const items = await response.json();
    for (const item of items) {
      const name = item.display_name || item.id;
      const hotspot = document.createElement('button');
      hotspot.type = 'button';
      hotspot.className = 'hotspot';
      hotspot.setAttribute('aria-label', `查看 ${name}`);
      hotspot.setAttribute('aria-haspopup', 'dialog');
      hotspot.title = name;
      for (const key of ['left', 'top', 'width', 'height']) hotspot.style[key] = `${item.hotspot[key]}%`;
      hotspot.addEventListener('click', () => openItem(item, hotspot));
      $('hotspots').append(hotspot);
      const li = document.createElement('li');
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'list-item';
      button.setAttribute('aria-haspopup', 'dialog');
      for (const [className, text] of [['number', item.id.slice(-3)], ['item-name', name], ['item-category', item.category]]) {
        const span = document.createElement('span');
        span.className = className;
        span.textContent = text;
        button.append(span);
      }
      button.addEventListener('click', () => openItem(item, button));
      li.append(button);
      $('item-list').append(li);
    }
    $('item-count').textContent = `${String(items.length).padStart(2, '0')} OBJECTS / COLLECTION INDEX`;
  } catch (error) {
    $('data-status').textContent = '收藏清单加载失败，请刷新重试。请通过本地静态服务器打开此页面。';
    console.error(error);
  }
}
init();
