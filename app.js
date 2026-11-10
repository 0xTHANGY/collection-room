'use strict';
const $ = (id) => document.getElementById(id);
const dialog = $('item-dialog');
let opener = null;
let imageRequest = 0;
const roomImage = $('room-image');
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const HANG = { spring: 8, damping: 1.1, max: 10, push: 0.2 };
const SKEW = { spring: 40, damping: 7, lag: 0.22, max: 2 };
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
  $('reveal').textContent = active ? '隐藏收藏位置' : '显示收藏位置';
});
$('collection').addEventListener('toggle', () => {
  document.querySelector('.summary-action').textContent = $('collection').open ? '收起' : '展开';
});
function isSafePath(value) {
  return typeof value === 'string' && value && !/^(?:[/\\]|[a-z][a-z\d+.-]*:)/i.test(value) && !value.includes(':') && !value.includes('\\') && !value.split('/').includes('..');
}
function validateRect(rect, label) {
  for (const key of ['left', 'top', 'width', 'height']) {
    const value = rect?.[key];
    if (!Number.isFinite(value) || value < 0 || value > 100 || (['width', 'height'].includes(key) && value === 0)) {
      throw new Error(`Invalid ${label} ${key}`);
    }
  }
}
function validateItem(item) {
  if (!isSafePath(item.image)) throw new Error('Image must be a same-site relative path');
  validateRect(item.hotspot, 'hotspot');
  if (typeof item.number !== 'string' || !/^\d{3}$/.test(item.number)) throw new Error('Invalid item number');
  if (item.layer) {
    if (!isSafePath(item.layer.image)) throw new Error('Layer image must be a same-site relative path');
    validateRect(item.layer, 'layer');
    if (!Number.isFinite(item.layer.pivotX) || !Number.isFinite(item.layer.pivotY)) throw new Error('Invalid layer pivot');
    if (item.layer.motion !== 'hang' && item.layer.motion !== 'rest') throw new Error('Invalid layer motion');
  }
}
function createLayer(item) {
  const layer = document.createElement('img');
  layer.className = 'layer';
  layer.src = item.layer.image;
  layer.alt = '';
  layer.draggable = false;
  for (const key of ['left', 'top', 'width', 'height']) layer.style[key] = `${item.layer[key]}%`;
  layer.style.transformOrigin = `${item.layer.pivotX}% ${item.layer.pivotY}%`;
  if (item.layer.motion === 'rest') layer.classList.add('layer-rest');
  $('layers').append(layer);
  return layer;
}
function segmentHitsRect(x0, y0, x1, y1, rect, pad) {
  const left = rect.left - pad, right = rect.right + pad, top = rect.top - pad, bottom = rect.bottom + pad;
  const dx = x1 - x0, dy = y1 - y0;
  let tMin = 0, tMax = 1;
  const clips = [[-dx, x0 - left], [dx, right - x0], [-dy, y0 - top], [dy, bottom - y0]];
  for (const [p, q] of clips) {
    if (p === 0) {
      if (q < 0) return false;
      continue;
    }
    const t = q / p;
    if (p < 0) {
      if (t > tMax) return false;
      tMin = Math.max(tMin, t);
    } else {
      if (t < tMin) return false;
      tMax = Math.min(tMax, t);
    }
  }
  return tMin <= tMax;
}
function attachHang(layer) {
  const state = { angle: 0, vel: 0, skew: 0, skewVel: 0, raf: 0, time: 0, pressed: false };
  const apply = () => {
    if (reduceMotion.matches) {
      layer.style.transform = '';
      return;
    }
    const settled = Math.abs(state.angle) < 0.02 && Math.abs(state.vel) < 0.05 && Math.abs(state.skew) < 0.02 && !state.pressed;
    if (settled) {
      layer.style.transform = '';
      state.angle = state.vel = state.skew = state.skewVel = 0;
      return false;
    }
    const scale = state.pressed ? 0.98 : 1;
    const skew = Math.max(-SKEW.max, Math.min(SKEW.max, state.skew));
    layer.style.transform = `rotate(${state.angle.toFixed(3)}deg) skewX(${skew.toFixed(3)}deg)${scale === 1 ? '' : ` scale(${scale})`}`;
    return true;
  };
  const step = (now) => {
    const dt = Math.min((now - state.time) / 1000, 1 / 30);
    state.time = now;
    state.vel += (-HANG.spring * state.angle - HANG.damping * state.vel) * dt;
    state.angle += state.vel * dt;
    state.angle = Math.max(-HANG.max, Math.min(HANG.max, state.angle));
    const target = Math.max(-SKEW.max, Math.min(SKEW.max, state.vel * SKEW.lag));
    state.skewVel += (SKEW.spring * (target - state.skew) - SKEW.damping * state.skewVel) * dt;
    state.skew += state.skewVel * dt;
    if (apply()) state.raf = requestAnimationFrame(step);
    else state.raf = 0;
  };
  const kick = () => {
    if (reduceMotion.matches) return;
    if (!state.raf) {
      state.time = performance.now();
      state.raf = requestAnimationFrame(step);
    }
  };
  const impulse = (dx, dtMs) => {
    if (reduceMotion.matches) return;
    const speed = Math.abs(dx) / Math.max(dtMs / 1000, 0.001);
    state.vel -= dx * HANG.push * Math.min(1, 0.3 + speed / 900);
    state.vel = Math.max(-HANG.max, Math.min(HANG.max, state.vel));
    kick();
  };
  const setPressed = (pressed) => {
    state.pressed = pressed;
    if (!apply()) return;
    kick();
  };
  return { layer, state, impulse, setPressed };
}
function setupMotion(entries) {
  const hang = [];
  const rest = [];
  for (const entry of entries) {
    if (!entry.layer || !entry.item.layer) continue;
    if (entry.item.layer.motion === 'hang') hang.push({ ...entry, spring: attachHang(entry.layer) });
    else rest.push(entry);
  }
  const setRest = (entry, className, on) => {
    if (reduceMotion.matches) return;
    entry.layer.classList.toggle(className, on);
  };
  for (const entry of rest) {
    entry.hotspot.addEventListener('pointerenter', () => setRest(entry, 'is-hover', true));
    entry.hotspot.addEventListener('pointerleave', () => {
      setRest(entry, 'is-hover', false);
      setRest(entry, 'is-press', false);
    });
    entry.hotspot.addEventListener('pointerdown', () => setRest(entry, 'is-press', true));
    entry.hotspot.addEventListener('pointerup', () => setRest(entry, 'is-press', false));
    entry.hotspot.addEventListener('pointercancel', () => setRest(entry, 'is-press', false));
  }
  for (const entry of hang) {
    entry.hotspot.addEventListener('pointerdown', () => entry.spring.setPressed(true));
    entry.hotspot.addEventListener('pointerup', () => entry.spring.setPressed(false));
    entry.hotspot.addEventListener('pointerleave', () => entry.spring.setPressed(false));
    entry.hotspot.addEventListener('pointercancel', () => entry.spring.setPressed(false));
  }
  let last = null;
  $('room').addEventListener('pointermove', (event) => {
    if (event.pointerType === 'touch' || reduceMotion.matches) {
      last = { x: event.clientX, y: event.clientY, t: event.timeStamp };
      return;
    }
    const prev = last;
    last = { x: event.clientX, y: event.clientY, t: event.timeStamp };
    if (!prev || event.timeStamp - prev.t > 150) return;
    for (const entry of hang) {
      const rect = entry.layer.getBoundingClientRect();
      if (!segmentHitsRect(prev.x, prev.y, last.x, last.y, rect, 6)) continue;
      entry.spring.impulse(last.x - prev.x, event.timeStamp - prev.t);
    }
  }, { passive: true });
  $('room').addEventListener('pointerleave', () => { last = null; });
}
async function init() {
  try {
    const response = await fetch('data/items.json');
    if (!response.ok) throw new Error('Could not load collection');
    const items = await response.json();
    let loaded = 0;
    const motionEntries = [];
    for (const item of items) {
      try {
        validateItem(item);
        const name = item.display_name || item.id;
        const layer = item.layer ? createLayer(item) : null;
        const hotspot = document.createElement('button');
        hotspot.type = 'button';
        hotspot.className = 'hotspot';
        hotspot.setAttribute('aria-label', `查看 ${name}`);
        hotspot.setAttribute('aria-haspopup', 'dialog');
        hotspot.title = name;
        for (const key of ['left', 'top', 'width', 'height']) hotspot.style[key] = `${item.hotspot[key]}%`;
        hotspot.addEventListener('click', () => openItem(item, hotspot));
        const li = document.createElement('li');
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'list-item';
        button.setAttribute('aria-haspopup', 'dialog');
        for (const [className, text] of [['number', item.number], ['item-name', name], ['item-category', item.category]]) {
          const span = document.createElement('span');
          span.className = className;
          if (className === 'item-name' || className === 'item-category') span.lang = 'en';
          span.textContent = text;
          button.append(span);
        }
        button.addEventListener('click', () => openItem(item, button));
        li.append(button);
        $('hotspots').append(hotspot);
        $('item-list').append(li);
        if (layer) motionEntries.push({ item, layer, hotspot });
        loaded++;
      } catch (error) {
        console.error(`Skipping collection item ${item?.id ?? '(unknown)'}`, error);
      }
    }
    $('item-count').textContent = `${String(loaded).padStart(2, '0')} OBJECTS / COLLECTION INDEX`;
    setupMotion(motionEntries);
  } catch (error) {
    $('data-status').textContent = '收藏清单加载失败，请刷新重试。请通过本地静态服务器打开此页面。';
    console.error(error);
  }
}
init();
