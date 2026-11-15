'use strict';
const $ = (id) => document.getElementById(id);
const dialog = $('item-dialog');
let opener = null;
let imageRequest = 0;
const roomImage = $('room-image');
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const HANG = { spring: 8, damping: 1.1, max: 10, push: 0.2 };
const SKEW = { spring: 40, damping: 7, lag: 0.22, max: 2 };
const FLIP_MS = 650;
const FLIP_EASE = 'cubic-bezier(0.22, 1, 0.36, 1)';
const layersById = new Map();
let flipState = 'idle';
let closeQueued = false;
let originSnapshot = null;
let currentItem = null;
function roomReady() {
  $('room').setAttribute('aria-busy', 'false');
  $('room-status').hidden = roomImage.naturalWidth > 0;
  if (!roomImage.naturalWidth) $('room-status').textContent = '房间图片暂时无法显示，请使用下方收藏清单。';
}
roomImage.addEventListener('load', roomReady);
roomImage.addEventListener('error', roomReady);
if (roomImage.complete) roomReady();
function invertTransform(from, to) {
  const sx = to.width ? from.width / to.width : 1;
  const sy = to.height ? from.height / to.height : 1;
  return `translate(${(from.left - to.left).toFixed(2)}px, ${(from.top - to.top).toFixed(2)}px) scale(${sx.toFixed(4)}, ${sy.toFixed(4)})`;
}
function originRectFor(trigger, item) {
  if (trigger?.classList.contains('list-item')) return trigger.getBoundingClientRect();
  const scope = trigger?.closest('#zone-view') || $('hub-view');
  const layer = item?.id ? scope.querySelector(`[data-layer-item-id="${item.id}"] .layer`) || layersById.get(item.id) : null;
  if (layer) return layer.getBoundingClientRect();
  return trigger?.getBoundingClientRect() ?? null;
}
function clearFlipStyles() {
  dialog.style.transition = 'none';
  dialog.style.transform = '';
  dialog.style.opacity = '';
}
function waitForFlip(result, callback) {
  let done = false;
  const finish = () => {
    if (done) return;
    done = true;
    window.clearTimeout(timer);
    callback();
  };
  const timer = window.setTimeout(finish, FLIP_MS + 80);
  if (result && typeof result.then === 'function') {
    result.then(finish, finish);
    return;
  }
  const onEnd = (event) => {
    if (event.target !== dialog) return;
    if (event.propertyName && event.propertyName !== 'transform' && event.propertyName !== 'opacity') return;
    dialog.removeEventListener('transitionend', onEnd);
    finish();
  };
  dialog.addEventListener('transitionend', onEnd);
}
function playFlip(from, opening) {
  if (reduceMotion.matches) {
    dialog.style.transform = '';
    const start = opening ? 0 : 1;
    const end = opening ? 1 : 0;
    dialog.style.opacity = String(start);
    const anim = dialog.animate(
      [{ opacity: start }, { opacity: end }],
      { duration: FLIP_MS, easing: FLIP_EASE, fill: 'forwards' }
    );
    return anim.finished;
  }
  dialog.style.transformOrigin = 'top left';
  if (opening) {
    dialog.style.transition = 'none';
    const last = dialog.getBoundingClientRect();
    dialog.style.transform = invertTransform(from, last);
    dialog.style.opacity = '0.4';
    dialog.getBoundingClientRect();
    requestAnimationFrame(() => {
      dialog.style.transition = `transform ${FLIP_MS}ms ${FLIP_EASE}, opacity ${FLIP_MS}ms ${FLIP_EASE}`;
      dialog.style.transform = 'none';
      dialog.style.opacity = '1';
    });
    return null;
  }
  const last = dialog.getBoundingClientRect();
  dialog.style.transition = `transform ${FLIP_MS}ms ${FLIP_EASE}, opacity ${FLIP_MS}ms ${FLIP_EASE}`;
  dialog.style.transform = invertTransform(from, last);
  dialog.style.opacity = '0.4';
  return null;
}
function finishOpen() {
  if (!dialog.open) return;
  flipState = 'open';
  if (closeQueued) {
    closeQueued = false;
    requestClose();
  }
}
function openItem(item, trigger) {
  if (flipState === 'opening' || flipState === 'closing') return;
  if (flipState === 'open') return;
  opener = trigger;
  if (item.card_images && location.hash !== `#/item/${item.id}`) history.pushState({ collectionCard: true }, '', `#/item/${item.id}`);
  currentItem = item;
  originSnapshot = originRectFor(trigger, item);
  dialog.dataset.itemId = item.id;
  $('detail-brand').textContent = item.brand || '';
  $('detail-brand').hidden = !item.brand;
  $('detail-id').textContent = `${item.id} / ${item.number}`;
  $('detail-name').textContent = item.display_name || item.id;
  $('detail-category').textContent = categoryLabel(item.category);
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
    photo.sizes = '(max-width: 650px) 90vw, 430px';
    photo.srcset = item.card_images ? srcset(item.card_images) : `${item.image} 800w`;
    photo.src = item.card_images ? item.card_images[1].src : item.image;
    photo.hidden = false;
    status.hidden = true;
  };
  loader.onerror = () => {
    if (request !== imageRequest) return;
    status.textContent = `照片暂时无法显示 · ${item.display_name || item.id}`;
  };
  loader.sizes = '(max-width: 650px) 90vw, 430px';
  loader.srcset = item.card_images ? srcset(item.card_images) : `${item.image} 800w`;
  loader.src = item.card_images ? item.card_images[1].src : item.image;
  flipState = 'opening';
  closeQueued = false;
  dialog.showModal();
  document.body.classList.add('modal-open');
  dialog.classList.add('is-open');
  const origin = originSnapshot;
  if (!origin || origin.width < 2 || origin.height < 2) {
    clearFlipStyles();
    finishOpen();
    return;
  }
  waitForFlip(playFlip(origin, true), finishOpen);
}
function requestClose() {
  if (flipState === 'idle' || flipState === 'closing') return;
  if (flipState === 'opening') {
    closeQueued = true;
    return;
  }
  flipState = 'closing';
  dialog.classList.remove('is-open');
  const origin = originRectFor(opener, currentItem) || originSnapshot;
  if (reduceMotion.matches || !origin || origin.width < 2) {
    if (reduceMotion.matches) {
      waitForFlip(playFlip(origin || { left: 0, top: 0, width: 1, height: 1 }, false), () => dialog.close());
      return;
    }
    dialog.close();
    return;
  }
  waitForFlip(playFlip(origin, false), () => dialog.close());
}
$('close-dialog').addEventListener('click', (event) => {
  event.preventDefault();
  requestClose();
});
dialog.addEventListener('click', (event) => { if (event.target === dialog) {
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) requestClose();
}});
dialog.addEventListener('cancel', (event) => {
  event.preventDefault();
  requestClose();
});
dialog.addEventListener('close', () => {
  if (!routing && currentItem?.card_images && location.hash === `#/item/${currentItem.id}`) {
    if (history.state?.collectionCard) history.back();
    else history.replaceState(null, '', zoneHash(catalog.zones.find(z => z.id === currentItem.zone)));
  }
  routing = false;
  imageRequest++;
  document.body.classList.remove('modal-open');
  dialog.classList.remove('is-open');
  clearFlipStyles();
  flipState = 'idle';
  closeQueued = false;
  originSnapshot = null;
  currentItem = null;
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
function createLayer(item, container = $('layers')) {
  const wrap = document.createElement('div');
  wrap.className = 'layer-wrap';
  wrap.dataset.layerItemId = item.id;
  if (Number.isFinite(item.layer.zIndex)) wrap.style.zIndex = item.layer.zIndex;
  for (const key of ['left', 'top', 'width', 'height']) wrap.style[key] = `${item.layer[key]}%`;
  const layer = document.createElement('img');
  layer.className = 'layer';
  layer.src = item.layer.image;
  layer.alt = '';
  layer.draggable = false;
  layer.style.transformOrigin = `${item.layer.pivotX}% ${item.layer.pivotY}%`;
  if (item.layer.motion !== 'hang') layer.classList.add('layer-rest');
  wrap.append(layer);
  container.append(wrap);
  if (item.layer.fallback) wrap.classList.add('layer-fallback');
  layer.srcset = `${item.layer.image} 1x`;
  layer.sizes = '(max-width: 700px) 100vw, 92vw';
  layer._wrap = wrap;
  layersById.set(item.id, layer);
  return layer;
}
function setLayerGlow(layer, glowing, focused) {
  const wrap = layer?._wrap;
  if (!wrap) return;
  wrap.classList.toggle('is-glow', glowing || focused);
  wrap.classList.toggle('is-focus', !!focused);
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
    state.vel -= dx * (layer._wrap?.classList.contains('layer-fallback') ? 0.25 : 1) * HANG.push * Math.min(1, 0.3 + speed / 900);
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
function setupMotion(entries, surface = $('room')) {
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
  const syncGlow = (entry) => {
    const focused = entry.hotspot.matches(':focus-visible');
    const hovered = entry.hotspot.matches(':hover');
    setLayerGlow(entry.layer, hovered || focused, focused);
  };
  for (const entry of entries) {
    if (!entry.layer) continue;
    entry.hotspot.addEventListener('pointerenter', () => syncGlow(entry));
    entry.hotspot.addEventListener('pointerleave', () => syncGlow(entry));
    entry.hotspot.addEventListener('focus', () => syncGlow(entry));
    entry.hotspot.addEventListener('blur', () => syncGlow(entry));
  }
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
  surface.addEventListener('pointermove', (event) => {
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
  surface.addEventListener('pointerleave', () => { last = null; });
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
        hotspot.setAttribute('aria-label', name);
        hotspot.dataset.itemId = item.id;
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
let routing = false;
let catalog = null;
let activeZone = null;
const srcset = (images) => [...new Map(images.map(i => [i.width, i])).values()].map(i => `${i.src} ${i.width}w`).join(', ');
const categoryLabel = (id) => catalog?.categories.find(c => c.id === id)?.label || id;
function makeButton(text, action, className = '') {
  const b = document.createElement('button'); b.type = 'button'; b.textContent = text; b.className = className; b.addEventListener('click', action); return b;
}
function zoneHash(z) { return `#/${z.category}/${z.letter}`; }
function renderZone(zone) {
  activeZone = zone.id;
  const view = $('zone-view'); view.replaceChildren();
  const nav = document.createElement('nav'); nav.className = 'zone-nav'; nav.setAttribute('aria-label', '分室导航');
  const home = document.createElement('a'); home.href = '#/'; home.textContent = '收藏房间 /'; nav.append(home);
  const title = document.createElement('h1'); title.textContent = `${categoryLabel(zone.category)} · ${zone.letter.toUpperCase()}`; nav.append(title); view.append(nav);
  const tabs = document.createElement('nav'); tabs.className = 'zone-tabs'; tabs.setAttribute('aria-label', '区域');
  const siblings = catalog.zones.filter(z => z.category === zone.category); const index = siblings.indexOf(zone);
  const prev = makeButton('← 上一区', () => { location.hash = zoneHash(siblings[index - 1]); }); prev.disabled = index === 0; tabs.append(prev);
  for (const z of siblings) { const a = document.createElement('a'); a.href = zoneHash(z); a.textContent = z.letter.toUpperCase(); if (z === zone) a.setAttribute('aria-current', 'page'); tabs.append(a); }
  const next = makeButton('下一区 →', () => { location.hash = zoneHash(siblings[index + 1]); }); next.disabled = index === siblings.length - 1; tabs.append(next); view.append(tabs);
  const room = document.createElement('div'); room.className = 'room zone-room'; room.style.aspectRatio = String(zone.aspect); room.dataset.zoneId = zone.id;
  const image = document.createElement('img'); image.src = zone.images[1].src; image.srcset = srcset(zone.images); image.sizes = '(max-width: 700px) 100vw, min(1360px, 92vw)'; image.width = zone.natural_size[0]; image.height = zone.natural_size[1]; image.alt = `${categoryLabel(zone.category)} · ${zone.letter.toUpperCase()}`; room.append(image);
  const layers = document.createElement('div'); layers.className = 'zone-layers'; layers.setAttribute('aria-hidden','true'); room.append(layers);
  const hotspots = document.createElement('div'); hotspots.className = 'zone-hotspots'; room.append(hotspots); view.append(room);
  const note = document.createElement('p'); note.className = 'list-note'; note.textContent = zone.has_layout ? '点击物件，或从清单打开收藏卡片。' : '物件位置待补充，请从清单打开收藏卡片。'; view.append(note);
  const list = document.createElement('ul'); list.className = 'zone-item-list'; view.append(list);
  const entries = [];
  for (const id of zone.items) {
    const item = catalog.items.find(i => i.id === id); let hotspot = null;
    const layer = item.layer ? createLayer(item, layers) : null;
    if (item.hotspot) {
      hotspot = makeButton('', () => openItem(item, hotspot), 'hotspot'); hotspot.dataset.itemId = id; hotspot.setAttribute('aria-label', item.name); hotspot.setAttribute('aria-haspopup','dialog');
      for (const key of ['left','top','width','height']) hotspot.style[key] = `${item.hotspot[key]}%`; hotspots.append(hotspot);
    }
    const li = document.createElement('li'); const b = makeButton(`${item.number}  ${item.name}`, () => openItem(item,b), 'list-item'); b.dataset.itemId = id; b.setAttribute('aria-haspopup','dialog'); li.append(b); list.append(li);
    if (layer && hotspot) entries.push({item,layer,hotspot});
  }
  setupMotion(entries, room);
}
function route() {
  if (!catalog) return;
  if (dialog.open) { routing = true; dialog.close(); }
  const parts = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean);
  const item = parts[0] === 'item' ? catalog.items.find(i => i.id === parts[1]) : null;
  const zone = item ? catalog.zones.find(z => z.id === item.zone) : catalog.zones.find(z => z.category === parts[0] && (!parts[1] || z.letter === parts[1] || z.id === parts[1]));
  const fromHub = !$('hub-view').hidden;
  $('hub-view').hidden = !!zone; $('zone-view').hidden = !zone;
  if (zone) {
    if (activeZone !== zone.id) { renderZone(zone); window.scrollTo(0, 0); }
    if (fromHub) window.scrollTo(0, 0);
    if (item) requestAnimationFrame(() => openItem(item, $('zone-view').querySelector(`button[data-item-id="${item.id}"]`)));
  } else window.scrollTo(0, 0);
}
async function initCatalog() {
  try {
    const response = await fetch('data/catalog.json'); if (!response.ok) throw new Error('Catalog unavailable'); catalog = await response.json();
    for (const c of catalog.categories) {
      const z = catalog.zones.find(z => z.id === c.cover_zone); const a = document.createElement('a'); a.className = 'room-tile'; a.href = `#/${c.id}`;
      const cover = z.cover || z.images[2];
      const img = document.createElement('img'); img.loading = 'lazy'; img.src = cover.src; img.srcset = `${cover.src} ${cover.width}w`; img.sizes = '(max-width: 650px) 45vw, 300px'; img.width = z.images[2].width; img.height = z.images[2].height; img.alt = '';
      const label = document.createElement('span'); label.textContent = `${c.label} / ${c.label_en}`; const count = document.createElement('small'); count.textContent = `${c.count} 件`; a.append(img,label,count); $('room-tiles').append(a);
    }
    route();
  } catch (error) { $('data-status').textContent = '分室清单加载失败，请刷新重试。'; console.error(error); }
}
window.addEventListener('hashchange', route);
init(); initCatalog();
