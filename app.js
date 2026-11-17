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
let closeQueued = false;  // kept for diagnostics only: a close during opening now interrupts instead of queueing
let flipAnim = null;      // the one running card flight (Web Animations)
let flipTimer = 0;        // wall-clock fallback that ends that flight
let ghost = null;         // visual stand-in that flies; the real dialog never moves
let veil = null;          // stand-in for the dialog backdrop while a closed card flies home
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
  stopFlip();
  dialog.style.transition = 'none';
  dialog.style.transform = '';
  dialog.style.opacity = '';
  dialog.style.willChange = '';
}
// v2.3.2 close fix. Before, the dialog itself flew (CSS transitions) and closing was a chain of deferred steps:
// a close asked for while the card was opening only set closeQueued and waited for the opening flight to end
// (transitionend or a 730 ms timer), then the closing flight ran and dialog.close() came from another
// transitionend/timer. When the page is starved (heavy parallel load) frames and timers are delayed by seconds
// while input still arrives, so after Esc / ✕ the dialog stayed open for seconds. Stray transitionend events
// of one flight could also end the next one, and the opening transition started in an unguarded rAF.
// Now the real dialog never moves: Esc / ✕ close it synchronously inside the input handler, and a ghost
// copy in the top layer plays the same flight (same keyframes, 650 ms, same easing). The ghost never takes
// pointer events, so the ✕ is clickable and stable from the first frame.
function removeGhost() {
  for (const el of [ghost, veil]) {
    if (!el) continue;
    try { el.hidePopover(); } catch (e) { /* not shown as popover */ }
    el.remove();
  }
  ghost = veil = null;
}
function stopFlip() {
  if (flipTimer) { window.clearTimeout(flipTimer); flipTimer = 0; }
  const anim = flipAnim;
  flipAnim = null;
  anim?.cancel();
  removeGhost();
}
function showOnTop(el) {
  el.setAttribute('aria-hidden', 'true');
  el.inert = true;
  el.popover = 'manual';
  document.body.append(el);
  try { el.showPopover(); } catch (e) { /* falls back to the fixed z-index layer */ }
}
function makeGhost(rect) {
  const g = document.createElement('div');
  g.className = 'card-ghost';
  const cs = getComputedStyle(dialog);
  for (const p of ['display', 'gridTemplateColumns', 'boxSizing', 'padding', 'borderTop', 'borderRight', 'borderBottom', 'borderLeft', 'borderRadius', 'background', 'color', 'boxShadow', 'font', 'lineHeight', 'letterSpacing']) g.style[p] = cs[p];
  Object.assign(g.style, {
    position: 'fixed', inset: 'auto', left: `${rect.left}px`, top: `${rect.top}px`, width: `${rect.width}px`, height: `${rect.height}px`,
    margin: '0', maxWidth: 'none', maxHeight: 'none', overflow: 'hidden', pointerEvents: 'none', transformOrigin: 'top left', zIndex: '2147483647'
  });
  for (const child of dialog.children) g.append(child.cloneNode(true));
  const brand = $('detail-brand');
  g.querySelectorAll('[id]').forEach((el) => {
    if (el.id === 'detail-brand') { el.style.fontSize = getComputedStyle(brand).fontSize; el.style.color = getComputedStyle(brand).color; }
    el.removeAttribute('id');
  });
  const closeButton = $('close-dialog');
  const ghostClose = g.querySelector('.close');
  if (ghostClose && closeButton.matches(':focus-visible')) {
    const fc = getComputedStyle(closeButton);
    Object.assign(ghostClose.style, { outline: fc.outline, outlineOffset: fc.outlineOffset });
  }
  g.querySelectorAll('[autofocus]').forEach((el) => el.removeAttribute('autofocus'));
  g.querySelectorAll('img').forEach((img) => { img.loading = 'eager'; });
  g.scrollTop = dialog.scrollTop;
  showOnTop(g);
  return g;
}
function fitGhost() {
  // The real card grows when its photo arrives; the flying copy follows, as the flying dialog used to.
  if (!ghost || flipState !== 'opening') return;
  const r = dialog.getBoundingClientRect();
  Object.assign(ghost.style, { left: `${r.left}px`, top: `${r.top}px`, width: `${r.width}px`, height: `${r.height}px` });
}
$('detail-image').addEventListener('load', fitGhost);
function syncGhostPhoto() {
  if (!ghost) return;
  const photo = $('detail-image');
  const img = ghost.querySelector('.detail-photo img');
  const status = ghost.querySelector('.detail-photo p');
  if (img) { img.sizes = photo.sizes; img.srcset = photo.srcset; img.src = photo.src; img.hidden = photo.hidden; }
  if (status) { status.hidden = $('photo-status').hidden; status.textContent = $('photo-status').textContent; }
  fitGhost();
}
function runFlip(target, keyframes, duration, done) {
  if (flipTimer) { window.clearTimeout(flipTimer); flipTimer = 0; }
  flipAnim?.cancel();
  const anim = target.animate(keyframes, { duration, easing: FLIP_EASE, fill: 'forwards' });
  flipAnim = anim;
  let finished = false;
  const finish = () => {
    if (finished || flipAnim !== anim) return;
    finished = true;
    if (flipTimer) { window.clearTimeout(flipTimer); flipTimer = 0; }
    done();
  };
  anim.finished.then(finish, () => {});  // a cancelled flight was superseded: nothing to do
  flipTimer = window.setTimeout(finish, duration + 80);
}
function openFlip(from) {
  const last = dialog.getBoundingClientRect();
  ghost = makeGhost(last);
  // v2.3.2 1px fix: until v2.3.1 the real dialog itself was animated, so it was rasterised as its own compositor
  // layer from its first frame (reduced motion: a filled opacity animation that stays on it; normal motion: the
  // transform/opacity transition, dropped when it ends). Text at fractional y (tag, brand, rule, story) snaps to
  // pixels differently with and without that layer, which moved the right column 1px. Keep the same layer state.
  dialog.style.willChange = reduceMotion.matches ? 'opacity' : 'transform, opacity';
  dialog.style.opacity = '0';  // the real card waits, in place and clickable, under its flying copy
  const frames = reduceMotion.matches
    ? [{ opacity: 0 }, { opacity: 1 }]
    : [{ transform: invertTransform(from, last), opacity: 0.4 }, { transform: 'none', opacity: 1 }];
  runFlip(ghost, frames, FLIP_MS, finishOpen);
}
function finishOpen() {
  if (!dialog.open || flipState !== 'opening') return;
  flipState = 'open';
  closeQueued = false;
  dialog.style.opacity = '';
  stopFlip();
  if (!reduceMotion.matches) dialog.style.willChange = '';  // as the old transition did when it ended
}
function openItem(item, trigger) {
  // v2.3 keyboard fix: dialog.close() dispatches 'close' asynchronously. A key press that lands
  // between the two used to see flipState === 'closing' and was ignored. Settle the close first.
  if (flipState !== 'idle' && !dialog.open) settleClose(false);  // closed elsewhere; its 'close' event is still queued
  if (flipState !== 'idle') return;
  stopFlip();  // a previous card may still be flying home
  opener = trigger;
  // v2.3 history fix: a card never adds a history entry and never calls history.back(), so no async
  // navigation can race a following zone switch. On a zone page the card replaces the zone URL with its
  // own (#/item/<id>, shareable); closing restores the zone URL. The hub keeps its URL unchanged.
  cardReturnHash = null;
  if (item.card_images && !$('zone-view').hidden) {
    const z = catalog.zones.find(z => z.id === item.zone);
    cardReturnHash = location.hash && location.hash !== `#/item/${item.id}` ? location.hash : zoneHash(z);
    // The entry is tagged so a later Forward onto it shows the zone only (v2.3.1).
    history.replaceState({ collectionCard: true }, '', `#/item/${item.id}`);
  }
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
    syncGhostPhoto();
  };
  loader.onerror = () => {
    if (request !== imageRequest) return;
    status.textContent = `照片暂时无法显示 · ${item.display_name || item.id}`;
    syncGhostPhoto();
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
  ++flipGen;
  openFlip(origin);
}
function requestClose() {
  if (flipState === 'idle' || flipState === 'closing') return;
  if (!dialog.open) { settleClose(true); return; }
  const last = dialog.getBoundingClientRect();  // the real dialog is never transformed
  const origin = originRectFor(opener, currentItem) || originSnapshot;
  const backdrop = Number(getComputedStyle(dialog, '::backdrop').opacity);
  // Build the flight first (it needs the open dialog's layout), then close for real, synchronously.
  let from = { transform: 'none', opacity: 1 };
  let duration = FLIP_MS;
  let g = null;
  if (flipState === 'opening' && ghost && flipAnim) {
    // Opening is interrupted, not waited for: fly back from where the copy is now, as fast as it came.
    const cs = getComputedStyle(ghost);
    from = { transform: cs.transform, opacity: Number(cs.opacity) };
    duration = Math.max(0, Math.min(FLIP_MS, Number(flipAnim.currentTime) || 0));
    flipAnim.cancel();
    flipAnim = null;
    g = ghost;
    ghost = null;
  } else {
    stopFlip();
    g = makeGhost(last);
  }
  flipState = 'closing';
  closeQueued = false;
  dialog.classList.remove('is-open');
  dialog.close();
  settleClose(true);  // URL, focus and state are restored now; the queued 'close' event finds nothing to do
  const animate = duration >= 17 && (reduceMotion.matches || (origin && origin.width >= 2 && origin.height >= 2));
  if (!animate) {
    try { g.hidePopover(); } catch (e) { /* not shown as popover */ }
    g.remove();
    return;
  }
  if (!reduceMotion.matches && backdrop > 0) {
    veil = document.createElement('div');
    veil.className = 'card-veil';
    Object.assign(veil.style, { position: 'fixed', inset: '0', width: 'auto', height: 'auto', margin: '0', padding: '0', border: '0', background: '#25271f80', backdropFilter: 'blur(4px)', pointerEvents: 'none', zIndex: '2147483646' });
    showOnTop(veil);
    veil.animate([{ opacity: backdrop }, { opacity: 0 }], { duration, easing: FLIP_EASE, fill: 'forwards' });
    if (g.matches(':popover-open')) { g.hidePopover(); g.showPopover(); }  // keep the card above its veil
  }
  ghost = g;
  const frames = reduceMotion.matches
    ? [{ opacity: from.opacity }, { opacity: 0 }]
    : [from, { transform: invertTransform(origin, last), opacity: 0.4 }];
  runFlip(g, frames, duration, removeGhost);
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
let cardReturnHash = null, flipGen = 0;
function settleClose(restoreFocus = true) {
  if (flipState === 'idle' && !currentItem) return;  // already settled (idempotent)
  flipGen++;  // invalidate any pending open/close animation callbacks
  // Synchronous URL restore (no history.back()): the zone URL comes back before any later click runs.
  if (!routing && cardReturnHash && currentItem && location.hash === `#/item/${currentItem.id}`) history.replaceState(null, '', cardReturnHash);
  cardReturnHash = null;
  routing = false;
  imageRequest++;
  document.body.classList.remove('modal-open');
  dialog.classList.remove('is-open');
  clearFlipStyles();
  flipState = 'idle';
  closeQueued = false;
  originSnapshot = null;
  currentItem = null;
  // Only return focus when it is not already somewhere the user chose (never steal a re-focus).
  const active = document.activeElement;
  if (restoreFocus && (!active || active === document.body || dialog.contains(active))) opener?.focus({preventScroll:true});
}
// A 'close' event that arrives after the card was already settled and a new card opened is stale.
dialog.addEventListener('close', () => { if (!dialog.open) settleClose(true); });
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
  const image = document.createElement('img'); image.src = (zone.images.find(variant => variant.width === 1024) || zone.images[0]).src; image.srcset = srcset(zone.images); image.sizes = '(max-width: 700px) 100vw, min(1360px, 92vw)'; image.width = zone.natural_size[0]; image.height = zone.natural_size[1]; image.alt = `${categoryLabel(zone.category)} · ${zone.letter.toUpperCase()}`; room.append(image);
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
function route(event) {
  if (!catalog) return;
  // A card opened right after our own back-navigation already owns this URL: nothing to route.
  if (dialog.open && currentItem && location.hash === `#/item/${currentItem.id}`) return;
  // Any real navigation (Back/Forward, zone tabs, links) closes an open or closing card immediately.
  if (dialog.open || flipState !== 'idle') { routing = true; if (dialog.open) dialog.close(); settleClose(false); }
  if (ghost || veil) stopFlip();  // a card still flying home does not follow us to another page
  const parts = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean);
  let item = parts[0] === 'item' ? catalog.items.find(i => i.id === parts[1]) : null;
  // Back/Forward onto an entry whose card was opened in-page (tagged state) returns to the zone only:
  // the card is not reopened and the entry is rewritten to the zone URL. Typed/shared #/item links
  // and a reload still open the card (no tag, or not a navigation event).
  const zone = item ? catalog.zones.find(z => z.id === item.zone) : catalog.zones.find(z => z.category === parts[0] && (!parts[1] || z.letter === parts[1] || z.id === parts[1]));
  if (item && zone && event && history.state?.collectionCard) {
    history.replaceState(null, '', zoneHash(zone));
    item = null;
  }
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
      const cover = z.cover || z.images[z.images.length - 1];
      const img = document.createElement('img'); img.loading = 'lazy'; img.src = cover.src; img.srcset = `${cover.src} ${cover.width}w`; img.sizes = '(max-width: 650px) 45vw, 300px'; img.width = cover.width; img.height = cover.height; img.alt = '';
      const label = document.createElement('span'); label.textContent = `${c.label} / ${c.label_en}`; const count = document.createElement('small'); count.textContent = `${c.count} 件`; a.append(img,label,count); $('room-tiles').append(a);
    }
    route();
  } catch (error) { $('data-status').textContent = '分室清单加载失败，请刷新重试。'; console.error(error); }
}
window.addEventListener('hashchange', route);
init(); initCatalog();
