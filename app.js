'use strict';
const $ = (id) => document.getElementById(id);
const dialog = $('item-dialog');
let opener = null;
let imageRequest = 0;
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
  g.querySelector('.detail-copy').scrollTop = dialog.querySelector('.detail-copy').scrollTop;
  return g;
}
function fitGhost() {
  // Keep the flying copy aligned if the real card resizes.
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
  if (item.zone && !$('zone-view').hidden) {
    const z = catalog.zones.find(z => z.id === item.zone);
    cardReturnHash = location.hash && location.hash !== `#/item/${item.id}` ? location.hash : zoneHash(z);
    // The entry is tagged so a later Forward onto it shows the zone only (v2.3.1).
    history.replaceState({ collectionCard: true }, '', `#/item/${item.id}`);
  }
  currentItem = item;
  originSnapshot = originRectFor(trigger, item);
  dialog.dataset.itemId = item.id;
  const brandText = (!item.brand || item.brand === 'Unknown') ? '品牌待补充' : item.brand;
  $('detail-brand').textContent = brandText;
  $('detail-brand').hidden = false;
  $('detail-brand').lang = (!item.brand || item.brand === 'Unknown') ? 'zh-CN' : 'en';
  $('detail-id').textContent = `${item.id} / ${item.number}`;
  const detailName = item.display_name || item.id;
  $('detail-name').textContent = detailName;
  $('detail-name').lang = hasCJK(detailName) ? 'zh-CN' : 'en';
  $('detail-category').textContent = categoryLabel(item.category);
  $('detail-story').textContent = typeof item.story === 'string' ? item.story : '故事待补充';
  const photo = $('detail-image');
  const status = $('photo-status');
  const request = ++imageRequest;
  photo.hidden = true;
  photo.alt = item.display_name || item.id;
  status.hidden = false;
  const cards = item.card_images;
  const source = cards?.[1]?.src || cards?.[0]?.src || item.image;
  const unavailable = `照片暂时无法显示 · ${item.display_name || item.id}`;
  status.textContent = unavailable;
  if (source) {
    status.textContent = '照片加载中…';
    const loader = new Image();
    loader.onload = () => {
      if (request !== imageRequest) return;
      photo.sizes = loader.sizes;
      photo.srcset = loader.srcset;
      photo.src = withV(source);
      photo.hidden = false;
      status.hidden = true;
      syncGhostPhoto();
    };
    loader.onerror = () => {
      if (request !== imageRequest) return;
      status.textContent = unavailable;
      syncGhostPhoto();
    };
    loader.sizes = '(max-width: 650px) 90vw, 430px';
    loader.srcset = cards?.length ? srcset(cards) : `${withV(source)} 800w`;
    loader.src = withV(source);
  } else {
    syncGhostPhoto();
  }
  flipState = 'opening';
  closeQueued = false;
  const copy = dialog.querySelector('.detail-copy');
  dialog.showModal();
  fitName($('detail-name'), 14);
  copy.scrollTop = 0;
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
const ASSET_V = '20261126g';
function withV(url) {
  if (typeof url !== 'string' || !url) return url;
  if (/[?&]v=/.test(url)) return url;
  return url + (url.includes('?') ? '&' : '?') + 'v=' + ASSET_V;
}
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
  if (item.hotspot.clip !== undefined && !(Array.isArray(item.hotspot.clip) && item.hotspot.clip.length >= 3 && item.hotspot.clip.every((p) => Array.isArray(p) && p.length === 2 && p.every((v) => Number.isFinite(v) && v >= 0 && v <= 100)))) throw new Error('Invalid hotspot clip');
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
  layer.src = withV(item.layer.image);
  layer.alt = '';
  layer.draggable = false;
  layer.style.transformOrigin = `${item.layer.pivotX}% ${item.layer.pivotY}%`;
  if (item.layer.motion !== 'hang') layer.classList.add('layer-rest');
  wrap.append(layer);
  container.append(wrap);
  if (item.layer.fallback) wrap.classList.add('layer-fallback');
  layer.srcset = `${withV(item.layer.image)} 1x`;
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
function setupMotion(entries, surface) {
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
let routing = false;
let catalog = null;
let activeZone = null;
const srcset = (images) => [...new Map(images.map(i => [i.width, i])).values()].map(i => `${withV(i.src)} ${i.width}w`).join(', ');
const categoryLabel = (id) => catalog?.categories.find(c => c.id === id)?.label || id;
function hasCJK(text) {
  return typeof text === 'string' && /[\u3400-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]/.test(text);
}
function itemCategoryLabel(item) {
  if (catalog) {
    const fromCat = catalog.items.find(i => i.id === item.id);
    const id = (fromCat || item).category;
    return categoryLabel(id);
  }
  return item.category || '';
}
const mobileNames = window.matchMedia('(max-width: 650px)');
// Phones: names use at most two lines. Shrink the font one pixel at a time (not below minPx);
// only if that is still not enough, clamp to two lines with an ellipsis. The full name stays in
// the DOM (and in a title) so nothing is rewritten.
function fitName(el, minPx, maxLines = 2) {
  if (!el) return;
  el.style.fontSize = ''; el.classList.remove('name-clamped'); el.removeAttribute('title');
  if (!mobileNames.matches || !el.textContent) return;
  const lines = () => { const cs = getComputedStyle(el); const lh = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.25; return Math.round(el.offsetHeight / lh); };
  let fs = parseFloat(getComputedStyle(el).fontSize);
  while (lines() > maxLines && fs > minPx) { fs = Math.max(minPx, fs - 1); el.style.fontSize = `${fs}px`; }
  if (lines() > maxLines) { el.classList.add('name-clamped'); el.title = el.textContent; }
}
function fitListNames(root) { for (const el of root.querySelectorAll('.list-item .item-name')) fitName(el, 12); }
let fitTimer = 0;
function refitNames() {
  clearTimeout(fitTimer);
  fitTimer = setTimeout(() => { const v = $('zone-view'); if (v && !v.hidden) fitListNames(v); if (dialog.open) fitName($('detail-name'), 14); }, 120);
}
window.addEventListener('resize', refitNames);
mobileNames.addEventListener?.('change', refitNames);
function fillListButton(button, item) {
  button.replaceChildren();
  button.type = 'button';
  button.className = 'list-item';
  button.setAttribute('aria-haspopup', 'dialog');
  button.dataset.itemId = item.id;
  const name = item.display_name || item.name || item.id;
  const rows = [
    ['number', item.number, null],
    ['item-name', name, hasCJK(name) ? 'zh-CN' : 'en'],
    ['item-category', itemCategoryLabel(item), 'zh-CN'],
  ];
  for (const [className, text, lang] of rows) {
    const span = document.createElement('span');
    span.className = className;
    if (lang) span.lang = lang;
    span.textContent = text;
    button.append(span);
  }
}
function makeButton(text, action, className = '') {
  const b = document.createElement('button'); b.type = 'button'; b.textContent = text; b.className = className; b.addEventListener('click', action); return b;
}
function zoneHash(z) { return `#/${z.category}/${z.letter}`; }

function inflateRect(rect, minSize) {
  const w = Math.max(rect.width, minSize);
  const h = Math.max(rect.height, minSize);
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  return { left: cx - w / 2, top: cy - h / 2, right: cx + w / 2, bottom: cy + h / 2 };
}
function pickHotspotAt(clientX, clientY, hotspotEls, minSize = 44) {
  const top = document.elementFromPoint(clientX, clientY);
  const direct = top && top.closest ? top.closest('.hotspot') : null;
  if (direct && hotspotEls.includes(direct)) return direct;
  let best = null, bestD = Infinity;
  for (const el of hotspotEls) {
    const r = el.getBoundingClientRect();
    const box = inflateRect(r, minSize);
    if (clientX < box.left || clientX >= box.right || clientY < box.top || clientY >= box.bottom) continue;
    const dx = clientX < r.left ? r.left - clientX : clientX >= r.right ? clientX - r.right : 0;
    const dy = clientY < r.top ? r.top - clientY : clientY >= r.bottom ? clientY - r.bottom : 0;
    const d = Math.hypot(dx, dy);
    if (d < bestD) { bestD = d; best = el; }
  }
  return best;
}
function renderZone(zone) {
  activeZone = zone.id;
  const view = $('zone-view'); view.replaceChildren();
  view.dataset.category = zone.category;
  const nav = document.createElement('nav'); nav.className = 'zone-nav'; nav.setAttribute('aria-label', '分室导航');
  const home = document.createElement('a'); home.href = '#/'; home.className = 'zone-back';
  const backMark = document.createElement('span'); backMark.setAttribute('aria-hidden', 'true'); backMark.textContent = '←';
  home.append(backMark, document.createTextNode('收藏房间')); nav.append(home);
  const title = document.createElement('h1');
  title.tabIndex = -1;
  title.append(document.createTextNode(`${categoryLabel(zone.category)} · ${zone.letter.toUpperCase()}`));
  const count = document.createElement('small'); count.className = 'zone-count'; count.textContent = `${zone.items.length} 件`;
  title.append(count); nav.append(title); view.append(nav);
  const tabs = document.createElement('nav'); tabs.className = 'zone-tabs'; tabs.setAttribute('aria-label', '区域');
  const siblings = catalog.zones.filter(z => z.category === zone.category); const index = siblings.indexOf(zone);
  const prev = makeButton('← 上一区', () => { location.hash = zoneHash(siblings[index - 1]); }, 'zone-tab-prev'); prev.disabled = index === 0; tabs.append(prev);
  const letters = document.createElement('div'); letters.className = 'zone-tab-letters';
  for (const z of siblings) { const a = document.createElement('a'); a.href = zoneHash(z); a.textContent = z.letter.toUpperCase(); if (z === zone) a.setAttribute('aria-current', 'page'); letters.append(a); }
  tabs.append(letters);
  const next = makeButton('下一区 →', () => { location.hash = zoneHash(siblings[index + 1]); }, 'zone-tab-next'); next.disabled = index === siblings.length - 1; tabs.append(next); view.append(tabs);
  view.dataset.zoneId = zone.id;
  const painting = zone.images?.[1] || zone.images?.[0];
  const hasLayout = painting && zone.has_layout;
  const note = document.createElement('p'); note.className = 'list-note'; note.textContent = hasLayout ? '点击物件，或从清单打开收藏卡片。' : '物件位置待补充，请从清单打开收藏卡片。';
  view.append(note);
  let room, layers, hotspots;
  if (painting) {
    room = document.createElement('div'); room.className = 'room zone-room'; room.style.setProperty('--room-aspect', zone.aspect);
    const stage = document.createElement('div'); stage.className = 'room-stage';
    const image = document.createElement('img'); image.src = withV(painting.src); image.srcset = srcset(zone.images); image.sizes = '(max-width: 1043px) 92vw, 960px'; image.width = zone.natural_size[0]; image.height = zone.natural_size[1]; image.alt = `${categoryLabel(zone.category)} · ${zone.letter.toUpperCase()}`; stage.append(image);
    layers = document.createElement('div'); layers.className = 'zone-layers'; layers.setAttribute('aria-hidden','true'); stage.append(layers);
    hotspots = document.createElement('div'); hotspots.className = 'zone-hotspots'; stage.append(hotspots);
    room.append(stage); view.append(room);
    room.addEventListener('click', (event) => {
      if (event.target.closest('.hotspot')) return;
      const pick = pickHotspotAt(event.clientX, event.clientY, [...hotspots.querySelectorAll('.hotspot')]);
      if (pick) pick.click();
    });
  }
  const list = document.createElement('ul'); list.className = 'zone-item-list'; view.append(list);
  const entries = [];
  for (const id of zone.items) {
    const item = catalog.items.find(i => i.id === id); let hotspot = null;
    const layer = layers && item.layer ? createLayer(item, layers) : null;
    if (hotspots && item.hotspot) {
      hotspot = makeButton('', () => openItem(item, hotspot), 'hotspot'); hotspot.dataset.itemId = id; hotspot.setAttribute('aria-label', item.name); hotspot.setAttribute('aria-haspopup','dialog');
      for (const key of ['left','top','width','height']) hotspot.style[key] = `${item.hotspot[key]}%`;
      if (item.hotspot.clip) Object.assign(hotspot.style, { borderRadius: '0', clipPath: `polygon(${item.hotspot.clip.map(([x, y]) => `${x}% ${y}%`).join(', ')})` });
      if (Number.isFinite(item.layer?.zIndex)) hotspot.style.zIndex = item.layer.zIndex;
      hotspots.append(hotspot);
      const indicator = document.createElement('span');
      indicator.className = 'hotspot-indicator';
      indicator.setAttribute('aria-hidden', 'true');
      for (const key of ['left','top','width','height']) indicator.style[key] = hotspot.style[key];
      hotspots.append(indicator);
    }
    const li = document.createElement('li'); const b = document.createElement('button'); fillListButton(b, item); b.addEventListener('click', () => openItem(item, b)); li.append(b); list.append(li);
    if (layer && hotspot) entries.push({item,layer,hotspot});
  }
  if (room) setupMotion(entries, room);
  requestAnimationFrame(() => fitListNames(view));
  const currentLetter = tabs.querySelector('.zone-tab-letters [aria-current]');
  if (currentLetter) requestAnimationFrame(() => {
    letters.scrollLeft = currentLetter.offsetLeft - (letters.clientWidth - currentLetter.offsetWidth) / 2;
  });
}
let keyboardRoomLink = null;
document.addEventListener('click', (event) => {
  const link = event.target.closest('a');
  keyboardRoomLink = event.detail === 0 && link?.matches('#room-tiles .room-tile, #zone-view .zone-back, .wordmark') ? link : null;
});
function route(event) {
  if (!catalog) return;
  // A card opened right after our own back-navigation already owns this URL: nothing to route.
  if (dialog.open && currentItem && location.hash === `#/item/${currentItem.id}`) return;
  // Any real navigation (Back/Forward, zone tabs, links) closes an open or closing card immediately.
  if (dialog.open || flipState !== 'idle') { routing = true; if (dialog.open) dialog.close(); settleClose(false); }
  if (ghost || veil) stopFlip();  // a card still flying home does not follow us to another page
  const rawHash = location.hash.replace(/^#\/?/, '');
  const redirectTo = catalog.redirects?.[rawHash] || catalog.redirects?.[rawHash.replace(/\/+$/, '')];
  if (redirectTo && redirectTo !== location.hash) {
    history.replaceState(history.state, '', redirectTo.startsWith('#') ? redirectTo : `#/${redirectTo}`);
  }
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
  const keyboardLink = keyboardRoomLink;
  keyboardRoomLink = null;
  const previousCategory = $('zone-view').dataset.category;
  $('hub-view').hidden = !!zone; $('zone-view').hidden = !zone;
  if (zone) {
    if (activeZone !== zone.id) { renderZone(zone); window.scrollTo(0, 0); }
    if (fromHub) window.scrollTo(0, 0);
    if (fromHub && keyboardLink) $('zone-view').querySelector('h1').focus({ preventScroll: true });
    if (item) requestAnimationFrame(() => openItem(item, $('zone-view').querySelector(`button[data-item-id="${item.id}"]`)));
  } else {
    window.scrollTo(0, 0);
    if (!fromHub && keyboardLink) $('room-tiles').querySelector(`[href="#/${previousCategory}"]`)?.focus();
  }
}
async function initCatalog() {
  try {
    const response = await fetch(withV('data/catalog.json')); if (!response.ok) throw new Error('Catalog unavailable'); catalog = await response.json();
    for (const c of catalog.categories) {
      const z = catalog.zones.find(z => z.id === c.cover_zone); const a = document.createElement('a'); a.className = 'room-tile'; a.href = `#/${c.id}`;
      const cover = z?.cover || z?.images?.at(-1);
      if (cover) {
        const img = document.createElement('img'); img.loading = 'lazy'; img.src = withV(cover.src); img.srcset = `${withV(cover.src)} ${cover.width}w`; img.sizes = '(max-width: 650px) 45vw, 300px'; img.width = cover.width; img.height = cover.height; img.alt = '';
        a.append(img);
      } else {
        a.classList.add('room-tile-text');
      }
      const label = document.createElement('span'); label.textContent = `${c.label} / ${c.label_en}`; const count = document.createElement('small'); count.textContent = `${c.count} 件`; a.append(label,count); $('room-tiles').append(a);
    }
    route();
  } catch (error) { $('data-status').textContent = '分室清单加载失败，请刷新重试。'; console.error(error); }
}
window.addEventListener('hashchange', route);
initCatalog();
