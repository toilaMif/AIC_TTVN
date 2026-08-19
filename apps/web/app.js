const COLORS = ['#e34d42', '#2d77c7', '#8a5bb4', '#14846d', '#c47a16'];
const state = { mode: 'whole', clauses: [], activeId: null, queryInputs: [] };
const $ = (id) => document.getElementById(id);
let nextClauseId = 1;
let timerId = null;
let nextInputId = 1;

function addQueryInput(modality = 'visual', text = '') {
  state.queryInputs.push({ id: nextInputId++, modality, text });
  renderQueryInputs();
}

function renderQueryInputs() {
  $('query-inputs').innerHTML = state.queryInputs.map((item) => `
    <div class="query-input" data-query-row="${item.id}">
      <div class="modality-row">
        ${['visual', 'asr', 'ocr', 'object'].map((modality) => `
          <label class="modality-option ${['ocr', 'object'].includes(modality) ? 'disabled' : ''}">
            <input type="radio" name="modality-${item.id}" value="${modality}" data-modality-id="${item.id}"
              ${item.modality === modality ? 'checked' : ''} ${['ocr', 'object'].includes(modality) ? 'disabled' : ''}>
            ${modality === 'visual' ? 'Visual / Text' : modality.toUpperCase()}
          </label>`).join('')}
      </div>
      <textarea data-query-id="${item.id}" rows="2" placeholder="${item.modality === 'asr' ? 'Vietnamese speech transcript...' : 'Describe the visual scene...'}">${escapeHtml(item.text)}</textarea>
      <div class="query-input-footer"><button class="remove-input" data-remove-query="${item.id}" type="button">Remove input</button></div>
    </div>`).join('');
  document.querySelectorAll('[data-query-id]').forEach((input) => input.oninput = (event) => {
    const item = state.queryInputs.find((entry) => entry.id === Number(event.target.dataset.queryId));
    item.text = event.target.value;
  });
  document.querySelectorAll('[data-modality-id]').forEach((radio) => radio.onchange = (event) => {
    const item = state.queryInputs.find((entry) => entry.id === Number(event.target.dataset.modalityId));
    item.modality = event.target.value;
    renderQueryInputs();
  });
  document.querySelectorAll('[data-remove-query]').forEach((button) => button.onclick = () => {
    if (state.queryInputs.length === 1) return setStatus('At least one query input is required.', true);
    state.queryInputs = state.queryInputs.filter((entry) => entry.id !== Number(button.dataset.removeQuery));
    renderQueryInputs();
  });
}

function newClause(text) {
  const offset = state.clauses.length * 0.06;
  return { id: nextClauseId++, text, x: 0.12 + offset, y: 0.14 + offset, w: 0.68, h: 0.34, color: COLORS[state.clauses.length % COLORS.length] };
}

function normalized(value, min, max) { return Math.min(max, Math.max(min, value)); }
function boxStyle(c) { return `left:${c.x * 100}%;top:${c.y * 100}%;width:${c.w * 100}%;height:${c.h * 100}%;--box-color:${c.color}`; }
function coordinates(c) { return `${c.x.toFixed(2)}, ${c.y.toFixed(2)}, ${(c.x + c.w).toFixed(2)}, ${(c.y + c.h).toFixed(2)}`; }

function renderEditor() {
  const showCoordinates = $('show-coordinates').checked;
  $('spatial-canvas').innerHTML = state.clauses.map((c) => `
    <div class="query-box ${c.id === state.activeId ? 'active' : ''}" data-box-id="${c.id}" style="${boxStyle(c)}">
      <span class="query-box-label">${escapeHtml(c.text || 'untitled')}</span>
      ${showCoordinates ? `<span class="query-box-coords">${coordinates(c)}</span>` : ''}
      <span class="resize-handle" data-resize-id="${c.id}"></span>
    </div>`).join('');
  $('clauses').innerHTML = state.clauses.map((c) => `
    <div class="clause ${c.id === state.activeId ? 'active' : ''}" data-row-id="${c.id}" style="--clause-color:${c.color}">
      <span class="clause-color"></span><input data-text-id="${c.id}" value="${escapeHtml(c.text)}"><button class="remove" data-remove-id="${c.id}">×</button>
    </div>`).join('');
  bindEditorEvents();
}

function escapeHtml(value) { const div = document.createElement('div'); div.textContent = value; return div.innerHTML; }
function clauseById(id) { return state.clauses.find((c) => c.id === Number(id)); }

function bindEditorEvents() {
  document.querySelectorAll('[data-box-id]').forEach((box) => box.addEventListener('pointerdown', beginDrag));
  document.querySelectorAll('[data-resize-id]').forEach((handle) => handle.addEventListener('pointerdown', beginResize));
  document.querySelectorAll('[data-row-id]').forEach((row) => row.onclick = () => { state.activeId = Number(row.dataset.rowId); renderEditor(); });
  document.querySelectorAll('[data-text-id]').forEach((input) => input.oninput = (event) => { clauseById(event.target.dataset.textId).text = event.target.value; renderCanvasOnly(); });
  document.querySelectorAll('[data-remove-id]').forEach((button) => button.onclick = (event) => { event.stopPropagation(); removeClause(Number(button.dataset.removeId)); });
}

function renderCanvasOnly() {
  const active = document.activeElement;
  const selection = active?.selectionStart;
  renderEditor();
  if (active?.dataset?.textId) { const input = document.querySelector(`[data-text-id="${active.dataset.textId}"]`); input?.focus(); if (selection !== undefined) input?.setSelectionRange(selection, selection); }
}

function pointerInteraction(event, mode) {
  event.preventDefault(); event.stopPropagation();
  const id = Number(mode === 'drag' ? event.currentTarget.dataset.boxId : event.currentTarget.dataset.resizeId);
  const clause = clauseById(id); if (!clause) return;
  state.activeId = id;
  const canvas = $('spatial-canvas').getBoundingClientRect();
  const start = { px: event.clientX, py: event.clientY, x: clause.x, y: clause.y, w: clause.w, h: clause.h };
  const move = (e) => {
    const dx = (e.clientX - start.px) / canvas.width; const dy = (e.clientY - start.py) / canvas.height;
    if (mode === 'drag') { clause.x = normalized(start.x + dx, 0, 1 - clause.w); clause.y = normalized(start.y + dy, 0, 1 - clause.h); }
    else { clause.w = normalized(start.w + dx, 0.12, 1 - clause.x); clause.h = normalized(start.h + dy, 0.12, 1 - clause.y); }
    renderEditor();
  };
  const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
  window.addEventListener('pointermove', move); window.addEventListener('pointerup', up); renderEditor();
}
function beginDrag(event) { if (event.target.dataset.resizeId) return; pointerInteraction(event, 'drag'); }
function beginResize(event) { pointerInteraction(event, 'resize'); }
function removeClause(id) { state.clauses = state.clauses.filter((c) => c.id !== id); if (state.activeId === id) state.activeId = state.clauses[0]?.id || null; renderEditor(); }

function addClause() {
  const text = $('new-clause-text').value.trim() || `query ${nextClauseId}`;
  const clause = newClause(text); state.clauses.push(clause); state.activeId = clause.id; $('new-clause-text').value = ''; renderEditor();
}

function setStatus(message, error = false) { $('status').textContent = message; $('status').style.color = error ? '#b33422' : ''; }
function startLoading() { const startedAt = performance.now(); $('loading-overlay').hidden = false; $('loading-time').textContent = '0.0s'; timerId = setInterval(() => { const elapsed = (performance.now() - startedAt) / 1000; $('loading-time').textContent = `${elapsed.toFixed(1)}s`; if (elapsed >= 8) $('loading-stage').textContent = 'Ranking and diversifying candidates...'; }, 100); return startedAt; }
function stopLoading(startedAt) { if (timerId) clearInterval(timerId); timerId = null; $('loading-overlay').hidden = true; const elapsed = (performance.now() - startedAt) / 1000; $('search-time').textContent = `${elapsed.toFixed(2)}s`; return elapsed; }
function renderResults(items) { $('count').textContent = `${items.length} results`; $('results').innerHTML = items.length ? items.map((item, index) => `<article class="result-card"><span class="rank">${index + 1}</span><img src="${item.frame_url || ''}" alt="${item.video_id} frame ${item.frame_idx}" loading="lazy"><div class="result-info"><strong>${item.video_id} · frame ${item.frame_idx}</strong><span>${Number(item.pts_time || 0).toFixed(2)}s · ${Number(item.score).toFixed(4)} · ${item.shot_id || ''}</span></div></article>`).join('') : '<div class="empty">No results</div>'; }

function renderMixedResults(groups) {
  const total = groups.reduce((sum, group) => sum + group.items.length, 0);
  $('count').textContent = `${total} results`;
  $('results').innerHTML = groups.map((group) => {
    const cards = group.items.map((item, index) => group.modality === 'asr'
      ? `<article class="result-card asr-result"><span class="rank">${index + 1}</span>${item.frame_url ? `<img src="${item.frame_url}" alt="${item.video_id} ASR frame" loading="lazy">` : ''}<div class="result-info"><strong>${item.video_id} · ${Number(item.start_time || 0).toFixed(2)}s - ${Number(item.end_time || 0).toFixed(2)}s</strong><span>${escapeHtml(item.text || '')}</span><small>${item.language || 'vi'} · score ${Number(item.score || 0).toFixed(4)}</small></div></article>`
      : `<article class="result-card"><span class="rank">${index + 1}</span><img src="${item.frame_url || ''}" alt="${item.video_id} frame ${item.frame_idx}" loading="lazy"><div class="result-info"><strong>${item.video_id} · frame ${item.frame_idx}</strong><span>${Number(item.pts_time || 0).toFixed(2)}s · ${Number(item.score || 0).toFixed(4)} · ${item.shot_id || ''}</span></div></article>`).join('');
    return `<section class="result-group"><h3>${group.label}</h3>${cards || '<div class="empty">No matches</div>'}</section>`;
  }).join('');
}
function nonEmptyInputs() { return state.queryInputs.filter((item) => item.text.trim()); }
function primaryVisualQuery() { return nonEmptyInputs().find((item) => item.modality === 'visual')?.text.trim() || ''; }

async function search() {
  const inputs = nonEmptyInputs(); if (!inputs.length) return setStatus('Enter at least one query.', true);
  $('result-title').textContent = inputs.map((item) => item.text.trim()).join(' + '); $('search').disabled = true; setStatus('Searching visual and ASR indexes...'); const startedAt = startLoading();
  try { const response = await fetch(`/search/visual?query=${encodeURIComponent(query)}&top_k=${Number($('top-k').value) || 20}`); if (!response.ok) throw new Error(await response.text()); const results = await response.json(); renderResults(results); const elapsed = stopLoading(startedAt); setStatus(`${results.length} results · completed in ${elapsed.toFixed(2)}s`); }
  catch (error) { const elapsed = stopLoading(startedAt); setStatus(`Search failed after ${elapsed.toFixed(2)}s: ${error.message}`, true); }
  finally { $('search').disabled = false; }
}

function spatialPayload() { return { global_query: $('query').value.trim(), mode: state.mode, clauses: state.clauses.map((c) => ({ text: c.text, box: [c.x, c.y, c.x + c.w, c.y + c.h], required: true, weight: 1.0 })), candidate_k: Number($('candidate-k').value), top_k: Number($('top-k').value) }; }

document.querySelectorAll('.mode').forEach((button) => button.onclick = () => { document.querySelectorAll('.mode').forEach((item) => item.classList.remove('active')); button.classList.add('active'); state.mode = button.dataset.mode; $('mode-pill').textContent = button.textContent.toUpperCase(); $('notice').textContent = state.mode === 'whole' ? 'Global OpenCLIP retrieval is active.' : 'Drag and resize query boxes. Spatial backend connection is the next stage.'; });
$('add-clause').onclick = addClause; $('new-clause-text').onkeydown = (event) => { if (event.key === 'Enter') addClause(); };
$('clear-clauses').onclick = () => { state.clauses = []; state.activeId = null; renderEditor(); };
$('show-coordinates').onchange = renderEditor;
$('submit-spatial').onclick = async () => {
  if (!state.clauses.length) return setStatus('Add at least one local query box.', true);
  const payload = spatialPayload(); $('search').disabled = true; setStatus('Localized reranking: loading candidate crops...'); const startedAt = startLoading();
  try {
    const response = await fetch('/search/localized', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    if (!response.ok) throw new Error(await response.text());
    const results = await response.json(); renderResults(results); const elapsed = stopLoading(startedAt); $('result-title').textContent = payload.global_query; setStatus(`${results.length} localized results · completed in ${elapsed.toFixed(2)}s`);
  } catch (error) { const elapsed = stopLoading(startedAt); setStatus(`Localized search failed after ${elapsed.toFixed(2)}s: ${error.message}`, true); }
  finally { $('search').disabled = false; }
};
// Multi-modality query flow. ASR and visual scores are rendered in separate groups.
function renderMixedResults(groups) {
  const total = groups.reduce((sum, group) => sum + group.items.length, 0);
  $('count').textContent = `${total} results`;
  $('results').innerHTML = groups.map((group) => `<section class="result-group"><h3>${group.label}</h3>${group.items.length ? group.items.map((item, index) => group.modality === 'asr' ? `<article class="result-card asr-result"><span class="rank">${index + 1}</span>${item.frame_url ? `<img src="${item.frame_url}" alt="ASR frame" loading="lazy">` : ''}<div class="result-info"><strong>${item.video_id} · ${Number(item.start_time || 0).toFixed(2)}s - ${Number(item.end_time || 0).toFixed(2)}s</strong><span>${escapeHtml(item.text || '')}</span></div></article>` : `<article class="result-card"><span class="rank">${index + 1}</span><img src="${item.frame_url || ''}" alt="${item.video_id} frame ${item.frame_idx}" loading="lazy"><div class="result-info"><strong>${item.video_id} · frame ${item.frame_idx}</strong><span>${Number(item.pts_time || 0).toFixed(2)}s · ${Number(item.score || 0).toFixed(4)}</span></div></article>`).join('') : '<div class="empty">No matches</div>'}</section>`).join('');
}
function queryItems() { return state.queryInputs.filter((item) => item.text.trim()); }
function visualQuery() { return queryItems().find((item) => item.modality === 'visual')?.text.trim() || ''; }
async function search() {
  const inputs = queryItems(); if (!inputs.length) return setStatus('Enter at least one query.', true);
  $('search').disabled = true; setStatus('Searching visual and ASR indexes...'); const startedAt = startLoading();
  try {
    const topK = Number($('top-k').value) || 20;
    const groups = await Promise.all(['visual', 'asr'].map(async (modality) => {
      const item = inputs.find((entry) => entry.modality === modality);
      if (!item) return { modality, label: modality === 'asr' ? 'ASR transcript' : 'Visual / Text', items: [] };
      const response = await fetch(`/search/${modality}?query=${encodeURIComponent(item.text.trim())}&top_k=${topK}`);
      if (!response.ok) throw new Error(await response.text());
      return { modality, label: modality === 'asr' ? 'ASR transcript' : 'Visual / Text', items: await response.json() };
    }));
    renderMixedResults(groups); const elapsed = stopLoading(startedAt); setStatus(`${groups.reduce((n, g) => n + g.items.length, 0)} results · completed in ${elapsed.toFixed(2)}s`);
  } catch (error) { const elapsed = stopLoading(startedAt); setStatus(`Search failed after ${elapsed.toFixed(2)}s: ${error.message}`, true); }
  finally { $('search').disabled = false; }
}
function spatialPayload() { return { global_query: visualQuery(), mode: state.mode, clauses: state.clauses.map((c) => ({ text: c.text, box: [c.x, c.y, c.x + c.w, c.y + c.h], required: true, weight: 1.0 })), candidate_k: Number($('candidate-k').value), top_k: Number($('top-k').value) }; }
$('add-input').onclick = () => addQueryInput();
$('search').onclick = search;
document.addEventListener('keydown', (event) => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') search(); });
if (!state.queryInputs.length) addQueryInput('visual', '');
if (!state.queryInputs.some((item) => item.modality === 'asr')) addQueryInput('asr', '');
state.clauses.push(newClause('diver')); state.clauses.push(newClause('turtle')); state.activeId = state.clauses[0].id; renderEditor();
