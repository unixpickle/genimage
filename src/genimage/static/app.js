const form = document.querySelector('#generation-form');
const aspect = document.querySelector('#aspect-ratio');
const widthInput = document.querySelector('#width');
const heightInput = document.querySelector('#height');
const customSize = document.querySelector('#custom-size');
const queueEl = document.querySelector('#queue');
const historyEl = document.querySelector('#history');
const dialog = document.querySelector('#job-dialog');
let state = { queue: [], history: [], worker: {}, model: {} };
let polling = false;

const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const relativeTime = seconds => {
  if (!seconds) return '';
  const delta = Math.max(0, Math.floor(Date.now() / 1000 - seconds));
  if (delta < 5) return 'now';
  if (delta < 60) return `${delta}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return new Date(seconds * 1000).toLocaleDateString();
};

function button(label, className, action, title = '') {
  const el = document.createElement('button');
  el.type = 'button'; el.className = className; el.textContent = label; el.title = title;
  el.addEventListener('click', action);
  return el;
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { const body = await response.json(); message = body.detail || message; } catch (_) {}
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

async function refresh() {
  if (polling) return;
  polling = true;
  try {
    state = await request('/api/state');
    render();
  } catch (error) {
    const status = document.querySelector('#worker-status');
    status.className = 'worker-status error';
    status.querySelector('span').textContent = 'Web server unavailable';
  } finally { polling = false; }
}

function renderWorker() {
  const el = document.querySelector('#worker-status');
  const active = state.queue.find(job => job.status === 'running');
  if (!state.model.downloaded) {
    el.className = 'worker-status error'; el.querySelector('span').textContent = 'Checkpoint missing';
  } else if (active) {
    el.className = 'worker-status busy'; el.querySelector('span').textContent = active.stage || 'Generating';
  } else if (!state.worker.alive) {
    el.className = 'worker-status error'; el.querySelector('span').textContent = 'Worker offline';
  } else if (state.worker.model_status === 'Load failed') {
    el.className = 'worker-status error'; el.querySelector('span').textContent = 'Model load failed';
  } else {
    el.className = 'worker-status ready'; el.querySelector('span').textContent = state.worker.model_status === 'Ready' ? 'Model ready' : 'Worker ready';
  }
}

function renderQueue() {
  queueEl.replaceChildren();
  document.querySelector('#queue-count').textContent = state.queue.length;
  document.querySelector('#queue-empty').hidden = state.queue.length > 0;
  const template = document.querySelector('#queue-template');
  state.queue.forEach((job, index) => {
    const node = template.content.cloneNode(true);
    const article = node.querySelector('article');
    article.classList.add(job.status);
    node.querySelector('.queue-position').textContent = job.status === 'running' ? '●' : String(index + 1).padStart(2, '0');
    node.querySelector('.status-pill').textContent = job.status === 'running' ? job.stage : 'Queued';
    node.querySelector('time').textContent = relativeTime(job.created_at);
    node.querySelector('.queue-prompt').textContent = job.prompt;
    const percent = job.progress_total ? Math.round(job.progress_current / job.progress_total * 100) : 0;
    node.querySelector('.progress-track').hidden = job.status !== 'running';
    node.querySelector('.progress-track span').style.width = `${percent}%`;
    node.querySelector('.queue-meta').textContent = `${job.width} × ${job.height} · ${job.steps} steps · seed ${job.seed}${job.status === 'running' ? ` · ${percent}%` : ''}`;
    const actions = node.querySelector('.queue-actions');
    actions.append(button('…', 'icon-button', () => showDetails(job), 'Show generation details'));
    if (job.status === 'running') {
      actions.append(button('■', 'icon-button', () => cancelJob(job.id), 'Cancel generation'));
    }
    actions.append(button('×', 'icon-button', () => deleteJob(job, false), 'Delete permanently'));
    queueEl.append(node);
  });
}

function renderHistory() {
  historyEl.replaceChildren();
  document.querySelector('#history-count').textContent = state.history.length;
  document.querySelector('#history-empty').hidden = state.history.length > 0;
  const template = document.querySelector('#card-template');
  state.history.forEach(job => {
    const node = template.content.cloneNode(true);
    const shell = node.querySelector('.image-shell');
    const imageButton = node.querySelector('.image-button');
    if (job.status === 'completed') {
      const img = new Image(); img.loading = 'lazy'; img.src = job.image_url; img.alt = job.prompt;
      shell.append(img); imageButton.addEventListener('click', () => showDetails(job));
    } else {
      shell.classList.add('failed-shell'); shell.textContent = job.status === 'failed' ? `Failed · ${job.error || 'Unknown error'}` : 'Cancelled';
      imageButton.style.cursor = 'pointer'; imageButton.addEventListener('click', () => showDetails(job));
    }
    node.querySelector('.card-prompt').textContent = job.prompt;
    node.querySelector('.card-meta span').textContent = job.status === 'completed' ? `${job.width} × ${job.height}` : job.status;
    node.querySelector('.card-meta time').textContent = relativeTime(job.finished_at || job.updated_at);
    const actions = node.querySelector('.card-actions');
    if (job.status === 'completed') {
      const download = document.createElement('a'); download.className = 'text-button'; download.href = job.download_url; download.textContent = 'Download';
      actions.append(download);
    }
    actions.append(button('Details', 'text-button', () => showDetails(job)));
    actions.append(button('Delete', 'text-button delete', () => deleteJob(job, true)));
    historyEl.append(node);
  });
}

function render() { renderWorker(); renderQueue(); renderHistory(); }

async function cancelJob(id) {
  try { await request(`/api/jobs/${id}/cancel`, { method: 'POST' }); await refresh(); }
  catch (error) { window.alert(error.message); }
}

async function deleteJob(job, confirmFirst) {
  if (confirmFirst && !window.confirm('Permanently delete this generation and its image file? This cannot be undone.')) return;
  try { await request(`/api/jobs/${job.id}`, { method: 'DELETE' }); await refresh(); }
  catch (error) { window.alert(error.message); }
}

function showDetails(job) {
  const content = document.querySelector('#dialog-content');
  const media = [];
  if (job.image_url) media.push(`<figure><figcaption>Generated</figcaption><img class="dialog-image" src="${job.image_url}" alt="${escapeHtml(job.prompt)}"></figure>`);
  if (job.input_image_url) media.push(`<figure><figcaption>Initial image</figcaption><img class="dialog-image" src="${job.input_image_url}" alt="Initial image for ${escapeHtml(job.prompt)}"></figure>`);
  content.innerHTML = `
    ${media.length ? `<div class="dialog-media">${media.join('')}</div>` : ''}
    <div class="dialog-details">
      <h3>${escapeHtml(job.prompt)}</h3>
      <div class="detail-grid">
        <div><small>Status</small><span>${escapeHtml(job.stage || job.status)}</span></div>
        <div><small>Canvas</small><span>${job.width} × ${job.height}</span></div>
        <div><small>Steps</small><span>${job.steps}</span></div>
        <div><small>Guidance</small><span>${job.guidance}</span></div>
        <div><small>Seed</small><span>${job.seed}</span></div>
        <div><small>Scheduler</small><span>${escapeHtml(job.scheduler)}</span></div>
        ${job.has_input ? `<div><small>Image strength</small><span>${job.image_strength}</span></div>` : ''}
        ${job.pid_decode ? `<div><small>PiD decode</small><span>On · σ ${job.pid_degrade_sigma}</span></div>` : ''}
      </div>
      ${job.negative_prompt ? `<p class="negative"><strong>Negative prompt:</strong> ${escapeHtml(job.negative_prompt)}</p>` : ''}
    </div>`;
  dialog.showModal();
}

aspect.addEventListener('change', () => {
  const isCustom = aspect.value === 'custom'; customSize.hidden = !isCustom;
  if (!isCustom) [widthInput.value, heightInput.value] = aspect.value.split('x');
});
document.querySelector('#steps').addEventListener('input', e => document.querySelector('#steps-output').textContent = e.target.value);
document.querySelector('#image-strength').addEventListener('input', e => document.querySelector('#strength-output').textContent = Number(e.target.value).toFixed(2));
document.querySelector('#random-seed').addEventListener('change', e => { document.querySelector('#seed').disabled = e.target.checked; });
document.querySelector('#seed').disabled = true;
document.querySelector('#input-image').addEventListener('change', e => {
  const hasFile = e.target.files.length > 0;
  document.querySelector('#file-label').textContent = hasFile ? e.target.files[0].name : 'Choose an image';
  document.querySelector('#clear-input').hidden = !hasFile;
  document.querySelector('#strength-field').classList.toggle('enabled', hasFile);
});
document.querySelector('.file-button').addEventListener('click', () => document.querySelector('#input-image').click());
document.querySelector('#clear-input').addEventListener('click', () => {
  const input = document.querySelector('#input-image');
  input.value = '';
  document.querySelector('#file-label').textContent = 'Choose an image';
  document.querySelector('#clear-input').hidden = true;
  document.querySelector('#strength-field').classList.remove('enabled');
});
document.querySelector('#refresh-button').addEventListener('click', refresh);
document.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });

form.addEventListener('submit', async event => {
  event.preventDefault();
  const submit = form.querySelector('button[type="submit"]');
  const message = document.querySelector('#form-message');
  submit.disabled = true; message.className = ''; message.textContent = 'Adding to queue…';
  const data = new FormData(form);
  data.set('width', widthInput.value); data.set('height', heightInput.value);
  if (document.querySelector('#random-seed').checked) data.delete('seed');
  if (!document.querySelector('#input-image').files.length) data.delete('input_image');
  if (!data.has('pid_decode')) data.set('pid_decode', 'false');
  try {
    await request('/api/jobs', { method: 'POST', body: data });
    message.textContent = 'Queued.'; form.querySelector('#prompt').value = '';
    await refresh(); setTimeout(() => { if (message.textContent === 'Queued.') message.textContent = ''; }, 1800);
  } catch (error) { message.className = 'error'; message.textContent = error.message; }
  finally { submit.disabled = false; }
});

document.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') form.requestSubmit();
});

refresh();
setInterval(refresh, 1500);
