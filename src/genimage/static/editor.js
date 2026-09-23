(() => {
  const mode = document.querySelector('#generation-mode');
  const sourceInput = document.querySelector('#input-image');
  const thumbnail = document.querySelector('#source-thumbnail');
  const referenceInput = document.querySelector('#reference-images');
  const references = [];
  let sourceURL = null;
  let sourceReady = false;
  let sourceVersion = 0;
  let sourceLoad = Promise.resolve();
  let loadingJob = false;
  const instructions = {
    generate: 'Describe the image you want to create.',
    img2img: 'Use a source image as a starting point. Strength controls how much it changes.',
    edit: 'Upload a source image and describe what to change. Add references for subjects, style, or details.',
    reference: 'Combine up to 10 reference images. Describe the new scene and how each image should be used.'
  };
  const needsSource = () => ['edit', 'img2img'].includes(mode.value);
  const usesReferences = () => ['edit', 'reference'].includes(mode.value);
  const maxReferences = () => mode.value === 'edit' ? 9 : 10;
  const notify = message => {
    const element = document.querySelector('#form-message');
    element.textContent = message; element.className = 'error';
  };

  function renderReferences() {
    const container = document.querySelector('#reference-previews');
    container.replaceChildren();
    references.forEach((item, index) => {
      const figure = document.createElement('figure'); figure.className = 'reference-card';
      const img = new Image(); img.src = item.url; img.alt = item.file.name;
      const label = document.createElement('figcaption');
      label.textContent = `Image ${index + (needsSource() ? 2 : 1)}`;
      const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'icon-button';
      remove.textContent = '×'; remove.setAttribute('aria-label', `Remove ${item.file.name}`);
      remove.addEventListener('click', () => { URL.revokeObjectURL(item.url); references.splice(index, 1); renderReferences(); });
      figure.append(img, label, remove); container.append(figure);
    });
  }

  function changeMode() {
    document.querySelector('#mode-help').textContent = instructions[mode.value];
    document.querySelector('#source-controls').hidden = !needsSource();
    sourceInput.required = needsSource();
    document.querySelector('#source-label').textContent = mode.value === 'img2img' ? 'Source image' : 'Source image · Image 1';
    document.querySelector('#strength-field').hidden = mode.value !== 'img2img';
    document.querySelector('#reference-controls').hidden = !usesReferences();
    document.querySelector('#reference-limit').textContent = `up to ${maxReferences()}`;
    thumbnail.hidden = !sourceReady || !['edit', 'img2img'].includes(mode.value);
    alignDimensions();
    renderReferences();
  }
  function alignDimensions() {
    for (const input of [widthInput, heightInput]) {
      input.step = usesReferences() ? '32' : '16';
      if (usesReferences() && Number(input.value) % 32) {
        input.value = Math.max(256, Math.round(Number(input.value) / 32) * 32);
        aspect.value = 'custom'; customSize.hidden = false;
      }
    }
  }
  mode.addEventListener('change', changeMode);
  aspect.addEventListener('change', alignDimensions);

  async function loadSource() {
    const version = ++sourceVersion;
    sourceReady = false;
    thumbnail.hidden = true;
    if (sourceURL) URL.revokeObjectURL(sourceURL);
    sourceURL = null;
    const file = sourceInput.files[0];
    if (!file) return;
    if (file.size > 25 * 1024 * 1024) { notify('Source image is larger than 25 MB'); return; }
    sourceURL = URL.createObjectURL(file);
    thumbnail.src = sourceURL;
    try {
      await thumbnail.decode();
      if (version !== sourceVersion) return;
      if (thumbnail.naturalWidth * thumbnail.naturalHeight > 16000000) throw new Error('Source images must be at most 16 megapixels');
      sourceReady = true;
      thumbnail.hidden = !['edit', 'img2img'].includes(mode.value);
    } catch (error) { if (version === sourceVersion) notify(error.message || 'Could not read source image'); }
  }
  sourceInput.addEventListener('change', () => { sourceLoad = loadSource(); });
  document.querySelector('#clear-input').addEventListener('click', loadSource);

  document.querySelector('#add-references').addEventListener('click', () => referenceInput.click());
  referenceInput.addEventListener('change', () => {
    const files = [...referenceInput.files]; referenceInput.value = '';
    if (references.length + files.length > maxReferences()) { notify(`This mode allows ${maxReferences()} reference images`); return; }
    if (files.some(file => file.size > 25 * 1024 * 1024)) { notify('Each reference must be at most 25 MB'); return; }
    for (const file of files) references.push({ file, url: URL.createObjectURL(file) });
    renderReferences();
  });

  async function decodeFile(file) {
    const url = URL.createObjectURL(file);
    try {
      const image = new Image(); image.src = url; await image.decode(); return image;
    } finally { URL.revokeObjectURL(url); }
  }

  async function fetchImageFile(url, name) {
    const response = await fetch(url);
    if (!response.ok) throw new Error('Could not load a saved image; it may have been deleted');
    return new File([await response.blob()], name, { type: 'image/png' });
  }

  async function setSourceFile(file) {
    const transfer = new DataTransfer();
    if (file) transfer.items.add(file);
    sourceInput.files = transfer.files;
    sourceInput.dispatchEvent(new Event('change'));
    await sourceLoad;
    if (file && !sourceReady) throw new Error('Could not load the source image');
  }

  function sizeFromImage(image) {
    const originalWidth = image.naturalWidth, originalHeight = image.naturalHeight;
    const scale = Math.min(1, 2048 / originalWidth, 2048 / originalHeight);
    const width = Math.max(256, Math.round(originalWidth * scale / 32) * 32);
    const height = Math.max(256, Math.round(originalHeight * scale / 32) * 32);
    setCanvasSize(width, height);
    const message = document.querySelector('#form-message');
    message.className = '';
    message.textContent = width === originalWidth && height === originalHeight ? '' :
      `Canvas set to ${width} × ${height}, the supported size closest to the image’s ${originalWidth} × ${originalHeight}.`;
  }

  window.imageEditor = {
    setSupported(supported) {
      for (const option of mode.options) if (['edit', 'reference'].includes(option.value)) option.disabled = !supported;
      if (!supported && usesReferences()) { mode.value = 'generate'; changeMode(); }
    },
    async appendTo(data) {
      if (loadingJob) throw new Error('Wait for the saved images to finish loading');
      data.set('mode', mode.value);
      data.delete('input_image'); data.delete('image_strength');
      if (needsSource()) {
        if (!sourceInput.files.length || !sourceReady) throw new Error('Choose a valid source image and wait for it to load');
        data.set('input_image', sourceInput.files[0]);
      }
      if (mode.value === 'img2img') data.set('image_strength', document.querySelector('#image-strength').value);
      if (usesReferences()) {
        if (references.length > maxReferences()) throw new Error(`Remove references until there are at most ${maxReferences()}`);
        if (mode.value === 'reference' && !references.length) throw new Error('Add at least one reference image');
        if (Number(data.get('width')) % 32 || Number(data.get('height')) % 32) throw new Error('Editing dimensions must be multiples of 32');
        for (const item of references) data.append('reference_images', item.file);
      }
    },
    async useJob(job, targetMode) {
      if (loadingJob) return;
      loadingJob = true;
      try {
        if (targetMode === 'reference' && references.length >= 10) throw new Error('Remove a reference before adding another');
        const file = await fetchImageFile(job.image_url, `generation-${job.id}.png`);
        const image = await decodeFile(file);
        mode.value = targetMode;
        if (targetMode === 'reference') {
          references.push({ file, url: URL.createObjectURL(file) });
        } else {
          await setSourceFile(file);
        }
        changeMode(); sizeFromImage(image); randomizeSeed(); focusComposer();
      } catch (error) { notify(error.message); } finally { loadingJob = false; }
    },
    async restoreJob(job) {
      if (loadingJob) throw new Error('Wait for the saved images to finish loading');
      loadingJob = true;
      try {
        const targetMode = (!job.mode || job.mode === 'generate') && job.input_image_url ? 'img2img' : (job.mode || 'generate');
        const option = [...mode.options].find(item => item.value === targetMode);
        if (!option || option.disabled) throw new Error('This generation mode is unavailable with the current model');
        const [source, savedReferences] = await Promise.all([
          job.input_image_url ? fetchImageFile(job.input_image_url, `source-${job.id}.png`) : null,
          Promise.all((job.reference_image_urls || []).map((url, index) => fetchImageFile(url, `reference-${index + 1}.png`))),
        ]);
        mode.value = targetMode;
        await setSourceFile(source);
        for (const item of references) URL.revokeObjectURL(item.url);
        references.splice(0, references.length, ...savedReferences.map(file => ({ file, url: URL.createObjectURL(file) })));
        changeMode();
      } finally { loadingJob = false; }
    }
  };
  changeMode();
})();
