<script lang="ts">
  import { onDestroy, onMount, tick } from 'svelte';
  import { browseAdapterRepo, deleteAdapter, downloadAdapter, jsonRequest, listAdapters,
    loadModel, models, runTask, unloadModel, uploadAdapter, uploadFile } from '$lib/api';
  import type { StoredAdapter } from '$lib/api';
  import MediaPreview from '$lib/MediaPreview.svelte';
  import type { Translator } from '$lib/i18n';
  import type { CatalogEntry, LoadedModel, ParamSpec, ServerHealth } from '$lib/types';

  export let lyrics = '';
  export let seed = 1234;
  export let loraUploading = false;
  export let busy = false;
  export let paramSpecs: ParamSpec[] = [];
  export let advancedValues: Record<string, unknown> = {};
  export let catalogEntries: CatalogEntry[] = [];
  export let loadedModels: LoadedModel[] = [];
  export let server: ServerHealth | null = null;
  export let modelPathFor: (entry: CatalogEntry) => string = (entry) => entry.path;
  export let sessionOptionsFor: (entry: CatalogEntry) => Record<string, string> = (entry) => entry.session_options || {};
  export let refreshModels: () => Promise<void> = async () => {};
  export let log: (message: string) => void = () => {};
  export let tr: Translator = (key, _values, fallback = key) => fallback;
  export let localizedParameterText:
    (spec: ParamSpec, field: 'label' | 'info' | 'placeholder', translate?: Translator) => string =
      (spec, field) => field === 'label' ? spec.name : '';
  export let setParameterValue: (spec: ParamSpec, value: unknown) => void = () => {};

  const componentParamNames = ['main_gguf', 'vae_gguf'];
  const coreParamNames = ['style', 'cot', 'guidance_scale', 'num_inference_steps'];
  const abcParamNames = ['abc', 'abc_file'];
  const semanticParamNames = [
    'semantic_temperature',
    'semantic_top_p',
    'semantic_top_k',
    'semantic_repetition_penalty',
    'semantic_penalty_window',
    'semantic_min_tokens',
    'semantic_max_tokens'
  ];
  const plannerParamNames = [
    'abc_temperature',
    'abc_top_p',
    'abc_top_k',
    'abc_repetition_penalty',
    'abc_penalty_window',
    'abc_min_tokens',
    'abc_max_tokens'
  ];

  let coverAudioFile: File | null = null;
  let loraError = '';
  let loraUpload: AbortController | null = null;
  let narLoraError = '';
  let narLoraUpload: AbortController | null = null;

  // Adapters stored in <model>/loras/. These persist across restarts, unlike the
  // temporary /v1/ui/upload path the old free-text field relied on.
  // `group` is the folder an adapter is filed under, decided when it is stored
  // rather than guessed from its tensor names afterwards. The server treats it
  // as an opaque name, so which slot a file belongs to is recorded, not derived.
  type LoraSlot = {
    name: 'ar_lora' | 'nar_lora';
    scale: 'ar_lora_scale' | 'nar_lora_scale';
    group: 'ar' | 'nar';
    title: string;
    blurb: string;
  };
  const loraSlots: LoraSlot[] = [
    { name: 'ar_lora', scale: 'ar_lora_scale', group: 'ar', title: 'AR LoRA',
      blurb: 'Changes what gets composed \u2014 melody, structure, arrangement.' },
    { name: 'nar_lora', scale: 'nar_lora_scale', group: 'nar', title: 'NAR LoRA',
      blurb: 'Changes how it is rendered \u2014 the sound, not the notes.' }
  ];
  // Svelte cannot bind:this to a ternary, so the two file inputs are keyed.
  let loraInputs: Record<string, HTMLInputElement | null> = {};
  let stored: StoredAdapter[] = [];
  let storeDirectory = '';
  let storeError = '';
  let repoId = '';
  let repoAdapters: StoredAdapter[] = [];
  let repoBusy = false;
  let repoError = '';
  let repoNotice = '';
  let downloading = '';
  let browseRequest: AbortController | null = null;
  onDestroy(() => { loraUpload?.abort(); narLoraUpload?.abort(); browseRequest?.abort(); });

  $: adaptersFor = (group: 'ar' | 'nar') =>
    stored.filter((a) => a.unfused_lora && (a.group === group || a.group === ''));

  function megabytes(bytes: number) {
    return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : `${Math.round(bytes / 1e6)} MB`;
  }

  function adapterOptionLabel(adapter: StoredAdapter) {
    const rank = adapter.rank ? `, rank ${adapter.rank}` : '';
    return `${adapter.file} (${megabytes(adapter.bytes)}${rank})`;
  }

  async function refreshStored() {
    storeError = '';
    const entry = catalogEntries.find((candidate) => candidate.family === 'yue2');
    if (!entry) return;
    try {
      const result = await listAdapters(modelPathFor(entry));
      stored = result.adapters;
      storeDirectory = result.directory;
    } catch (error) {
      storeError = error instanceof Error ? error.message : String(error);
    }
  }

  onMount(refreshStored);

  async function browseRepo() {
    repoError = ''; repoNotice = ''; repoAdapters = [];
    const repo = repoId.trim();
    if (!repo) return;
    repoBusy = true;
    browseRequest?.abort();
    browseRequest = new AbortController();
    try {
      const result = await browseAdapterRepo(repo, browseRequest.signal);
      repoAdapters = result.adapters;
      const usable = repoAdapters.filter((a) => a.unfused_lora).length;
      repoNotice = usable
        ? `${usable} of ${repoAdapters.length} files in ${repo} are unfused adapters. Whether one fits YuE2 shows when the model loads it.`
        : `None of the ${repoAdapters.length} safetensors in ${repo} are unfused adapters this engine can read.`;
    } catch (error) {
      if (!browseRequest.signal.aborted) repoError = error instanceof Error ? error.message : String(error);
    } finally {
      repoBusy = false; browseRequest = null;
    }
  }

  async function fetchAdapter(adapter: StoredAdapter, slot: LoraSlot) {
    const entry = catalogEntries.find((candidate) => candidate.family === 'yue2');
    if (!entry || !adapter.repo_file) return;
    repoError = '';
    downloading = adapter.repo_file;
    try {
      const saved = await downloadAdapter(repoId.trim(), adapter.repo_file, modelPathFor(entry), slot.group);
      await refreshStored();
      setNamedParameter(slot.name, saved.path);
      repoNotice = `Stored ${saved.file} as a ${slot.group.toUpperCase()} adapter. Reload the model to apply it.`;
    } catch (error) {
      repoError = error instanceof Error ? error.message : String(error);
    } finally {
      downloading = '';
    }
  }

  async function removeAdapter(path: string, name: 'ar_lora' | 'nar_lora') {
    const entry = catalogEntries.find((candidate) => candidate.family === 'yue2');
    if (!entry) return;
    // path is "loras/<file>" or "loras/<group>/<file>"; the server wants them apart.
    const parts = path.split('/');
    const file = parts[parts.length - 1];
    const group = parts.length > 2 ? parts[1] : '';
    try {
      await deleteAdapter(modelPathFor(entry), file, group);
      if (String(advancedValues[name] ?? '') === path) setNamedParameter(name, '');
      await refreshStored();
    } catch (error) {
      storeError = error instanceof Error ? error.message : String(error);
    }
  }

  // Both adapters upload through the same route; `name` picks which field and which
  // error slot they land in. The engine rejects anything but an unfused .safetensors
  // (yue2/lora.cpp), so the extension check is the one thing worth catching early --
  // the `_comfyui` files in the upstream adapter repos are fused and will not load.
  async function selectLora(file: File | null, name: 'ar_lora' | 'nar_lora' = 'ar_lora') {
    if (!file) return;
    const isNar = name === 'nar_lora';
    const setError = (message: string) => { if (isNar) narLoraError = message; else loraError = message; };
    setError('');
    if (!file.name.toLowerCase().endsWith('.safetensors')) {
      setError(`Select an unfused ${isNar ? 'NAR' : 'AR'} .safetensors adapter.`);
      return;
    }
    loraUploading = true;
    const upload = new AbortController();
    if (isNar) narLoraUpload = upload; else loraUpload = upload;
    try {
      const entry = catalogEntries.find((candidate) => candidate.family === 'yue2');
      if (!entry) throw new Error('no YuE2 model in the catalog to store the adapter beside');
      const slot = loraSlots.find((candidate) => candidate.name === name);
      const saved = await uploadAdapter(file, modelPathFor(entry), slot?.group ?? '', upload.signal);
      await refreshStored();
      setNamedParameter(name, saved.path);
      log(`YuE2 ${isNar ? 'NAR' : 'AR'} LoRA stored: ${saved.file}`);
    } catch (error) {
      if (!upload.signal.aborted) {
        setError(error instanceof Error ? error.message : String(error));
      }
    } finally {
      loraUploading = false;
      if (isNar) narLoraUpload = null; else loraUpload = null;
      const input = loraInputs[name];
      if (input) input.value = '';
    }
  }
  let coverAudioInput: HTMLInputElement | null = null;
  let coverRunning = false;
  const unloadSettingKey = 'audiocpp.ui.yue2.unloadSheetSageAfterConversion';
  let unloadAfterConversion = true;
  onMount(() => {
    unloadAfterConversion = localStorage.getItem(unloadSettingKey) !== 'false';
  });
  let coverStatus = '';
  let coverError = '';
  let abcDraft = '';
  let abcPreviewElement: HTMLDivElement | null = null;
  let lastRenderedAbc = '';
  let abcRenderError = '';
  let abcRender: ((target: HTMLElement, abc: string, options?: Record<string, unknown>) => unknown) | null = null;

  function specsByName(names: string[], specs: ParamSpec[]) {
    return names
      .map((name) => specs.find((spec) => spec.name === name))
      .filter((spec): spec is ParamSpec => spec !== undefined);
  }

  function specByName(name: string) {
    return paramSpecs.find((spec) => spec.name === name);
  }

  function setNamedParameter(name: string, value: unknown) {
    const spec = specByName(name);
    if (spec) setParameterValue(spec, value);
  }

  function decodedArtifactPayload(payload: string) {
    try {
      return atob(payload);
    } catch {
      return payload;
    }
  }

  function abcFromResult(result: Record<string, unknown>) {
    if (typeof result.text === 'string' && result.text.trim()) return result.text;
    const artifacts = Array.isArray(result.artifacts) ? result.artifacts : [];
    for (const artifact of artifacts) {
      if (!artifact || typeof artifact !== 'object') continue;
      const entry = artifact as { id?: unknown; payload?: unknown; meta?: Record<string, unknown> };
      const format = String(entry.meta?.format || entry.meta?.extension || entry.id || '');
      if (/abc|score/i.test(format) && typeof entry.payload === 'string') {
        return decodedArtifactPayload(entry.payload);
      }
    }
    return '';
  }

  function sheetSageModel() {
    return catalogEntries.find((entry) => entry.family === 'sheetsage2') || null;
  }

  async function ensureSheetSageLoaded(entry: CatalogEntry) {
    if (loadedModels.some((model) => model.id === entry.id && model.loaded)) return;
    const current = await models();
    const resident = current.find((model) => model.id === entry.id && model.loaded);
    if (resident) return;
    if (!server?.ui_management) {
      if (!current.some((model) => model.id === entry.id)) {
        throw new Error('SheetSage2 is not registered. Add SheetSage2 to server config.');
      }
      return;
    }
    await loadModel({
      id: entry.id,
      path: modelPathFor(entry),
      family: entry.family,
      task: entry.task,
      mode: entry.mode || 'offline',
      load_options: entry.load_options || {},
      session_options: sessionOptionsFor(entry)
    });
  }

  async function transcribeCoverScore() {
    if (!coverAudioFile || coverRunning) return;
    coverRunning = true;
    coverError = '';
    coverStatus = 'Preparing SheetSage2 cover score transcription...';
    const shouldUnload = unloadAfterConversion;
    let cleanupModelId: string | null = null;
    try {
      const entry = sheetSageModel();
      if (!entry) throw new Error('SheetSage2 is not available in the model catalog.');
      await ensureSheetSageLoaded(entry);
      cleanupModelId = entry.id;
      await refreshModels();
      coverStatus = 'Uploading source song...';
      const audio = await uploadFile(coverAudioFile);
      coverStatus = 'Transcribing source song to ABC with SheetSage2...';
      const result = await runTask({
        model: entry.id,
        request: {
          audio,
          options: {}
        }
      });
      const abc = abcFromResult(result);
      if (!abc.trim()) throw new Error('SheetSage2 did not return an ABC score.');
      abcDraft = abc;
      setNamedParameter('abc', abc);
      setNamedParameter('abc_file', '');
      setNamedParameter('cot', 'melody');
      coverStatus = 'ABC score imported. Review/edit the sheet before generating the cover.';
      log('SheetSage2 cover score imported into Yue2 ABC conditioning.');
    } catch (error) {
      coverError = error instanceof Error ? error.message : String(error);
      coverStatus = '';
      log(`SheetSage2 cover transcription failed: ${coverError}`);
    } finally {
      if (shouldUnload && cleanupModelId) {
        try {
          if (server?.ui_management) {
            await unloadModel(cleanupModelId);
          } else {
            await jsonRequest('/v1/tasks/unload_models', {
              method: 'POST',
              body: JSON.stringify({ model_ids: [cleanupModelId] })
            });
          }
          log('SheetSage2 unloaded after cover transcription.');
        } catch (error) {
          const message = `SheetSage2 unload failed: ${error instanceof Error ? error.message : String(error)}`;
          coverError = [coverError, message].filter(Boolean).join(' ');
          log(message);
        }
        try {
          await refreshModels();
        } catch (error) {
          const message = `Model status refresh failed: ${error instanceof Error ? error.message : String(error)}`;
          coverError = [coverError, message].filter(Boolean).join(' ');
          log(message);
        }
      }
      coverRunning = false;
    }
  }

  function clearCoverAudio() {
    coverAudioFile = null;
    if (coverAudioInput) coverAudioInput.value = '';
  }

  function updateAbc(value: string) {
    abcDraft = value;
    setNamedParameter('abc', value);
    if (value.trim()) setNamedParameter('abc_file', '');
  }

  async function renderAbcPreview(value: string) {
    const abc = value.trim();
    if (!abcPreviewElement) return;
    if (!abc) {
      abcPreviewElement.replaceChildren();
      lastRenderedAbc = '';
      abcRenderError = '';
      return;
    }
    if (abc === lastRenderedAbc) return;
    await tick();
    if (!abcPreviewElement) return;
    try {
      if (!abcRender) {
        const mod = await import('abcjs');
        abcRender = mod.renderAbc;
      }
      abcPreviewElement.replaceChildren();
      abcRender(abcPreviewElement, abc, {
        add_classes: true,
        responsive: 'resize'
      });
      lastRenderedAbc = abc;
      abcRenderError = '';
    } catch (error) {
      abcPreviewElement.replaceChildren();
      lastRenderedAbc = '';
      abcRenderError = error instanceof Error ? error.message : String(error);
    }
  }

  $: componentSpecs = specsByName(componentParamNames, paramSpecs);
  $: coreSpecs = specsByName(coreParamNames, paramSpecs);
  $: abcSpecs = specsByName(abcParamNames, paramSpecs);
  $: semanticSpecs = specsByName(semanticParamNames, paramSpecs);
  $: plannerSpecs = specsByName(plannerParamNames, paramSpecs);
  $: if (String(advancedValues.abc || '') !== abcDraft && !coverRunning) {
    abcDraft = String(advancedValues.abc || '');
  }
  $: void renderAbcPreview(abcDraft);
</script>

<div class="model-form yue2-form">
  <div class="yue2-field wide">
    <label for="lyrics">{tr('request.lyrics')} <span>{tr('voice.required')}</span></label>
    <textarea id="lyrics" rows="5" bind:value={lyrics} required aria-required="true"
      placeholder="[Verse]&#10;...&#10;[Chorus]&#10;..."></textarea>
  </div>

  <div class="yue2-grid yue2-grid-components">
    <div class="yue2-field">
      <label for="seed">{tr('request.seed')} <span>{tr('request.randomSeed')}</span></label>
      <input id="seed" type="number" min="-1" max="4294967295" step="1" bind:value={seed} />
    </div>
    {#each componentSpecs as spec}
      <div class="yue2-field">
        <label for={'param-' + spec.name}>{localizedParameterText(spec, 'label', tr)}</label>
        <select id={'param-' + spec.name} value={String(advancedValues[spec.name] ?? '')}
          on:change={(event) => setParameterValue(spec, event.currentTarget.value)}>
          {#each spec.choices || [] as choice}<option value={choice}>{choice}</option>{/each}
        </select>
        {#if localizedParameterText(spec, 'info', tr)}<small>{localizedParameterText(spec, 'info', tr)}</small>{/if}
      </div>
    {/each}
  </div>

  {#if specByName('ar_lora') || specByName('nar_lora')}
    <div class="yue2-grid">
      {#each loraSlots as slot}
        {#if specByName(slot.name)}
          <div class="yue2-field">
            <label for={'param-' + slot.name}>{slot.title} adapter</label>
            <select id={'param-' + slot.name}
              disabled={!server?.ui_management || busy || loraUploading}
              value={String(advancedValues[slot.name] ?? '')}
              on:change={(event) => setNamedParameter(slot.name, event.currentTarget.value)}>
              <option value="">None</option>
              {#each adaptersFor(slot.group) as adapter}
                <option value={adapter.path}>{adapterOptionLabel(adapter)}</option>
              {/each}
              {#if advancedValues[slot.name] && !stored.some((a) => a.path === advancedValues[slot.name])}
                <option value={String(advancedValues[slot.name])}>{String(advancedValues[slot.name])}</option>
              {/if}
            </select>
            <small>{slot.blurb}</small>
            <input class="file file-native" type="file" accept=".safetensors"
              bind:this={loraInputs[slot.name]}
              disabled={!server?.ui_management || busy || loraUploading}
              on:change={(event) => selectLora(event.currentTarget.files?.[0] || null, slot.name)} />
            <div class="media-actions">
              <button type="button" disabled={!server?.ui_management || busy || loraUploading}
                on:click={() => loraInputs[slot.name]?.click()}>
                {loraUploading ? 'Uploading...' : 'Upload a file'}</button>
              <button type="button"
                disabled={!server?.ui_management || busy || loraUploading || !advancedValues[slot.name]}
                on:click={() => setNamedParameter(slot.name, '')}>Clear</button>
              {#if String(advancedValues[slot.name] ?? '').startsWith('loras/')}
                <button type="button" disabled={!server?.ui_management || busy}
                  on:click={() => removeAdapter(String(advancedValues[slot.name]), slot.name)}>Delete stored</button>
              {/if}
            </div>
            {#if slot.name === 'ar_lora' && loraError}<span class="yue2-error" role="alert">{loraError}</span>{/if}
            {#if slot.name === 'nar_lora' && narLoraError}<span class="yue2-error" role="alert">{narLoraError}</span>{/if}
          </div>
          <div class="yue2-field">
            <label for={'param-' + slot.scale}>{slot.title} strength</label>
            <input id={'param-' + slot.scale} type="number" step="0.1"
              disabled={!server?.ui_management || busy || loraUploading || !advancedValues[slot.name]}
              value={Number(advancedValues[slot.scale] ?? 1)}
              on:change={(event) => {
                if (Number.isFinite(event.currentTarget.valueAsNumber)) {
                  setNamedParameter(slot.scale, event.currentTarget.valueAsNumber);
                }
              }} />
            {#if slot.name === 'nar_lora'}
              <small>Scales the LoRA deltas only; full vae2llm/llm2vae replacements stay at full strength. 0 disables the whole adapter.</small>
            {:else}
              <small>0 disables the adapter.</small>
            {/if}
          </div>
        {/if}
      {/each}
    </div>

    <details class="yue2-details yue2-lora-fetch">
      <summary>Get adapters from Hugging Face <span>{stored.length} stored</span></summary>
      <div class="yue2-field wide">
        <label for="yue2-lora-repo">Repository</label>
        <div class="media-actions">
          <input id="yue2-lora-repo" type="text" placeholder="owner/model, e.g. Mothersuperior/YuE2-instrumental-cot-full-loras"
            bind:value={repoId} disabled={!server?.ui_management || repoBusy}
            on:keydown={(event) => { if (event.key === 'Enter') browseRepo(); }} />
          <button type="button" disabled={!server?.ui_management || repoBusy || !repoId.trim()}
            on:click={browseRepo}>{repoBusy ? 'Reading...' : 'List adapters'}</button>
        </div>
        <small>Each file is identified by reading its header, so nothing large is downloaded to find out what it is. Only unfused adapters are offered; pick the slot it is for, and whether it suits YuE2 shows when the model loads it.</small>
        {#if repoError}<span class="yue2-error" role="alert">{repoError}</span>{/if}
        {#if repoNotice}<small>{repoNotice}</small>{/if}
      </div>
      {#if repoAdapters.length}
        <div class="yue2-field wide">
          {#each repoAdapters.filter((a) => a.unfused_lora) as adapter}
            <div class="media-actions">
              {#each loraSlots as slot}
                <button type="button"
                  disabled={!server?.ui_management || Boolean(downloading)
                    || stored.some((a) => a.file === adapter.file && a.group === slot.group)}
                  on:click={() => fetchAdapter(adapter, slot)}>
                  {downloading === adapter.repo_file ? 'Downloading...'
                    : stored.some((a) => a.file === adapter.file && a.group === slot.group) ? `Stored as ${slot.group.toUpperCase()}`
                    : `Download as ${slot.group.toUpperCase()}`}</button>
              {/each}
              <span>{adapter.file} &middot; {megabytes(adapter.bytes)}{adapter.rank ? ` · rank ${adapter.rank}` : ''}</span>
            </div>
          {/each}
          {#if repoAdapters.some((a) => !a.unfused_lora)}
            <small>{repoAdapters.filter((a) => !a.unfused_lora).length} other file(s) in this repo are not unfused adapters (ComfyUI layouts, or other components).</small>
          {/if}
        </div>
      {/if}
      {#if storeDirectory}<small>Stored in {storeDirectory}</small>{/if}
      {#if storeError}<span class="yue2-error" role="alert">{storeError}</span>{/if}
    </details>

  {/if}

  <div class="yue2-grid yue2-grid-core">
    {#each coreSpecs as spec}
      <div class="yue2-field" class:wide={spec.type === 'text'}>
        <label for={'param-' + spec.name}>{localizedParameterText(spec, 'label', tr)}</label>
        {#if spec.type === 'choice'}
          <select id={'param-' + spec.name} value={String(advancedValues[spec.name] ?? '')}
            on:change={(event) => setParameterValue(spec, event.currentTarget.value)}>
            {#each spec.choices || [] as choice}<option value={choice}>{choice}</option>{/each}
          </select>
        {:else if spec.type === 'slider'}
          <div class="yue2-range">
            <input id={'param-' + spec.name} type="range" min={spec.minimum} max={spec.maximum} step={spec.step}
              value={Number(advancedValues[spec.name] ?? spec.default)}
              on:input={(event) => setParameterValue(spec, event.currentTarget.valueAsNumber)} />
            <output>{String(advancedValues[spec.name])}</output>
          </div>
        {:else}
          <input id={'param-' + spec.name} type={spec.type === 'number' ? 'number' : 'text'}
            min={spec.minimum} max={spec.maximum} step={spec.step}
            value={String(advancedValues[spec.name] ?? '')}
            placeholder={localizedParameterText(spec, 'placeholder', tr)}
            on:input={(event) => setParameterValue(spec,
              spec.type === 'number' ? event.currentTarget.valueAsNumber : event.currentTarget.value)} />
        {/if}
        {#if localizedParameterText(spec, 'info', tr)}<small>{localizedParameterText(spec, 'info', tr)}</small>{/if}
      </div>
    {/each}
  </div>

  <details class="yue2-details">
    <summary>Yue2 ABC conditioning <span>{abcSpecs.length}</span></summary>
    <div class="yue2-cover">
      <section class="yue2-cover-card">
        <div class="yue2-cover-head">
          <div>
            <strong>Cover source</strong>
            <small>Use SheetSage2 + MERT2 to extract an editable ABC score from a song.</small>
          </div>
          <button type="button" disabled={!coverAudioFile || coverRunning} on:click={transcribeCoverScore}>
            {coverRunning ? 'Transcribing...' : 'Extract ABC'}
          </button>
        </div>
        <label class="yue2-unload-toggle">
          <input type="checkbox" role="switch"
            checked={unloadAfterConversion}
            disabled={coverRunning}
            on:change={(event) => {
              unloadAfterConversion = event.currentTarget.checked;
              localStorage.setItem(unloadSettingKey, String(unloadAfterConversion));
            }} />
          <span>Unload SheetSage2 after conversion</span>
        </label>
        <small class="yue2-error">Warning: VRAM may remain in use after unloading.</small>
        <input id="yue2-cover-audio" class="file file-native" type="file" accept="audio/*"
          bind:this={coverAudioInput}
          on:change={(event) => coverAudioFile = event.currentTarget.files?.[0] || null} />
        <label class="file-picker yue2-cover-picker" for="yue2-cover-audio">
          <strong>Choose source song</strong>
          <span>{coverAudioFile?.name || tr('file.none')}</span>
        </label>
        <div class="media-actions">
          <button type="button" disabled={!coverAudioFile || coverRunning} on:click={clearCoverAudio}>Clear</button>
          {#if coverStatus}<span>{coverStatus}</span>{/if}
          {#if coverError}<span class="yue2-error">{coverError}</span>{/if}
        </div>
        <MediaPreview file={coverAudioFile} kind="audio" label={tr('file.preview')} />
      </section>

      <section class="yue2-cover-card">
        <div class="yue2-cover-head">
          <div>
            <strong>ABC score editor</strong>
            <small>Edit the extracted score here. The sheet preview updates from this ABC.</small>
          </div>
        </div>
        <textarea id="param-abc" rows="8"
          value={abcDraft}
          placeholder="X:1&#10;T:Cover melody&#10;M:4/4&#10;L:1/8&#10;K:C&#10;C D E F | G A B c |"
          on:input={(event) => updateAbc(event.currentTarget.value)}></textarea>
        {#if specByName('abc_file')}
          <div class="yue2-field">
            <label for="param-abc_file">{localizedParameterText(specByName('abc_file')!, 'label', tr)}</label>
            <input id="param-abc_file" type="text"
              value={String(advancedValues.abc_file ?? '')}
              placeholder={localizedParameterText(specByName('abc_file')!, 'placeholder', tr)}
              on:input={(event) => setNamedParameter('abc_file', event.currentTarget.value)} />
          </div>
        {/if}
      </section>

      <section class="yue2-cover-card yue2-sheet-card">
        <div class="yue2-cover-head">
          <div>
            <strong>Sheet preview</strong>
            <small>Rendered from the editable ABC score.</small>
          </div>
        </div>
        <div class="yue2-sheet-preview" bind:this={abcPreviewElement}>
          {#if !abcDraft.trim()}<span>Extract or paste ABC to preview the score.</span>{/if}
        </div>
        {#if abcRenderError}<span class="yue2-error">{abcRenderError}</span>{/if}
      </section>
    </div>
  </details>

  <details class="yue2-details">
    <summary>Yue2 semantic sampling <span>{semanticSpecs.length}</span></summary>
    <div class="yue2-grid">
      {#each semanticSpecs as spec}
        <div class="yue2-field">
          <label for={'param-' + spec.name}>{localizedParameterText(spec, 'label', tr)}</label>
          {#if spec.type === 'slider'}
            <div class="yue2-range">
              <input id={'param-' + spec.name} type="range" min={spec.minimum} max={spec.maximum} step={spec.step}
                value={Number(advancedValues[spec.name] ?? spec.default)}
                on:input={(event) => setParameterValue(spec, event.currentTarget.valueAsNumber)} />
              <output>{String(advancedValues[spec.name])}</output>
            </div>
          {:else}
            <input id={'param-' + spec.name} type="number" min={spec.minimum} max={spec.maximum} step={spec.step}
              value={String(advancedValues[spec.name] ?? '')}
              on:input={(event) => setParameterValue(spec, event.currentTarget.valueAsNumber)} />
          {/if}
        </div>
      {/each}
    </div>
  </details>

  <details class="yue2-details">
    <summary>Yue2 ABC planner sampling <span>{plannerSpecs.length}</span></summary>
    <div class="yue2-grid">
      {#each plannerSpecs as spec}
        <div class="yue2-field">
          <label for={'param-' + spec.name}>{localizedParameterText(spec, 'label', tr)}</label>
          {#if spec.type === 'slider'}
            <div class="yue2-range">
              <input id={'param-' + spec.name} type="range" min={spec.minimum} max={spec.maximum} step={spec.step}
                value={Number(advancedValues[spec.name] ?? spec.default)}
                on:input={(event) => setParameterValue(spec, event.currentTarget.valueAsNumber)} />
              <output>{String(advancedValues[spec.name])}</output>
            </div>
          {:else}
            <input id={'param-' + spec.name} type="number" min={spec.minimum} max={spec.maximum} step={spec.step}
              value={String(advancedValues[spec.name] ?? '')}
              on:input={(event) => setParameterValue(spec, event.currentTarget.valueAsNumber)} />
          {/if}
        </div>
      {/each}
    </div>
  </details>
</div>

<style>
  .yue2-form {
    display: grid;
    gap: 14px;
  }

  .yue2-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px 12px;
    align-items: start;
  }

  .yue2-grid-core,
  .yue2-grid-components {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .yue2-field {
    min-width: 0;
  }

  .yue2-field.wide {
    grid-column: 1 / -1;
  }

  .yue2-field label {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
    min-height: 16px;
    margin: 0 0 5px;
    color: var(--ink);
    font-size: 11px;
    font-weight: 700;
    line-height: 1.3;
  }

  .yue2-field label span,
  .yue2-field small {
    color: var(--muted);
    font-size: 10px;
    font-weight: 500;
    line-height: 1.35;
  }

  .yue2-field small {
    display: block;
    min-height: 14px;
    margin-top: 5px;
  }

  .yue2-form :global(input),
  .yue2-form :global(select),
  .yue2-form :global(textarea) {
    width: 100%;
    min-width: 0;
    font-size: 12px;
    line-height: 1.35;
  }

  .yue2-form :global(textarea) {
    resize: vertical;
  }

  .yue2-range {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 62px;
    gap: 10px;
    align-items: center;
  }

  .yue2-range :global(input) {
    padding: 0;
    accent-color: var(--cyan);
  }

  .yue2-range output {
    color: var(--cyan);
    font: 12px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    text-align: right;
  }

  .yue2-details {
    margin: 0;
  }

  .yue2-details > .yue2-grid {
    padding: 0 10px 10px;
  }

  /* This section holds bare fields rather than a .yue2-grid, so it needs the
     same horizontal inset the grid-based sections get. */
  .yue2-lora-fetch > .yue2-field,
  .yue2-lora-fetch > small,
  .yue2-lora-fetch > .yue2-error {
    display: block;
    padding: 0 10px 10px;
  }

  .yue2-lora-fetch > .yue2-field + .yue2-field {
    padding-top: 2px;
  }

  /* A direct child, so it misses the .yue2-field small styling and would
     otherwise render louder than the helper text it sits beneath. */
  .yue2-lora-fetch > small {
    color: var(--muted);
    font-size: 10px;
    font-weight: 500;
    line-height: 1.35;
    overflow-wrap: anywhere;
  }

  .yue2-lora-fetch > :last-child {
    padding-bottom: 12px;
  }

  .yue2-cover {
    display: grid;
    gap: 12px;
    padding: 0 10px 10px;
  }

  .yue2-cover-card {
    display: grid;
    gap: 10px;
    min-width: 0;
    padding: 10px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--control-bg);
  }

  .yue2-sheet-card {
    overflow: hidden;
  }

  .yue2-sheet-preview {
    min-height: 180px;
    max-height: 420px;
    overflow: auto;
    padding: 10px;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: #f8fbff;
    color: #0e1726;
  }

  .yue2-sheet-preview span {
    color: var(--muted);
    font-size: 12px;
  }

  .yue2-sheet-preview :global(svg) {
    display: block;
    max-width: 100%;
  }

  .yue2-cover-head {
    display: flex;
    align-items: start;
    justify-content: space-between;
    gap: 12px;
  }

  .yue2-cover-head strong {
    display: block;
    color: var(--text-strong);
    font-size: 12px;
    line-height: 1.3;
  }

  .yue2-cover-head small {
    display: block;
    margin-top: 3px;
    color: var(--muted);
    font-size: 10px;
    line-height: 1.35;
  }

  .yue2-cover-head button {
    width: auto;
    min-width: 110px;
  }

  .yue2-cover-picker {
    margin: 0;
  }

  .yue2-unload-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    line-height: 1.4;
  }

  .yue2-unload-toggle input {
    appearance: none;
    position: relative;
    flex: 0 0 34px;
    width: 34px;
    height: 20px;
    padding: 0;
    margin: 0;
    border: 1px solid var(--line);
    border-radius: 10px;
    background: var(--muted);
    cursor: pointer;
  }

  .yue2-unload-toggle input::before {
    content: '';
    position: absolute;
    top: 2px;
    left: 2px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: white;
  }

  .yue2-unload-toggle input:checked {
    background: var(--cyan);
  }

  .yue2-unload-toggle input:checked::before {
    transform: translateX(14px);
  }

  .yue2-unload-toggle input:focus-visible {
    outline: 2px solid var(--cyan);
    outline-offset: 2px;
  }

  .yue2-unload-toggle input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .yue2-error {
    color: var(--danger);
  }

  .yue2-details > summary {
    min-height: 42px;
    font-size: 11px;
    line-height: 1.3;
  }

  @media (max-width: 760px) {
    .yue2-grid,
    .yue2-grid-components,
    .yue2-grid-core {
      grid-template-columns: 1fr;
    }
  }
</style>
