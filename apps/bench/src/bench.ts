// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The bench page. It reads models/index.json and each model's manifest from the extension package, plans one row
// per model x variant x runtime, and runs each row in a worker of its own (so a hung row can be given up on and
// the page stays alive), then shows a table and a plain-text summary to copy. The page itself never touches
// onnxruntime-web; the workers do (bench-*.ts).
//
// Options for looking into things, in the page's address: ?runtimes=wasm,webgpu,webgpu-jsep (only these),
// ?quick=1 (one warm-up and one timed run per batch), ?stall=SECONDS (how long a silent worker is waited for).

import { estimates, timedRows, type FrameItems } from './estimate';
import { collectEnv, supportOf } from './env';
import { parseIndex, parseManifest } from './model-manifest';
import { planRows, type ModelEntry, type RowPlan } from './plan';
import { RUNTIMES, runtimeOf, type RuntimeId } from './runtimes';
import { REPEATS } from './stats';
import { checkText, envLines, fmtMs, frameLines, renderSummary, runtimeName, sortRows, type EnvInfo } from './summary';
import { blankRow, type FromWorker, type RowJob, type RowResult, type ToWorker } from './types';

// A worker that has said nothing for this long is given up on (?stall=SECONDS for a machine so slow that one step takes longer).
const STALL_MS = (Number(new URLSearchParams(location.search).get('stall')) || 300) * 1000;

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const runButton = $<HTMLButtonElement>('run');
const stopButton = $<HTMLButtonElement>('stop');
const copyButton = $<HTMLButtonElement>('copy');
const statusEl = $('status');
const copiedEl = $('copied');
const envEl = $('env');
const bodyEl = $('rows');
const frameEl = $('frame');
const summaryEl = $<HTMLTextAreaElement>('summary');

let env: EnvInfo | null = null;
let entries: ModelEntry[] = [];
let rows: RowResult[] = [];
let problems: string[] = [];
let state: 'ready' | 'running' | 'done' | 'stopped' = 'ready';
let startedAt = 0;
let elapsedMs: number | null = null;
let hiddenDuringRun = false;
let current: { job: RowJob; giveUp: (why: string) => void } | null = null;
let stopRequested = false;

const modelsUrl = new URL('models/', location.href).href;
// ?quick=1: one warm-up and one timed run per batch, to see that the models run and check (slow machines, debugging)
const quick = new URLSearchParams(location.search).get('quick') === '1';

// ?runtimes=wasm,webgpu limits the run (for looking into one runtime); by default all of them run
function chosenRuntimes(): RuntimeId[] {
  const asked = new URLSearchParams(location.search).get('runtimes');
  if (!asked) return RUNTIMES.map((r) => r.id);
  const ids = asked.split(',').map((s) => s.trim());
  return RUNTIMES.filter((r) => ids.includes(r.id)).map((r) => r.id);
}

/** Models the per-frame estimate cannot include, so that it is not mistaken for a whole frame's cost. */
function leftOut(): string[] {
  return entries.flatMap((e) => {
    if (!e.manifest) return [`${e.name} could not be read`];
    return e.manifest.perFrame === null ? [`${e.manifest.id} has no per_frame items`] : [];
  });
}

function frameItems(): FrameItems[] {
  return entries.flatMap((e) => (e.manifest && e.manifest.perFrame !== null ? [{ model: e.manifest.id, items: e.manifest.perFrame }] : []));
}

function status(text: string): void {
  statusEl.textContent = text;
}

// what the row is doing, and for how long: a session for a big model takes many seconds, and the line must not look stuck
let step = '';
let stepAt = 0;
function setStep(text: string): void {
  step = text;
  stepAt = performance.now();
  status(text);
}
setInterval(() => {
  const seconds = Math.round((performance.now() - stepAt) / 1000);
  if (state === 'running' && step && seconds >= 3) status(`${step} (${seconds} s)`);
}, 1000);

// a laptop that dims and sleeps in the middle of a run gives numbers that mean little
let awake: { release(): Promise<void> } | null = null;
async function keepAwake(): Promise<void> {
  try {
    awake = (await (navigator as unknown as { wakeLock?: { request(type: 'screen'): Promise<{ release(): Promise<void> }> } }).wakeLock?.request('screen')) ?? null;
  } catch {
    awake = null;
  }
}

async function fetchJson(name: string): Promise<unknown> {
  const res = await fetch(new URL(name, modelsUrl));
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/** True when the file is in the models folder (the first bytes are not read: the body is cancelled). */
async function isPresent(file: string): Promise<boolean> {
  try {
    const res = await fetch(new URL(file, modelsUrl));
    void res.body?.cancel();
    return res.ok;
  } catch {
    return false;
  }
}

async function loadEntries(): Promise<ModelEntry[]> {
  let names: string[];
  try {
    names = parseIndex(await fetchJson('index.json'));
  } catch (e) {
    problems.push(`models/index.json could not be read (${e instanceof Error ? e.message : String(e)}). Unzip the whole folder and load it as it is.`);
    return [];
  }
  if (names.length === 0) problems.push('models/index.json lists no model.');
  const out: ModelEntry[] = [];
  for (const name of names) {
    try {
      out.push({ name, manifest: parseManifest(await fetchJson(name)), error: null });
    } catch (e) {
      out.push({ name, manifest: null, error: `${name} could not be read: ${e instanceof Error ? e.message : String(e)}` });
    }
  }
  return out;
}

function skippedRow(p: RowPlan): RowResult {
  const row = blankRow({ model: p.model, title: p.title, precision: p.precision, runtime: p.runtime });
  row.status = p.action === 'error' ? 'error' : 'skipped';
  row.note = p.note;
  if (p.action === 'error') row.errors.push(p.note);
  return row;
}

/** Runs one row in a fresh worker; the worker is given up on (and the row says so) if it goes quiet or dies. */
function runInWorker(job: RowJob, onProgress: (text: string) => void): Promise<RowResult> {
  return new Promise((resolve) => {
    const worker = new Worker(new URL(runtimeOf(job.runtime).worker, location.href), { type: 'module', name: `bench-${job.runtime}` });
    let timer = 0;
    let last = 'starting';
    let finished = false;
    const finish = (row: RowResult) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      worker.terminate();
      current = null;
      resolve(row);
    };
    const giveUp = (why: string) => {
      const row = blankRow(job);
      row.status = 'error';
      row.errors.push(`${why} (last step: ${last})`);
      if (job.manifest.check) row.check = { metric: job.manifest.check.metric, batch: job.manifest.check.batch, value: NaN, pass: false, outputs: [], note: '', error: 'not run' };
      finish(row);
    };
    const arm = () => {
      clearTimeout(timer);
      timer = window.setTimeout(() => giveUp(`no word from the worker for ${STALL_MS / 1000} s`), STALL_MS);
    };
    worker.onmessage = (e: MessageEvent<FromWorker>) => {
      arm();
      if (e.data.kind === 'progress') {
        last = e.data.text;
        onProgress(e.data.text);
      } else {
        finish(e.data.row);
      }
    };
    worker.onerror = (ev) => giveUp(`worker error: ${ev.message || 'the worker script did not start'}`);
    worker.onmessageerror = () => giveUp('worker message could not be read');
    current = { job, giveUp };
    arm();
    const msg: ToWorker = { kind: 'run', job };
    worker.postMessage(msg);
  });
}

// ---- drawing ----

function cell(tr: HTMLElement, text: string, cls = ''): HTMLTableCellElement {
  const td = document.createElement('td');
  td.textContent = text;
  if (cls) td.className = cls;
  tr.appendChild(td);
  return td;
}

/** A row of the table, and under it (when it has errors) a row of its own that spans the table. */
function drawRow(r: RowResult): HTMLElement[] {
  const tr = document.createElement('tr');
  tr.className = r.status;
  cell(tr, r.model).title = r.title;
  cell(tr, r.precision);
  cell(tr, runtimeName(r));
  if (r.status === 'skipped') {
    cell(tr, r.note, 'note').colSpan = 5;
    return [tr];
  }
  cell(tr, fmtMs(r.initMs), 'num');
  cell(tr, fmtMs(r.loadMs), 'num');
  cell(tr, fmtMs(r.firstMs), 'num');
  const times = cell(tr, '', 'times');
  for (const b of r.batches) {
    const line = document.createElement('div');
    line.textContent = b.error
      ? `b${b.batch}  error`
      : `b${b.batch}  ${fmtMs(b.medianMs)} / ${fmtMs(b.p90Ms)} ms  (${fmtMs(b.perItemMs)} / ${fmtMs(b.p90PerItemMs)} per item${b.runs === REPEATS ? '' : `, only ${b.runs} runs`})`;
    times.appendChild(line);
  }
  const check = cell(tr, r.check ? checkText(r.check) : 'no check in the manifest', r.check ? (r.check.pass ? 'pass check' : 'fail check') : 'note check');
  check.title = 'the worst output against its tolerance';
  if (r.errors.length === 0 && r.warnings.length === 0) return [tr];
  const detail = document.createElement('tr');
  detail.className = 'detail';
  const td = document.createElement('td');
  td.colSpan = 8;
  for (const [cls, mark, texts] of [['err', '!', r.errors], ['warn', '~', r.warnings]] as const) {
    for (const t of texts) {
      const line = document.createElement('div');
      line.className = cls;
      line.textContent = `${mark} ${t}`;
      td.appendChild(line);
    }
  }
  detail.appendChild(td);
  return [tr, detail];
}

function render(): void {
  if (env) envEl.textContent = envLines(env).join('\n');
  const ordered = sortRows(rows, entries.map((e) => (e.manifest ? e.manifest.id : e.name.replace(/\.bench\.json$/, ''))));
  bodyEl.replaceChildren(...ordered.flatMap(drawRow));
  const frame = frameItems();
  const est = estimates(timedRows(rows), frame, chosenRuntimes());
  // before a run there is nothing to estimate
  frameEl.textContent = state === 'ready' ? '' : [...problems.map((p) => `PROBLEM: ${p}`), ...frameLines(frame, est, leftOut())].join('\n');
  if (env) {
    summaryEl.value = renderSummary({
      env,
      rows,
      modelOrder: entries.map((e) => (e.manifest ? e.manifest.id : e.name.replace(/\.bench\.json$/, ''))),
      frame,
      est,
      state,
      elapsedMs,
      hiddenDuringRun,
      problems,
      leftOut: leftOut(),
      quick,
    });
  }
}

function setBusy(busy: boolean): void {
  runButton.disabled = busy;
  stopButton.disabled = !busy;
}

// ---- the run ----

async function run(): Promise<void> {
  setBusy(true);
  state = 'running';
  stopRequested = false;
  rows = [];
  problems = [];
  entries = [];
  hiddenDuringRun = document.hidden;
  startedAt = performance.now();
  step = '';
  void keepAwake();
  elapsedMs = null;
  copiedEl.textContent = '';
  try {
    status('Looking at this browser...');
    env = await collectEnv(__ORT_VERSION__, __BENCH_VERSION__);
    render();

    status('Reading the models folder...');
    entries = await loadEntries();
    const present = new Map<string, boolean>();
    for (const e of entries) for (const v of e.manifest?.variants ?? []) present.set(v.file, await isPresent(v.file));
    const plan = planRows(entries, (f) => present.get(f) === true, supportOf(env), chosenRuntimes());
    rows = plan.filter((p) => p.action !== 'run').map(skippedRow);
    render();

    const todo = plan.filter((p) => p.action === 'run');
    const manifests = new Map(entries.flatMap((e) => (e.manifest ? [[e.manifest.id, e.manifest] as const] : [])));
    for (const [i, p] of todo.entries()) {
      if (stopRequested) break;
      const manifest = manifests.get(p.model)!;
      const runtime = p.runtime as RuntimeId;
      const job: RowJob = {
        model: p.model,
        title: p.title,
        precision: p.precision,
        runtime,
        ep: runtimeOf(runtime).ep,
        manifest,
        file: p.file,
        modelsUrl,
        ortBase: new URL('ort/', location.href).href,
        threads: env.threads,
        quick,
      };
      const head = `Row ${i + 1} of ${todo.length}`;
      setStep(`${head}: ${p.model} ${p.precision} on ${runtime}...`);
      const row = await runInWorker(job, (text) => setStep(`${head}: ${text}`));
      rows.push(row);
      render();
    }
    state = stopRequested ? 'stopped' : 'done';
  } catch (e) {
    problems.push(`the bench stopped early: ${e instanceof Error ? e.message : String(e)}`);
    state = 'stopped';
  } finally {
    elapsedMs = performance.now() - startedAt;
    step = '';
    void awake?.release().catch(() => undefined);
    awake = null;
    setBusy(false);
    render();
    const bad = rows.filter((r) => r.status === 'error').length;
    status(`${state === 'done' ? 'Done' : 'Stopped'} in ${(elapsedMs / 1000).toFixed(0)} s: ${rows.length} rows${bad ? `, ${bad} with errors` : ''}. Press Copy results, then paste the text back.`);
  }
}

async function copyResults(): Promise<void> {
  const text = summaryEl.value;
  try {
    await navigator.clipboard.writeText(text);
    copiedEl.textContent = `Copied (${text.length} characters).`;
  } catch {
    summaryEl.focus();
    summaryEl.select();
    copiedEl.textContent = document.execCommand('copy') ? `Copied (${text.length} characters).` : 'Copying was refused: select the text in the box and copy it by hand.';
  }
}

runButton.addEventListener('click', () => void run());
stopButton.addEventListener('click', () => {
  stopRequested = true;
  status('Stopping...');
  current?.giveUp('stopped by the user');
});
copyButton.addEventListener('click', () => void copyResults());
document.addEventListener('visibilitychange', () => {
  if (document.hidden && state === 'running') hiddenDuringRun = true;
});

// what this browser is, before anything runs
void collectEnv(__ORT_VERSION__, __BENCH_VERSION__).then((e) => {
  if (state !== 'ready') return;
  env = e;
  render();
});
