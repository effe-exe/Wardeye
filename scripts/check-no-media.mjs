// Fails when git tracks footage, card art, audio, datasets or model weights (decision D-006).
// Our own documentation images may live under docs/, and extension icons under apps/*/icons/.
import { execFileSync } from 'node:child_process';

const BLOCKED =
  /\.(jpe?g|png|webp|gif|bmp|tiff?|avif|heic|mp4|m4v|mkv|mov|webm|flv|avi|m3u8|m4s|wav|mp3|aac|flac|ogg|opus|onnx|ort|pt|pth|ckpt|safetensors|tflite|engine|npy|npz|parquet|h5)$/i;
const ALLOWED = [/^docs\//, /^apps\/[^/]+\/icons\//];

const files = execFileSync('git', ['ls-files', '-z'], { encoding: 'utf8' }).split('\0').filter(Boolean);
const offending = files.filter((f) => BLOCKED.test(f) && !ALLOWED.some((re) => re.test(f)));

if (offending.length > 0) {
  console.error('These files must not be in git (see docs/decisions.md, D-006):');
  for (const f of offending) console.error(`  ${f}`);
  console.error('Keep media, datasets and weights in private storage and reference them by manifest.');
  process.exit(1);
}
console.log(`check:media ok (${files.length} tracked files, none blocked)`);
