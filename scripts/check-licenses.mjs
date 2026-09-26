// Fails when a shipped (non-dev) npm dependency is not permissively licensed (decision D-002).
// Dev tooling (compilers, test runners) is not shipped and is not checked here.
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const ALLOWED = new Set([
  'MIT', 'ISC', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', '0BSD', 'Zlib',
  'Unlicense', 'CC0-1.0', 'BlueOak-1.0.0', 'Python-2.0',
]);

/** Accepts SPDX expressions such as "(MIT OR Apache-2.0)" or "MIT AND Zlib". */
function isAllowed(expr) {
  if (!expr || typeof expr !== 'string') return false;
  const clean = expr.replace(/[()]/g, ' ').trim();
  if (/\bOR\b/.test(clean)) return clean.split(/\bOR\b/).some((part) => isAllowed(part));
  if (/\bAND\b/.test(clean)) return clean.split(/\bAND\b/).every((part) => isAllowed(part));
  return ALLOWED.has(clean.trim());
}

let tree;
try {
  tree = JSON.parse(execFileSync('npm', ['ls', '--omit=dev', '--all', '--json'], { encoding: 'utf8' }));
} catch (e) {
  // npm ls exits non-zero on peer-dependency warnings but still prints the tree.
  tree = JSON.parse(e.stdout || '{}');
}

const seen = new Map();
function walk(deps) {
  for (const [name, info] of Object.entries(deps || {})) {
    if (name.startsWith('@rifteye/')) {
      walk(info.dependencies);
      continue;
    }
    const key = `${name}@${info.version}`;
    if (seen.has(key)) continue;
    let license = null;
    try {
      const pkg = JSON.parse(readFileSync(join('node_modules', name, 'package.json'), 'utf8'));
      license = typeof pkg.license === 'string' ? pkg.license : pkg.license?.type ?? null;
    } catch {
      license = null;
    }
    seen.set(key, license);
    walk(info.dependencies);
  }
}
walk(tree.dependencies);

const bad = [...seen].filter(([, license]) => !isAllowed(license));
if (bad.length > 0) {
  console.error('Shipped dependencies with a non-permissive or unknown licence (see D-002):');
  for (const [pkg, license] of bad) console.error(`  ${pkg}: ${license ?? 'unknown'}`);
  process.exit(1);
}
console.log(`check:licenses ok (${seen.size} shipped dependencies, all permissive)`);
