import { readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import { validateReviewAnswers, validateReviewPack, type ReviewAnswers, type ReviewPack } from '../../../packages/schema/src/index';

const APP = new URL('../dist/index.html', import.meta.url).href;

/** A labelled coloured rectangle, so the test needs no image fixtures in git. */
const tile = (text: string, hue: number) =>
  'data:image/svg+xml;base64,' +
  Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="120" height="168"><rect width="120" height="168" fill="hsl(${hue} 50% 40%)"/>` +
      `<text x="10" y="90" fill="#fff" font-size="16">${text}</text></svg>`,
  ).toString('base64');

const CARDS = [
  { value: 'FAK-001', label: 'Fake Hero' },
  { value: 'FAK-002', label: 'Fake Legend' },
  { value: 'FAK-003', label: 'Heroic Strike' },
  { value: 'FAK-004', label: 'Mystic Shield' },
];

function makePack(): ReviewPack {
  const files: Record<string, string> = {};
  CARDS.forEach((c, i) => (files[`art/${c.value}.svg`] = tile(c.label, i * 80)));
  const withArt = (c: (typeof CARDS)[number]) => ({ ...c, image: `art/${c.value}.svg` });
  const items = [0, 1, 2, 3].map((i) => {
    files[`crop/${i}.svg`] = tile(`crop ${i}`, 200 + i * 10);
    const proposal = withArt(CARDS[i]!);
    return {
      id: `item-${i}`,
      images: [`crop/${i}.svg`],
      proposal,
      confidence: 0.3 + i * 0.1,
      alternatives: CARDS.filter((c) => c.value !== proposal.value).slice(0, 3).map(withArt),
      note: `frame ${i}`,
    };
  });
  return {
    schema: 'rifteye.reviewpack',
    version: 1,
    id: 'e2e-pack',
    kind: 'identity',
    question: 'Is this the card?',
    items,
    vocabulary: CARDS,
    files,
    createdAt: '2026-09-26T00:00:00Z',
  };
}

test('review a pack from the keyboard, export valid answers, and continue after a reload', async ({ page }) => {
  const pack = makePack();
  expect(validateReviewPack(pack)).toEqual([]);
  const packFile = { name: 'e2e.reviewpack.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(pack)) };
  await page.goto(APP);
  page.on('dialog', (d) => void d.accept());

  await page.setInputFiles('#open-pack', packFile);
  await expect(page.locator('#question')).toHaveText('Is this the card?');
  await expect(page.locator('#position')).toHaveText('1 / 4');
  await expect(page.locator('#proposal .label')).toHaveText('Fake Hero');
  await expect(page.locator('#proposal .confidence')).toContainText('30% sure');
  await expect(page.locator('#alternatives .option')).toHaveCount(3);
  // Every picture comes from the pack itself.
  await expect(page.locator('#evidence img')).toHaveAttribute('src', /^data:image\/svg\+xml/);

  await page.keyboard.press('y'); // item 0: correct
  await expect(page.locator('#position')).toHaveText('2 / 4');
  await page.keyboard.press('2'); // item 1: wrong, it is the 2nd alternative (Heroic Strike)
  await expect(page.locator('#position')).toHaveText('3 / 4');

  await page.keyboard.press('n'); // item 2: wrong, typed name
  await expect(page.locator('#ask-input')).toBeFocused();
  await page.keyboard.type('myst');
  // The catalogue match first, then the typed name itself for cards the list does not have.
  await expect(page.locator('#suggestions li')).toHaveText(['Mystic ShieldFAK-004', 'Use “myst”not in the list']);
  await page.keyboard.press('Enter');
  await expect(page.locator('#position')).toHaveText('4 / 4');

  await page.keyboard.press('n'); // item 3: wrong, then Esc goes back without answering
  await page.keyboard.press('Escape');
  await expect(page.locator('#ask')).toBeHidden();
  await expect(page.locator('#position')).toHaveText('4 / 4');
  await page.keyboard.press('n'); // a card the list does not have: keep the typed name
  await page.keyboard.type('mech token');
  await expect(page.locator('#suggestions li')).toHaveText(['Use “mech token”not in the list']);
  await page.keyboard.press('Enter');
  await expect(page.locator('#done')).toBeVisible();
  await page.keyboard.press('ArrowLeft');
  await expect(page.locator('#answered')).toHaveText('You said: wrong, it is “mech token” (not in the list).');
  await page.keyboard.press('s'); // changed my mind: can't tell
  await expect(page.locator('#done')).toBeVisible();
  await expect(page.locator('#done-counts')).toHaveText("4 of 4 answered: 1 correct · 2 wrong · 1 can't tell.");

  await page.keyboard.press('z'); // undo restores what "can't tell" replaced: the typed name
  await expect(page.locator('#position')).toHaveText('4 / 4');
  await expect(page.locator('#answered')).toHaveText('You said: wrong, it is “mech token” (not in the list).');
  await page.keyboard.press('z'); // and once more: item 3 is open again
  await expect(page.locator('#answered')).toHaveText('');
  await page.locator('#keys button[data-key="n"]').click();
  await page.keyboard.press('Enter'); // empty box: wrong, don't know
  await expect(page.locator('#done')).toBeVisible();

  await page.keyboard.press('ArrowLeft'); // look back at item 3
  await expect(page.locator('#answered')).toHaveText("You said: wrong (you didn't know what it is).");

  await page.fill('#reviewer', 'e2e');
  await page.locator('#reviewer').dispatchEvent('change');
  const [download] = await Promise.all([page.waitForEvent('download'), page.click('#export')]);
  expect(download.suggestedFilename()).toBe('e2e-pack.answers.json');
  const doc = JSON.parse(readFileSync((await download.path())!, 'utf8')) as ReviewAnswers;
  expect(validateReviewAnswers(doc)).toEqual([]);
  expect(doc.reviewer).toBe('e2e');
  expect(doc.answers.map((a) => [a.itemId, a.verdict, a.value ?? null])).toEqual([
    ['item-0', 'correct', null],
    ['item-1', 'wrong', 'FAK-003'],
    ['item-2', 'wrong', 'FAK-004'],
    ['item-3', 'wrong', null],
  ]);

  // Autosave: reopening the same pack offers to continue (the dialog handler accepts).
  await page.reload();
  await page.setInputFiles('#open-pack', packFile);
  await expect(page.locator('#done')).toBeVisible();
  await page.keyboard.press('ArrowLeft');
  await expect(page.locator('#answered')).toHaveText("You said: wrong (you didn't know what it is).");
});
