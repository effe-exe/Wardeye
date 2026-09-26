import { describe, expect, it } from 'vitest';
import {
  newId,
  validateIndexManifest,
  validateLayoutPreset,
  validateReviewAnswers,
  validateReviewPack,
  validateTimelineDocument,
  validateTimelineEvent,
  type ReviewAnswers,
  type ReviewPack,
  type TimelineDocument,
  type TimelineEvent,
} from '@rifteye/schema';

const event = (over: Partial<TimelineEvent> = {}): TimelineEvent => ({
  id: newId(0),
  matchId: 'm1',
  t: 12.5,
  clock: 'media',
  type: 'card_played',
  player: 'A',
  zone: 'base',
  card: { cardId: 'fake-unit-001', confidence: 1 },
  evidence: ['manual'],
  engine: 'test',
  ...over,
});

const doc = (events: TimelineEvent[]): TimelineDocument => ({
  schema: 'rifteye.timeline',
  version: 1,
  match: { matchId: 'm1', source: { kind: 'file', ref: 'match.mp4' } },
  events,
  createdAt: '2026-09-26T00:00:00.000Z',
  updatedAt: '2026-09-26T00:00:00.000Z',
});

describe('validateTimelineEvent', () => {
  it('accepts a well-formed event', () => {
    expect(validateTimelineEvent(event())).toEqual([]);
  });

  it('reports every problem, with paths', () => {
    const issues = validateTimelineEvent({ ...event(), t: -1, type: 'nope', player: 'C', evidence: [] });
    expect(issues.map((i) => i.path).sort()).toEqual(['event.evidence', 'event.player', 'event.t', 'event.type']);
  });

  it('requires retracts on retracted events and score on score changes', () => {
    expect(validateTimelineEvent(event({ type: 'retracted' })).map((i) => i.path)).toEqual(['event.retracts']);
    expect(validateTimelineEvent(event({ type: 'score_changed' })).map((i) => i.path)).toEqual(['event.score']);
    expect(validateTimelineEvent(event({ type: 'score_changed', score: { A: 3, B: 2 } }))).toEqual([]);
  });

  it('allows an unidentified card and rejects bad confidence or quads', () => {
    expect(validateTimelineEvent(event({ card: { cardId: null, confidence: 0 } }))).toEqual([]);
    expect(validateTimelineEvent(event({ card: { cardId: 'x', confidence: 1.5 } })).map((i) => i.path)).toEqual([
      'event.card.confidence',
    ]);
    const quad: NonNullable<TimelineEvent['quad']> = [0, 0, 1, 0, 1, 1, 0, 2];
    expect(validateTimelineEvent(event({ quad })).map((i) => i.path)).toEqual(['event.quad']);
  });
});

describe('validateTimelineDocument', () => {
  it('accepts a valid document', () => {
    expect(validateTimelineDocument(doc([event(), event({ id: newId(1), t: 20 })]))).toEqual([]);
  });

  it('catches duplicate ids and events from another match', () => {
    const a = event();
    const issues = validateTimelineDocument(doc([a, { ...a }, event({ id: newId(5), matchId: 'other' })]));
    expect(issues.map((i) => i.message)).toContain('duplicate id');
    expect(issues.map((i) => i.message)).toContain('must equal match.matchId');
  });

  it('rejects the wrong schema tag', () => {
    expect(validateTimelineDocument({ ...doc([]), schema: 'x' }).map((i) => i.path)).toContain('schema');
  });
});

describe('validateLayoutPreset', () => {
  const layout = {
    schema: 'rifteye.layout',
    version: 1,
    id: 'example',
    match: { channels: ['example'] },
    regions: { overhead: [0.2, 0.05, 0.8, 0.95], handcam: [0, 0.7, 0.18, 1] },
    playerA: 'bottom',
  };

  it('accepts a valid preset', () => {
    expect(validateLayoutPreset(layout)).toEqual([]);
  });

  it('requires an overhead region and well-ordered rectangles', () => {
    expect(validateLayoutPreset({ ...layout, regions: { handcam: [0.5, 0, 0.1, 1] } }).map((i) => i.path).sort()).toEqual([
      'regions.handcam',
      'regions.overhead',
    ]);
  });
});

describe('validateIndexManifest', () => {
  const manifest = {
    schema: 'rifteye.index',
    version: 1,
    catalogVersion: '2026-09-26',
    model: 'dinov2-s14/test',
    modelSha256: 'abc',
    dim: 256,
    dtype: 'float16',
    rows: ['A-001', 'A-002'],
    createdAt: '2026-09-26T00:00:00Z',
  };

  it('accepts a valid manifest and rejects duplicate rows', () => {
    expect(validateIndexManifest(manifest)).toEqual([]);
    expect(validateIndexManifest({ ...manifest, rows: ['A-001', 'A-001'] }).map((i) => i.path)).toEqual(['rows']);
  });
});

describe('newId', () => {
  it('is 26 Crockford characters and sorts by time', () => {
    const a = newId(1_000);
    const b = newId(2_000);
    expect(a).toMatch(/^[0-9A-HJKMNP-TV-Z]{26}$/);
    expect(a < b).toBe(true);
  });
});

describe('review packs and answers', () => {
  const pixel = 'data:image/svg+xml;base64,PHN2Zy8+';
  const pack = (over: Partial<ReviewPack> = {}): ReviewPack => ({
    schema: 'rifteye.reviewpack',
    version: 1,
    id: 'pack-1',
    kind: 'identity',
    question: 'Is this the card?',
    items: [
      {
        id: 'i1',
        images: ['img/crop-1.jpg'],
        proposal: { value: 'FAK-001', label: 'Fake Hero', image: 'img/art-1.jpg' },
        confidence: 0.4,
        alternatives: [{ value: 'FAK-002', label: 'Fake Legend', image: 'img/art-2.jpg' }],
      },
      { id: 'i2', images: ['img/crop-2.jpg'], proposal: { value: 'FAK-002', label: 'Fake Legend' }, alternatives: [] },
    ],
    vocabulary: [
      { value: 'FAK-001', label: 'Fake Hero' },
      { value: 'FAK-002', label: 'Fake Legend' },
    ],
    createdAt: '2026-09-26T00:00:00Z',
    ...over,
  });
  const answers = (over: Partial<ReviewAnswers> = {}): ReviewAnswers => ({
    schema: 'rifteye.reviewanswers',
    version: 1,
    packId: 'pack-1',
    answers: [
      { itemId: 'i1', verdict: 'wrong', value: 'FAK-002', ms: 2100 },
      { itemId: 'i2', verdict: 'correct' },
    ],
    exportedAt: '2026-09-26T00:10:00Z',
    ...over,
  });

  it('accepts a pack, with or without embedded files', () => {
    expect(validateReviewPack(pack())).toEqual([]);
    const files = { 'img/crop-1.jpg': pixel, 'img/crop-2.jpg': pixel, 'img/art-1.jpg': pixel, 'img/art-2.jpg': pixel };
    expect(validateReviewPack(pack({ files }))).toEqual([]);
  });

  it('requires every picture to be embedded once files are', () => {
    const issues = validateReviewPack(pack({ files: { 'img/crop-1.jpg': pixel, 'img/x.jpg': 'http://example.com/x.jpg' } }));
    expect(issues.map((i) => i.path).sort()).toEqual([
      'files.img/x.jpg',
      'items[0].alternatives[0].image',
      'items[0].proposal.image',
      'items[1].images[0]',
    ]);
  });

  it('reports bad items with paths', () => {
    const [first] = pack().items;
    const issues = validateReviewPack(
      pack({ kind: 'nope' as ReviewPack['kind'], items: [{ ...first!, images: [], confidence: 2 }, { ...first! }] }),
    );
    expect(issues.map((i) => `${i.path}: ${i.message}`)).toEqual([
      'kind: must be one of identity, event',
      'items[0].images: must be a non-empty array of paths',
      'items[0].confidence: must be in [0, 1]',
      'items[1].id: duplicate id',
    ]);
    expect(validateReviewPack(pack({ items: [] })).map((i) => i.path)).toEqual(['items']);
  });

  it('accepts answers and rejects a value on a correct one', () => {
    expect(validateReviewAnswers(answers())).toEqual([]);
    const bad = answers({ answers: [{ itemId: 'i1', verdict: 'correct', value: 'FAK-001' }, { itemId: '', verdict: 'maybe' as 'correct' }] });
    expect(validateReviewAnswers(bad).map((i) => i.path)).toEqual(['answers[0].value', 'answers[1].itemId', 'answers[1].verdict']);
  });
});
