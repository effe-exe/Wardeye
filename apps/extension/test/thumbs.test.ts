import { describe, expect, it } from 'vitest';
import { thumbName, thumbPath } from '../src/thumbs';

// the same cases as ml/tests/test_web_assets.py: the two sides name a file the same way
describe('the name of a hover picture', () => {
  it('is the printing id, with what a file name may not hold written as _ and hex', () => {
    expect(thumbName('SFD-195a')).toBe('SFD-195a');
    expect(thumbName('OGN-299*')).toBe('OGN-299_2a');
    expect(thumbName('a_b')).toBe('a_5f62'.replace('62', 'b'));
    expect(thumbName('é/')).toBe('_c3_a9_2f');
    expect(thumbPath('OGN-299*')).toBe('data/thumbs/OGN-299_2a.jpg');
  });
});
