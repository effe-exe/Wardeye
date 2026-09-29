import { describe, expect, it } from 'vitest';
import { bytesOfBase64 } from '../src/decode';

describe('a page\'s frame as bytes', () => {
  it('reads base64', () => {
    expect(Array.from(bytesOfBase64('/9j/4AAQ'))).toEqual([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]);
    expect(bytesOfBase64('')).toEqual(new Uint8Array(0));
  });
});
