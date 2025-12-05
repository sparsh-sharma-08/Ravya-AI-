import crypto from 'crypto';

/**
 * Deterministic local embedding generator for offline use.
 * - Produces 768-dimensional float32 vectors (embedding_dim = 768).
 * - Deterministic per chunk text via md5 seed.
 * - Normalized to unit length.
 *
 * This stands in for "intfloat/e5-multilingual" for offline runs.
 */
export const EMBEDDING_DIM = 768;

function xorshift32(seed: number) {
  // returns PRNG function that yields float in [-1,1)
  let x = seed || 2463534242;
  return function () {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    // convert to 32-bit unsigned, then to float in (-1,1)
    return ((x >>> 0) / 0xffffffff) * 2 - 1;
  };
}

export function md5Hex(s: string) {
  return crypto.createHash('md5').update(s, 'utf8').digest('hex');
}

/**
 * deterministicEmbedding(text)
 * - text -> md5 -> seed -> produce 768 floats -> normalize
 */
export function deterministicEmbedding(text: string) {
  const hash = md5Hex(text);
  // derive 32-bit seed from first 8 chars
  const seed = parseInt(hash.slice(0, 8), 16) >>> 0;
  const rnd = xorshift32(seed);
  const vec = new Float32Array(EMBEDDING_DIM);
  let norm = 0;
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    const v = rnd();
    vec[i] = v;
    norm += v * v;
  }
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < EMBEDDING_DIM; i++) vec[i] = vec[i] / norm;
  return Array.from(vec);
}