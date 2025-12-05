export type Vector = number[];
export interface VectorMetadata { [k: string]: any; }

// Simple in-memory VectorStore (implements upsert, get, delete, query)
export class VectorStore {
  private store: Map<string, { vector: Vector; meta?: VectorMetadata }>;

  constructor() {
    this.store = new Map();
  }

  upsert(id: string, vector: Vector, meta?: VectorMetadata) {
    this.store.set(id, { vector, meta });
  }

  get(id: string) {
    return this.store.get(id) ?? null;
  }

  delete(id: string) {
    return this.store.delete(id);
  }

  query(qvec: Vector, k = 5) {
    const dist = (a: Vector, b: Vector) => {
      let s = 0;
      const n = Math.min(a.length, b.length);
      for (let i = 0; i < n; i++) s += (a[i] - b[i]) ** 2;
      return Math.sqrt(s);
    };

    const items = Array.from(this.store.entries()).map(([id, v]) => ({
      id,
      dist: dist(qvec, v.vector),
      meta: v.meta,
    }));

    items.sort((a, b) => a.dist - b.dist);
    return items.slice(0, k);
  }
}