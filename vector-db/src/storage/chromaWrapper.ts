import axios, { AxiosInstance } from 'axios';
import { md5Hex } from '../embeddings/localEmbedding';

export type Vector = number[];
export interface ChunkMetadata {
  id: string;
  class: number;
  subject: string;
  chapter: number;
  textbook: string;
  language: string;
  tokens: number;
  hash: string;
  text?: string;
}

/**
 * ChromaWrapper
 * - Enforces collection naming: class_<grade>_<subject>_<lang>
 * - Batch upsert via /collections/{name}/add (100-1000 per request)
 * - Query requires metadata 'where' filter (no global search)
 * - Query applies similarity threshold: if top1 dist > 0.6 => REFER_TEACHER
 */
export class ChromaWrapper {
  private client: AxiosInstance;
  public readonly embedding_dim: number;

  constructor(baseUrl = process.env.CHROMA_URL || 'http://localhost:8000', embedding_dim = 768) {
    this.client = axios.create({ baseURL: baseUrl, timeout: 30_000 });
    this.embedding_dim = embedding_dim;
  }

  collectionNameFor(meta: { class: number; subject: string; language: string }) {
    const cls = String(meta.class);
    const subject = String(meta.subject).toLowerCase().replace(/\s+/g, '_');
    const lang = String(meta.language).toLowerCase();
    return `class_${cls}_${subject}_${lang}`;
  }

  async initCollection(collectionName: string) {
    // POST /collections { name } is idempotent; ignore if exists
    await this.client.post('/collections', { name: collectionName }).catch((err: any) => {
      // if not reachable, surface error
      if (err.code === 'ECONNREFUSED' || err.response?.status === 404) {
        throw new Error(`Chroma server not reachable at ${this.client.defaults.baseURL}`);
      }
      // ignore existing collection errors
    });
  }

  private validateBatchArgs(ids: string[], embeddings: Vector[], metadatas: any[], documents: (string | null)[]) {
    if (!(ids.length && embeddings.length)) throw new Error('ids and embeddings are required and non-empty');
    if (ids.length !== embeddings.length || ids.length !== metadatas.length || ids.length !== documents.length) {
      throw new Error('ids, embeddings, metadatas and documents must have the same length');
    }
    if (ids.length < 1) throw new Error('batch empty');
    if (ids.length > 1000) throw new Error('batch too large; use <= 1000');
    // dimension check
    for (const e of embeddings) {
      if (!Array.isArray(e) || e.length !== this.embedding_dim) throw new Error(`embedding dimension mismatch: expected ${this.embedding_dim}`);
    }
  }

  /**
   * upsertBatch - send 1..1000 vectors per request
   */
  async upsertBatch(collectionName: string, ids: string[], embeddings: Vector[], metadatas: any[], documents: (string | null)[]) {
    this.validateBatchArgs(ids, embeddings, metadatas, documents);
    const body = { ids, embeddings, metadatas, documents };
    await this.client.post(`/collections/${encodeURIComponent(collectionName)}/add`, body, { timeout: 60_000 });
  }

  /**
   * ingestChunks - convenience to ingest many chunks grouped by collection
   * - chunksByCollection: { collectionName: [{ id, text, metadata }] }
   * - uses batch size 256 by default
   */
  async ingestChunksGrouped(chunksByCollection: Record<string, { id: string; text: string; metadata: any; embedding: Vector }[]>, batchSize = 256) {
    for (const [collection, chunks] of Object.entries(chunksByCollection)) {
      await this.initCollection(collection);
      for (let i = 0; i < chunks.length; i += batchSize) {
        const batch = chunks.slice(i, i + batchSize);
        const ids = batch.map((c) => c.id);
        const embeddings = batch.map((c) => c.embedding);
        const metadatas = batch.map((c) => {
          // metadata must follow schema (excluding text)
          return {
            id: c.id,
            class: c.metadata.class,
            subject: c.metadata.subject,
            chapter: c.metadata.chapter,
            textbook: c.metadata.textbook,
            language: c.metadata.language,
            tokens: c.metadata.tokens,
            hash: c.metadata.hash,
          };
        });
        const documents = batch.map((c) => c.text ?? null);
        await this.upsertBatch(collection, ids, embeddings, metadatas, documents);
      }
    }
  }

  /**
   * query - must include 'where' metadata filter.
   * - where must include class, subject, language (chapter optional)
   * - returns { status: 'OK'|'REFER_TEACHER', results: [...] }
   */
  async query(collectionName: string, qvec: Vector, k = 5, where?: Record<string, any>) {
    if (!where || !where.class || !where.subject || !where.language) {
      throw new Error('Metadata filter required: class, subject, language');
    }
    if (!Array.isArray(qvec) || qvec.length !== this.embedding_dim) {
      throw new Error(`query vector must be array of length ${this.embedding_dim}`);
    }

    const body: any = {
      query_embeddings: [qvec],
      n_results: k,
      include: ['metadatas', 'distances', 'ids', 'documents'],
      where: where,
    };

    const resp = await this.client.post(`/collections/${encodeURIComponent(collectionName)}/query`, body);
    const r = resp.data?.result;
    if (!r) return { status: 'OK', results: [] };
    const ids: string[] = r.ids?.[0] ?? [];
    const distances: number[] = r.distances?.[0] ?? [];
    const metadatas: any[] = r.metadatas?.[0] ?? [];
    const documents: (string | null)[] = r.documents?.[0] ?? [];

    const results = ids.map((id, i) => ({ id, dist: distances[i], meta: metadatas[i], document: documents[i] }));

    // Similarity threshold control: if top1 distance > 0.6 => REFER_TEACHER
    if (results.length > 0 && typeof results[0].dist === 'number' && results[0].dist > 0.6) {
      return { status: 'REFER_TEACHER', results: [] };
    }

    return { status: 'OK', results };
  }

  /**
   * getAll - retrieve all items metadata+embeddings from a collection (for export).
   * Uses /collections/{name}/get with include
   */
  async getAll(collectionName: string) {
    const body = { include: ['ids', 'embeddings', 'metadatas', 'documents'] };
    const resp = await this.client.post(`/collections/${encodeURIComponent(collectionName)}/get`, body, { timeout: 60_000 });
    return resp.data?.result ?? null;
  }
}