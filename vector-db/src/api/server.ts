import express, { Application } from 'express';
import dotenv from 'dotenv';
dotenv.config();

export function createServer(app?: Application, vectorStore?: any) {
  const serverApp = app ?? express();
  serverApp.use(express.json());

  // dev-friendly CSP (non-production)
  if (process.env.NODE_ENV !== 'production') {
    serverApp.use((req, res, next) => {
      res.setHeader('Content-Security-Policy', "default-src 'self' 'unsafe-inline' http: https:; connect-src *");
      next();
    });
    serverApp.get('/.well-known/appspecific/com.chrome.devtools.json', (_req, res) => {
      return res.status(204).send();
    });
  }

  serverApp.get('/health', (_req, res) => {
    res.json({ status: 'ok' });
  });

  // Upsert batch endpoint (required: provide metadata to determine collection)
  serverApp.post('/vectors/upsert_batch', async (req, res) => {
    if (!vectorStore) return res.status(500).json({ error: 'vector store not initialized' });
    const { chunks } = req.body;
    // chunks: [{ text, class, subject, chapter, textbook, language, tokens }]
    if (!Array.isArray(chunks) || chunks.length === 0) return res.status(400).json({ error: 'chunks required' });

    try {
      // map and validate
      const grouped: Record<string, { id: string; text: string; metadata: any; embedding: number[] }[]> = {};
      for (const c of chunks) {
        if (!c.text || !c.class || !c.subject || !c.language) {
          return res.status(400).json({ error: 'chunk missing required fields: text,class,subject,language' });
        }
        const metadata = {
          class: Number(c.class),
          subject: String(c.subject),
          chapter: Number(c.chapter || 0),
          textbook: String(c.textbook || 'ncert'),
          language: String(c.language),
          tokens: Number(c.tokens || Math.ceil(String(c.text).length / 4)),
          hash: require('crypto').createHash('md5').update(String(c.text), 'utf8').digest('hex'),
        };
        const id = `${metadata.class}_${metadata.subject}_${metadata.chapter}_${metadata.hash}`;
        // create deterministic embedding in-process or call external adapter (we expect deterministicEmbedding to exist)
        const { deterministicEmbedding } = require('../embeddings/localEmbedding');
        const embedding = deterministicEmbedding(String(c.text));
        const coll = vectorStore.collectionNameFor({ class: metadata.class, subject: metadata.subject, language: metadata.language });
        if (!grouped[coll]) grouped[coll] = [];
        grouped[coll].push({ id, text: String(c.text), metadata: { ...metadata, id }, embedding });
      }

      // call ingestChunksGrouped (uses batching internally)
      await vectorStore.ingestChunksGrouped(grouped, 256);
      return res.json({ ok: true, collections: Object.keys(grouped) });
    } catch (err: any) {
      console.error(err);
      return res.status(500).json({ error: err.message || String(err) });
    }
  });

  // Query endpoint: must provide class, subject, language (chapter optional)
  serverApp.post('/vectors/query', async (req, res) => {
    if (!vectorStore) return res.status(500).json({ error: 'vector store not initialized' });
    try {
      const { query, k = 5, class: cls, subject, language, chapter } = req.body;
      if (!Array.isArray(query)) return res.status(400).json({ error: 'query vector required' });
      if (!cls || !subject || !language) return res.status(400).json({ error: 'class,subject,language metadata filter required' });

      const collectionName = vectorStore.collectionNameFor({ class: Number(cls), subject: String(subject), language: String(language) });
      const whereFilter: any = { class: Number(cls), subject: String(subject), language: String(language) };
      if (typeof chapter !== 'undefined') whereFilter.chapter = Number(chapter);

      const out = await vectorStore.query(collectionName, query, Number(k), whereFilter);
      if (out.status === 'REFER_TEACHER') return res.json({ status: 'REFER_TEACHER' });
      return res.json({ status: 'OK', results: out.results });
    } catch (err: any) {
      console.error(err);
      return res.status(500).json({ error: err.message || String(err) });
    }
  });

  return serverApp;
}

export default createServer;