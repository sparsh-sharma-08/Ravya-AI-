/**
 * ingest_to_chroma.ts
 *
 * CLI usage:
 *   npx ts-node src/ingest/ingest_to_chroma.ts /path/to/chunks.jsonl
 *
 * Input format (JSONL) — each line must include:
 * {
 *   "text": "chunk text ...",
 *   "class": 8,
 *   "subject": "science",
 *   "chapter": 3,
 *   "textbook": "ncert",
 *   "language": "en",
 *   "tokens": 123
 * }
 *
 * The script:
 *  - computes id = md5(text)
 *  - computes deterministic embedding (768)
 *  - groups by collection (class_subject_language)
 *  - batches upserts (256 per request)
 */
import fs from 'fs';
import readline from 'readline';
import path from 'path';
import { deterministicEmbedding, md5Hex } from '../embeddings/localEmbedding';
import { ChromaWrapper } from '../storage/chromaWrapper';

async function main() {
  const infile = process.argv[2];
  if (!infile) {
    console.error('Usage: npx ts-node src/ingest/ingest_to_chroma.ts /path/to/chunks.jsonl');
    process.exit(2);
  }
  if (!fs.existsSync(infile)) {
    console.error('File not found:', infile);
    process.exit(2);
  }

  const chroma = new ChromaWrapper(process.env.CHROMA_URL || 'http://localhost:8000', 768);

  const rl = readline.createInterface({ input: fs.createReadStream(infile), crlfDelay: Infinity });

  const chunksByCollection: Record<string, { id: string; text: string; metadata: any; embedding: number[] }[]> = {};

  for await (const line of rl) {
    if (!line.trim()) continue;
    const obj = JSON.parse(line);
    const text = String(obj.text || '');
    if (!text) continue;
    const md5 = md5Hex(text);
    const id = `${obj.class}_${obj.subject}_${obj.chapter}_${md5}`; // unique chunk id
    const metadata = {
      id,
      class: Number(obj.class),
      subject: String(obj.subject),
      chapter: Number(obj.chapter),
      textbook: String(obj.textbook || 'ncert'),
      language: String(obj.language || 'en'),
      tokens: Number(obj.tokens || Math.ceil(text.length / 4)),
      hash: md5,
    };
    const embedding = deterministicEmbedding(text);
    const coll = chroma.collectionNameFor({ class: metadata.class, subject: metadata.subject, language: metadata.language });
    if (!chunksByCollection[coll]) chunksByCollection[coll] = [];
    chunksByCollection[coll].push({ id, text, metadata, embedding });
  }

  // ingest grouped
  await chroma.ingestChunksGrouped(chunksByCollection, 256);
  console.log('Ingest completed. Collections created:', Object.keys(chunksByCollection));
}

if (require.main === module) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}