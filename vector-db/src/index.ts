import express from 'express';
import { createServer } from './api/server';
import { VectorStore } from './core/vectorStore';

const app = express();
const PORT = process.env.PORT || 3000;

// Initialize the vector store
const vectorStore = new VectorStore();

// Set up the server
createServer(app, vectorStore);

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});