import { VectorStore } from '../src/core/vectorStore';
import { FsStore } from '../src/storage/fsStore';
import { OpenAIAdapter } from '../src/embeddings/openaiAdapter';

async function main() {
    // Initialize the file system store
    const fsStore = new FsStore('./data/vectors.json');
    
    // Initialize the vector store with the file system store
    const vectorStore = new VectorStore(fsStore);
    
    // Initialize the OpenAI adapter for embeddings
    const embeddingAdapter = new OpenAIAdapter('YOUR_OPENAI_API_KEY');

    // Example vector data
    const vectorData = {
        id: '1',
        vector: [0.1, 0.2, 0.3],
        metadata: { description: 'Example vector' }
    };

    // Add a vector to the store
    await vectorStore.add(vectorData, embeddingAdapter);

    // Retrieve the vector from the store
    const retrievedVector = await vectorStore.get('1');
    console.log('Retrieved Vector:', retrievedVector);
}

main().catch(console.error);