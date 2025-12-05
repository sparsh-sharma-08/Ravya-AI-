import { Router } from 'express';
import { VectorStore } from '../../core/vectorStore';

const router = Router();
const vectorStore = new VectorStore();

// Route to add a vector
router.post('/vectors', async (req, res) => {
    try {
        const vector = req.body.vector;
        await vectorStore.addVector(vector);
        res.status(201).json({ message: 'Vector added successfully' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to add vector' });
    }
});

// Route to retrieve a vector by ID
router.get('/vectors/:id', async (req, res) => {
    try {
        const vectorId = req.params.id;
        const vector = await vectorStore.getVector(vectorId);
        if (vector) {
            res.status(200).json(vector);
        } else {
            res.status(404).json({ error: 'Vector not found' });
        }
    } catch (error) {
        res.status(500).json({ error: 'Failed to retrieve vector' });
    }
});

// Route to list all vectors
router.get('/vectors', async (req, res) => {
    try {
        const vectors = await vectorStore.listVectors();
        res.status(200).json(vectors);
    } catch (error) {
        res.status(500).json({ error: 'Failed to retrieve vectors' });
    }
});

export default router;