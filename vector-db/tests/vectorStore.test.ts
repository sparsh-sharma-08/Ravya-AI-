import { VectorStore } from '../src/core/vectorStore';

describe('VectorStore', () => {
    let vectorStore: VectorStore;

    beforeEach(() => {
        vectorStore = new VectorStore();
    });

    test('should add a vector', () => {
        const vector = { id: '1', values: [0.1, 0.2, 0.3] };
        vectorStore.addVector(vector);
        expect(vectorStore.getVector('1')).toEqual(vector);
    });

    test('should retrieve a vector', () => {
        const vector = { id: '2', values: [0.4, 0.5, 0.6] };
        vectorStore.addVector(vector);
        const retrievedVector = vectorStore.getVector('2');
        expect(retrievedVector).toEqual(vector);
    });

    test('should return undefined for a non-existent vector', () => {
        const retrievedVector = vectorStore.getVector('non-existent-id');
        expect(retrievedVector).toBeUndefined();
    });

    test('should remove a vector', () => {
        const vector = { id: '3', values: [0.7, 0.8, 0.9] };
        vectorStore.addVector(vector);
        vectorStore.removeVector('3');
        const retrievedVector = vectorStore.getVector('3');
        expect(retrievedVector).toBeUndefined();
    });

    test('should update a vector', () => {
        const vector = { id: '4', values: [1.0, 1.1, 1.2] };
        vectorStore.addVector(vector);
        const updatedVector = { id: '4', values: [1.3, 1.4, 1.5] };
        vectorStore.updateVector(updatedVector);
        expect(vectorStore.getVector('4')).toEqual(updatedVector);
    });
});