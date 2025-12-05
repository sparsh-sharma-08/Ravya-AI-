class LocalAdapter {
    private embeddings: Map<string, number[]>;

    constructor() {
        this.embeddings = new Map();
    }

    addEmbedding(id: string, vector: number[]): void {
        this.embeddings.set(id, vector);
    }

    getEmbedding(id: string): number[] | undefined {
        return this.embeddings.get(id);
    }

    deleteEmbedding(id: string): void {
        this.embeddings.delete(id);
    }

    listEmbeddings(): Array<{ id: string; vector: number[] }> {
        return Array.from(this.embeddings.entries()).map(([id, vector]) => ({ id, vector }));
    }
}

export default LocalAdapter;