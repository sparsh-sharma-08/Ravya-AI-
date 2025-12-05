export interface Vector {
    id: string;
    values: number[];
    metadata?: Record<string, any>;
}

export interface VectorStore {
    addVector(vector: Vector): Promise<void>;
    getVector(id: string): Promise<Vector | null>;
    deleteVector(id: string): Promise<void>;
    listVectors(): Promise<Vector[]>;
}

export interface EmbeddingAdapter {
    embed(text: string): Promise<number[]>;
}

export interface Storage {
    saveVector(vector: Vector): Promise<void>;
    loadVector(id: string): Promise<Vector | null>;
    deleteVector(id: string): Promise<void>;
    listVectors(): Promise<Vector[]>;
}