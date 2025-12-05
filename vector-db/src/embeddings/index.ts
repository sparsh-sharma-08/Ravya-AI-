export interface Embedding {
    embed(text: string): Promise<number[]>;
}

export function normalize(vector: number[]): number[] {
    const length = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return vector.map(val => val / length);
}