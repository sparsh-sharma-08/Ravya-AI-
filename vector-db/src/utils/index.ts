export const generateRandomId = (): string => {
    return Math.random().toString(36).substr(2, 9);
};

export const calculateDistance = (vec1: number[], vec2: number[]): number => {
    if (vec1.length !== vec2.length) {
        throw new Error("Vectors must be of the same length");
    }
    return Math.sqrt(vec1.reduce((sum, val, index) => sum + Math.pow(val - vec2[index], 2), 0));
};

export const normalizeVector = (vector: number[]): number[] => {
    const length = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return length ? vector.map(val => val / length) : vector;
};