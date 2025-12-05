export class OpenAIAdapter {
    private apiKey: string;

    constructor(apiKey: string) {
        this.apiKey = apiKey;
    }

    async embed(text: string): Promise<number[]> {
        const response = await fetch('https://api.openai.com/v1/embeddings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiKey}`,
            },
            body: JSON.stringify({
                input: text,
                model: 'text-embedding-ada-002',
            }),
        });

        if (!response.ok) {
            throw new Error(`Error fetching embeddings: ${response.statusText}`);
        }

        const data = await response.json();
        return data.data[0].embedding;
    }
}