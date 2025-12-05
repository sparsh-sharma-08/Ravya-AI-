class SqliteStore {
    private db: any;

    constructor(databasePath: string) {
        const sqlite3 = require('sqlite3').verbose();
        this.db = new sqlite3.Database(databasePath);
        this.initialize();
    }

    private initialize() {
        this.db.run(`CREATE TABLE IF NOT EXISTS vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vector BLOB NOT NULL,
            metadata TEXT
        )`);
    }

    public addVector(vector: Buffer, metadata: string) {
        const stmt = this.db.prepare("INSERT INTO vectors (vector, metadata) VALUES (?, ?)");
        stmt.run(vector, metadata);
        stmt.finalize();
    }

    public getVector(id: number, callback: (vector: Buffer | null, metadata: string | null) => void) {
        this.db.get("SELECT vector, metadata FROM vectors WHERE id = ?", [id], (err: Error, row: any) => {
            if (err) {
                callback(null, null);
                return;
            }
            callback(row ? row.vector : null, row ? row.metadata : null);
        });
    }

    public close() {
        this.db.close();
    }
}

export default SqliteStore;