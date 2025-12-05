# Vector Database Project

## Overview
This project implements a vector database that allows for efficient storage, retrieval, and management of vector embeddings. It provides various storage options and integrates with embedding APIs.

## Project Structure
```
vector-db
├── src
│   ├── index.ts                # Entry point of the application
│   ├── core
│   │   ├── index.ts            # Core functionalities of the vector database
│   │   └── vectorStore.ts      # Class for managing vectors
│   ├── storage
│   │   ├── fsStore.ts          # File system-based storage implementation
│   │   └── sqliteStore.ts      # SQLite-based storage implementation
│   ├── embeddings
│   │   ├── index.ts            # Embedding interface and utilities
│   │   ├── openaiAdapter.ts     # OpenAI embedding API interface
│   │   └── localAdapter.ts      # Local embedding functionalities
│   ├── api
│   │   ├── server.ts           # Express server setup
│   │   └── routes
│   │       └── vectors.ts      # API routes for vector handling
│   ├── utils
│   │   └── index.ts            # Utility functions
│   └── types
│       └── index.ts            # TypeScript interfaces and types
├── tests
│   └── vectorStore.test.ts     # Unit tests for VectorStore
├── examples
│   └── basic-usage.ts          # Example usage of the vector database
├── package.json                 # npm configuration file
├── tsconfig.json                # TypeScript configuration file
├── docker-compose.yml           # Docker configuration
├── .env.example                 # Example environment variables
└── README.md                   # Project documentation
```

## Setup Instructions

1. **Clone the Repository**
   ```
   git clone <repository-url>
   cd vector-db
   ```

2. **Install Dependencies**
   Ensure you have Node.js and npm installed. Then run:
   ```
   npm install
   ```

3. **Configure Environment Variables**
   Copy the `.env.example` file to `.env` and update the values as needed.

4. **Run the Application**
   You can start the application using:
   ```
   npm start
   ```

5. **Run Tests**
   To run the unit tests, use:
   ```
   npm test
   ```

6. **Example Usage**
   Check the `examples/basic-usage.ts` file for a simple demonstration of how to use the vector database.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.