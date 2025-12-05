class Chunk:
    def __init__(self, text, class_num, subject, chapter, language, textbook, tokens, unique_hash):
        self.text = text
        self.class_num = class_num
        self.subject = subject
        self.chapter = chapter
        self.language = language
        self.textbook = textbook
        self.tokens = tokens
        self.unique_hash = unique_hash

class Query:
    def __init__(self, class_num, subject, language, chapter=None):
        self.class_num = class_num
        self.subject = subject
        self.language = language
        self.chapter = chapter

class Manifest:
    def __init__(self, class_num, subject, chapter, language, embedding_model, embedding_dim, chunk_count, chunk_strategy, created_at, version, hash_strategy):
        self.class_num = class_num
        self.subject = subject
        self.chapter = chapter
        self.language = language
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.chunk_count = chunk_count
        self.chunk_strategy = chunk_strategy
        self.created_at = created_at
        self.version = version
        self.hash_strategy = hash_strategy