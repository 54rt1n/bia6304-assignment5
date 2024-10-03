# assignment/dqm.py

import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Callable, Optional

from .config import ChatConfig
from .embedding import HuggingFaceEmbedding


class DocumentQueryModel:
    """
    A simplified Document Query Model that uses a single collection with a flexible pipeline strategy 
    for document indexing and query preprocessing.
    
    Attributes:
        ef: The embedding function to use for indexing documents.
        preprocess: A callable strategy for preprocessing and indexing documents and queries.

    Usage:
        embedding_function = lambda x: np.random.rand(1, 100)  # Example embedding function that returns random embeddings
        preprocess = lambda x: x.split()  # Example preprocessing function
        dqm = DocumentQueryModel(preprocess, embedding_function, db_path="my_db_path")

        dqm = DocumentQueryModel(preprocess, HuggingFaceEmbedding(), db_path="my_db_path")
        # Index a document
        dqm.insert("doc1", "This is a sample document.")

        print(f"Document count: {dqm.document_count}")

        # Query the document
        query_result = dqm.query("This is a sample query.")

        # Delete all documents
        dqm.clear()

    """

    def __init__(self, data: pd.DataFrame, db_path: str, embedding_function: Callable[[str], np.ndarray]):
        """
        Initializes the DocumentQueryModel with a simple pandas DataFrame.

        Args:
            data: The indexed data to use for the document query model.
            db_path: The path to the data file.
            embedding_function: A callable strategy (function) that calculates the embedding for a document.
        """
        # Our data
        self.data = data
        self.db_path = db_path

        # Initialize the embedding function
        self.ef = embedding_function

    @classmethod
    def from_config(cls, config: ChatConfig) -> 'DocumentQueryModel':
        """
        Initializes the DocumentQueryModel from a file. Since we have embeddings we use a binary format.
        """
        if not config.db_path.endswith(".pkl"):
            raise ValueError(f"Invalid file format in '{config.db_path}'. Expected .pkl file.")
        
        if os.path.exists(config.db_path):
            data = pd.read_pickle(config.db_path)
        else:
            data = cls.new()
        # Validate headers are "embedding" and "document", and the index is "doc_id"
        if not all(col in data.columns for col in ["embedding", "content"]):
            raise ValueError("Invalid data format. Expected columns: 'embedding', 'content'")
        if data.index.name != "doc_id":
            raise ValueError("Invalid data format. Expected index name: 'doc_id'")

        embedding_function = HuggingFaceEmbedding(model_name=config.embedding_model)

        return cls(
            data=data,
            db_path=config.db_path,
            embedding_function=embedding_function,
        )

    @classmethod
    def new(cls) -> pd.DataFrame:
        """
        Initializes a new DocumentQueryModel with an empty data frame.
        """
        data = pd.DataFrame(columns=["embedding", "content"])
        data.index.name = "doc_id"
        return data

    def load_jsonl(self, file_path: str, id_key: str, content_key: str) -> None:
        """
        Load data from a JSONL file
        """
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    self.insert(data[id_key], content=data[content_key])
                except ValueError as ve:
                    print(f"Insert Error: {ve}")

    def save(self):
        """
        Saves the indexed data to a file.
        """
        self.data.to_pickle(self.db_path)

    @property
    def document_count(self) -> int:
        """
        Returns the total number of documents in the collection.

        Returns:
            int: The count of documents in the collection.

        Usage:
            document_count = dqm.document_count
            print(f"Total documents in the collection: {document_count}")
        """
        return self.data.shape[0]

    def insert(self, doc_id: str, content: str) -> np.ndarray:
        """
        Inserts a document into the collection after preprocessing with the pipeline strategy.

        Args:
            doc_id: A unique identifier for the document.
            content: The document to insert.
        """

        # Calculate the embedding for the document
        embedding = self.ef(content)

        # Add the document and its embedding to the collection
        self.data.loc[doc_id] = [embedding, content]

        return embedding

    def query(self, query_text: str, top_n: int = 5) -> pd.DataFrame:
        """
        Queries the indexed documents using the DQM preprocessing/embedding strategy.

        Args:
            query_text: The query as a string.
            top_n: Number of top results to return (default: 5).

        Returns:
            A list of document IDs of the top-k results based on similarity, and their distances.
        """

        # Calculate the embedding for the query, reshaped to a 2D array
        query = self.ef(query_text).reshape(1, -1)

        if self.data.empty:
            return pd.DataFrame()
        embeddings = np.stack(self.data['embedding'].values)

        search = self.data.copy()
        search['distance'] = cosine_similarity(query, embeddings)[0]
        return search.sort_values(by='distance', ascending=False).head(top_n)

    def get_document(self, doc_id: str) -> Optional[str]:
        """
        Retrieves a document from the collection.

        Args:
            doc_id: The ID of the document to retrieve.

        Returns:
            The document content as a string, or None if the document is not found.
        """
        try:
            result = self.data.loc[doc_id]
            return result['content']
        except IndexError:
            return None
    
    def clear(self):
        """
        Clears the collection.
        """
        self.data = self.new()