import pandas as pd
import numpy as np
import nmslib
from sentence_transformers import SentenceTransformer

# TODO: Use pipe, remove embeddings


class SentenceEncoder:
    """Encodes the querry and papers data set and finds elements with the lowest cosine similarity
    This application uses Sentence-BERT embeddings.
    Sentence Embedding is achieved here via Siamese BERT-Networks from https://arxiv.org/abs/1908.10084
    The implementation used is that of SBERT.net (https://www.sbert.net/)
    """

    def load_and_encode(self):
        """prepare data before running search querry"""
        # load
        df = self._load()
        # encode
        df, model, embeddings = self._encode_papers(df)
        return df, model, embeddings

    def transform(self, df, querry, model, embeddings):
        """main querry pipeline"""
        # create_index
        emb_querry = self._econde_querry(querry, model)
        # search
        result = self._make_search(df, emb_querry, embeddings)
        # add_relevant_columns
        df = self._add_relevant_columns(df, result)
        return df, result

    def _load(self):
        # Load data
        df = pd.read_csv("data/arxiv.csv")
        return df

    def _encode_papers(self, df):
        # Encode the papers title
        checkpoint = "distilbert-base-uncased"
        model = SentenceTransformer(checkpoint)
        embeddings = model.encode(df["title"], convert_to_tensor=True)
        # embeddings column
        df["embeddings"] = np.array(embeddings).tolist()
        return df, model, embeddings

    def _econde_querry(self, querry, model):
        # Encode the querry
        emb_querry = model.encode([querry])
        return emb_querry

    def _make_search(self, df, emb_querry, embeddings):
        """search for nearest K neighbours in the embedding space"""
        # initialize a new index, using a HNSW index on Cosine Similarity
        index = nmslib.init(method="hnsw", space="cosinesimil")
        index.addDataPointBatch(embeddings)
        index.createIndex({"post": 2}, print_progress=True)
        # search
        result = self._extract_search_result(index, emb_querry, df, k=10)
        return result

    def _extract_search_result(self, index, emb_querry, df, k):
        data = []
        idx, distances = index.knnQuery(emb_querry, k=k)
        for i, j in zip(idx, distances):
            data.append(
                {
                    "index": i,
                    "title": df.title[i],
                    "abstract": df.abstract[i],
                    "similarity": 1.0 - j,
                }
            )
        return pd.DataFrame(data)

    def _add_relevant_columns(self, df, result):
        """post processing"""
        # get categories
        df["categories_parsed"] = (
            df.categories.str.split()
            .apply(lambda x: x[0])
            .str.split(".")
            .apply(lambda x: x[0])
        )
        # create columns for plotting
        df["index_papers"] = df.index
        df["selected"] = df.index_papers.apply(lambda x: x in list(result["index"]))
        return df
