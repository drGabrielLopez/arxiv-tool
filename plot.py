import umap
import plotly.express as px
import numpy as np


class EmbeddingPlotter:
    """Lower the dimensionality of the representation from 768 -> 2, over the surface of the sphere"""

    def transform(self, df, embeddings):
        """Plotting pipleine"""
        df = self.umap_embedding(df, embeddings)
        fig1, fig2 = self.plot(df)
        return fig1, fig2

    def umap_embedding(self, df, embeddings):
        """Generate 2D embeddings"""
        # UMAP - Spherical
        sphere_mapper = umap.UMAP(output_metric="haversine", random_state=42).fit(
            np.array(embeddings)
        )
        df["spherical_emb_X"] = np.sin(sphere_mapper.embedding_[:, 0]) * np.cos(
            sphere_mapper.embedding_[:, 1]
        )
        df["spherical_emb_Y"] = np.sin(sphere_mapper.embedding_[:, 0]) * np.sin(
            sphere_mapper.embedding_[:, 1]
        )
        df["spherical_emb_Z"] = np.cos(sphere_mapper.embedding_[:, 0])
        # UMAP - Lambert Conformal
        df["lambert_conformal_emb_x"] = np.arctan2(
            df["spherical_emb_X"], df["spherical_emb_Y"]
        )
        df["lambert_conformal_emb_y"] = -np.arccos(df["spherical_emb_Z"])
        return df

    def plot(self, df):
        """Plot embeddings over representation space"""
        # on the 3d sphere
        fig1 = px.scatter_3d(
            df,
            x="spherical_emb_X",
            y="spherical_emb_Y",
            z="spherical_emb_Z",
            color="categories_parsed",
        )
        # on the projected spehre
        fig2 = px.scatter(
            data_frame=df,
            x="lambert_conformal_emb_x",
            y="lambert_conformal_emb_y",
            color="categories_parsed",
        )
        return fig1, fig2
