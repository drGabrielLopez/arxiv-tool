import gradio as gr
from arxiv_tool.core import SentenceEncoder
from arxiv_tool.plot import EmbeddingPlotter

TITLE = "Search tool for ArXiv papers"
DESCRIPTION = "<center>Find your most beloved ArXiv papers!</center>"
EXAMPLES = [
    "RoBERTa optimisation",
    "Permutation invariant AI models",
    "Gradient descent",
    "Black hole information theory",
]
ARTICLE = r"""<center>
              This application uses Sentence-BERT embeddings.
              Sentence Embedding is achieved via Siamese BERT-Networks from  <a href=https://arxiv.org/abs/1908.10084>this paper</a> <br>
              After embedding, encoded papers are projected into the unit sphere and a nearest neighbours search is done to extract best matching results.<br>
              Done by dr. Gabriel Lopez<br> 
              For more please visit: <a href='https://sites.google.com/view/dr-gabriel-lopez/home'>My Page</a><br>
              </center>"""

# interface function
def search_and_plot(querry):
    # search
    df, model, embeddings = SentenceEncoder().load_and_encode()
    df, result = SentenceEncoder().transform(df, querry, model, embeddings)
    # plot
    fig1, fig2 = EmbeddingPlotter().transform(df, embeddings)
    return result[["title", "similarity"]], fig1, fig2


# gradio elements
in_textbox = gr.Textbox(
    label="Search on ArXiv:", placeholder="what do you want to learn today?...", lines=1
)
out_dataframe = gr.DataFrame(label="Most similar papers on ArXiv:")
out_plot_sphere = gr.Plot(label="Embedding projection over a unit sphere")
out_plot_projected_sphere = gr.Plot(
    label="Lambert-conformal projection over a plane", visible=False
)

# launch interface
gr.Interface(
    inputs=in_textbox,
    outputs=[out_dataframe, out_plot_sphere, out_plot_projected_sphere],
    examples=EXAMPLES,
    fn=search_and_plot,
    title=TITLE,
    description=DESCRIPTION,
    article=ARTICLE,
).launch()
