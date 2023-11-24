import streamlit as st
import pandas as pd
import plotly.express as px
from umap.umap_ import UMAP
import numpy as np

st.set_page_config(layout="wide")

margins_css = """
        <style>
        .main > div {
            padding-left: 2rem;
            padding-right: 0rem;
        }
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 0rem;
            padding-right: 5rem;
        }
        </style>
        """

st.markdown(margins_css, unsafe_allow_html=True)

# Title of the web app
st.title('Feature Engineering Instructions Embedding Space')

# Load embeddings
df = pd.read_csv('special/feature_engineering_instructions_embeddings.csv', sep=";")

# Set subheader
st.subheader('OpenAI ADA embeddings with UMAP projection')

# Get lists from df
embeddings = embeddings = [np.array(eval(e)) for e in df["embedding"]]
texts = df["text"].tolist()

# Compute UMAP projection
reducer = UMAP()
embeddings_umap = reducer.fit_transform(embeddings)

# Convert reduced embeddings into a DataFrame for easier plotting
df_embeddings_umap = pd.DataFrame(embeddings_umap, columns=['x', 'y'])

# Create an interactive scatter plot using Plotly
fig = px.scatter(df_embeddings_umap, x='x', y='y')

# Update the hovertemplate to only show text
fig.update_traces(hovertemplate='%{hovertext}', hovertext=texts)

# Set size of the plot
fig.update_layout(
    autosize=False,
    width=1400,
    height=1000,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ),
)
st.plotly_chart(fig)


# Optional: Display the raw data as a table (comment out if not needed)
df_display = df[["instruction", "tags"]]
if st.checkbox('Show raw data'):
    st.write(df_display)
