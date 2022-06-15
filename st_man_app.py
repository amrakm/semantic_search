import os
import streamlit as st
import pinecone
import tensorflow_hub as hub
import numpy as np

API_KEY = st.secrets['pinecone_api_key']


pinecone_env=  'us-west1-gcp' #'guides-2d52ac8.svc.us-west1-gcp.pinecone.io'
index_name = 'guides'


@st.experimental_singleton
def init_sentence_encoder():
    # initialize sentence encoder
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    return embed

@st.experimental_singleton
def init_pinecone():

    pinecone.init(
        api_key=API_KEY,
        environment=pinecone_env
    )
    index = pinecone.Index(index_name)
    return index

def card(name, introduction, score):
    st.markdown(f"""
    <div class="card" style="margin:1rem;">
        <div class="card-body">
            <h5 class="card-title">{name}</h5>
            <p class="card-text">{introduction}</p>
            <h6 class="card-subtitle mb-2 text-muted">matching score = {score} %</h6>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
""", unsafe_allow_html=True)

# initialize the index and retriever components
setnence_encoder = init_sentence_encoder()
index = init_pinecone()


st.write("""
# what is your device name and what seem to be the problem with it? 
""")

query = st.text_input("Search!", "")


if query != "":
    # encode the query as sentence vector
    query_emb = setnence_encoder([query]).numpy().tolist()
    # get relevant contexts
    xc = index.query(query_emb, top_k=5,
                     include_metadata=True)

    # print(xc.__dir__())
    # print(xc)
    # display each context (NEW PART)
    for context in xc['matches']:
        card(
            context['metadata']['title'],
            context['metadata']['introduction'],
            round(context['score'] * 100, 2)
        )
