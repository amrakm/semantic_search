import os
import streamlit as st
import pinecone
import tensorflow_hub as hub
import numpy as np

API_KEY = st.secrets['pinecone_api_key']

@st.experimental_singleton
def init_sentence_encoder():
    # initialize sentence encoder
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    return embed

@st.experimental_singleton
def init_pinecone():

    pinecone.init(
        api_key=API_KEY,
        environment='us-west1-gcp'
    )
    index = pinecone.Index('qa-index')
    return index

def card(name, score, website_url, yelp_url):
    st.markdown(f"""
    <div class="card" style="margin:1rem;">
        <div class="card-body">
            <h5 class="card-title">{name}</h5>
            <h6 class="card-subtitle mb-2 text-muted">matching score = {score} %</h6>
            <a href={website_url}>website_url</a>
            <br> 
            <a href={yelp_url}>yelp_url</a>
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
# eezy restaurants text recommender
Describe your ideal restaurant! you can be creative here, describe the food, vibe or occasion!
""")

query = st.text_input("Search!", "")


if query != "":
    # encode the query as sentence vector
    query_emb = setnence_encoder([query]).numpy().tolist()
    # get relevant contexts
    xc = index.query(query_emb, top_k=5,
                     include_metadata=True)
    # display each context (NEW PART)
    for context in xc['results'][0]['matches']:
        card(
            context['metadata']['name'],
            round(context['score'] * 100, 2),
            context['metadata']['website_links'],
            context['metadata']['yelp_url']
        )
