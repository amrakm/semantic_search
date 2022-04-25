# Semantic Search
Template streamlit app to embed and search documents by text query

to run the app:
- install requirements
- sign up for [pinecone](https://www.pinecone.io/) and save the API key in `./secret`
- create embeddings for a sample dataset from huggingface datasets and save embedding vectors to pinecone
`python create_index.py`
- run streamlit app
- `streamlit run st_app.py`
