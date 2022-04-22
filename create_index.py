# source: https://gist.github.com/jamescalam/2123ce0bb8a871f48a151a023a7ece67
# https://www.youtube.com/watch?v=U0EoaFFGyTg

import pinecone
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

model = SentenceTransformer('pinecone/mpnet-retriever-squad2')

squad_dev = load_dataset('squad_v2', split='validation')

# extract all unique contexts
unique_contexts = []
unique_ids = []

# make list of IDs that represent only first instance of
# each context
for row in squad_dev:
    if row['context'] not in unique_contexts:
        unique_contexts.append(row['context'])
        unique_ids.append(row['id'])

# now filter out any samples that aren't included in unique IDs
squad_dev = squad_dev.filter(lambda x: True if x['id'] in unique_ids else False)

# and now encode the unique contexts
squad_dev = squad_dev.map(lambda x: {
    'encoding': model.encode(x['context']).tolist()
}, batched=True, batch_size=4)


with open('./secret', 'r') as fp:
    # get key from app.pinecone.io
    API_KEY = fp.read()

# initialize connection to pinecone
pinecone.init(
    api_key=API_KEY,
    environment='us-west1-gcp'
)
# check if index already exists and if not we create it
if 'qa-index' not in pinecone.list_indexes():
    pinecone.create_index(
        name='qa-index',
        dimension=model.get_sentence_embedding_dimension()
    )
# connect to the index
index = pinecone.Index('qa-index')

# prep data and upsert to pinecone
upserts = [(v['id'], v['encoding'], {'text': v['context'], 'title':v['title']}) for v in squad_dev]
# now upsert in chunks
for i in tqdm(range(0, len(upserts), 50)):
    i_end = i + 50
    if i_end > len(upserts): i_end = len(upserts)
    index.upsert(vectors=upserts[i:i_end])

