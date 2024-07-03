import pandas as pd
from sentence_transformers import SentenceTransformer

#embedding model
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
def generate_vector(text):
    return model.encode(text, normalize_embeddings=True)

#load data
df = pd.read_json('quotes.json')

#drop rows with missing values
df = df.dropna()

#get rid of duplicates
df = df.drop_duplicates(subset='Quote')
df = df.reset_index(drop=True)

# drop unneeded column
df = df.drop(columns=['Tags','Category', 'Popularity'])

#generate vectors
df['Vector'] = df['Quote'].apply(generate_vector)

#save vectors
df.to_pickle("quotes_with_vectors.pkl")