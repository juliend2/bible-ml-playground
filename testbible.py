from IPython.display import display, HTML
import bible
import pickle
import numpy as np
from os import path



DATA_DIR = './data'
BIBLE_DIR = path.join(DATA_DIR, 'bible')

EMBEDDINGS_PATH = path.join(DATA_DIR, 'embeddings.pkl')

# Open the pickle file in binary mode for reading
with open(EMBEDDINGS_PATH, "rb") as file:
    # Load the data from the pickle file
    data = pickle.load(file)
    script_id_embedding_idx = data['script_id_embedding_idx']
    embedding_idx_script_id = data['embedding_idx_script_id']
    scriptures = data['scriptures']
    embeddings = data['embeddings']

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

WIKIPEDIA_W2V_PATH = '/home/julien/julien-data/_DATA/enwiki.model'
model = Word2Vec.load(WIKIPEDIA_W2V_PATH)

# Get word embeddings
word_embeddings = model.wv

desired_embedding = word_embeddings["peace"] - word_embeddings["birds"]

# array = np.ones(400, dtype=int)
nearest_word_index = word_embeddings.similar_by_vector(desired_embedding, topn=1)
# nearest_word_index = word_embeddings.similar_by_vector(array, topn=1)
nearest_word = nearest_word_index[0][0]

print(nearest_word)

exit()

# Calculate cosine similarity
similarity = cosine_similarity([word_embeddings["love"]], [word_embeddings["enjoy"]])
print(similarity)
similarity = cosine_similarity([word_embeddings["love"]], [word_embeddings["favorite"]])
print(similarity)

similarity = cosine_similarity([word_embeddings["love"]], [word_embeddings["data"]])
print(similarity)

exit()




# Now you can work with the loaded data
# For example, you can print the data
print(data)

def similar_scriptures(scripture_id, top_k=0, drop_first=True):
    """Assumes all embeddings have been normalized."""
    
    embedding_idx = script_id_embedding_idx[scripture_id]
    embedding = embeddings[embedding_idx]
    
    cosines = embeddings.dot(embedding)
    indexes = np.argsort(-cosines)

    if drop_first:
        indexes = indexes[1:]

    if top_k > 0:
        indexes = indexes[:top_k]
    return [(embedding_idx_script_id[index], scriptures[embedding_idx_script_id[index]], cosines[index], index)
            for index in indexes]

def similar_scriptures_html(scripture_id, top_k=15):   
    def row_html(s):
        row_pattern = """
            <td>{s_id}</td>
            <td>{scripture}</td>
            <td>{score}</td>
        """
        row = row_pattern.format(s_id=s[0], scripture=s[1],
                                 score=np.round(s[2], decimals=2))
        return row
    
    similar = similar_scriptures(scripture_id, top_k=top_k)
    rows = map(row_html, similar)
    rows = ['<tr>\n%s\n</tr>' % r for r in rows]
    
    columns = ['Scripture ID', 'Scripture', 'Score']
    headers = ['<th>%s</th>' % h for h in columns]
    header_row = '<tr>\n%s\n</tr>' % '\n'.join(headers)

    rows = [header_row] + rows
    table = '<table>\n{}\n</table>'.format('\n'.join(rows))
    header = '<h3>Query: (%s) %s</h3>' % (scripture_id, scriptures[scripture_id])
    return header + table

scripture_id = bible.get_scripture_id('Genesis', 1, 1)

print(HTML(similar_scriptures_html(scripture_id)).data)
