from gensim.models import Word2Vec
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
from os import path
import glob
import pickle
import bible
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)

WIKIPEDIA_W2V_PATH = '/home/julien/julien-data/_DATA/enwiki.model'

DATA_DIR = './data'
BIBLE_DIR = path.join(DATA_DIR, 'bible')

EMBEDDINGS_PATH = path.join(DATA_DIR, 'embeddings.pkl')

def read_bible():
    """Returns dictionary of { scripture_id: verse }."""
    
    pattern = path.join(BIBLE_DIR, '*')
    book_names = glob.glob(pattern)

    book_names = list(map(path.basename, book_names))
    sciptures = map(read_book, book_names)
    
    bible_verses = {}
    for book_idx, book in enumerate(sciptures):
        for chapter_idx, chapter in enumerate(book):
            for verse_idx, verse in enumerate(chapter):
                book_name = book_names[book_idx]
                s_id = bible.get_scripture_id(book_name, chapter_idx + 1, verse_idx + 1)
                bible_verses[s_id] = verse
    
    return bible_verses

def read_book(book):
    pattern = path.join(BIBLE_DIR, '%s/*.txt' % book)
    n_chapters = len(glob.glob(pattern))
    chapters = [read_chapter(book, n) for n in range(1, n_chapters+1)]
    return chapters
    
def read_chapter(book, chapter):
    filename = path.join(BIBLE_DIR, '%s/%s%d.txt' % (book, book, chapter))
    with open(filename, 'rt') as f:
        lines = f.readlines()
        lines = [re.sub(r'\d+\s', '', l.rstrip()) for l in lines]
    return lines


def word_vec2(model, words):
    word_vectors = []
    
    for word in words:
        if word in model.wv['vocab']:
            vector = model.wv[word]
            word_vectors.append(vector)
        else:
            # Handle out-of-vocabulary words
            word_vectors.append(None)
    
    return word_vectors



def word_vec(model, words, normalize=False):
    words = filter(lambda w: w in model.wv['vocab'], words)
    vecs = np.array([model[w] for w in words])
    vec = vecs.mean(axis=0)
    if normalize:
        vec = vec / np.linalg.norm(vec)
    return vec

def tokenize(line, stemmer=None, stopwords=None):
    line = re.sub(r'[^\w\s]+', ' ', line)
    line = re.sub(r'\s+', ' ', line)
    line = line.strip().lower()
    words = line.split()    

    if stemmer is not None:
        words = [stemmer.stem(w) for w in words if w not in stopwords]

    if stopwords is not None:
        words = [w for w in words if w not in stopwords]
    
    return words


model = Word2Vec.load(WIKIPEDIA_W2V_PATH)
stemmer = SnowballStemmer('english')
nltk.download('stopwords')
stops = stopwords.words('english')


duple = word_vec2(model, ['moab', 'rebel', 'israel', 'death', 'ahab'])

print(duple)

scriptures = read_bible()

print('%s scriptures' % len(scriptures))

embeddings = [word_vec(model, tokenize(verse, stemmer=stemmer, stopwords=stops), normalize=True)
                       for verse in scriptures.values()]

def nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

# HACK: If word_vec(...) returns nan, turn it into a nan row so we can vstack. Improve this.
embeddings = [nans(400) if np.isnan(e).any() else e for e in embeddings]
embeddings = np.vstack(embeddings)

print('Embeddings shape: ', embeddings.shape)

script_ids = scriptures.keys()
embedding_idxs = range(embeddings.shape[0])

script_id_embedding_idx = dict(zip(script_ids, embedding_idxs))

embedding_idx_script_id = dict(zip(embedding_idxs, script_ids))

with open(EMBEDDINGS_PATH, 'wb') as f:
    pickle.dump({
        'scriptures': scriptures,
        'embeddings': embeddings,
        'script_id_embedding_idx': script_id_embedding_idx,
        'embedding_idx_script_id': embedding_idx_script_id
    }, f)
