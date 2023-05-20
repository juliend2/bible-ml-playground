import sys
import os
import tempfile
import multiprocessing
import logging

from gensim.corpora import WikiCorpus
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec

# WIKIPEDIA_DUMP_PATH = '/home/julien/julien-data/_DATA/enwiki-20230501-pages-articles.xml.bz2'
WIKIPEDIA_DUMP_PATH = '/home/julien/julien-data/_DATA/frwiki-20230501-pages-articles.xml.bz2'

TMP_DIR = '/home/julien/julien-data/_DATA/tmp'

# Choose a path that the word2vec model should be saved to
# (during training), and read from afterwards.
# WIKIPEDIA_W2V_PATH = '/home/julien/julien-data/_DATA/enwiki.model'
WIKIPEDIA_W2V_PATH = '/home/julien/julien-data/_DATA/frwiki.model'

def write_wiki_corpus(wiki, text_output_file):
    """Write a WikiCorpus as plain text to file."""
    
    i = 0
    for text in wiki.get_texts():
        text_output_file.write(b' '.join([item.encode('utf-8') for item in text]) + b'\n')
        i = i + 1
        if (i % 10000 == 0):
            print('\rSaved %d articles' % i, end='', flush=True)
            
    print('\rFinished saving %d articles' % i, end='', flush=True)
    
def build_trained_model(text_file):
    """Reads text file and returns a trained model."""
    
    sentences = LineSentence(text_file)
    model = Word2Vec(sentences, vector_size=400, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())

    # Trim unneeded model memory to reduce RAM usage
    model.init_sims(replace=True)
    return model



logging_format = '%(asctime)s : %(levelname)s : %(message)s'
logging.basicConfig(format=logging_format, level=logging.INFO)

with tempfile.NamedTemporaryFile(dir=TMP_DIR, suffix='.txt') as text_output_file:
    # Create wiki corpus, and save text to temp file
    wiki_corpus = WikiCorpus(WIKIPEDIA_DUMP_PATH, dictionary={})
    write_wiki_corpus(wiki_corpus, text_output_file)
    del wiki_corpus

    # Train model on wiki corpus
    model = build_trained_model(text_output_file)    
    model.save(WIKIPEDIA_W2V_PATH)


# started again on 2023-05-18 23:15