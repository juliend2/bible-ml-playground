import sys
import os
import tempfile
import multiprocessing
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Embedding, Linear, CrossEntropyLoss
import torch.optim as optim

print(torch.cuda.is_available())
exit()

from gensim.corpora import WikiCorpus
from gensim.models.word2vec import LineSentence

WIKIPEDIA_DUMP_PATH = '/home/julien/julien-data/_DATA/enwiki-20230501-pages-articles.xml.bz2'
TMP_DIR = '/home/julien/julien-data/_DATA/tmp'
WIKIPEDIA_W2V_PATH = '/home/julien/julien-data/_DATA/enwiki.model'


def write_wiki_corpus(wiki, text_output_file):
    """Write a WikiCorpus as plain text to file."""
    
    i = 0
    for text in wiki.get_texts():
        text_output_file.write(b' '.join([item.encode('utf-8') for item in text]) + b'\n')
        i = i + 1
        if (i % 10000 == 0):
            print('\rSaved %d articles' % i, end='', flush=True)
            
    print('\rFinished saving %d articles' % i, end='', flush=True)
    
class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = Embedding(vocab_size, embedding_dim)
        self.linear = Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        output = self.linear(embeds)
        return output

class Word2VecDataset(Dataset):
    def __init__(self, corpus):
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        return self.corpus[idx]

def train_word2vec(corpus, vocab_size, embedding_dim, batch_size, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Word2Vec(vocab_size, embedding_dim).to(device)
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    dataset = Word2VecDataset(corpus)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs in dataloader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), inputs.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

    return model

logging_format = '%(asctime)s : %(levelname)s : %(message)s'
logging.basicConfig(format=logging_format, level=logging.INFO)

with tempfile.NamedTemporaryFile(dir=TMP_DIR, suffix='.txt') as text_output_file:
    wiki_corpus = WikiCorpus(WIKIPEDIA_DUMP_PATH, dictionary={})
    write_wiki_corpus(wiki_corpus, text_output_file)
    del wiki_corpus

    sentences = LineSentence(text_output_file.name)
    vocab = set([word for sentence in sentences for word in sentence])
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for i, word in enumerate(vocab)}
    corpus = [[word2idx[word] for word in sentence] for sentence in sentences]

    vocab_size = len(vocab)
    embedding_dim = 400
    batch_size = 128
    num_epochs = 10

    model = train_word2vec(corpus, vocab_size, embedding_dim, batch_size, num_epochs)
    embeddings = model.embeddings.weight.data

    # Save the trained Word2Vec model
    torch.save(model.state_dict(), WIKIPEDIA_W2V_PATH)