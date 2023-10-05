import os
import json
import nltk
from collections import Counter
from pycocotools.coco import COCO

nltk.download('punkt')

class Vocabulary():
    def __init__(
        self, 
        vocab_threshold,
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>"
    ):
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.vocab_threshold = vocab_threshold
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.get_vocab()
    
    def __len__(self):
        return len(self.word2idx)
    
    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def add_captions(self):
        coco = COCO("coco_dataset/annotations/captions_train2017.json")
        counter = Counter()
        ids = coco.anns.keys()
        for i, idx in enumerate(ids):
            caption = str(coco.anns[idx]["caption"])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
        words = [word for word, count in counter.items() if count >= self.vocab_threshold]

        for _, word in enumerate(words):
            self.add_word(word)

    def build_vocab(self):
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()
    
    def get_vocab(self):
        if os.path.exists("./vocab.json"):
            with open("./vocab.json", "r") as f:
                vocab = json.load(f)
            self.word2idx = vocab['word2idx']
            self.idx2word = vocab['idx2word']
        
        else:
            self.build_vocab()
            vocab = {
                'word2idx': self.word2idx,
                'idx2word': self.idx2word
            }
            with open('./vocab.json', 'w') as f:
                json.dump(vocab, f)