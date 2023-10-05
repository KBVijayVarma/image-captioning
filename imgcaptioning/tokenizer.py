import nltk, torch
from imgcaptioning.coco_dataset import CocoDataset
from imgcaptioning.vocabulary import Vocabulary

class Tokenizer(CocoDataset):
    def __init__(
        self, 
        img_dir, 
        annotations_file,
        vocab_threshold, 
        transform=None,
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>"
    ):
        super().__init__(img_dir, annotations_file, transform)
        self.vocab_threshold = vocab_threshold
        self.vocab = Vocabulary(self.vocab_threshold, start_word, end_word, unk_word)
    
    def __getitem__(self, idx):
        img, cap = super().__getitem__(idx)
        cap_tokens = nltk.word_tokenize(str(cap.lower()))
        cap_tokens.insert(0, self.vocab.start_word)
        cap_tokens.append(self.vocab.end_word)
        cap_tokens = [self.vocab(cap) for cap in cap_tokens]
        cap_tokens = torch.tensor(cap_tokens)
        return img, cap_tokens