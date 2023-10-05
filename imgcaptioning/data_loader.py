from torch.utils.data import DataLoader
from imgcaptioning.tokenizer import Tokenizer
from imgcaptioning.utils import get_batches

def get_loader(transform, batch_size):
    dataset = Tokenizer(
        "coco_dataset/train2017",
        "coco_dataset/annotations/captions_train2017.json",
        5, #vocab_threshold
        transform
    )
    batches = get_batches(dataset.captions_len, batch_size)
    data_loader = DataLoader(dataset, batch_sampler=batches)
    return data_loader, len(batches)