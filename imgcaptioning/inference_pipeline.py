import os
import io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from .data_loader import get_loader
from .model import CNN, RNN
from .vocabulary import Vocabulary


def inference(image):
    transform_test = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )

    embed_size = 256
    hidden_size = 512
    vocab_threshold = 5

    voc = Vocabulary(vocab_threshold)

    encoder_file = "encoder.pkl"
    decoder_file = "decoder.pkl"

    encoder = CNN(embed_size=embed_size)
    decoder = RNN(len(voc), embed_size, hidden_size)

    encoder.load_state_dict(torch.load(os.path.join("./models", encoder_file), map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load(os.path.join("./models", decoder_file), map_location=torch.device('cpu')))

    encoder.eval()
    decoder.eval()
    img = Image.open(io.BytesIO(image.read())).convert('RGB')
    img = transform_test(img=img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        features = encoder(img).unsqueeze(1)
        output = decoder.sample(features)
    l = [voc.idx2word[str(idx)] for idx in output]
    return " ".join(l[1:-1])
