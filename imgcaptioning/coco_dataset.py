import os
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from imgcaptioning.utils import caption_len

class CocoDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.annotations_file = annotations_file
        self.coco = COCO(annotations_file)
        self.transform = transform
        self.idx2ann_id = self.coco.getAnnIds()
        self.captions_len = [caption_len(self.coco.loadAnns(ann_id)[0]['caption']) for ann_id in self.idx2ann_id]

    def __len__(self):
        return len(self.idx2ann_id)
    
    def __getitem__(self, idx):
        ann_id = self.idx2ann_id[idx]
        ann = self.coco.loadAnns(ann_id)[0]

        img_id = ann['image_id']
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        caption = ann['caption']

        return image, caption