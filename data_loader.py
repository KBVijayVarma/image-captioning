import os
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CocoDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.annotations_file = annotations_file
        self.coco = COCO(annotations_file)
        self.transform = transform

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, idx):
        img_id = self.coco.getImgIds()[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        captions_ids = self.coco.getAnnIds(imgIds=img_id)
        captions = [self.coco.loadAnns(c_id)[0]['caption']
                    for c_id in captions_ids]

        return image, captions
