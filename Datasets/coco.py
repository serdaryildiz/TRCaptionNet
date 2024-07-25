import json
import os.path
import re

import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from Datasets.caption_utils import pre_caption


class COCOKarpathyTrain(Dataset):

    def __init__(self, dataset_root: str, json_path: str, tokenizer=None, transforms=None):
        self.dataset_root = dataset_root
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.transforms = transforms

        # load captions
        self.labels = json.load(open(json_path, 'r'))
        self.prompt = ""

        print(f"Train Sample Size : {len(self.labels)}")
        return

    def __getitem__(self, index):
        sample = self.labels[index]
        img = Image.open(os.path.join(self.dataset_root, sample["image"])).convert('RGB')

        caption = sample["caption"] if not type(sample["caption"]) == list else ""
        caption = self.prompt + pre_caption(caption, 35)
        img_id = int(sample["image_id"].replace("coco_", ""))

        # transform
        if self.transforms is not None:
            img = self.transforms(img)

        # tokenize
        if self.tokenizer is not None:
            caption = self.tokenizer(caption, padding='max_length', truncation=True, max_length=25, return_tensors="pt")

        return img, caption, img_id

    def __len__(self):
        return len(self.labels)


class COCOKarpathyTest(Dataset):

    def __init__(self, dataset_root: str, json_path: str, transforms=None):
        self.dataset_root = dataset_root
        self.json_path = json_path
        self.transforms = transforms

        # load captions
        self.labels = json.load(open(json_path, 'r'))
        self.image_ids = [sample["id"] for sample in self.labels["images"]]

        print(f"Test Sample Size : {len(self.image_ids)}")
        return

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        img_path = os.path.join(self.dataset_root, 'val2014/COCO_val2014_' + str(img_id).zfill(12) + '.jpg')
        img = Image.open(img_path).convert('RGB')

        # transform
        if self.transforms is not None:
            img = self.transforms(img)

        return img, img_id

    def __len__(self):
        return len(self.image_ids)




def test():
    root_path = "Data/coco/"
    json_path = "Data/coco/annotations/coco_karpathy_val_gt_tr.json"

    clip_model, preprocess = clip.load("ViT-B/32", device="cuda:0", jit=False)

    dataset = COCOKarpathyTest(dataset_root=root_path,
                               json_path=json_path,
                               transforms=preprocess)

    loader = DataLoader(
        dataset,
        batch_size=5,
        num_workers=0,
        pin_memory=True,
        shuffle=True)

    for img, img_id in loader:
        pass
        break

    json_path = "Data/coco/annotations/coco_karpathy_train_tr.json"

    dataset = COCOKarpathyTrain(dataset_root=root_path,
                                json_path=json_path,
                                tokenizer=None,
                                transforms=preprocess)

    loader = DataLoader(
        dataset,
        batch_size=5,
        num_workers=0,
        pin_memory=True,
        shuffle=True)

    for img, caption, img_id in loader:
        pass

    return


if __name__ == '__main__':
    test()
