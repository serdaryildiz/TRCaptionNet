import json
import os

from PIL import Image
from torch.utils.data import Dataset

from Datasets.caption_utils import pre_caption


class TasvirEtTrain(Dataset):

    def __init__(self, dataset_root: str, json_path: str, transforms=None):
        self.dataset_root = dataset_root
        self.json_path = json_path
        self.transforms = transforms

        # load captions
        self.labels = json.load(open(json_path, 'r'))
        self.image_ids = [sample["id"] for sample in self.labels["annotations"]]
        self.file_names = [f'{sample["filename"].split("_")[0]}.jpg'for sample in self.labels["annotations"]]
        self.captions = [sample["caption"] for sample in self.labels["annotations"]]

        print(f"Train Sample Size : {len(self.image_ids)}")
        return

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        img_path = os.path.join(self.dataset_root, self.file_names[index])
        img = Image.open(img_path).convert('RGB')

        caption = self.captions[index]
        caption = pre_caption(caption, 35)

        # transform
        if self.transforms is not None:
            img = self.transforms(img)

        return img, caption, img_id

    def __len__(self):
        return len(self.image_ids)


class TasvirEtTest(Dataset):

    def __init__(self, dataset_root: str, json_path: str, transforms=None):
        self.dataset_root = dataset_root
        self.json_path = json_path
        self.transforms = transforms

        # load captions
        self.labels = json.load(open(json_path, 'r'))
        self.image_ids = [sample["imgid"] for sample in self.labels["images"]]
        self.file_names = [sample["filename"] for sample in self.labels["images"]]

        print(f"Test Sample Size : {len(self.image_ids)}")
        return

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        img_path = os.path.join(self.dataset_root, f'{self.file_names[index].split("_")[0]}.jpg')
        img = Image.open(img_path).convert('RGB')

        # transform
        if self.transforms is not None:
            img = self.transforms(img)

        return img, img_id

    def __len__(self):
        return len(self.image_ids)
