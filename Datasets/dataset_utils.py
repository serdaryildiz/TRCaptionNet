
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .coco import COCOKarpathyTrain, COCOKarpathyTest
from .flickr import FlickrTrain, FlickrTest
from .tasviret import TasvirEtTrain, TasvirEtTest
from Model import clip
from transform.randaugment import RandomAugment


def getTrainTransform():
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(384, scale=(0.5, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToPILImage()
    ])
    return transform_train


def getTestTransforms(vision_model=None):
    if vision_model is None:
        _, preprocess = clip.load("ViT-B/32", jit=False)
    else:
        _, preprocess = clip.load(vision_model, jit=False)
    return preprocess


def getTrainDataset(dataset_name, dataset_root, train_json_path, vision_model=None):
    test_transforms = getTestTransforms(vision_model)
    # train_transforms = torchvision.transforms.Compose([getTrainTransform(),
    #                                                    test_transforms])
    train_transforms = test_transforms
    if dataset_name.lower() == 'coco-karphaty':
        train_dataset = COCOKarpathyTrain(dataset_root=dataset_root,
                                          json_path=train_json_path,
                                          tokenizer=None,
                                          transforms=train_transforms)
    elif dataset_name.lower() == 'tasvir-et':
        train_dataset = TasvirEtTrain(dataset_root=dataset_root,
                                      json_path=train_json_path,
                                      transforms=train_transforms)
    elif dataset_name.lower() == 'flickr30k':
        train_dataset = FlickrTrain(dataset_root=dataset_root,
                                    json_path=train_json_path,
                                    transforms=train_transforms)
    else:
        raise Exception(f"Unknown dataset : {dataset_name}")
    return train_dataset


def getTestDataset(dataset_name, dataset_root, test_json_path, vision_model=None):
    test_transforms = getTestTransforms(vision_model)
    if dataset_name.lower() == 'coco-karphaty':
        test_dataset = COCOKarpathyTest(dataset_root=dataset_root,
                                        json_path=test_json_path,
                                        transforms=test_transforms)
    elif dataset_name.lower() == 'tasvir-et':
        test_dataset = TasvirEtTest(dataset_root=dataset_root,
                                    json_path=test_json_path,
                                    transforms=test_transforms)
    elif dataset_name.lower() == 'flickr30k':
        test_dataset = FlickrTest(dataset_root=dataset_root,
                                  json_path=test_json_path,
                                  transforms=test_transforms)
    else:
        raise Exception(f"Unknown dataset : {dataset_name}")
    return test_dataset


def getCocoDataset(dataset_root, train_json_path, test_json_path, vision_model=None):
    test_transforms = getTestTransforms(vision_model)
    train_dataset = COCOKarpathyTrain(dataset_root=dataset_root,
                                      json_path=train_json_path,
                                      tokenizer=None,
                                      transforms=test_transforms)

    test_dataset = COCOKarpathyTest(dataset_root=dataset_root,
                                    json_path=test_json_path,
                                    transforms=test_transforms)
    return train_dataset, test_dataset


def getTasvirEtDataset(dataset_root, train_json_path, test_json_path, vision_model=None):
    test_transforms = getTestTransforms(vision_model)
    train_dataset = TasvirEtTrain(dataset_root=dataset_root,
                                  json_path=train_json_path,
                                  transforms=test_transforms)

    test_dataset = TasvirEtTest(dataset_root=dataset_root,
                                json_path=test_json_path,
                                transforms=test_transforms)
    return train_dataset, test_dataset
