import argparse
import glob
import os

import cv2
import numpy
import torch
from PIL import Image

from Model import TRCaptionNet, clip_transform


def demo(opt):
    preprocess = clip_transform(224)
    model = TRCaptionNet({
        "max_length": 35,
        "clip": "ViT-L/14",
        "bert": "dbmdz/bert-base-turkish-cased",
        "proj": True,
        "proj_num_head": 16
    })
    device = torch.device(opt.device)
    model.load_state_dict(torch.load(opt.model_ckpt, map_location=device)["model"], strict=True)
    model = model.to(device)
    model.eval()

    image_paths = glob.glob(os.path.join(opt.input_dir, '*.jpg'))

    for image_path in sorted(image_paths):
        img_name = image_path.split('/')[-1]
        img0 = Image.open(image_path)
        batch = preprocess(img0).unsqueeze(0).to(device)
        caption = model.generate(batch, min_length=11, repetition_penalty=1.6)[0]
        print(f"{img_name} :", caption)

        orj_img = numpy.array(img0)[:, :, ::-1]
        h, w, _ = orj_img.shape
        new_h = 800
        new_w = int(new_h * (w / h))
        orj_img = cv2.resize(orj_img, (new_w, new_h))

        cv2.imshow("image", orj_img)
        cv2.waitKey(0)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turkish-Image-Captioning!')
    parser.add_argument('--model-ckpt', type=str, default='./checkpoints/TRCaptionNet_L14_berturk.pth')
    parser.add_argument('--input-dir', type=str, default='/home/serdar/huggingface/TRCaptionNet/images')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    demo(args)
    # dm()
