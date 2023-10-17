import os.path

import gdown
import gradio as gr
import torch

from Model import TRCaptionNet, clip_transform

model_ckpt = "./checkpoints/TRCaptionNet_L14_berturk.pth"
if not os.path.exists(model_ckpt):
    os.makedirs("./checkpoints/", exist_ok=True)
    url = 'https://drive.google.com/u/0/uc?id=14Ll1PIQhsMSypHT34Rt9voz_zaAf4Xh9&export=download&confirm=t&uuid=9b4bf589-d438-4b4f-a37c-fc34b0a63a5d&at=AB6BwCAY8xK0EZiPGv2YT7isL8pG:1697575816291'
    gdown.download(url, model_ckpt, quiet=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

preprocess = clip_transform(224)
model = TRCaptionNet({
    "max_length": 35,
    "clip": "ViT-L/14",
    "bert": "dbmdz/bert-base-turkish-cased",
    "proj": True,
    "proj_num_head": 16
})
model.load_state_dict(torch.load(model_ckpt, map_location=device)["model"], strict=True)
model = model.to(device)
model.eval()


def inference(raw_image, min_length, repetition_penalty):
    batch = preprocess(raw_image).unsqueeze(0).to(device)
    caption = model.generate(batch, min_length=min_length, repetition_penalty=repetition_penalty)[0]
    return caption


inputs = [gr.Image(type='pil', interactive=False,),
          gr.Slider(minimum=6, maximum=22, value=11, label="MINIMUM CAPTION LENGTH", step=1),
          gr.Slider(minimum=1, maximum=2, value=1.6, label="REPETITION PENALTY")]
outputs = gr.components.Textbox(label="Caption")
title = "TRCaptionNet"
paper_link = ""
github_link = "https://github.com/serdaryildiz/TRCaptionNet"
description = f"<p style='text-align: center'><a href='{github_link}' target='_blank'>TRCaptionNet</a> : A novel and accurate deep Turkish image captioning model with vision transformer based image encoders and deep linguistic text decoders"
examples = [["images/test1.jpg"], ["images/test2.jpg"], ["images/test3.jpg"], ["images/test4.jpg"]]
article = f"<p style='text-align: center'><a href='{paper_link}' target='_blank'>Paper</a> | <a href='{github_link}' target='_blank'>Github Repo</a></p>"
css = ".output-image, .input-image, .image-preview {height: 600px !important}"

iface = gr.Interface(fn=inference,
                     inputs=inputs,
                     outputs=outputs,
                     title=title,
                     description=description,
                     examples=examples,
                     article=article,
                     css=css)
iface.launch()
