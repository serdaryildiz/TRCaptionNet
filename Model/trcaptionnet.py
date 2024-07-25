import os
import numpy

import torch
from torch import nn
from PIL import Image
from transformers import BertTokenizer

from Model import clip
from Model.bert import BertLMHeadModel, BertConfig
from Model.clip.model import Transformer


class Proj(nn.Module):

    def __init__(self, encoder_output_size, num_head=16):
        super().__init__()
        self.encoder_output_size = encoder_output_size

        self.transformer = Transformer(encoder_output_size, 1, num_head)
        self.linear = nn.Linear(encoder_output_size, 768)
        return

    def forward(self, x):
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return self.linear(x)


class TRCaptionNet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # parameters
        self.max_length = config["max_length"]
        self.proj_flag = config["proj"]
        assert type(self.proj_flag) == bool
        self.proj_num_head = config["proj_num_head"]

        # vision encoder
        self.vision_encoder, preprocess = clip.load(config["clip"], jit=False)
        self.vision_encoder.eval()
        self.vision_encoder = self.vision_encoder.visual
        with torch.no_grad():
            dummy_input_image = preprocess(Image.fromarray(numpy.zeros((512, 512, 3), dtype=numpy.uint8))).to(next(self.parameters()).device).half()
            encoder_output_size = self.vision_encoder(dummy_input_image.unsqueeze(0)).shape[-1]
        self.vision_encoder = self.vision_encoder.float()

        # language decoder
        if not os.path.isfile(config["bert"]):
            self.language_decoder = BertLMHeadModel.from_pretrained(config["bert"],
                                                                    is_decoder=True,
                                                                    add_cross_attention=True)
            self.tokenizer = BertTokenizer.from_pretrained(config["bert"])
        else:
            med_config = BertConfig.from_json_file(config["bert"])
            self.language_decoder = BertLMHeadModel(config=med_config)
            self.tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

        # proj
        if self.proj_flag:
            if self.proj_num_head is None:
                self.proj = nn.Linear(encoder_output_size, 768)
            else:
                self.proj = Proj(encoder_output_size, self.proj_num_head)
        else:
            self.proj = None
        return

    @torch.no_grad()
    def generate(self, images, max_length: int = None, min_length: int = 12, num_beams: int = 3,
                 repetition_penalty: float = 1.1):
        image_embeds = self.vision_encoder(images)

        if self.proj is not None:
            image_embeds = self.proj(image_embeds)

        image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long).to(images.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        input_ids = torch.ones((image_embeds.shape[0], 1), device=images.device, dtype=torch.long)
        input_ids *= 2

        outputs = self.language_decoder.generate(input_ids=input_ids,
                                                 max_length=self.max_length if max_length is None else max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)

        captions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return captions

    def forward(self, images, captions):
        with torch.no_grad():
            image_embeds = self.vision_encoder(images).detach()

        image_embeds = self.proj(image_embeds)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(images.device)

        captions = self.tokenizer(captions, padding='longest', truncation=True, max_length=self.max_length,
                                  return_tensors="pt").to(images.device)

        captions.input_ids[:, 0] = 2
        decoder_targets = captions.input_ids.masked_fill(captions.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, 0] = -100

        decoder_output = self.language_decoder(input_ids=captions.input_ids,
                                               attention_mask=captions.attention_mask,
                                               encoder_hidden_states=image_embeds,
                                               encoder_attention_mask=image_atts,
                                               labels=decoder_targets,
                                               return_dict=True,
                                               )

        loss_lm = decoder_output.loss
        return loss_lm

def test():
    model = TRCaptionNet({
        "max_length": 35,
        "clip": "ViT-B/32",
        "bert": "dbmdz/bert-base-turkish-cased"
    })

    return


if __name__ == '__main__':
    test()
