import imageio
import json
import numpy as np
import os
import torch
import random
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from PIL import Image
import math
import re
from seq2seq import Vocab


def get_digits(n, b, k=4):
    if n == 0:
        return [0] * 4
    digits = [0] * 4
    i = 0
    while n:
        digits[i] = int(n % b)
        n //= b
        i += 1
    return digits



def preprocess_scan(folder="scan/images"):
    rex = re.compile(r'image(\d+).png')

    objs = ["ice-lolly", "hat", "suitcase"]
    colors = ["color"+str(i) for i in range(16)]

    data = []
    splits = {"train":[], "test":[]}

    for imname in os.listdir(folder):

        if not imname.endswith(".png"):
            continue

        id = int(rex.match(imname).groups()[0])

        obj_color, wall_color, floor_color, obj_id = get_digits(id, 16)

        text = colors[obj_color] + " " + objs[obj_id] + " , " + \
               colors[wall_color] + " " + "walls" + " , " + \
               colors[floor_color] + " "  + "floor"

        data.append({"text": text, "image": os.path.join("images", imname)})

        if colors[obj_color] == "color1" and objs[obj_id] == "hat":
            splits["test"].append(len(data)-1)
        else:
            splits["train"].append(len(data)-1)

    with open('scan/data.json', 'w') as f:
        json.dump(data, f)

    with open('scan/splits.json', 'w') as f:
        json.dump(splits, f)


class SCANDataset(object):
    def __init__(self, root="data/scan/", split="train", transform=None, vocab=None, color="HSV", size=(80,80)):
        self.root = root
        self.split = split
        self.color = color
        self.size = self.size

        with open(self.root+"data.json") as reader:
            self.annotations = json.load(reader)

        with open(self.root+"splits.json") as reader:
            self.annotations = [self.annotations[i] for i in json.load(reader)[self.split]]

        if vocab is None:
            self.vocab = Vocab()
            for annotation in self.annotations:
                desc = annotation["text"]
                for tok in desc.split():
                    self.vocab.add(tok)
        else:
            self.vocab = vocab

        random.shuffle(self.annotations)
        print(f"{split}: {len(self.annotations)}")

        if transform is None:
            T = transforms.ToTensor()

            running_mean = torch.zeros(3, dtype=torch.float32)
            for i in range(len(self.annotations)):
                img = T(Image.open(os.path.join(self.root,
                           self.annotations[i]["image"])).convert(self.color))
                running_mean += img.mean(dim=(1, 2))
            self.mean = running_mean / len(self.annotations)

            running_var = torch.zeros(3, dtype=torch.float32)
            for i in range(len(self.annotations)):
                img = T(Image.open(os.path.join(self.root,
                            self.annotations[i]["image"])).convert(self.color))
                running_var += ((img - self.mean[:, None, None]) ** 2).mean(dim=(1, 2))
            var = running_var / len(self.annotations)
            self.std = torch.sqrt(var)

            self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(self.mean, self.std)])
        else:
            self.transform  = transform


    def __getitem__(self, i):
        annotation = self.annotations[i]
        file = annotation["image"]
        desc = annotation["text"]
        image = self.transform(Image.open(os.path.join(self.root,file)).convert(self.color))
        return desc.split(), image, file

    def __len__(self):
        return len(self.annotations)

    def decode(self, cmd):
        return self.vocab.decode(cmd)

    def collate(self, batch):
        cmds, imgs, files = zip(*batch)
        # enc_cmds = [torch.tensor([self.vocab[w] for w in cmd]) for cmd in cmds]
        enc_cmds = [torch.tensor(self.vocab.encode(cmd)) for cmd in cmds]
        pad_cmds = pad_sequence(enc_cmds, padding_value=self.vocab.pad())
        return pad_cmds, torch.stack(imgs, dim=0), files


if __name__=="__main__":
    preprocess_scan()
