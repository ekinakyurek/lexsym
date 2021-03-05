import imageio
import json
import numpy as np
import os
import torch
import random
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from PIL import Image
import pdb

class ShapeDataset(object):
    def __init__(self, root="data/shapes/", split="train", transform=None, vocab=None, color="RGB"):
        self.root = root
        self.split = split
        self.color = color
        with open(self.root + "data.json") as reader:
            self.annotations = json.load(reader)
        with open(self.root+"splits.json") as reader:
             self.annotations = [self.annotations[i] for i in json.load(reader)[self.split]]

        if vocab is None:
            self.vocab = {"*pad*": 0}
            for annotation in self.annotations:
                desc = annotation["description"]
                for tok in desc.split():
                    if tok not in self.vocab:
                        self.vocab[tok] = len(self.vocab)
        else:
            self.vocab = vocab

        self.rev_vocab = {v: k for k, v in self.vocab.items()}

        random.shuffle(self.annotations)
        print(f"{split}: {len(self.annotations)}")

        if transform is None:
            T = transforms.ToTensor()

            running_mean = torch.zeros(3, dtype=torch.float32)
            for i in range(len(self.annotations)):
                img = T(Image.open(os.path.join(self.root,
                           self.annotations[i]["image"])).convert(self.color))
                running_mean += img.mean(dim=(1,2))
            self.mean = running_mean / len(self.annotations)

            running_var = torch.zeros(3, dtype=torch.float32)
            for i in range(100):
                img = T(Image.open(os.path.join(self.root,
                            self.annotations[i]["image"])).convert(self.color))
                running_var += ((img - self.mean[:,None,None]) ** 2).mean(dim=(1,2))
            var = running_var / len(self.annotations)
            self.std = np.sqrt(var)

            self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(self.mean, self.std)])
        else:
            self.transform = transform

    def __getitem__(self, i):
        annotation = self.annotations[i]
        image = annotation["image"]
        desc = annotation["description"]
        image = self.transform(Image.open(os.path.join(self.root, image)).convert(self.color))
        return desc.split(), image

    def __len__(self):
        return len(self.annotations)

    def decode(self, cmd):
        return [self.rev_vocab[int(i)] for i in cmd]

    def collate(self, batch):
        cmds, imgs = zip(*batch)
        enc_cmds = [torch.tensor([self.vocab[w] for w in cmd]) for cmd in cmds]
        pad_cmds = pad_sequence(enc_cmds, padding_value=0)
        return pad_cmds, torch.stack(imgs, dim=0)
