import imageio
import json
import numpy as np
import os
import torch
import random
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from PIL import Image
from seq2seq import Vocab
import pdb

class CLEVRDataset(object):
    def __init__(self, root="data/clevr/", split="trainA", transform=None, vocab=None, color="RGB", size=(128,128)):
        self.root  = root
        self.split = split
        self.color = color
        self.size = size

        with open(os.path.join(self.root,"questions",f"CLEVR_{split}_questions.json")) as reader:
            self.annotations = json.load(reader)["questions"]

        # with open(self.root+"splits.json") as reader:
        #      self.annotations = [self.annotations[i] for i in json.load(reader)[self.split]]

        if vocab is None:
            self.vocab = Vocab()

        for annotation in self.annotations:
            desc = annotation["question"]
            desc = desc[:-1] + " ?"
            if "answer" in annotation:
                desc = desc + " " + annotation["answer"]
            annotation["desc"]  = desc
            annotation["image"] = os.path.join(root, "images", split, annotation["image_filename"])
            if vocab is None:
                for tok in desc.split():
                    self.vocab.add(tok)

        random.shuffle(self.annotations)
        print(f"{split}: {len(self.annotations)}")

        if transform is None:
            T = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomResizedCrop(self.size)])
            running_mean = torch.zeros(3, dtype=torch.float32)
            N  = 250
            for i in range(N):
                img = T(Image.open(self.annotations[i]["image"]).convert(self.color))
                running_mean += img.mean(dim=(1,2))
            self.mean = running_mean / N

            running_var = torch.zeros(3, dtype=torch.float32)
            for i in range(N):
                img = T(Image.open(self.annotations[i]["image"]).convert(self.color))
                running_var += ((img - self.mean[:,None,None]) ** 2).mean(dim=(1,2))
            var = running_var / N
            self.std = torch.sqrt(var)

            self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.RandomResizedCrop(self.size),
                                                transforms.Normalize(self.mean, self.std)])
        else:
            self.transform  = transform


    def __getitem__(self, i):
        annotation = self.annotations[i]
        desc  = annotation["desc"].split(" ")
        image = self.transform(Image.open( annotation["image"]).convert(self.color))
        return desc, image

    def __len__(self):
        return len(self.annotations)

    def decode(self, cmd):
        return [self.rev_vocab[int(i)] for i in cmd]

    def collate(self, batch):
        cmds, imgs = zip(*batch)
        enc_cmds = [torch.tensor(self.vocab.encode(cmd)) for cmd in cmds]
        pad_cmds = pad_sequence(enc_cmds, padding_value=0)
        return pad_cmds, torch.stack(imgs, dim=0)
