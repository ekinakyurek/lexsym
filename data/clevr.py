import json
import os
import torch
import random
import functools
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from PIL import Image
from seq2seq import Vocab
import math


class CLEVRDataset(object):
    def __init__(self,
                 root="data/clevr/",
                 split="trainA",
                 transform=None,
                 vocab=None,
                 color="RGB",
                 size=(128, 128),
                 vqa=False,
                 img2code=False,
                 answer_vocab=None,
                 no_images=False,
                 ):

        self.root = root
        self.split = split
        self.color = color
        self.size = size
        self.vqa = vqa
        self.img2code = img2code
        self.no_images = no_images

        with open(os.path.join(self.root, "scenes", f"CLEVR_{split}_scenes.json")) as reader:
            self.annotations = json.load(reader)["scenes"]

        if img2code or vqa:
            with open(os.path.join(self.root, "questions", f"CLEVR_{split}_questions.json")) as reader:
                self.questions = json.load(reader)["questions"]
        #
        # self.annotations = list(filter(lambda a: len(a['objects']) < 5,
        #                           self.annotations))

        # with open(self.root+"splits.json") as reader:
        #      self.annotations = [self.annotations[i] for i in json.load(reader)[self.split]]

        if vocab is None:
            self.vocab = Vocab()
            if vqa:
                self.answer_vocab = Vocab()
            else:
                self.answer_vocab = None
        else:
            self.vocab = vocab
            self.answer_vocab = answer_vocab

        for annotation in self.annotations:
            objects = annotation["objects"]
            objs = []
            for obj in objects:
                objs.append(" ".join((obj['size'],
                                     obj['color'],
                                     obj['material'],
                                     obj['shape']))
                            )

            text = " | ".join(objs) + " ."
            annotation["text"] = text
            annotation["image"] = os.path.join(root,
                                               "images",
                                               split,
                                               annotation["image_filename"])
            if not (vqa or img2code):
                if vocab is None:
                    for tok in text.split():
                        self.vocab.add(tok)
            
            annotation.pop('relationships')
            annotation.pop('directions')
            annotation.pop('objects')
            annotation.pop('image_index')
            annotation.pop('split')

        random.shuffle(self.annotations)

        if not no_images and transform is None:
            T = transforms.Compose([transforms.ToTensor(),

                                    transforms.Resize(int(math.ceil(self.size[0]*1.1))),
                                    transforms.CenterCrop(self.size)])
            running_mean = torch.zeros(3, dtype=torch.float32)
            N = min(250, len(self.annotations))
            for i in range(N):
                with Image.open(self.annotations[i]["image"]) as image:
                    img = T(image.convert(self.color))
                    running_mean += img.mean(dim=(1, 2))
            self.mean = running_mean / N

            running_var = torch.zeros(3, dtype=torch.float32)
            for i in range(N):
                with Image.open(self.annotations[i]["image"]) as image:
                    img = T(image.convert(self.color))
                    running_var += ((img - self.mean[:, None, None]) ** 2).mean(dim=(1, 2))
            var = running_var / N
            self.std = torch.sqrt(var)

            rcrop = functools.partial(transforms.functional.resized_crop,
                                      top=10,
                                      left=20,
                                      width=440,
                                      height=300,
                                      size=self.size,
                                      interpolation=transforms.functional.InterpolationMode.BICUBIC)

            self.transform = transforms.Compose(
                                  [transforms.ToTensor(),
                                   rcrop,
                                   transforms.Normalize(self.mean, self.std)])
        elif transform is None:
            self.mean = self.std = self.transform = None
        else:
            self.mean = transform.transforms[-1].mean
            self.std = transform.transforms[-1].std
            self.transform = transform

        if vqa or img2code:
            for question in self.questions:
                question_processed = question['question'].lower().replace('?', '').strip()
                question_processed = question_processed.replace(';', ' ;')
                answer_processed = question.get('answer', '?').lower().strip()
                text = question_processed + " | " + answer_processed
                question['text'] = text
                question["image"] = os.path.join(root,
                                                 "images",
                                                 split,
                                                 question["image_filename"])
                question['question_processed'] = question_processed
                question['answer_processed'] = answer_processed
                if vocab is None:
                    for tok in text.split():
                        self.vocab.add(tok)
                    if vqa:
                        self.answer_vocab.add(answer_processed)
                question.pop('program')
                question.pop('split')
                question.pop('question_family_index')
                question.pop('question_index')
                question.pop('image_index')
                question.pop('question')
                question.pop('answer')

            self.annotations = self.questions
            random.shuffle(self.annotations)


    def __getitem__(self, i):
        annotation = self.annotations[i]
        text = annotation["text"].split(" ")
        file = annotation["image"]
        if not self.no_images:
            with Image.open(file) as image:
                img = self.transform(image.convert(self.color))
        else:
            img = None

        if self.vqa:
            question = annotation['question_processed'].split(" ")
            answer = annotation["answer_processed"]
            return question, img, answer, file

        return text, img, file

    def __len__(self):
        return len(self.annotations)

    def decode(self, cmd):
        return self.vocab.decode(cmd)

    def collate(self, batch):
        if self.vqa:
            questions, imgs, answers, files = zip(*batch)
            enc_q = [torch.tensor(self.vocab.encode(q)) for q in questions]
            pad_q = pad_sequence(enc_q, padding_value=self.vocab.pad())
            enc_a = torch.tensor(self.answer_vocab.encode(answers))
            if not self.no_images:
                imgs = torch.stack(imgs, dim=0)
            return pad_q, imgs, enc_a, files
        else:
            cmds, imgs, files = zip(*batch)
            enc_cmds = [torch.tensor(self.vocab.encode(cmd)) for cmd in cmds]
            pad_cmds = pad_sequence(enc_cmds, padding_value=self.vocab.pad())
            return pad_cmds, torch.stack(imgs, dim=0), files
