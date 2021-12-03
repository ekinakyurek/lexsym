import os
from data.shapes import ShapeDataset
from data.set import SetDataset
from data.scan import SCANDataset
from data.clevr import CLEVRDataset

from absl import flags
from absl import logging
FLAGS = flags.FLAGS


flags.DEFINE_string('datatype', default='setpp',
                    help='Sets which dataset to use.')

flags.DEFINE_string('dataroot', default='None',
                    help='Sets which dataset to use.')


def get_data(**kwargs):
    if FLAGS.datatype == "setpp":
        root = "data/setpp/" if FLAGS.dataroot is None else FLAGS.dataroot
        train = SetDataset(root, split="train", **kwargs)
        test = SetDataset(root,
                          split="test",
                          transform=train.transform,
                          vocab=train.vocab,
                          **kwargs)
        val = test
    elif FLAGS.datatype == "shapes":
        root = "data/shapes/" if FLAGS.dataroot is None else FLAGS.dataroot
        train = ShapeDataset(root, split="train", **kwargs)
        test = ShapeDataset(root,
                            split="test",
                            transform=train.transform,
                            vocab=train.vocab,
                            **kwargs)
        val = test
    elif FLAGS.datatype == "clevr":
        root = "data/clevr/" if FLAGS.dataroot is None else FLAGS.dataroot
        if os.path.isdir(os.path.join(root, 'images', 'trainA')):
            splits = {'train': 'trainA', 'val': 'valA', 'test': 'valB'}
        else:
            splits = {'train': 'train', 'val': 'val', 'test': 'val'}
        logging.info(f"Root data dir:  {root}")
        train = CLEVRDataset(root, split=splits['train'], **kwargs)
        test = CLEVRDataset(root,
                            split=splits['test'],
                            transform=train.transform,
                            vocab=train.vocab,
                            answer_vocab=train.answer_vocab,
                            **kwargs)
        val = CLEVRDataset(root,
                           split=splits['val'],
                           transform=train.transform,
                           vocab=train.vocab,
                           answer_vocab=train.answer_vocab,
                           **kwargs)
    else:
        root = "data/scan/" if FLAGS.dataroot is None else FLAGS.dataroot
        train = SCANDataset(root, split="train", **kwargs)
        test = SCANDataset(root,
                           split="test",
                           transform=train.transform,
                           vocab=train.vocab, **kwargs)
        val = test
    return train, val, test
