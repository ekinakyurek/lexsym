import os
from seq2seq import hlog
from data.shapes import ShapeDataset
from data.set import SetDataset
from data.scan import SCANDataset
from data.clevr import CLEVRDataset


from absl import flags

FLAGS = flags.FLAGS


flags.DEFINE_string('datatype', default='setpp',
                    help='Sets which dataset to use.')

flags.DEFINE_string('dataroot', default='None',
                    help='Sets which dataset to use.')


def get_data(datatype="clevr", dataroot="data/clevr/", **kwargs):
    if datatype == "setpp":
        root = "data/setpp/" if dataroot is None else dataroot
        train = SetDataset(root, split="train", **kwargs)
        test = SetDataset(root,
                          split="test",
                          transform=train.transform,
                          vocab=train.vocab,
                          **kwargs)
        val = test
    elif datatype == "shapes":
        root = "data/shapes/" if dataroot is None else dataroot
        train = ShapeDataset(root, split="train", **kwargs)
        test = ShapeDataset(root,
                            split="test",
                            transform=train.transform,
                            vocab=train.vocab,
                            **kwargs)
        val = test
    elif datatype == "clevr":
        root = "data/clevr/" if dataroot is None else dataroot
        if os.path.isfile(os.path.join(root, 'scenes', 'CLEVR_trainA_scenes.json')):
            splits = {'train': 'trainA', 'val': 'valA', 'test': 'valB'}
        else:
            splits = {'train': 'train', 'val': 'val', 'test': 'val'}
        hlog.log(f"Root data dir:  {root}")
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
        root = "data/scan/" if dataroot is None else dataroot
        train = SCANDataset(root, split="train", **kwargs)
        test = SCANDataset(root,
                           split="test",
                           transform=train.transform,
                           vocab=train.vocab, **kwargs)
        val = test
    return train, val, test
