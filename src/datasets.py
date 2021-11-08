from data.shapes import ShapeDataset
from data.set import SetDataset
from data.scan import SCANDataset
from data.clevr import CLEVRDataset

from absl import flags
FLAGS = flags.FLAGS


flags.DEFINE_string('datatype', default='setpp',
                    help='Sets which dataset to use.')


def get_data():
    if FLAGS.datatype == "setpp":
        train = SetDataset("data/setpp/", split="train")
        test = SetDataset("data/setpp/",
                          split="test",
                          transform=train.transform,
                          vocab=train.vocab)
    elif FLAGS.datatype == "shapes":
        train = ShapeDataset("data/shapes/", split="train")
        test = ShapeDataset("data/shapes/",
                            split="test",
                            transform=train.transform,
                            vocab=train.vocab)
    elif FLAGS.datatype == "clevr":
        train = CLEVRDataset("data/clevr/", split="trainA")
        test = CLEVRDataset("data/clevr/",
                            split="valB",
                            transform=train.transform,
                            vocab=train.vocab)
    else:
        train = SCANDataset("data/scan/", split="train")
        test = SCANDataset("data/scan/",
                           split="test",
                           transform=train.transform,
                           vocab=train.vocab)
    return train, test
