from absl import flags, app
from src import lexutils
import json
import numpy as np
import random
import sys

FLAGS = flags.FLAGS

flags.DEFINE_string("lexfile", default=None,
                    help='Seed lexicon path')

flags.DEFINE_string("trainfile", default=None,
                    help='qa and code files')

flags.DEFINE_string("testfile", default=None,
                    help='qa and code files')


def read_file(path, seperator=' ||| '):
    inputs, outputs = [], []
    with open(path) as f:
        for d in f:
            input, output, *_ = d.split(seperator)
            inputs.append(input.strip().split(' '))
            outputs.append(output.strip().split(' '))
    return (inputs, outputs)


MASK = "UNK"


def masked_fill_(array, mask, value):
    for i, m in enumerate(mask):
        if m:
            array[i] = value
    

def swap_ids(tensor, id1, id2, substitute=False):
    if substitute:
        masked_fill_(tensor, tensor == id1, id2)
    else:
        masked_fill_(tensor, tensor == id1, MASK)
        masked_fill_(tensor, tensor == id2, id1)
        masked_fill_(tensor, tensor == MASK, id2)


def make_a_swap_single(inp, out, lex_and_swaps, steps=0, substitute=False):
    lexicon, swapables = lex_and_swaps['lexicon'], lex_and_swaps['swapables']
    
    keys = list(filter(lambda k: k in inp, lexicon.keys()))

    if len(keys) != 0:
        k1 = random.choice(keys)
        weights = [1 / max(iter(lexicon[k].values())) for k in swapables[k1]]
        k2 = random.choices(swapables[k1], weights=weights, k=1)[0]
        ks = [k1, k2]
    else:
        return
        
    ks_q_id = ks    
    swap_ids(inp, *ks_q_id, substitute=substitute)
    
    if substitute:
        for v, _ in lexicon[ks[0]].items():
            code2 = random.choice(list(lexicon[ks[1]].keys()))
            masked_fill_(out, out == v, code2)
    else:
        for v, _ in lexicon[ks[0]].items():
            masked_fill_(out, out == v, MASK)

        for v, _ in lexicon[ks[1]].items():
            code1 = random.choice(list(lexicon[ks[0]].keys()))
            masked_fill_(out, out == v, code1)

        code2 = random.choice(list(lexicon[ks[1]].keys()))

        masked_fill_(out, out == MASK, code2)


def main(_):
    lex = json.load(open(FLAGS.lexfile))
    for k, v in lex["swapables"].items():
        lex["swapables"][k] = np.unique(v).tolist()
    for k in list(lex["swapables"].keys()):
        if len(lex["swapables"][k]) == 0:
            del lex["swapables"][k]
            del lex["lexicon"][k]
    # print(lex["lexicon"])
    inputs, outputs = read_file(FLAGS.trainfile, seperator='\t')
    test_inputs, test_outputs = read_file(FLAGS.testfile, seperator='\t')
    test_inputs = [" ".join(x) for x in test_inputs]
    test_outputs = [" ".join(x) for x in test_outputs]
    
    train_outputs = set([" ".join(x) for x in outputs])
    train_inputs = set([" ".join(x) for x in inputs])
    test_inputs_set = set(test_inputs)
    #test_compounds = set(list(open("/afs/csail.mit.edu/u/a/akyurek/akyurek/git/align/CoGnition/data/cg-test/cg-test.compound").readlines()))

    generated = set([])
    for i in range(10):
        for inp, out in zip(inputs, outputs):
            aug_inp = np.array(inp.copy(), dtype=object)
            aug_out = np.array(out.copy(),  dtype=object)
            make_a_swap_single(aug_inp, aug_out, lex)
            aug_inp = " ".join(aug_inp)
            aug_out = " ".join(aug_out)
            inp = " ".join(inp)
            out = " ".join(out)
            if aug_inp not in generated and aug_out not in train_outputs and aug_inp not in train_inputs:
                generated.add(aug_inp)
                print(aug_inp, "\t", aug_out, '\t', inp, '\t', out)
            if len(generated) > 1000:
                break
        if len(generated) > 1000:
            break


if __name__ == "__main__":
    app.run(main)
