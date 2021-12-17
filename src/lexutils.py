import json
import itertools
import copy
import random
from absl import logging
from absl import flags
import torch


FLAGS = flags.FLAGS

flags.DEFINE_bool("substitute", default=False,
                    help='substitute instead of swap')


def filter_lexicon(lexicon):
    keys_to_hold = "yellow,red,green,cyan,purple,blue,gray,brown".split(",")
    deleted_keys = set()
    for k in lexicon.keys():
        if k not in keys_to_hold:
            deleted_keys.add(k)

    for k in deleted_keys:
        del lexicon[k]

    return lexicon


def load_lexicon(lexicon_path, train_path):
    lexicon = json.load(open(lexicon_path))
    inputs = []
    with open(train_path, 'r') as f:
        for line in f:
            inputs.append(line.split('\t')[0])
    return lexicon, inputs

def filter_uncommon_tokens(lexicon, threshold):
    # Filter uncommon tokens
    deleted_keys = set()
    
    for (k1, v1) in lexicon.items():
        deleted_codes = set()
        
        for c, count in v1.items():
            
            if count < threshold[k1] / 100:
                deleted_codes.add(c)
        
        for k in deleted_codes:
            del v1[k]
            
        if len(v1) == 0:
            deleted_keys.add(k1)
            
    for k in deleted_keys:
        del lexicon[k]
        
    return lexicon


def filter_intersected_tokens(lexicon):
    deleted_keys = set()
    for (k1, v1) in lexicon.items():
        for ci, count in v1.items():
            for (k2, v2) in lexicon.items():
                if k2 == k1:
                    continue
                if ci in v2:
                    deleted_keys.add(k1)
                    deleted_keys.add(k2)
    for k in deleted_keys:
        del lexicon[k]
    return lexicon
    

def get_swapables(lexicon, inputs):
    inputs = copy.deepcopy(inputs)
    random.shuffle(inputs)
    swapables = {k: [] for k in lexicon.keys()}
    for k1 in lexicon.keys():
        for k2 in lexicon.keys():
            if k1 != k2:
                if k1 in swapables[k2]:
                    swapables[k1].append(k2)
                else:   
                    x1s = itertools.islice(filter(lambda x: k1 in x, inputs), 5000)
                    x2s = itertools.islice(filter(lambda x: k2 in x, inputs), 5000)
                    for (x1, x2) in itertools.product(x1s, x2s):
                        if x1.replace(k1, k2) == x2:
                            swapables[k1].append(k2)
                            print(f"Linked {k1} - {k2}")
                            break
    deleted_keys = set()               
    for k, v in swapables.items():
        if len(v) == 0:
            deleted_keys.add(k)
            
    for k in deleted_keys:
        del lexicon[k]
        del swapables[k]
             
    return (lexicon, swapables)

def propagate_swaps(swapables): 
    for k1, swaps in swapables.items():
        for k2 in swaps:
            swaps2 = swapables[k2]
            if k1 in swaps2 and k2 not in swaps:
                swaps.append(k2)
            elif k2 in swaps and k1 not in swaps2:
                swaps2.append(k1)
    return swapables
    

def get_counts(lexicon, inputs):
    counts = {k: 0 for k in lexicon.keys()}
    for inp in inputs:
        for k in counts.keys():
            if k in inp:
                counts[k]+= 1
    return counts

    
def filter_lexicon_v2(lexicon, inputs):
    lexicon = copy.deepcopy(lexicon)
    counts = get_counts(lexicon, inputs)
    lexicon = filter_uncommon_tokens(lexicon, counts)
    lexicon = filter_intersected_tokens(lexicon)
    lexicon, swapables = get_swapables(lexicon, inputs)
    return lexicon, propagate_swaps(swapables)


def swap_ids(tensor, id1, id2, substitute=False):
    if not substitute:
        tensor.masked_fill_(tensor == id1, -1)
        tensor.masked_fill_(tensor == id2, id1)
        tensor.masked_fill_(tensor == -1, id2)
    else:
        tensor.masked_fill_(tensor == id1, id2)


def swap_codes(codes, lexicon, token1, token2, substitute=False):
    if not substitute:
        for v, _ in lexicon[token1].items():
            codes.masked_fill_(codes == int(v), -1)

        for v, _ in lexicon[token2].items():
            region2 = codes == int(v)
            length2 = int(region2.sum().item())
            if length2 > 0:
                possible_codes1 = list(lexicon[token1].keys())
                code_scores1 = list(lexicon[token1].values())
                codes1 = random.choices(possible_codes1,
                                        weights=code_scores1,
                                        k=length2)

                codes[region2] = torch.tensor(list(map(int, codes1)))
        
        region1 = codes == -1
        length1 = int(region1.sum().item())
        if length1 > 0:
            possible_codes2 = list(lexicon[token2].keys())
            code_scores2 = list(lexicon[token2].values())
            codes2 = random.choices(possible_codes2,
                                    weights=code_scores2,
                                    k=length1)
            codes[region1] = torch.tensor(list(map(int, codes2)))
    else:
        for v, _ in lexicon[token1].items():
            region1 = codes == int(v)
            length1 = int(region1.sum().item())
            possible_codes2 = list(lexicon[token2].keys())
            code_scores2 = list(lexicon[token2].values())
            
            codes2 = random.choices(possible_codes2,
                                    weights=code_scores2,
                                    k=length1)
            
            codes[region1] = torch.tensor(list(map(int, codes2)))
        
        
def random_swap(lexicon_and_swapables, question, vocab, answer, answer_vocab, codes):
    lexicon, swapables = lexicon_and_swapables['lexicon'], lexicon_and_swapables['swapables']
    
    keys = list(filter(lambda k: vocab[k] in question or vocab[k] in answer, lexicon.keys()))
    
    if len(keys) != 0:
        k1 = random.choice(keys)
        k2 = random.choice(swapables[k1])
        ks = [k1, k2]
    else:
        k1 = random.choice(list(lexicon.keys()))
        k2 = random.choice(swapables[k1])
        ks = [k1, k2]
        
    ks_q_id = [vocab[k] for k in ks]
    ks_a_id = [answer_vocab[k] for k in ks]
    swap_ids(question, *ks_q_id, substitute=FLAGS.substitute)
    swap_ids(answer, *ks_a_id, substitute=FLAGS.substitute)
    swap_codes(codes, lexicon, *ks, substitute=FLAGS.substitute)
    
