import json
import itertools
import copy
import random
from seq2seq import hlog
from absl import flags
import torch
import functools
from multiprocessing import Pool

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


def load_lexicon(lexicon_path, train_path, seperator=" ||| "):
    lexicon = json.load(open(lexicon_path))
    inputs = []
    with open(train_path, 'r') as f:
        for line in f:
            # line = line.lower()
            inputs.append(line.split(seperator)[0].strip().split(' '))

    # for k in list(lexicon.keys()):
    #     lexicon[k.lower()] = {ki.lower(): ci for ki, ci in lexicon.pop(k).items()}
    return lexicon, inputs


def filter_uncommon_tokens(lexicon, threshold, percent=1/100):
    # Filter uncommon tokens
    deleted_keys = set()

    for (k1, v1) in lexicon.items():
        deleted_codes = set()

        for c, count in v1.items():

            if count < threshold[k1] * percent:
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
                if (k2.lower() == k1.lower()
                    or k1[:-1].lower() == k2.lower()
                    or k2[:-1].lower() == k1.lower()):
                    continue
                if ci in v2:
                    print(f"{k1} and {k2} is deleted because intersecting")
                    deleted_keys.add(k1)
                    deleted_keys.add(k2)
    for k in deleted_keys:
        del lexicon[k]
    return lexicon


def find_swapables_single(k1,
                          lexicon=None,
                          inputs=None,
                          window_size=2,
                          search_size=10000):
    hlog.log(f"Processing {k1}")
    swapables = []
    x1s = list(itertools.islice(
            filter(lambda x: k1 in x and len(x) > 1, inputs), search_size))
    x1ids = [d.index(k1) for d in x1s]
    for k2 in lexicon.keys():
        if k1 != k2:
            x2s = itertools.islice(
                filter(lambda x: k2 in x and len(x) > 1, inputs), search_size)
            for x2 in x2s:
                idx2 = x2.index(k2)
                if idx2 != 0 and idx2 != len(x2)-1:
                    x2 = x2[min(0, idx2-window_size):
                            max(len(x2), idx2 + window_size + 1)]
                    for idx1, x1 in zip(x1ids, x1s):
                        if idx1 != 0 and idx1 != len(x1)-1:
                            x1 = x1[min(0, idx1-window_size):
                                    max(len(x1), idx1+window_size+1)]
                        if len(x1) == len(x2):
                            if " ".join(x1).replace(k1, k2) == " ".join(x2):
                                swapables.append(k2)
                                break

    return "||".join(set(swapables))


def get_swapables_mp(lexicon, inputs):
    inputs = copy.deepcopy(inputs)
    random.shuffle(inputs)
    finder = functools.partial(find_swapables_single,
                               lexicon=lexicon,
                               inputs=inputs)
    keys = list(lexicon.keys())
    with Pool(72) as p:
        swaps = p.map(finder, keys)

    swapables = {k: [k2 for k2 in v.split("||") if k2 in lexicon]
                 for k, v in zip(keys, swaps)}

    deleted_keys = set()
    for k, v in swapables.items():
        if len(v) == 0:
            deleted_keys.add(k)

    for k in deleted_keys:
        del lexicon[k]
        del swapables[k]
        for (k2, v2) in swapables.items():
            try:
                idx = v2.index(k)
                del v2[idx]
            except:
                pass

    return (lexicon, swapables)


def propagate_swaps(swapables, full_prop=False):
    for k1, swaps in list(swapables.items()):
        for k2 in swaps:
            if k2 not in swapables:
                swapables[k2] = []
            swaps2 = swapables[k2]
            if k1 in swaps2 and k2 not in swaps:
                swaps.append(k2)
            elif k2 in swaps and k1 not in swaps2:
                swaps2.append(k1)

    if full_prop:
        for k1, swaps in swapables.items():
            for k2 in swaps:
                for k3 in swapables[k2]:
                    if k3 != k2 and k3 not in swaps:
                        swaps.append(k3)

    return swapables


def filter_top_n(my_dict, n):
    from collections import Counter
    c = Counter(my_dict)
    most_common = c.most_common(n)
    return {k: v for k, v in most_common}


def get_counts(lexicon, inputs):
    counts = {k: 0 for k in lexicon.keys()}
    for inp in inputs:
        for k in counts.keys():
            if k in inp:
                counts[k] += 1
    return counts


def get_swapables_v3(lexicon, inputs, search_size=10):
    import spacy
    nlp = spacy.load("en_core_web_lg")
    pos_list = {}
    for k1 in lexicon.keys():
        x1s = itertools.islice(filter(lambda x: k1 in x, inputs), search_size)
        for x1 in x1s:
            x1 = " ".join(x1)
            doc = nlp(x1)
            for token in doc:
                if token.text == k1:
                    # if token.pos_ == "VERB":
                    #     tag = token.pos_
                    #     if token.morph.get("Tense"):
                    #         tag += '|' + token.morph.get("Tense")[0]
                    # else:
                    #     tag = token.pos_
                    tag = str(token.morph)

                    if tag not in pos_list:
                        pos_list[tag] = set()

                    pos_list[tag].add(k1)

    swapables = {k: set() for k in lexicon.keys()}

    for k, v in pos_list.items():
        for vi in v:
            for vj in v:
                if vi != vj:
                    swapables[vi].add(vj)
                    swapables[vj].add(vi)

    for (k, v) in swapables.items():
        swapables[k] = list(v)

    return lexicon, swapables


def filter_non_alpha(lexicon):
    for k in list(lexicon.keys()):
        if not k.isalpha():
            del lexicon[k]


def filter_lexicon_v2(lexicon, inputs, full_prop=False):
    lexicon = copy.deepcopy(lexicon)
    filter_non_alpha(lexicon)
    # print(list(lexicon.keys()))
    counts = get_counts(lexicon, inputs)
    lexicon = filter_uncommon_tokens(lexicon, counts)
    # print(list(lexicon.keys()))
    lexicon = {k: filter_top_n(lexicon[k], 3) for k in lexicon.keys()}
    # print(list(lexicon.keys()))
    lexicon = filter_intersected_tokens(lexicon)
    # print(list(lexicon.keys()))
    lexicon, swapables = get_swapables_v3(lexicon, inputs, search_size=10)
    lexicon, swapables2 = get_swapables_mp(lexicon, inputs)

    for k in lexicon.keys():
        swapables[k] = list(set(swapables[k]).intersection(swapables2[k]))

    deleted_keys = set()
    for k, _ in swapables.items():
        if k not in lexicon:
            deleted_keys.add(k)

    for k in deleted_keys:
        del swapables[k]
        for (k2, v2) in swapables.items():
            try:
                idx = v2.index(k)
                del v2[idx]
            except:
                pass

    swapables = propagate_swaps(swapables, full_prop=full_prop)

    return lexicon, swapables


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
    lexicon, swapables =\
        lexicon_and_swapables['lexicon'], lexicon_and_swapables['swapables']

    keys = list(filter(lambda k: vocab[k] in question or vocab[k] in answer,
                       lexicon.keys()))

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