from absl import flags, app
from src import lexutils
import json
import os
import numpy as np
import pdb


FLAGS = flags.FLAGS

flags.DEFINE_string("prediction_file", default=None,
                    help='Seed lexicon path')


flags.DEFINE_string("dataroot", default=None,
                    help='data root for clevr')


def main(_):
    root = "data/clevr/" if FLAGS.dataroot is None else FLAGS.dataroot
    if os.path.isdir(os.path.join(root, 'images', 'trainA')):
        splits = {'train': 'trainA', 'val': 'valA', 'test': 'valB'}
    else:
        splits = {'train': 'train', 'val': 'val', 'test': 'val'}
    
    if 'test' in FLAGS.prediction_file:
        eval_split(FLAGS.dataroot, splits['test'])
    else:
        eval_split(FLAGS.dataroot, splits['val'])
        

def eval_split(root, split):
    
    with open(os.path.join(root, "questions", f"CLEVR_{split}_questions.json")) as reader:
        data = json.load(reader)
        questions = data['questions']
    
    hashmap = {}
    for i, question in enumerate(questions):
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
        if question["image"] in hashmap:
            hashmap[question["image"]].append(i)
        else:
            hashmap[question["image"]] = [i]
            
    category_map = {}
    with open(FLAGS.prediction_file) as f:
        for line in f:
            question, answer, pred, image = line.strip().split('\t')
            question = question.replace('<unk>', '')
            ids = hashmap[image]
            pquestions = [questions[i]['question_processed'] for i in ids]
            try:
                i = pquestions.index(question)
                question_info = questions[ids[i]]
            except:
                print(question)
                print(pquestions)
                pdb.set_trace()
            family_index = question_info['program'][-1].get('function')
            if family_index not in category_map:
                category_map[family_index] = [answer==pred]
            else:
                category_map[family_index].append(answer==pred)
    
    for k, v in category_map.items():
        category_map[k] = np.mean(v)
    
    with open(FLAGS.prediction_file.replace('predictions', 'metrics').replace('.txt', '.json'), "w") as f:
        json.dump(category_map, f)
            
    
if __name__ == "__main__":
    app.run(main)
