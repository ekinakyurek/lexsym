from absl import flags, app
from src import lexutils
import json
import os
import numpy as np
import pdb


FLAGS = flags.FLAGS


flags.DEFINE_string("metric_output", default=None,
                    help='output for average metrics')



def single_file_eval(prediction_file, root, split):
    
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
    with open(prediction_file) as f:
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
                category_map['all'] = [answer==pred]
            else:
                category_map[family_index].append(answer==pred)
                category_map['all'].append(answer==pred)
    
    for k, v in category_map.items():
        category_map[k] = np.mean(v)
    
    with open(prediction_file.replace('predictions', 'metrics').replace('.txt', '.json'), "w") as f:
        json.dump(category_map, f)
        
    return category_map
          

def _single_eval(prediction_file, dataroot):
    root = "data/clevr/" if dataroot is None else dataroot
    if os.path.isdir(os.path.join(root, 'images', 'trainA')):
        splits = {'train': 'trainA', 'val': 'valA', 'test': 'valB'}
    else:
        splits = {'train': 'train', 'val': 'val', 'test': 'val'}
    
    if 'test' in prediction_file:
        return single_file_eval(prediction_file, dataroot, splits['test'])
    else:
        return single_file_eval(prediction_file, dataroot, splits['val'])
        

def exp_eval(prediction_files, dataroot):
    evals = []
    for file in prediction_files:
        eval_seed = _single_eval(file, dataroot)
        evals.append(eval_seed)
    
    avg_eval = {}
    for k in evals[0].keys():
        avg_eval[k] = np.mean([eval[k] for eval in evals]) 
    
    return avg_eval

from prettytable import PrettyTable

import pdb
def main(_):
    results = {}
    x = PrettyTable()
    field_names_set = False
    for dataroot in ("original_clevr",):
        if dataroot not in results:
            results[dataroot] = {}
        for k in ("_no_swap_", "_"):
            prediction_files = []
            for seed in range(0, 4):
                prediction_template = f"clip_exp{k}vqa_seed_{seed}_{dataroot}/clevr/VQA/beta_1.0_ldim_64_dim_128_lr_1.0/predictions_test_792000.txt"
                prediction_files.append(prediction_template)
                
            results[dataroot][k] = exp_eval(prediction_files, "data/"+dataroot)
            
            if not field_names_set:
                x.field_names = ['data', 'model'] + list(results[dataroot][k].keys())
                field_names_set = True
            x.add_row([dataroot, k] + [str(round(results[dataroot][k][i],3)) for i in x.field_names[2:]])
            print(x)
            
    print(x)
    with open(FLAGS.metric_output, "w") as f:
        json.dump(results, f)      
    
if __name__ == "__main__":
    app.run(main)
