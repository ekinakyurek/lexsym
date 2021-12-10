import sys
import os
import io
import base64
import json

from flask import Flask
from flask import render_template, request

import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import make_grid
from torch import nn

from absl import flags, logging

import options
import vae_train
from seq2seq import Vocab
from src import utils
from src import lexutils
from src.datasets import get_data
from src.vqvae import VectorQuantizedVAE

FLAGS = flags.FLAGS

flags.DEFINE_string("lex_and_swaps_path", default='',
                    help='A prelearned lexicon path to be used in text-image '
                         'vqvae models')

flags.DEFINE_string('vae_path', default='',
                    help='A pretrained vae path for conditional vae models.')

app = Flask(__name__)


def init_fn(_):
    args = utils.flags_to_args()
    vis_folder = utils.flags_to_path()
    img_size = tuple(map(int, args.imgsize.split(',')))
    train, val, test = get_data(size=img_size)
    os.makedirs(vis_folder, exist_ok=True)
    print("vis folder:", vis_folder)


    if args.modeltype == "VQVAE":
        model = VectorQuantizedVAE(3,
                                   args.h_dim,
                                   args.n_latent,
                                   n_codes=args.n_codes,
                                   cc=args.commitment_cost,
                                   decay=args.decay,
                                   epsilon=args.epsilon,
                                   beta=args.beta,
                                   cmdproc=False,
                                   size=train.size,
                                   )
        model = nn.DataParallel(model)
        utils.resume(model, args, mark='iter')
        model = model.module
        model.to(utils.device())
        model.eval()
    else:
        raise ValueError(f"Model type {args.modeltype} not available for this")
    
    if args.lex_and_swaps_path is not None:
        with open(args.lex_and_swaps_path) as f:
            lexicon = json.load(f)
        logging.info(f'Lex and Swaps:\n{lexicon}')

    return model, train, val, test, vis_folder, lexicon


def img2str(img):
    file_object = io.BytesIO()
    img.save(file_object, 'JPEG')
    img_str = base64.b64encode(file_object.getvalue())
    return img_str.decode('utf-8')


def encode_next():
    cmd, img, names = next(generator)
    cmd = cmd.transpose(0, 1)
    img = img.to(utils.device())
    cmd = cmd.to(utils.device())
    _, recon, _, _, _, encodings = model(**dict(img=img, cmd=cmd))
    recon = recon.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
    recon = recon.clip_(0, 1)
    encodings = encodings.flatten().cpu().tolist()
    img = T(make_grid(recon)).convert("RGB")
    return img2str(img), encodings


@app.route('/get_next', methods=['GET', 'POST'])
def get_next():
    img, encodings = encode_next()
    data = {"img": img,
            "encodings": encodings,
            "matchings": app.config['MATCHINGS']}
    # print(data)
    return json.dumps(data)


@app.route('/decode', methods=['GET', 'POST'])
def decode():
    encodings = np.array([int(request.form['cell'+str(i)]) for i in range(app.config['GRID_SIZE']**2)])
    encodings = torch.from_numpy(encodings).to(utils.device())
    encodings = encodings.view(1, -1)
    print(encodings)
    quantized = model.codebook1._embedding(encodings)  # B,HW,C
    C = quantized.shape[-1]
    z_rnn = quantized.transpose(1, 2).contiguous().view(1, C, app.config['GRID_SIZE'], app.config['GRID_SIZE'])
    recon = model.decode(z_rnn)
    recon = recon.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
    recon = recon.clip_(0, 1)
    img = T(make_grid(recon)).convert("RGB")
    return json.dumps({"img": img2str(img)})

def get_dummy_question_answer():
    vocab = Vocab()
    answer_vocab = Vocab()
    answer = question = "blue yellow red green brown gray purple cyan"
    for word in question.split():
        vocab.add(word)
    for word in reversed(question.split()):
        answer_vocab.add(word)
    answer = torch.tensor(answer_vocab.encode(answer.split()))
    question = torch.tensor(vocab.encode(question.split()))
    return question, vocab, answer, answer_vocab


@app.route('/swap', methods=['GET', 'POST'])
def swap():
    encodings = np.array([int(request.form['cell'+str(i)]) for i in range(app.config['GRID_SIZE']**2)])
    encodings = torch.from_numpy(encodings)
    encodings = encodings.view(1, -1)
    print("Before swap (encodings):", encodings)
    question, vocab, answer, answer_vocab = get_dummy_question_answer()
    print("Before swap (questions):", vocab.decode(question.numpy()))
    print("Before swap (answer):", answer_vocab.decode(answer.numpy()))
    lexutils.random_swap(app.config['LEXICON'], question, vocab, answer, answer_vocab, encodings)
    print("After swap:", encodings)
    print("After swap (questions):", vocab.decode(question.numpy()))
    print("After swap (answer):", answer_vocab.decode(answer.numpy()))
    encodings = encodings.to(utils.device())
    quantized = model.codebook1._embedding(encodings)  # B,HW,C
    C = quantized.shape[-1]
    z_rnn = quantized.transpose(1, 2).contiguous().view(1, C, app.config['GRID_SIZE'], app.config['GRID_SIZE'])
    recon = model.decode(z_rnn)
    recon = recon.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
    recon = recon.clip_(0, 1)
    img = T(make_grid(recon)).convert("RGB")
    data = {"img": img2str(img),
            "encodings": encodings.flatten().cpu().tolist()
            }
    return json.dumps(data)


@app.route('/')
@app.route('/index')
def index():
    return render_template("demo.html", vis_folder=app.config['VIS_FOLDER'], grid_size=model.latent_shape[1])


if __name__ == "__main__":
    flags.FLAGS(sys.argv)
    model, train, val, test, vis_folder, lexicon = init_fn(sys.argv)
    test_loader = DataLoader(train, batch_size=1, shuffle=False, collate_fn=train.collate)
    generator = iter(test_loader)
    T = torchvision.transforms.ToPILImage(mode=train.color)
    app.config['VIS_FOLDER'] = vis_folder
    app.config['DATA_FOLDER'] = os.path.join(test.root, "images", test.split)
    app.config['GRID_SIZE'] = model.latent_shape[1]
    app.config['MATCHINGS'] = lexicon['lexicon']
    app.config['LEXICON'] = lexicon
    app.run(host='0.0.0.0', port=6632)
