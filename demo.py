import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import make_grid, save_image
from absl import app, flags
FLAGS = flags.FLAGS
from main import set_seed, flags_to_path, device
from main import VectorQuantizedVAE, CLEVRDataset, SetDataset, ShapeDataset
from flask import Flask, render_template, request, jsonify
from absl import app as fapp
import os
import io, base64
import json
import numpy as np
app = Flask(__name__)

def init_fn(args):
    set_seed(FLAGS.seed)

    if FLAGS.datatype == "setpp":
        train  = SetDataset("data/setpp/", split="train")
        test   = SetDataset("data/setpp/", split="test", transform=train.transform, vocab=train.vocab)
    elif FLAGS.datatype == "shapes":
        train  = ShapeDataset("data/shapes/",split="train")
        test   = ShapeDataset("data/shapes/",split="test", transform=train.transform, vocab=train.vocab)
    elif FLAGS.datatype == "clevr":
        train  = CLEVRDataset("data/clevr/", split="trainA")
        test   = CLEVRDataset("data/clevr/", split="valB", transform=train.transform, vocab=train.vocab)
    else:
        train  = SCANDataset("data/scan/",split="train")
        test   = SCANDataset("data/scan/",split="test", transform=train.transform, vocab=train.vocab)

    vis_folder = flags_to_path()
    os.makedirs(vis_folder,exist_ok = True)
    print("vis folder:", vis_folder)

    if FLAGS.modeltype == "VQVAE":
        model = VectorQuantizedVAE(3, FLAGS.h_dim,
                                     FLAGS.n_latent,
                                     n_codes=FLAGS.n_codes,
                                     cc=FLAGS.commitment_cost,
                                     decay=FLAGS.decay,
                                     epsilon=FLAGS.epsilon,
                                     beta=FLAGS.beta,
                                     cmdproc=False,
                                     size=train.size,
                                   ).to(device)
        model.load_state_dict(torch.load(FLAGS.vae_path).state_dict())
        model.eval()
    else:
        error("Not available for imgcode")
    with open(FLAGS.lex_path,"r") as f:
        matchings = json.load(f)
    return model, train, test, vis_folder, matchings

def img2str(img):
    file_object = io.BytesIO()
    img.save(file_object, 'JPEG')
    img_str = base64.b64encode(file_object.getvalue())
    return img_str.decode('utf-8')

def encode_next():
    cmd, img, names = next(generator)
    img = img.to(device)
    cmd = cmd.to(device)
    _, recon, _, _, _, encodings = model(img, cmd)
    recon = recon.cpu().data * train.std[None,:,None,None] + train.mean[None,:,None,None]
    encodings = encodings.flatten().cpu().tolist()
    img = T(make_grid(recon)).convert("RGB")
    return img2str(img), encodings

@app.route('/get_next', methods=['GET', 'POST'])
def get_next():
    img, encodings = encode_next()
    data = {"img":img, "encodings":encodings, "matchings":app.config['MATCHINGS']}
    # print(data)
    return  json.dumps(data)

@app.route('/decode', methods=['GET', 'POST'])
def decode():
    encodings = np.array([int(request.form['cell'+str(i)]) for i in range(app.config['GRID_SIZE']**2)])
    encodings = torch.from_numpy(encodings).to(device)
    encodings = encodings.view(1,-1)
    print(encodings)
    quantized = model.codebook1._embedding(encodings) # B,HW,C
    C = quantized.shape[-1]
    z_rnn = quantized.transpose(1,2).contiguous().view(1,C,app.config['GRID_SIZE'],app.config['GRID_SIZE'])
    recon = model.decode(z_rnn)
    recon = recon.cpu().data * train.std[None,:,None,None] + train.mean[None,:,None,None]
    img = T(make_grid(recon)).convert("RGB")
    return json.dumps({"img": img2str(img)})


@app.route('/')
@app.route('/index')
def index():
    return render_template("demo.html", vis_folder=app.config['VIS_FOLDER'], grid_size=model.latent_shape[1])


import sys
if __name__=="__main__":
    flags.FLAGS(sys.argv)
    model, train, test, vis_folder, matchings = init_fn(sys.argv)
    test_loader = DataLoader(train, batch_size=1, shuffle=False, collate_fn=train.collate)
    generator = iter(test_loader)
    T = torchvision.transforms.ToPILImage(mode=train.color)
    app.config['VIS_FOLDER']  = vis_folder
    app.config['DATA_FOLDER'] = os.path.join(test.root, "images", test.split)
    app.config['GRID_SIZE'] = model.latent_shape[1]
    app.config['MATCHINGS'] = matchings
    app.run(host='0.0.0.0', port=6635);
