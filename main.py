import os
import torch
import imageio
import json
import random
import numpy as np
from torch import optim
from torch.utils import data as torch_data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.lex import FilterModel
from src.vqvae import VectorQuantizedVAE
from src.vae import VAE, CVAE
from src.dae import DAE

from data.shapes import ShapeDataset
from data.set import SetDataset
from data.scan import SCANDataset

import torchvision
from torchvision.utils import make_grid, save_image
import itertools
from absl import app, flags

FLAGS = flags.FLAGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE: ", device)

def train_lexgen():
    if FLAGS.datatype == "setpp":
        train  = SetDataset("data/setpp/", split="train")
        test   = SetDataset("data/setpp/", split="test", transform = train.transform, vocab = train.vocab)
    else:
        train  = ShapeDataset(root="data/shapes/", split="train")
        test   = ShapeDataset(root="data/shapes/", split="test", transform = train.transform, vocab = train.vocab)

    vis_folder = os.path.join("vis",FLAGS.datatype,"FilterModel")
    os.makedirs(vis_folder,exist_ok = True)
    print("vis folder:", vis_folder)

    loader = DataLoader(train, batch_size=FLAGS.n_batch, collate_fn=train.collate)
    test_loader = DataLoader(test, batch_size=32, collate_fn=train.collate)

    model = FilterModel(
        vocab=train.vocab,
        n_downsample=2,
        n_latent=FLAGS.n_latent,
        n_steps=10
    ).cuda()

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

    for i in range(50):
        print(i)
        for j, (cmd, img) in enumerate(loader):
            loss, *_ = model(cmd.to(device), img.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss.item())
        if i%5 == 0:
            visualizations = []

            itloader = iter(loader)
            for i in range(3):
                cmd, img = next(itloader)
                print(cmd)
                loss, _, results, attentions, text_attentions = model(cmd.to(device), img.to(device))
                visualizations.append(visualize(
                    train,
                    f"train-{i}",
                    train.decode(cmd[:, 0]),
                    results,
                    attentions,
                    text_attentions
                ))

            print("\n-----------------\n")
            itloader = iter(test_loader)
            for i in range(3):
                cmd, img = next(itloader)
                print(cmd)
                loss, _, results, attentions, text_attentions = model(cmd.to(device), img.to(device))
                visualizations.append(visualize(
                    train,
                    f"test-{i}",
                    test.decode(cmd[:, 0]),
                    results,
                    attentions,
                    text_attentions,
                ))

            render_html(visualizations)

def visualize(dataset, name, command, results, attentions, text_attentions):
    def prep(img):
        img = img[0, ...].cpu().detach()
        if img.shape[0] == 3:
            img = img * dataset.std[:, None, None] + dataset.mean[:, None, None]
            img = torch.clip(img, 0, 1)
        img = img.numpy().transpose(1, 2, 0)
        if img.shape[2] == 1:
            img = img[:, :, 0]
        return img

    imageio.imwrite(os.path.join(vis_folder, name + f".result-0.png"), prep(results[0]))
    example = {
        "name": name,
        "command": command,
        "init-result": name + f".result-0.png",
        "attentions": [],
        "text_attentions": [],
        "results": [],
    }
    for i in range(1, len(results)):
        example["attentions"].append(name + f".att-{i}.png")
        t_att = text_attentions[i-1][:, 0].detach().cpu().numpy().ravel().tolist()
        t_att = " | ".join([f"{a:.1f}" for a in t_att])
        example["text_attentions"].append(t_att)
        example["results"].append(name + f".result-{i}.png")
        imageio.imwrite("vis/" + name + f".att-{i}.png", prep(attentions[i-1]))
        imageio.imwrite("vis/" + name + f".result-{i}.png", prep(results[i]))
    return example

def render_html(visualizations):
    with open("vis/index.html", "w") as writer:
        for vis in visualizations:
            print("<p>", vis["name"], vis["command"], "</p>", file=writer)
            print("<table>", file=writer)
            init = vis["init-result"]
            print(f"<tr><td></td><td><img src='{init}' height=30></td></tr>", file=writer)
            for att, result, t_att in zip(vis["attentions"], vis["results"], vis["text_attentions"]):
                print(f"<tr><td><img src='{att}' height=30></td><td><img src='{result}' height=30></td><td>{t_att}</td></tr>", file=writer)
            print("</table>", file=writer)
            print("<br>", file=writer)



flags.DEFINE_integer("h_dim", 32, "")
flags.DEFINE_integer("rnn_dim", 256, "")
flags.DEFINE_integer("rnn_n_layers", 2, "")
flags.DEFINE_float("rnn_drop", 0.1, "")
flags.DEFINE_integer("n_latent", 24, "")
flags.DEFINE_integer("n_batch", 128, "")
flags.DEFINE_integer("n_iter", 25000, "")
flags.DEFINE_integer("n_codes", 10, "")
flags.DEFINE_integer("seed", 0, "")
flags.DEFINE_float("beta", 5.0, "")
flags.DEFINE_float("commitment_cost", 0.25, "")
flags.DEFINE_string("datatype","setpp","")
flags.DEFINE_string("modeltype","VQVAE","")
flags.DEFINE_float("decay",0.99,"")
flags.DEFINE_float("lr",1e-3,"")
flags.DEFINE_float("epsilon",1e-5,"")
flags.DEFINE_bool("debug", False,"")
flags.DEFINE_bool("highdrop", False,"")
flags.DEFINE_bool("highdroptest", False,"")
flags.DEFINE_float("highdropvalue",0.,"")
flags.DEFINE_bool("copy", False,"")
flags.DEFINE_string("vae_path","","pretrained vae path")

def flags_to_path():
    root = os.path.join("vis", FLAGS.datatype, FLAGS.modeltype)
    if FLAGS.modeltype == "VQVAE":
        return os.path.join(root,
                    f"beta_{FLAGS.beta}_ncodes_{FLAGS.n_codes}_ldim_{FLAGS.n_latent}_dim_{FLAGS.h_dim}_lr_{FLAGS.lr}")
    else:
        return os.path.join(root,
                    f"beta_{FLAGS.beta}_ldim_{FLAGS.n_latent}_dim_{FLAGS.h_dim}_lr_{FLAGS.lr}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_vae():
    set_seed(FLAGS.seed)
    if FLAGS.datatype == "setpp":
        train  = SetDataset("data/setpp/", split="train")
        test   = SetDataset("data/setpp/", split="test", transform=train.transform, vocab=train.vocab)
    elif FLAGS.datatype == "shapes":
        train  = ShapeDataset("data/shapes/",split="train")
        test   = ShapeDataset("data/shapes/",split="test", transform=train.transform, vocab=train.vocab)
    else:
        train  = SCANDataset("data/scan/",split="train")
        test   = SCANDataset("data/scan/",split="test", transform=train.transform, vocab=train.vocab)

    vis_folder = flags_to_path()
    os.makedirs(vis_folder,exist_ok = True)
    print("vis folder:", vis_folder)


    loader = DataLoader(train, batch_size=FLAGS.n_batch, shuffle=True, collate_fn=train.collate, pin_memory=True)
    test_loader = DataLoader(test, batch_size=32, shuffle=True,collate_fn=train.collate, pin_memory=True)

    if FLAGS.modeltype == "VQVAE":
        model = VectorQuantizedVAE(3, FLAGS.h_dim,
                                     FLAGS.n_latent,
                                     n_codes=FLAGS.n_codes,
                                     cc=FLAGS.commitment_cost,
                                     decay=FLAGS.decay,
                                     epsilon=FLAGS.epsilon,
                                     beta=FLAGS.beta,
                                     cmdproc=False,
                                   ).to(device)
    elif FLAGS.modeltype == "VAE":
        model = VAE(3, FLAGS.h_dim, FLAGS.n_latent, beta=FLAGS.beta).to(device)
    elif FLAGS.modeltype ==  "DAE":
        model = DAE(3, latentdim=FLAGS.n_latent*4*4).to(device)
    else:
        error("Unknown Model Type")

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

    train_res_recon_error = 0.
    cnt = 0.
    model.train()
    iterator = itertools.cycle(iter(loader))
    for i in range(FLAGS.n_iter):
        (cmd, img) = next(iterator)
        img = img.to(device)
        cmd = cmd.to(device)
        optimizer.zero_grad()
        loss, _, recon_error, *_ = model(img, cmd)
        loss.backward()
        optimizer.step()

        train_res_recon_error += recon_error.item()
        cnt += img.shape[0]
        # train_res_nll.append(nll.item())

        if (i+1) % 100 == 0:
            with torch.no_grad():
                print('%d iterations' % (i+1))
                print('recon_error: %.6f' % (train_res_recon_error / cnt))
                #print('nll: %.6f' % np.mean(train_res_nll[-100:]))
                print(len(test_loader))
                val_recon, val_perp = evaluate_vqvae(model, test_loader)
                print('val_recon_error: %.6f' % val_recon)
                print('val_nll: %.6f' % val_perp)
                test_iter = itertools.cycle(iter(test_loader))
                if (i+1) % 500 == 0:
                    T = torchvision.transforms.ToPILImage(mode=train.color)
                    for j in range(5):
                        cmd, img = next(test_iter)
                        loss, recon, recon_error, *_= model(img.to(device), cmd.to(device))
                        recon = recon.cpu().data * train.std[None,:,None,None] + train.mean[None,:,None,None]
                        img   = img.cpu().data * train.std[None,:,None,None] + train.mean[None,:,None,None]
                        res   =  torch.cat((recon,img),0).clip_(0,1)
                        T(make_grid(res)).convert("RGB").save(os.path.join(vis_folder,f"{i}_{j}.png"))
                        #if FLAGS.modeltype=="VAE":
                        sample, _ = model.sample(B=32)
                        sample = sample.cpu().data * train.std[None,:,None,None] + train.mean[None,:,None,None]
                        T(make_grid(sample.clip_(0,1))).convert("RGB").save(os.path.join(vis_folder,f"prior_{i}_{j}.png"))

    torch.save(model,os.path.join(vis_folder,f"model.pt"))

def train_cvae():
    assert FLAGS.vae_path != ""
    set_seed(FLAGS.seed)
    if FLAGS.datatype == "setpp":
        train  = SetDataset("data/setpp/", split="train")
        test   = SetDataset("data/setpp/", split="test", transform=train.transform, vocab=train.vocab)
    elif FLAGS.datatype == "shapes":
        train  = ShapeDataset("data/shapes/",split="train")
        test   = ShapeDataset("data/shapes/",split="test", transform=train.transform, vocab=train.vocab)
    else:
        train  = SCANDataset("data/scan/",split="train")
        test   = SCANDataset("data/scan/",split="test", transform=train.transform, vocab=train.vocab)

    vis_folder = flags_to_path()
    os.makedirs(vis_folder,exist_ok = True)
    print("vis folder:", vis_folder)
    loader = DataLoader(train, batch_size=FLAGS.n_batch, shuffle=True, collate_fn=train.collate, pin_memory=True)
    test_loader = DataLoader(test, batch_size=32, shuffle=True,collate_fn=train.collate, pin_memory=True)

    if FLAGS.modeltype == "CVQVAE":
        error("Unknown Model Type")
        # model = VectorQuantizedVAE(3, FLAGS.h_dim,
        #                              FLAGS.n_latent,
        #                              n_codes=FLAGS.n_codes,
        #                              cc=FLAGS.commitment_cost,
        #                              decay=FLAGS.decay,
        #                              epsilon=FLAGS.epsilon,
        #                              beta=FLAGS.beta,
        #                              cmdproc=False,
        #                            ).to(device)
    elif FLAGS.modeltype == "CVAE":
        model = CVAE(3, FLAGS.h_dim, FLAGS.n_latent, beta=FLAGS.beta, rnn_dim=FLAGS.rnn_dim, vocab=train.vocab)
        model.vae.load_state_dict(torch.load(FLAGS.vae_path).state_dict())
        model = model.to(device)
        for p in model.vae.parameters():
            p.requires_grad = False
    else:
        error("Unknown Model Type")

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

    train_loss= 0.
    cnt = 0.
    model.train()
    iterator = itertools.cycle(iter(loader))
    for i in range(FLAGS.n_iter):
        (cmd, img) = next(iterator)
        img = img.to(device)
        cmd = cmd.to(device)
        optimizer.zero_grad()
        loss = model(img, cmd)
        loss.backward()
        optimizer.step()

        train_loss += loss * img.shape[0]
        cnt += img.shape[0]

        if (i+1) % 100 == 0:
            with torch.no_grad():
                print('%d iterations' % (i+1))
                print(len(test_loader))
                val_recon, val_loss = evaluate_cvae(model, test_loader)
                print('val_recon_error: %.6f' % val_recon)
                print('val_loss: %.6f' % val_loss)
                test_iter = itertools.cycle(iter(test_loader))
                if (i+1) % 500 == 0:
                    T = torchvision.transforms.ToPILImage(mode=train.color)
                    for j in range(5):
                        cmd, img = next(test_iter)
                        recon = model.predict(cmd.to(device))
                        recon = recon.cpu().data * train.std[None,:,None,None] + train.mean[None,:,None,None]
                        img   = img.cpu().data * train.std[None,:,None,None] + train.mean[None,:,None,None]
                        res   = torch.cat((recon,img),0).clip_(0,1)
                        T(make_grid(res)).convert("RGB").save(os.path.join(vis_folder,f"{i}_{j}.png"))

    torch.save(model,os.path.join(vis_folder,f"model.pt"))

def evaluate_vqvae(model,loader):
    val_res_recon_error = 0.0
    val_res_nll = 0.0
    model.eval()
    cnt = 0
    for (cmd, img) in loader:
        img = img.to(device)
        cmd = cmd.to(device)
        cnt += img.shape[0]
        loss, data_recon, recon_error, *_ = model(img, cmd)
        nll = model.nll(img,cmd)
        val_res_recon_error += recon_error.item()
        val_res_nll += nll.item()
    model.train()
    return val_res_recon_error / cnt, val_res_nll / cnt

def evaluate_cvae(model,loader):
    val_recon_error = 0.0
    val_loss = 0.0
    model.eval()
    cnt = 0
    for (cmd, img) in loader:
        img = img.to(device)
        cmd = cmd.to(device)
        cnt += img.shape[0]
        loss  = model(img, cmd)
        x_tilde = model.predict(cmd)
        val_recon_error += (x_tilde-img).pow(2).sum().item()
        val_loss += loss.item()
    model.train()
    return val_recon_error / cnt, val_loss / cnt

def main(argv):
    if FLAGS.modeltype.startswith("C"):
        train_cvae()
    else:
        train_vae()

if __name__ == "__main__":
    app.run(main)
    #train_vae(FLAGS.datatype="setpp",FLAGS.modeltype="DAE")
    #train_vae(FLAGS.datatype="setpp",FLAGS.modeltype="VQVAE")
    #train_vae(FLAGS.datatype="SCAN",FLAGS.modeltype="VQVAE")
    #train_lexgen(FLAGS.datatype="shapes")
    #train_vae(FLAGS.datatype="shapes",FLAGS.modeltype="VQVAE")
    #train_lexgen(FLAGS.datatype="setpp")

    #train_vae(FLAGS.datatype="shapes",FLAGS.modeltype="VAE")
