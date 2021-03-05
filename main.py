import os
import torch
import imageio
import json
import numpy as np
from torch import optim
from torch.utils import data as torch_data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.lex import FilterModel
from src.vqvae import VectorQuantizedVAE
from src.vae import VAE
from src.dae import DAE

from data.shapes import ShapeDataset
from data.set import SetDataset
from data.scan import SCANDataset

import torchvision
from torchvision.utils import make_grid, save_image
import itertools


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE: ", device)

def train_lexgen(datatype="Shapes"):
    if datatype == "Set++":
        train  = SetDataset("data/setpp/", split="train")
        test   = SetDataset("data/setpp/", split="test", transform = train.transform, vocab = train.vocab)
    else:
        train  = ShapeDataset(root="data/shapes/", split="train")
        test   = ShapeDataset(root="data/shapes/", split="test", transform = train.transform, vocab = train.vocab)

    loader = DataLoader(train, batch_size=64, collate_fn=train.collate)
    test_loader = DataLoader(test, batch_size=32, collate_fn=train.collate)

    model = FilterModel(
        vocab=train.vocab,
        n_downsample=2,
        n_latent=32,
        n_steps=10
    ).cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.0003)

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

    imageio.imwrite("vis/" + name + f".result-0.png", prep(results[0]))
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



def train_vae(datatype="Shapes", modeltype="VQVAE"):
    batch_size = 128
    num_training_updates = 15000
    dim = 16
    num_residual_hiddens = 16
    embedding_dim=16
    K = 12
    commitment_cost = 0.25
    decay = 0.99
    learning_rate = 1e-3
    epsilon=1e-5
    print(datatype)

    if datatype == "Set++":
        train  = SetDataset("data/setpp/", split="train")
        test   = SetDataset("data/setpp/", split="test", transform=train.transform, vocab=train.vocab)
    elif datatype == "Shapes":
        train  = ShapeDataset("data/shapes/",split="train")
        test   = ShapeDataset("data/shapes/",split="test", transform=train.transform, vocab=train.vocab)
    else:
        train  = SCANDataset("data/scan/",split="train")
        test   = SCANDataset("data/scan/",split="test", transform=train.transform, vocab=train.vocab)



    loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=train.collate, pin_memory=True)
    test_loader = DataLoader(test, batch_size=32, shuffle=True,collate_fn=train.collate, pin_memory=True)

    if modeltype == "VQVAE":
        model = VectorQuantizedVAE(3, dim, embedding_dim,
                                   K=K,
                                   cc=commitment_cost,
                                   decay=decay,
                                   epsilon=epsilon).to(device)
    elif modeltype == "VAE":
        model = VAE(3, dim, embedding_dim).to(device)
    elif modeltype ==  "DAE":
        model = DAE(3, latentdim=embedding_dim*4*4).to(device)
    else:
        error("Unknown Model Type")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    train_res_recon_error = []
    # train_res_perplexity = []
    validation_res_recon_error = []
    # validation_res_perplexity = []

    model.train()
    iterator = itertools.cycle(iter(loader))
    for i in range(num_training_updates):
        (_, img) = next(iterator)
        img = img.to(device)
        optimizer.zero_grad()
        loss, _, recon_error, *_ = model(img)
        loss.backward()
        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        # train_res_perplexity.append(perplexity.item())

        if (i+1) % 100 == 0:
            with torch.no_grad():
                print('%d iterations' % (i+1))
                print('recon_error: %.6f' % np.mean(train_res_recon_error[-100:]))
                # print('perplexity: %.6f' % np.mean(train_res_perplexity[-100:]))
                print(len(test_loader))
                val_recon = evaluate_vqvae(model, test_loader)
                validation_res_recon_error.append(val_recon)
                # validation_res_perplexity.append(val_perp)
                test_iter = itertools.cycle(iter(test_loader))
                if (i+1) % 500 == 0:
                    T = torchvision.transforms.ToPILImage(mode=train.color)
                    for j in range(5):
                        _, img = next(test_iter)
                        loss, recon, recon_error, *_= model(img.to(device))
                        recon = recon.cpu().data * train.std[None,:,None,None] + train.mean[None,:,None,None]
                        img   = img.cpu().data * train.std[None,:,None,None] + train.mean[None,:,None,None]
                        T(make_grid(torch.clip(recon,0,1))).convert("RGB").save(f"vis/{datatype}_{modeltype}_{i}_{j}.png")
                        T(make_grid(torch.clip(img,0,1))).convert("RGB").save(f"vis/{datatype}_{i}_{j}.png")



def evaluate_vqvae(model,loader):
    val_res_recon_error = []
    # val_res_perplexity = []
    model.eval()
    for (_, img) in loader:
        img = img.to(device)
        loss, data_recon, recon_error, *_= model(img)
        val_res_recon_error.append(recon_error.item())
        # val_res_perplexity.append(perplexity.item())


    print('val recon_error: %.6f' % np.mean(val_res_recon_error))
    # print('val perplexity: %.6f' % np.mean(val_res_perplexity))
    print()
    model.train()
    return np.mean(val_res_recon_error)


if __name__ == "__main__":
    #train_vae(datatype="Set++",modeltype="DAE")
    train_vae(datatype="Set++",modeltype="VQVAE")
    #train_vae(datatype="SCAN",modeltype="VQVAE")
    #train_lexgen(datatype="Shapes")
    #train_vae(datatype="Shapes",modeltype="VQVAE")
    #train_lexgen(datatype="Set++")
    #train_vae(datatype="Set++",modeltype="VAE")
    #train_vae(datatype="Shapes",modeltype="VAE")
