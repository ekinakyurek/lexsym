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
from src.vqvae import VectorQuantizedVAE, CVQVAE
from src.vae import VAE, CVAE
from src.dae import DAE
# from src.utils import make_number_grid

from data.shapes import ShapeDataset
from data.set import SetDataset
from data.scan import SCANDataset
from data.clevr import CLEVRDataset
import torchvision
from torchvision.utils import make_grid, save_image
import itertools
import functools
from absl import app, flags, logging

FLAGS = flags.FLAGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"DEVICE: {device}")

flags.DEFINE_integer("h_dim", 32, "")
flags.DEFINE_integer("rnn_dim", 256, "")
flags.DEFINE_integer("rnn_n_layers", 2, "")
flags.DEFINE_float("rnn_drop", 0.1, "")
flags.DEFINE_integer("n_latent", 24, "")
flags.DEFINE_integer("n_batch", 128, "")
flags.DEFINE_integer("n_iter", 100000, "")
flags.DEFINE_integer("n_epoch", 50, "")
flags.DEFINE_integer("n_codes", 10, "")
flags.DEFINE_integer("seed", 0, "")
flags.DEFINE_float("beta", 1.0, "")
flags.DEFINE_float("commitment_cost", 0.25, "")
flags.DEFINE_string("datatype", "setpp", "")
flags.DEFINE_string("modeltype", "VQVAE", "")
flags.DEFINE_float("decay", 0.99, "")
flags.DEFINE_float("lr", 1e-3, "")
flags.DEFINE_float("epsilon", 1e-5, "")
flags.DEFINE_bool("debug", False, "")
flags.DEFINE_bool("highdrop", False, "")
flags.DEFINE_bool("highdroptest", False, "")
flags.DEFINE_float("highdropvalue", 0., "")
flags.DEFINE_bool("copy", False, "")
flags.DEFINE_string("vae_path", "", "pretrained vae path")
flags.DEFINE_string("lex_path", "", "prelearned lexicon")
flags.DEFINE_string("model_path", "", "prelearned model")
flags.DEFINE_bool("extract_codes", False, "")
flags.DEFINE_bool("filter_model", False, "")
flags.DEFINE_bool("test", False, "")


def train_lexgen():
    if FLAGS.datatype == "setpp":
        train = SetDataset("data/setpp/", split="train")
        test = SetDataset("data/setpp/",
                          split="test",
                          transform=train.transform,
                          vocab=train.vocab)
    else:
        train = ShapeDataset(root="data/shapes/", split="train")
        test = ShapeDataset(root="data/shapes/",
                            split="test",
                            transform=train.transform,
                            vocab=train.vocab)

    vis_folder = os.path.join("vis", FLAGS.datatype, "FilterModel")
    os.makedirs(vis_folder, exist_ok=True)
    logging.info("vis folder: %s", vis_folder)

    loader = DataLoader(train,
                        batch_size=FLAGS.n_batch,
                        shuffle=True,
                        collate_fn=train.collate)

    test_loader = DataLoader(test, batch_size=32, collate_fn=train.collate)

    model = FilterModel(
        vocab=train.vocab,
        n_downsample=2,
        n_latent=FLAGS.n_latent,
        n_steps=10
    ).cuda()

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

    for i in range(FLAGS.n_epoch):
        for j, (cmd, img, _) in enumerate(loader):
            loss, *_ = model(cmd.to(device), img.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logging.info("Loss %.4f", loss.item())
        if i % 5 == 0:
            visualizations = []
            itloader = iter(loader)
            for j in range(3):
                cmd, img, _ = next(itloader)
                loss, _, results, attentions, text_attentions = model(
                                                                cmd.to(device),
                                                                img.to(device))
                visualizations.append(visualize(
                    train,
                    f"train-{i}-{j}",
                    train.decode(cmd[:, 0].detach().cpu().numpy()),
                    results,
                    attentions,
                    text_attentions,
                    vis_folder,
                ))

            # logging.info("\n-----------------\n")
            itloader = iter(test_loader)
            for j in range(3):
                cmd, img, _ = next(itloader)
                loss, _, results, attentions, text_attentions = model(
                                                                cmd.to(device),
                                                                img.to(device))
                visualizations.append(visualize(
                    train,
                    f"test-{i}-{j}",
                    test.decode(cmd[:, 0].detach().cpu().numpy()),
                    results,
                    attentions,
                    text_attentions,
                    vis_folder,
                ))

            render_html(visualizations, vis_folder)


def prep(img, *, mean=[0.], std=[0.], transform=None):
    img = img[0, ...].cpu().detach()
    if img.shape[0] == 3:
        img = img * std[:, None, None] + mean[:, None, None]
        img = torch.clip(img, 0, 1)
    if transform is not None:
        img = transform(img)
    else:
        img = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        if img.shape[2] == 1:
            img = img[:, :, 0]
    return img


def visualize(dataset,
              name,
              command,
              results,
              attentions,
              text_attentions,
              vis_folder):

    # transform = torchvision.transforms.ToPILImage(mode=dataset.color)

    prep1 = functools.partial(prep,
                              mean=dataset.mean,
                              std=dataset.std)

    imageio.imwrite(os.path.join(vis_folder, name + ".result-{0}.png"),
                    prep1(results[0]))

    example = {
        "name": name,
        "command": command,
        "init-result": name + ".result-0.png",
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
        imageio.imwrite(os.path.join(vis_folder, name + f".att-{i}.png"),
                        prep(attentions[i-1]))
        imageio.imwrite(os.path.join(vis_folder, name + f".result-{i}.png"),
                        prep1(results[i]))
    return example

def render_html(visualizations, vis_folder):
    with open(os.path.join(vis_folder, 'index.html'), "w") as writer:
        writer = functools.partial(print, file=writer)
        for vis in visualizations:
            writer("<p>", vis["name"], vis["command"], "</p>")
            writer("<table>")
            init = vis["init-result"]
            writer(f"<tr><td></td><td><img src='{init}' height=30></td></tr>")
            for att, result, t_att in zip(vis["attentions"], vis["results"], vis["text_attentions"]):
                writer(f"<tr><td><img src='{att}' height=30></td><td><img src='{result}' height=30></td><td>{t_att}</td></tr>")
            writer("</table>")
            writer("<br>")


def flags_to_path():
    root = os.path.join("vis", FLAGS.datatype, FLAGS.modeltype)
    if "VQVAE" in FLAGS.modeltype:
        return os.path.join(root,
                            (f"beta_{FLAGS.beta}_ncodes_{FLAGS.n_codes}_"
                             f"ldim_{FLAGS.n_latent}_dim_{FLAGS.h_dim}_"
                             f"lr_{FLAGS.lr}")
                            )
    else:
        return os.path.join(root,
                            (f"beta_{FLAGS.beta}_ldim_{FLAGS.n_latent}_"
                             f"dim_{FLAGS.h_dim}"
                             f"_lr_{FLAGS.lr}")
                            )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_vae():
    set_seed(FLAGS.seed)

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

    vis_folder = flags_to_path()
    os.makedirs(vis_folder, exist_ok=True)
    logging.info("vis folder: %s", vis_folder)

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
    elif FLAGS.modeltype == "VAE":
        model = VAE(3,
                    FLAGS.h_dim,
                    FLAGS.n_latent,
                    beta=FLAGS.beta,
                    size=train.size).to(device)
    elif FLAGS.modeltype == "DAE":
        model = DAE(3,
                    latentdim=FLAGS.n_latent*4*4,
                    size=train.size).to(device)
    else:
        raise ValueError(f"Unknown model type {FLAGS.modeltype}")

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

    train_res_recon_error = 0.
    cnt = 0.

    loader = DataLoader(train,
                        batch_size=FLAGS.n_batch,
                        shuffle=True,
                        collate_fn=train.collate,
                        num_workers=4)

    test_loader = DataLoader(test,
                             batch_size=32,
                             shuffle=True,
                             collate_fn=train.collate,
                             num_workers=4)

    generator = iter(loader)

    model.train()

    for i in range(FLAGS.n_iter):
        try:
            cmd, img, _ = next(generator)
        except StopIteration:
            generator = iter(loader)
            cmd, img, _ = next(generator)

        img = img.to(device)
        cmd = cmd.to(device)
        optimizer.zero_grad()
        loss, _, recon_error, *_ = model(img, cmd)
        loss.backward()
        optimizer.step()

        train_res_recon_error += recon_error.item()
        cnt += img.shape[0]
        # train_res_nll.append(nll.item())

        if (i+1) % 1000 == 0:
            with torch.no_grad():
                logging.info('%d iterations', (i+1))
                logging.info('recon_error: %.6f', (train_res_recon_error / cnt))
                # #print('nll: %.6f' % np.mean(train_res_nll[-100:]))
                # print(len(test_loader))
                # val_recon, val_perp = evaluate_vqvae(model, test_loader)
                # print('val_recon_error: %.6f' % val_recon)
                # print('val_nll: %.6f' % val_perp)
                if (i+1) % 1000 == 0:
                    T = torchvision.transforms.ToPILImage(mode=train.color)
                    test_iter = iter(test_loader)
                    for j in range(5):
                        cmd, img, _ = next(test_iter)
                        img = img.to(device)
                        cmd = cmd.to(device)
                        loss, recon, recon_error, *_ = model(img, cmd)
                        recon = recon.cpu().data * train.std[None, :, None, None] + train.mean[None, abs:, None, None]
                        img = img.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
                        res = torch.cat((recon, img), 0).clip_(0, 1)
                        T(make_grid(res)).convert("RGB").save(os.path.join(vis_folder, f"{i}_{j}.png"))
                        sample, _ = model.sample(B=32)
                        sample = sample.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
                        T(make_grid(sample.clip_(0, 1))).convert("RGB").save(os.path.join(vis_folder, f"prior_{i}_{j}.png"))

    torch.save(model, os.path.join(vis_folder, "model.pt"))


def train_cvae():
    assert FLAGS.vae_path != ""
    set_seed(FLAGS.seed)
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

    vis_folder = flags_to_path()
    os.makedirs(vis_folder, exist_ok=True)
    logging.info("vis folder: %s", vis_folder)
    if FLAGS.test:
        model = torch.load(FLAGS.model_path)
        model = model.to(device)

        loader = DataLoader(train,
                            batch_size=FLAGS.n_batch,
                            shuffle=True,
                            collate_fn=train.collate,
                            num_workers=4)

        test_loader = DataLoader(test,
                                 batch_size=36,
                                 shuffle=True,
                                 collate_fn=train.collate,
                                 num_workers=4)
    else:
        if FLAGS.modeltype == "CVQVAE":
            model = CVQVAE(3,
                           FLAGS.h_dim,
                           FLAGS.n_latent,
                           train.vocab,
                           rnn_dim=FLAGS.rnn_dim,
                           n_codes=FLAGS.n_codes,
                           cc=FLAGS.commitment_cost,
                           decay=FLAGS.decay,
                           epsilon=FLAGS.epsilon,
                           beta=FLAGS.beta,
                           cmdproc=False,
                           size=train.size,
                           ).to(device)
            model.vqvae.load_state_dict(torch.load(FLAGS.vae_path).state_dict())
            for p in model.vqvae.parameters():
                p.requires_grad = False
            if FLAGS.lex_path != "":
                model.set_src_copy(model.get_src_lexicon(FLAGS.lex_path))

            model = model.to(device)

            loader = DataLoader(train,
                                batch_size=FLAGS.n_batch,
                                shuffle=True, collate_fn=train.collate,
                                num_workers=4)
            test_loader = DataLoader(test,
                                     batch_size=32,
                                     shuffle=True,
                                     collate_fn=train.collate,
                                     num_workers=4)

            model.eval()
            T = torchvision.transforms.ToPILImage(mode=train.color)
            with torch.no_grad():
                sample, _ = model.vqvae.sample(B=32)
                sample = sample.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
                T(make_grid(sample.clip_(0, 1))).convert("RGB").save(os.path.join(vis_folder, "initial_samples.png"))
                generator = iter(test_loader)
                cmd, img, _ = next(generator)
                cmd = cmd.to(device)
                img = img.to(device)
                _, recon, *_ = model.vqvae(img, cmd)
                recon = recon.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
                img = img.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
                res = torch.cat((recon, img), 0).clip_(0, 1)
                T(make_grid(res)).convert("RGB").save(os.path.join(vis_folder, f"initial_recons.png"))

        elif FLAGS.modeltype == "CVAE":
            model = CVAE(3,
                         FLAGS.h_dim,
                         FLAGS.n_latent,
                         train.vocab,
                         beta=FLAGS.beta,
                         rnn_dim=FLAGS.rnn_dim,
                         size=train.size)

            model.vae.load_state_dict(torch.load(FLAGS.vae_path).state_dict())
            model = model.to(device)
            for p in model.vae.parameters():
                p.requires_grad = False
        else:
            raise ValueError(f"Unknown model type {FLAGS.modeltype}")

        optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
        train_loss = 0.
        cnt = 0.

        generator = iter(loader)
        model.train()
        model.vqvae.eval()
        for i in tqdm(range(FLAGS.n_iter)):
            try:
                cmd, img, _ = next(generator)
            except StopIteration:
                generator = iter(loader)
                cmd, img, _ = next(generator)

            cmd = cmd.to(device)
            img = img.to(device)

            optimizer.zero_grad()
            loss = model(img, cmd)
            loss.backward()
            optimizer.step()

            train_loss += loss * img.shape[0]
            cnt += img.shape[0]

            if i == 10 or (i+1) % 100 == 0:
                model.eval()
                with torch.no_grad():
                    logging.info('%d iterations', (i+1))
                    logging.info('%.6f train loss', (train_loss / cnt))
                    val_recon, val_loss = evaluate_cvae(model, test_loader)
                    logging.info('val_recon_error: %.6f', val_recon)
                    logging.info('val_loss: %.6f', val_loss)
                    T = torchvision.transforms.ToPILImage(mode=train.color)
                    test_iter = iter(test_loader)
                    for j in range(5):
                        cmd, img, _ = next(test_iter)
                        cmd = cmd.to(device)
                        img = img.to(device)
                        recon, pred_encodings, copy_probs = model.predict(cmd, sample=True, top_k=10)
                        pred_encodings = model.make_number_grid(pred_encodings.cpu())
                        copy_heatmap = model.make_copy_grid(copy_probs.cpu())
                        _, encodings, *_ = model.vqvae.encode(img)
                        encodings = model.make_number_grid(encodings.cpu())
                        encodings = torch.cat((copy_heatmap, pred_encodings, encodings), 0)
                        recon = recon.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
                        img = img.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
                        res = torch.cat((recon, img), 0).clip_(0, 1)
                        T(make_grid(encodings, nrow=encodings.shape[0]//3)).convert("RGB").save(os.path.join(vis_folder, f"{i}_{j}_encodings.png"))
                        T(make_grid(res, nrow=res.shape[0]//2)).convert("RGB").save(os.path.join(vis_folder, f"{i}_{j}.png"))
                        logging.info("saved")
                model.train()
                model.vqvae.eval()
        torch.save(model, os.path.join(vis_folder, f"model.pt"))

    model.eval()
    with torch.no_grad():
        val_recon, val_loss = evaluate_cvae(model, test_loader)
        logging.info('val_recon_error: %.6f', val_recon)
        logging.info('val_loss: %.6f', val_loss)
        T = torchvision.transforms.ToPILImage(mode=train.color)
        test_iter = iter(test_loader)
        for j in range(5):
            cmd, img, _ = next(test_iter)
            cmd = cmd.to(device)
            img = img.to(device)
            logging.info("sampling")
            recon, *_ = model.predict(cmd, top_k=10, sample=True)
            logging.info("sampled")
            recon = recon.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
            img = img.cpu().data * train.std[None, :, None, None] + train.mean[None, :, None, None]
            res = torch.cat((recon, img), 0).clip_(0, 1)
            logging.info("saving samples")
            T(make_grid(res, nrow=res.shape[0]//2)).convert("RGB").save(os.path.join(vis_folder, f"eval_{j}.png"))
            logging.info("saved")


def evaluate_vqvae(model, loader):
    val_res_recon_error = 0.0
    val_res_nll = 0.0
    cnt = 0
    for (cmd, img, _) in loader:
        img = img.to(device)
        cmd = cmd.to(device)
        cnt += img.shape[0]
        loss, data_recon, recon_error, *_ = model(img, cmd)
        nll = model.nll(img, cmd)
        val_res_recon_error += recon_error.item()
        val_res_nll += nll.item()
    model.train()
    return val_res_recon_error / cnt, val_res_nll / cnt


def evaluate_cvae(model,loader):
    val_recon_error = 0.0
    val_loss = 0.0
    cnt = 0
    for (cmd, img, _) in loader:
        img = img.to(device)
        cmd = cmd.to(device)
        cnt += img.shape[0]
        loss = model(img, cmd)
        x_tilde, *_ = model.predict(cmd)
        val_recon_error += (x_tilde-img).pow(2).sum().item()
        val_loss += loss.item() * img.shape[0]
        if cnt > 100:
            break
    return val_recon_error / cnt, val_loss / cnt


def img2code():
    set_seed(FLAGS.seed)

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

    vis_folder = flags_to_path()
    os.makedirs(vis_folder, exist_ok=True)
    logging.info("vis folder: %s", vis_folder)

    if FLAGS.modeltype == "VQVAE":
        model = VectorQuantizedVAE(3,
                                   FLAGS.h_dim,
                                   FLAGS.n_latent,
                                   n_codes=FLAGS.n_codes,
                                   cc=FLAGS.commitment_cost,
                                   decay=FLAGS.decay,
                                   epsilon=FLAGS.epsilon,
                                   beta=FLAGS.beta,
                                   cmdproc=False,
                                   size=train.size).to(device)
        model.load_state_dict(torch.load(FLAGS.vae_path).state_dict())
        model.eval()
    else:
        raise ValueError(f"model type not available for this {FLAGS.modeltype}")

    train_loader = DataLoader(train,
                              batch_size=FLAGS.n_batch,
                              shuffle=False,
                              collate_fn=train.collate,
                              num_workers=4)
    test_loader = DataLoader(test,
                             batch_size=FLAGS.n_batch,
                             shuffle=False,
                             collate_fn=train.collate,
                             num_workers=4)
    for (split, loader) in zip(("train", "test"), (train_loader, test_loader)):
        generator = iter(loader)
        with open(os.path.join(vis_folder, f"{split}_encodings.txt"), "w") as f:
            for i in range(len(generator)):
                try:
                    cmd, img, names = next(generator)
                except StopIteration:
                    generator = iter(loader)
                    cmd, img, names = next(generator)

                img = img.to(device)
                cmd = cmd.to(device)
                _, _, recon_error, _, _, encodings = model(img, cmd)
                for k in range(img.shape[0]):
                    encc = encodings[k].flatten().detach().cpu().numpy()
                    encc = [str(e) for e in encc]
                    cmdc = train.vocab.decode_plus(cmd[:,k].detach().cpu().numpy())
                    line = " ".join(cmdc) + "\t" + " ".join(encc) + "\t" + names[k] + "\n"
                    f.write(line)


def main(argv):
    if FLAGS.filter_model:
        return train_lexgen()
    if FLAGS.extract_codes:
        return img2code()
    else:
        if FLAGS.modeltype.startswith("C"):
            return train_cvae()
        else:
            return train_vae()

if __name__ == "__main__":
    app.run(main)
