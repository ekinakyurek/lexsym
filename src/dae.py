import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import View
import math

class DAE(nn.Module):
    def __init__(self, input_dim, encdims=[32,32,64,64], decdims=[64,64,32,32], latentdim=16*4*4, size=(64,64)):
        super().__init__()
        self.encdims = [input_dim, *encdims]
        self.decdims = [*decdims, input_dim]

        self.down_blocks = []
        for in_layer,out_layer in zip(self.encdims,self.encdims[1:]):
            self.down_blocks.append(nn.Conv2d(in_layer,out_layer,4,stride=2,padding=1))
            self.down_blocks.append(nn.ELU())

        self.encoder=nn.Sequential(*self.down_blocks)

        with torch.no_grad():
            mu = self.encoder(torch.ones(1,3,*size))
            self.pre_shape = mu.shape[1:]

        self. proj = nn.Sequential(nn.Flatten(),
                                   nn.Linear(math.prod(self.pre_shape), latentdim)
                                   )

        self.latent_shape = (latentdim,)
        self.up_blocks = []

        for in_layer,out_layer in zip(self.decdims,self.decdims[1:]):
            self.up_blocks.append(nn.ConvTranspose2d(in_layer,out_layer,4,stride=2,padding=1))
            self.up_blocks.append(nn.ELU())

        self.up_blocks.pop()

        self.decoder=nn.Sequential(
                        nn.Linear(latentdim,self.pre_shape[1] * self.pre_shape[2] * self.decdims[0]),
                        View(-1, self.decdims[0], self.pre_shape[1], self.pre_shape[2]),
                        *self.up_blocks,
                        )

    def forward(self,blocked_image):
        encoded = self.encoder(blocked_image)
        z = self.proj(encoded)
        decoded=self.decoder(z)
        loss = F.mse_loss(decoded, blocked_image)
        return loss, decoded, loss, None
