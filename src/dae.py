
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import View

class DAE(nn.Module):
    def __init__(self, input_dim, encdims=[32,32,64,64], decdims=[64,64,32,32], latentdim=16*4*4):
        super().__init__()
        self.encdims = [input_dim, *encdims]
        self.decdims = [*decdims, input_dim]

        self.down_blocks = []
        for in_layer,out_layer in zip(self.encdims,self.encdims[1:]):
            self.down_blocks.append(nn.Conv2d(in_layer,out_layer,4,stride=2,padding=1))
            self.down_blocks.append(nn.ELU())

        self.encoder=nn.Sequential(*self.down_blocks,
                                    nn.Flatten(),
                                    nn.Linear(4 * 4 * self.encdims[-1], latentdim) #FIXME: Here 4x4 is function of input H-W
                                   )

        self.up_blocks = []

        for in_layer,out_layer in zip(self.decdims,self.decdims[1:]):
            self.up_blocks.append(nn.ConvTranspose2d(in_layer,out_layer,4,stride=2,padding=1))
            self.up_blocks.append(nn.ELU())

        self.up_blocks.pop() # remove last activation

        self.decoder=nn.Sequential(
                        nn.Linear(latentdim,4*4*self.decdims[0]),
                        View(-1, 64, 4, 4),
                        *self.up_blocks,
                        )

    def forward(self,blocked_image):
        encoded=self.encoder(blocked_image)
        decoded=self.decoder(encoded)
        loss = F.mse_loss(decoded, blocked_image)
        return loss, decoded, loss, None
