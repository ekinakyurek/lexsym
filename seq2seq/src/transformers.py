import copy
from typing import Optional, Any
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import ModuleList, MultiheadAttention,Module
from torch.nn.init import xavier_uniform_
from torch.nn import LayerNorm, Linear, Dropout
import torch.nn as nn

class TransformerDecoderLayerv2(nn.TransformerDecoderLayer):
    def __init__(self, *args, return_attentions=False, **kwargs):
        super(TransformerDecoderLayerv2, self).__init__(*args, **kwargs)
        self.return_attentions = return_attentions

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2, self_att_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, src_att_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if self.return_attentions:
            return tgt, src_att_weights, self_att_weights
        else:
            return tgt

class TransformerDecoderv2(nn.TransformerDecoder):
        def __init__(self, *args, return_attentions=False, **kwargs):
            super(TransformerDecoderv2, self).__init__(*args, **kwargs)
            self.return_attentions = return_attentions
            self.layers[-1].return_attentions = return_attentions

        def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                        memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                        memory_key_padding_mask: Optional[Tensor] = None):
                r"""Pass the inputs (and mask) through the decoder layer in turn.

                Args:
                    tgt: the sequence to the decoder (required).
                    memory: the sequence from the last layer of the encoder (required).
                    tgt_mask: the mask for the tgt sequence (optional).
                    memory_mask: the mask for the memory sequence (optional).
                    tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
                    memory_key_padding_mask: the mask for the memory keys per batch (optional).

                Shape:
                    see the docs in Transformer class.
                """
                output = tgt

                for (i,mod) in enumerate(self.layers):
                    if self.return_attentions and i == len(self.layers)-1:
                        output, src_att_weights, self_att_weights = mod(output, memory, tgt_mask=tgt_mask,
                                     memory_mask=memory_mask,
                                     tgt_key_padding_mask=tgt_key_padding_mask,
                                     memory_key_padding_mask=memory_key_padding_mask)
                    else:
                        output = mod(output, memory, tgt_mask=tgt_mask,
                                     memory_mask=memory_mask,
                                     tgt_key_padding_mask=tgt_key_padding_mask,
                                     memory_key_padding_mask=memory_key_padding_mask)



                if self.norm is not None:
                    output = self.norm(output)

                if self.return_attentions:
                    return output, src_att_weights, self_att_weights
                else:
                    return output
