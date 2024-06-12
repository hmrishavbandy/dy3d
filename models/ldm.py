# From HuggingFace

from dataclasses import dataclass

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
import math
from models.transformer import make_self_decoder


@dataclass
class MLPConditionOutput(BaseOutput):
    """
    The output of [`UNet2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.FloatTensor

@dataclass
class MLPOutput(BaseOutput):
    """
    The output of [`UNet2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.FloatTensor

def positionalencoding1d(d_model = 64, length = 16):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def check_params():
    model = MLPConditionModel()
    return sum(torch.numel(param) for param in model.parameters())/1000000

class MLPConditionModel(ModelMixin, ConfigMixin):
    
    @register_to_config
    def __init__(
        self,
        freq_shift = 0,
        flip_sin_to_cos = True,
        time_dim = 224, 
        cond_dim=256,
        size = 128, # -> Increase to scale the model
    ):
        super().__init__()

        
        time_embed_dim = 64


        self.time_proj = Timesteps(time_dim, flip_sin_to_cos, freq_shift)
        
        self.time_embedding = TimestepEmbedding(time_dim, time_embed_dim, post_act_fn = None)
        self.x_embeddings_beg = nn.Linear(512+64, size)
        self.cond_proj = nn.Linear(cond_dim, size)
        
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        self.lin_out = nn.Linear(size+64, size)
        self.lin_out_2 = nn.Linear(size, size)
        self.lin_out_3 = nn.Linear(size, 512)
        
        
        self.norm_1 = nn.BatchNorm1d(16)
        self.norm_2 = nn.BatchNorm1d(16)
        self.norm_3 = nn.BatchNorm1d(16)
        self.norm_4 = nn.BatchNorm1d(16)
        self.norm_5 = nn.BatchNorm1d(16)

        self.pos_embed = positionalencoding1d(d_model = 64, length = 16).cuda()

        self.decoder_mha = make_self_decoder(d_model=size+64)

    
    

    def forward(
        self,
        sample,
        timestep,
        condition_emb,
    ):

        skip_sample = sample
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)
    
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb)
        emb = emb.unsqueeze(1).repeat(1,16,1)

        sample = torch.cat((emb,sample), dim = -1)
        


        sample_enc = self.act(self.x_embeddings_beg(sample))
        sample_enc = self.norm_1(sample_enc)

        
        pos_batch = self.pos_embed.repeat(sample.shape[0],1,1)

        emb_cat1 = torch.cat((pos_batch,sample_enc), dim = -1)
        
        emb_cat2 = self.cond_proj(condition_emb)
        emb_cat2 = self.act(emb_cat2) 
        emb_cat2 = self.norm_2(emb_cat2)

        emb_cat2 = torch.cat((pos_batch, emb_cat2), dim = -1)
        
        emb_out = self.decoder_mha(emb_cat2, emb_cat1, None, None)

        sample = self.norm_3(self.act(self.lin_out(emb_out)))
        
        
        sample = self.lin_out_2(sample) 
        sample = self.act(sample) 
        sample = self.norm_4(sample) 

        sample = self.lin_out_3(sample)
        sample = self.act(sample) 
        sample = self.norm_5(sample) 



        sample = sample+skip_sample

        return MLPConditionOutput(sample=sample)
