import torch
from torch import nn
import json
import os

class VQVAE(nn.Module):
    def __init__(self,
                    enc_config,
                    dec_config,
                    quantize_config):
        super().__init__()
        
        self.enc = new_module(enc_config)
        self.dec = new_module(dec_config)
        self.quantize = new_module(quantize_config)

    def forward(self, input):
        quant_t, diff_t, _ = self.encode(input)

        return self.decode(quant_t), diff_t
    
    def encode(self, input):
        logits = self.enc(input)
        quant_t, diff_t, id_t = self.quantize.forward(logits)
        quant_t = quant_t.permute(0, 3, 1, 2)
        # diff_t = diff_t.unsqueeze(0)

        return quant_t, diff_t, id_t

    def decode(self, code):
        return [self.dec(code)]
    
    def decode_code(self, code_t):
        quant_t = self.quantize.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        return self.decode(quant_t)
    
    def get_last_layer(self):
        return self.dec.get_last_layer()
    

class HVQVAE(nn.Module):
    def __init__(
        self,
        levels,
        embedding_dim,
        enc_config,
        quantize_config,
        down_sampler_configs,
        dec_configs,
        codebook_scale=1.
    ):
        super().__init__()
        
        self.levels = levels

        self.enc = new_module(enc_config)
            
        self.decs = nn.ModuleList()
        for i in range(levels):
            self.decs.append(new_module(dec_configs[i]))
            
        self.quantize = new_module(quantize_config)
        self.down_samplers = nn.ModuleList()
        for i in range(levels-1):
            self.down_samplers.append(new_module(down_sampler_configs[i]))
        self.codebook_scale = codebook_scale
            
    def forward(self, input):        
        quants, diffs, ids = self.encode(input)
        dec_outputs = self.decode(quants[::-1])
        
        total_diff = diffs[0]
        scale = 1.
        for diff in diffs[1:]:
            scale *= self.codebook_scale
            total_diff = total_diff + diff * scale
        return dec_outputs, total_diff

    def encode(self, input):
        enc_output = self.enc(input)
        enc_outputs = [enc_output]
        for l in range(self.levels-1):
            enc_outputs.append(self.down_samplers[l](enc_outputs[-1]))
       
        quants, diffs, ids = [], [], []
        for enc_output in enc_outputs:
            quant, diff, id = self.quantize(enc_output)
            quants.append(quant.permute(0, 3, 1, 2))
            diffs.append(diff)
            ids.append(id)
            
        return quants, diffs, ids
        
    def decode(self, quants):
        dec_outputs = []
        for l in range(self.levels-1, -1, -1):
            dec_outputs.append(self.decs[l](quants[l]))
            
        return dec_outputs

    def decode_code(self, codes):
        quants = []
        for l in range(self.levels):
            quants.append(self.quantize.embed_code(codes[l]).permute(0, 3, 1, 2))
        dec_outputs = self.decode(quants)

        return dec_outputs
    
    def get_last_layer(self):
        return self.decs[-1].get_last_layer()