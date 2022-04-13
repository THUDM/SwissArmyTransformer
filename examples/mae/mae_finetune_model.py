import torch

class MAE_finetune(torch.nn.Module):
    def __init__(self, encoder, hidden_size, num_classes):
        super().__init__()
        self.final = torch.nn.Linear(hidden_size, num_classes)
        self.encoder = encoder
    def forward(self, tokens, position_ids, attention_mask, **kwargs):
        x, *mem = self.encoder(tokens, position_ids, attention_mask, **kwargs)
        x = self.final(x[:, 0])
        return x
    def disable_untrainable_params(self):
        pass