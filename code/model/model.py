import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,emb_size=768):
        super(MLP,self).__init__()
        self.lang_emb_layer = nn.Linear(emb_size, emb_size)
        self.meaning_emb_layer = nn.Linear(emb_size, emb_size)
        self.lang_iden_layer = nn.Linear(emb_size, 2)

    def forward(self, x):
        lang_emb = self.lang_emb_layer(x)
        meaning_emb = self.meaning_emb_layer(x)
        lang_iden = self.lang_iden_layer(lang_emb)
        return lang_emb, meaning_emb, lang_iden