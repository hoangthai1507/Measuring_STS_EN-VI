


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, src_emb, trg_emb, src_lang, trg_lang):
        self.src_emb = src_emb
        self.trg_emb = trg_emb
        self.src_lang = src_lang
        self.trg_lang = trg_lang

    def __len__(self):
        return len(self.src_emb)

    def __getitem__(self, idx):
        return {
            "src_emb": self.src_emb[idx],
            "trg_emb": self.trg_emb[idx],
            "src_lang": self.src_lang[idx],
            "trg_lang": self.trg_lang[idx],
        }


def embed(sentence):
    inputs=tokenizer(sentence,padding='max_length',truncation=True,return_tensors="pt",max_length=128)
    with torch.no_grad():
        outputs = embed_model(**inputs.to(device))
    sentence_embedding = torch.index_select(outputs[0],1,torch.tensor([0]).to(device)).squeeze()
    return sentence_embedding

