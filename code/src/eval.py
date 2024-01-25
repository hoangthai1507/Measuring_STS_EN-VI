class SimilarityFunction(Enum):
    COSINE = 0
    EUCLIDEAN = 1
    MANHATTAN = 2
    DOT_PRODUCT = 3


class STSBenchmarkReader:
    """
    STS Benchmark reader to prep the data for evaluation.
    """

    def __init__(self, data_path: str = None):
        assert data_path != None and os.path.isfile(data_path)
        self.data_path = data_path
        data_dict = dict(sent1=[], sent2=[], scores=[])

        with open(data_path) as fopen:
            dataset = list(filter(None, fopen.read().split('\n')))

        sent1 = []
        sent2 = []
        scores = []

        
        for data in dataset:
            data_list = data.split('|')
            sent1.append(data_list[0])
            sent2.append(data_list[1])
            scores.append(data_list[2])
            
        data_dict['sent1'] = sent1
        data_dict['sent2'] = sent2
        data_dict['scores'] = scores
        # sanity check
        assert len(data_dict['sent1']) == len(data_dict['sent2'])
        assert len(data_dict['sent1']) == len(data_dict['scores'])

        self.data = data_dict
        
class EmbeddingSimilarityEval_STSB:
    def __init__(self, embeddings1,embeddings2, score, batch_size: int = 16, 
                 main_similarity: SimilarityFunction = SimilarityFunction.COSINE, 
                 name: str = '', show_progress_bar: bool = False, write_csv: bool = True):

        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2
        self.scores = [float(i) for i in score]
        self.write_csv = write_csv
        self.main_similarity = main_similarity
        self.name = name
        self.batch_size = batch_size

        self.csv_file = "similarity_evaluation" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["model", "stsb_dataset_name", "cosine_pearson", "cosine_spearman", "euclidean_pearson", "euclidean_spearman", "manhattan_pearson", "manhattan_spearman", "dot_pearson", "dot_spearman"]
    
    def encode_embeddings(self, save_path=None):
        embeddings1_cpu = [emb.detach().cpu().numpy() for emb in self.embeddings1]
        embeddings2_cpu = [emb.detach().cpu().numpy() for emb in self.embeddings2]
        
        embeddings1 = np.array(embeddings1_cpu)
        embeddings2 = np.array(embeddings2_cpu) 
          
        print(len(embeddings1[0]))
        print(len(embeddings2[0]))

        if save_path:
            with open(save_path, 'wb') as file:
                pickle.dump((embeddings1, embeddings2), file)
        
        return embeddings1, embeddings2
    
    def run_eval(self, output_path: str = None):
        embeddings1, embeddings2 = self.encode_embeddings()
        labels = self.scores
        eval_cosine = dict()
        eval_manhattan = dict()
        eval_euclidean = dict()
        eval_dot = dict()
        
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]
        
        eval_cosine['pearson'], _ = pearsonr(labels, cosine_scores)
        eval_cosine['spearman'], _ = spearmanr(labels, cosine_scores)
        
        eval_manhattan['pearson'], _ = pearsonr(labels, manhattan_distances)
        eval_manhattan['spearman'], _ = spearmanr(labels, manhattan_distances)

        eval_euclidean['pearson'], _ = pearsonr(labels, euclidean_distances)
        eval_euclidean['spearman'], _ = spearmanr(labels, euclidean_distances)

        eval_dot['pearson'], _ = pearsonr(labels, dot_products)
        eval_dot['spearman'], _ = spearmanr(labels, dot_products)

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)
                    
                writer.writerow([eval_cosine['pearson'], eval_cosine['spearman'], eval_euclidean['pearson'],
                                 eval_euclidean['spearman'], eval_manhattan['pearson'], eval_manhattan['spearman'], eval_dot['pearson'], eval_dot['spearman']])


        if self.main_similarity == SimilarityFunction.COSINE:
            print("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
                eval_cosine['pearson'], eval_cosine['spearman']))
            return eval_cosine
        elif self.main_similarity == SimilarityFunction.EUCLIDEAN:
            return eval_euclidean
        elif self.main_similarity == SimilarityFunction.MANHATTAN:
            return eval_manhattan
        elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
            return eval_dot
        elif self.main_similarity is None:
            return max(eval_cosine, eval_manhattan, eval_euclidean, eval_dot)
        else:
            raise ValueError("Unknown main_similarity value")
    


def embedding(tokenizer, base_model, model, sentences, batch_size, device):
    base_model.to(device)
    model.to(device)

    all_embeddings = []

    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

    for i in range(-(-len(sentences) // batch_size)):
        sentence_batch = sentences_sorted[i * batch_size : (i + 1) * batch_size]

        encoded = tokenizer(sentence_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = base_model(**encoded.to(device))

        # last hidden state
        embeddings = outputs[0][:, 0, :]

        embeddings = model(embeddings)[0]

        all_embeddings.extend(embeddings)

    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    all_embeddings = torch.stack(all_embeddings)
    
    return all_embeddings


if __name__ == '__main__':
    eval_data_path = '/kaggle/input/sts-vi-en/STS benchmark/test_1000c.csv'
    data_dict  = STSBenchmarkReader(eval_data_path)
    
    src_sentences = data_dict.data['sent1']
    trg_sentences = data_dict.data['sent2']
    score = data_dict.data['scores']
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModel.from_pretrained(base_model_name)
    base_model.to(device)
    
    model = MLP()
    model.load_state_dict(torch.load("/kaggle/input/baseline-1-10/train_model"))
    model.to(device)
    batch_size = 512
    source_embedding = embedding(tokenizer, base_model, model, src_sentences, batch_size, device)
    target_embedding = embedding(tokenizer, base_model, model, trg_sentences, batch_size, device)
    
    sts_eval = EmbeddingSimilarityEval_STSB(source_embedding,target_embedding,score,main_similarity=SimilarityFunction.COSINE, name="multilang_models", show_progress_bar=True, write_csv=True)
    sts_eval.run_eval(output_path='/kaggle/working/')