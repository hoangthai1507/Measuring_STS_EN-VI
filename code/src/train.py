from datetime import datetime



def train_model(
    model,
    dataset_train,
    dataset_valid,
    optimizer,
    batch_size,
    save_model_path,
    #tmp_path
):

    model.to(device)
    
    sampler_train = SubsetRandomSampler(range(len(dataset_train)))
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    
    random_indices = torch.randperm(len(dataset_train))
    sampler_random = SubsetRandomSampler(random_indices)
    random_dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler_random)
    

    
    
    check_all_batches_different(dataloader_train,random_dataloader_train)
    
    sampler_valid = SubsetRandomSampler(range(len(dataset_valid)))
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    
    random_indices_valid = torch.randperm(len(dataset_valid))
    sampler_random_valid = SubsetRandomSampler(random_indices_valid)
    random_dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler_random_valid)
        
    check_all_batches_different(dataloader_valid,random_dataloader_valid)

    
    
    train_num = len(dataloader_train)
    dev_num = len(dataloader_valid)
    
    min_valid_loss = float("inf")
    
    for epoch in range(1000):
        # train
        s_time = time.time()
        train_loss = 0
        for batch,ran_batch in zip(dataloader_train,random_dataloader_train):
        #for batch in dataloader_train:
            src_emb = batch["src_emb"].to(device)
            trg_emb = batch["trg_emb"].to(device)
            src_lang_batch = batch["src_lang"].to(device)
            trg_lang_batch = batch["trg_lang"].to(device)
            
            
            ran_src_emb = ran_batch["src_emb"].to(device)
            ran_trg_emb = ran_batch["trg_emb"].to(device)
            #ran_src_lang_batch = ran_batch["src_lang"].to(device)
            #ran_trg_lang_batch = ran_batch["trg_lang"].to(device)
            

            optimizer.zero_grad()
            loss = cal_loss(src_emb,trg_emb, 
                            ran_src_emb, ran_trg_emb,
                            src_lang_batch, trg_lang_batch)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # eval  
        #print("Validating")
        with torch.no_grad():
            valid_loss = 0
            for batch, rand_batch in zip(dataloader_valid,random_dataloader_valid):
            #for batch in dataloader_valid:
                src_emb = batch["src_emb"].to(device)
                trg_emb = batch["trg_emb"].to(device)
                src_lang_batch = batch["src_lang"].to(device)
                trg_lang_batch = batch["trg_lang"].to(device)
                
                ran_src_emb = rand_batch["src_emb"].to(device)
                ran_trg_emb = rand_batch["trg_emb"].to(device)
                
                loss = cal_loss(
                    src_emb,trg_emb, 
                    ran_src_emb, ran_trg_emb, 
                    src_lang_batch, trg_lang_batch
                )
                valid_loss += loss.item()
            
            print(
                f"epoch:{epoch + 1: <2}, "
                f"train_loss: {train_loss / train_num:.5f}, "
                f"valid_loss: {valid_loss / dev_num:.5f}, "
                f"{(time.time() - s_time) / 60:.1f}[min]"
            )
            #torch.save(model.state_dict(), tmp_path+'model_by_epoch'+str(epoch+1)+'.pt')
            if valid_loss < min_valid_loss:
                epochs_no_improve = 0
                min_valid_loss = valid_loss
                torch.save(model.state_dict(), save_model_path)
                model.to(device)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= 10:
                print("Model no improve")
                break
        



if __name__ == '__main__':
    formatted_time = datetime.now().strftime("%d_%m_%Y")
    PATH = f'../../checkpoint/{data_train}_{formatted_time}/'
    #tmp_path= '/kaggle/working/model/'
    
    data_train = torch.load("/kaggle/input/data-en-vi-embedding-1-10/train")
    dataset_train = TextDataset(
        data_train["src_emb"], data_train["trg_emb"], data_train["src_lang"], data_train["trg_lang"]
    )
    
    data_valid = torch.load("/kaggle/input/data-en-vi-embedding-1-10/valid")
    dataset_valid = TextDataset(
            data_valid["src_emb"], data_valid["trg_emb"], data_valid["src_lang"], data_valid["trg_lang"]
    )
    
    maxlen=2000
    random.seed(9001)
    seed = 9001
    torch.manual_seed(seed)
    
    model = MLP()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    cos_fn = nn.CosineEmbeddingLoss()
    cos_fn_m = nn.CosineEmbeddingLoss()
    cross_fn = nn.CrossEntropyLoss()
    epochs_no_improve = 0
    batch_size = 512
    train_model(
        model,
        dataset_train,
        dataset_valid,
        optimizer,
        batch_size,
        PATH,
        #tmp_path
    )