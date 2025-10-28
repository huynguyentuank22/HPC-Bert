Run file `finetune_SBERT.py`
    
```bash
    python .\finetune_SBERT.py 
        --input .\data_train\24_03.parquet
```

Run file `finetune_SBERT-Kmeans.py`

```bash
    python .\finetune_SBERT-Kmeans.py
        --input .\data_train\24_03.parquet
        --exclude_cols jid,usr,jnam,jobenv_req,adt,qdt,schedsdt,deldt,sdt,edt,embedding,pclass,exit state,duration,avgpcon,minpcon,maxpcon
        --n_triplets_per_anchor 2
```