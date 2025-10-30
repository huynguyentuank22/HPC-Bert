Run file `finetune_SBERT.py`
    
```bash
    python .\finetune_SBERT.py `
        --input .\data_train\24_03.parquet
```

Run file `finetune_SBERT-Kmeans.py`

```bash
    python .\finetune_SBERT-Kmeans.py `
        --input .\data_train\24_03.parquet `
        --exclude_cols "jid,cnumr,cnumat,cnumut,nnumr,adt,qdt,schedsdt,deldt,ec,elpl,sdt,edt,nnuma,idle_time_ave,nnumu,perf1,perf2,perf3,perf4,perf5,perf6,mszl,pri,econ,msza,mmszu,uctmut,sctmut,usctmut,freq_req,freq_alloc,flops,mbwidth,opint,embedding," `
        --n_triplets_per_anchor 2 `
        --max_k 7 `
        --epoch 2 `
        --batch_size 32 `
        --lr 3e-5 `
        --warmup_ratio 0.2 `
```