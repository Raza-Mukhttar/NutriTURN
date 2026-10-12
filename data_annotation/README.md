# data_annotation — stage 2 of 3

Labels every (claim, record) association with the direction of the reported association:
`PROTECTIVE` / `HARMFUL` / `NULL` / `UNCLEAR`. **Needs GPUs** and the abstracts from stage 1.

```bash
bash data_annotation/run_data_annotation.sh             # every cell, one GPU
NSHARD=5 bash data_annotation/run_data_annotation.sh    # fan each cell across 5 GPUs
```


