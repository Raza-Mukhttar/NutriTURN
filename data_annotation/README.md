# data_annotation — stage 2 of 3

Labels every (claim, record) association with the direction of the reported association:
`PROTECTIVE` / `HARMFUL` / `NULL` / `UNCLEAR`. **Needs GPUs** and the abstracts from stage 1.

```bash
bash data_annotation/run_data_annotation.sh             # every cell, one GPU
NSHARD=5 bash data_annotation/run_data_annotation.sh    # fan each cell across 5 GPUs
```

## The cells

A cell is one (model, prompt, read-out) combination. Everything else is held identical — 3,000
character text budget, 1,600-token truncation, bfloat16, a single forward pass, no sampling — which
is what makes a difference between two cells attributable to the annotator rather than to the
harness. `config/annotation_models.json` is the machine-readable index.

| Cell | Model | Prompt | Read-out | Invocation |
|---|---|---|---|---|
| primary | Llama-3.1-8B-Instruct | P1 | first-token | `01_annotate_primary.py --gpu 0` |
| D | Qwen2.5-7B-Instruct | P2 | full-string | `02_annotate_second_pipeline.py --gpu 0` |
| C | Qwen2.5-7B-Instruct | P1 | full-string | `03_annotate_second_pipeline_origprompt.py --gpu 0` |
| A′ | Llama-3.1-8B-Instruct | P1 | full-string | `04_annotate_controlled_cells.py --cell Aprime --gpu 0` |
| B | Llama-3.1-8B-Instruct | P2 | full-string | `04_annotate_controlled_cells.py --cell B --gpu 0` |
| M | Mistral-7B-Instruct-v0.3 | P2 | full-string | `05_annotate_third_family.py --cell M --gpu 0` |

**The two read-outs.** *First-token* takes a softmax over the logits of the leading token of each
label at the answer position. *Full-string* sums the log-probabilities of all of a label's tokens
and takes a softmax over the four sums, **without** length normalisation. `UNCLEAR` tokenises to
more than one token in most vocabularies, so the two read-outs are not interchangeable — which is
precisely why the paper separates them.

**Prompts.** P1 is the deployed prompt. P2 adds explicit label definitions to the system prompt
(`NULL` = the abstract directly reports no association for the named pair; `UNCLEAR` = the pair is
absent or its direction cannot be determined).

## Operational notes that cost us time

- **The primary cell checks every prompt** for the claim's own outcome term before scoring. That
  guard is what the corrected extractor fixed; keep it.
- **Records that already have a label keep it.** Only unlabelled records are scored, so a cell is
  resumable — re-running it after an interruption costs only the remainder.
- **The third family needs `text_full.json`, not the records snapshot.** The snapshot carries
  identifiers, years, titles and publication types but **no abstracts**. Pointing a third-family run
  at it produces a cell in which every abstract is empty, and the failure is silent: the model still
  returns a label for every item. Always check the realised abstract coverage of a new cell.
- **Shard with verified-disjoint key lists.** When fanning a cell across GPUs, confirm that the
  union of the shards equals the item set and that no key appears twice. Duplicate keys across
  shards do not crash anything; they quietly change the cell.
- **Gated weights.** Llama-3.1-8B-Instruct needs a Hugging Face token whose scope includes
  gated-repository read access. That permission is separate from being granted access to the
  repository itself, and the error when it is missing looks like a 403 on download.

## External re-annotation

```bash
python data_annotation/external/build_items.py
python data_annotation/external/annotate_external.py --gpu 0
```

This runs on the retained External-1/2 retrieval texts and reproduces the archived labels at
κ = 0.959 (External-1) and 0.967 (External-2). It establishes population alignment, not identical
input text.
