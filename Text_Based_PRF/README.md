# Text Based PRF
The code for text based PRF

### Command
`python3 main.py --prf 20 --prf_method CT --filetype TREC2019`

For different text handling method:

`CT`: Concatenate Truncate \
`CA`: Concatenate Aggregate \
`SW`: Sliding Window

For different datasets:

`TREC2019`: TREC Deep Learning 2019 \
`TREC2020`: TREC Deep Learning 2020 \
`CAST`: TREC CAsT 2019 \
`WEBAP`: Web Answer Passages \
`DLHARD`: Deep Learning Hard Queries 

After generating result crumb files, use:

`python3 generate_prf_results.py -bert -i bm25_default_bert_prf20_sliding_window.w65.s32_crumbs.txt -agg avg`

to aggregate the results.

If based on BERT, add `-bert` flag \
If based on BM25, add `-bm25` flag

For different aggregate methods:

`avg`: Average \
`max`: Max \
`borda`: Borda