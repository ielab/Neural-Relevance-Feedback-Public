# Vector Based PRF

The code for vector based PRF.

## Index Generating Command

### RepBERT

`python3 repbert_index_embedding_generator.py --collection_file ./data/trec_deep_2019/collection_jsonl --model_path ./RepBERT_Model --embedding_output ./data/trec_deep_2019/index --per_gpu_batch_size 128`

### ANCE

`python3 ance_index_embedding_generator.py --encoder ./ANCE_Model --corpus ./data/trec_deep_2019/collection_jsonl --index ./data/trec_deep_2019/index --batch 128`

## Command

The output file is res files following TREC standards.

### Retrieval Task

For retrieval task, two methods are implemented, avg and rocchio. `MODEL_NAME` can be `ANCE` or `RepBERT`.

#### No PRF Retrieval

For no PRF retrieval, `--prf`, `--iterations`, `--prf_method`, `--iteration_out`, `--rocchio_beta`, `--has_alpha`  parameters are not needed, `--mapper_path` is only for trec_cast and webap, other datasets do not need a mapper.

`python3 main.py --result_output RESULT_OUTPUT_PATH --query_tsv_file QUERY_TSV_FILEPATH --collection_index COLLECTION_INDEX_PATH --model MODEL_NAME --model_path MODEL_PATH --collection_file COLLECTION_TSV_FILEPATH --mapper_path MAPPER_JSON_PATH`

#### AVG PRF Retrieval

For avg retrieval, `--rocchio_beta`, `--has_alpha`  parameters are not needed, `--mapper_path` is only for trec_cast and webap, other datasets do not need a mapper.

`python3 main.py --result_output RESULT_OUTPUT_PATH --iteration_out ITERATION_OUTPUT_PATH --prf_method avg --iterations ITERATION_NUMBER --prf PRF_DEPTH --query_tsv_file QUERY_TSV_FILEPATH --collection_index COLLECTION_INDEX_PATH --model MODEL_NAME --model_path MODEL_PATH --collection_file COLLECTION_TSV_FILEPATH --mapper_path MAPPER_JSON_PATH`

#### Rocchio PRF Retrieval

For Rocchio retrieval, two subtypes are implemented:

For varying beta only, `--has_alpha`  parameters are not needed, `--mapper_path` is only for trec_cast and webap, other datasets do not need a mapper.

`python3 main.py --result_output RESULT_OUTPUT_PATH --iteration_out ITERATION_OUTPUT_PATH --prf_method rocchio --rocchio_beta BETA --iterations ITERATION_NUMBER --prf PRF_DEPTH --query_tsv_file QUERY_TSV_FILEPATH --collection_index COLLECTION_TSV_FILEPATH --model MODEL_NAME --model_path MODEL_PATH --collection_file COLLECTION_TSV_FILEPATH --mapper_path MAPPER_JSON_PATH`

For varying both alpha and beta, all parameters are needed, `--mapper_path` is only for trec_cast and webap, other datasets do not need a mapper.

`python3 main.py --result_output RESULT_OUTPUT_PATH --iteration_out ITERATION_OUTPUT_PATH --prf_method rocchio --rocchio_beta BETA --has_alpha --iterations ITERATION_NUMBER --prf PRF_DEPTH --query_tsv_file QUERY_TSV_FILEPATH --collection_index COLLECTION_TSV_FILEPATH --model MODEL_NAME --model_path MODEL_PATH --collection_file COLLECTION_TSV_FILEPATH --mapper_path MAPPER_JSON_PATH`

### Rerank Task

For rerank task, two methods are implemented, avg and rocchio. `MODEL_NAME` can be `ANCE` or `RepBERT`. `--vector_rerank` parameter needs to be added.
`BASE_RES_FILE` is the filepath to the base result file, e.g. `bm25_default.res`.

#### AVG PRF Rerank

For avg rerank, `--rocchio_beta`, `--has_alpha`  parameters are not needed, `--vector_rerank_mapper` is reverse mapper only for trec_cast and webap, other datasets do not need a REVERSE mapper.

`python3 main.py --result_output RESULT_OUTPUT_PATH --vector_rerank --vector_rerank_basefile BASE_RES_FILE --prf_method avg --prf PRF_DEPTH --query_tsv_file QUERY_TSV_FILEPATH --collection_index COLLECTION_TSV_FILEPATH --model MODEL_NAME --model_path MODEL_PATH --vector_rerank_mapper REVERSE_MAPPER_JSON_PATH`

#### Rocchio PRF Rerank

For Rocchio rerank, two subtypes are implemented:

For varying beta only, `--has_alpha`  parameters are not needed, `--vector_rerank_mapper` is reverse mapper only for trec_cast and webap, other datasets do not need a REVERSE mapper.

`python3 main.py --result_output RESULT_OUTPUT_PATH --vector_rerank --vector_rerank_basefile BASE_RES_FILE --prf_method rocchio --rocchio_beta BETA --prf PRF_DEPTH --query_tsv_file QUERY_TSV_FILEPATH --collection_index COLLECTION_TSV_FILEPATH --model MODEL_NAME --model_path MODEL_PATH --vector_rerank_mapper REVERSE_MAPPER_JSON_PATH`

For varying alpha and beta, all parameters are needed, `--vector_rerank_mapper` is reverse mapper only for trec_cast and webap, other datasets do not need a REVERSE mapper.

`python3 main.py --result_output RESULT_OUTPUT_PATH --vector_rerank --vector_rerank_basefile BASE_RES_FILE --prf_method rocchio --rocchio_beta BETA --has_alpha --prf PRF_DEPTH --query_tsv_file QUERY_TSV_FILEPATH --collection_index COLLECTION_TSV_FILEPATH --model MODEL_NAME --model_path MODEL_PATH --vector_rerank_mapper REVERSE_MAPPER_JSON_PATH`