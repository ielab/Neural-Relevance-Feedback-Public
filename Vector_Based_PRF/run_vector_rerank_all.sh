#!/bin/zsh

for prf in 1 3 5 10
do
  # TREC DL 2019 AVG PRF
  python3 main.py --result_output ./result/trec_deep_2019_vector_rerank/ANCE_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_deep_2019/bm25+bert/bm25_default_bert.res --prf_method avg --prf "${prf}" --query_tsv_file ./data/trec_deep_2019/query/msmarco-2019test-queries.tsv --collection_index ./data/trec_deep_2019/index/ANCE_FlatIP_collection.index --model ANCE --model_path ./ANCE_Model
  python3 main.py --result_output ./result/trec_deep_2019_vector_rerank/RepBERT_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_deep_2019/bm25+bert/bm25_default_bert.res --prf_method avg --prf "${prf}" --query_tsv_file ./data/trec_deep_2019/query/msmarco-2019test-queries.tsv --collection_index ./data/trec_deep_2019/index/RepBERT_FlatIP_collection.index --model RepBERT --model_path ./RepBERT_Model

  # TREC DL 2020 AVG PRF
  python3 main.py --result_output ./result/trec_deep_2020_vector_rerank/ANCE_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_deep_2020/bm25+bert/bm25_default_bert.res --prf_method avg --prf "${prf}" --query_tsv_file ./data/trec_deep_2020/query/msmarco-test2020-queries.tsv --collection_index ./data/trec_deep_2019/index/ANCE_FlatIP_collection.index --model ANCE --model_path ./ANCE_Model
  python3 main.py --result_output ./result/trec_deep_2020_vector_rerank/RepBERT_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_deep_2020/bm25+bert/bm25_default_bert.res --prf_method avg --prf "${prf}" --query_tsv_file ./data/trec_deep_2020/query/msmarco-test2020-queries.tsv --collection_index ./data/trec_deep_2019/index/RepBERT_FlatIP_collection.index --model RepBERT --model_path ./RepBERT_Model

  # WebAP AVG PRF
  python3 main.py --result_output ./result/webap_vector_rerank/ANCE_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/webap/bm25+bert/bm25_default_bert.res --prf_method avg --prf "${prf}" --query_tsv_file ./data/webap/query/full_queries.tsv --collection_index ./data/webap/index/ANCE_FlatIP_collection.index --model ANCE --model_path ./ANCE_Model --vector_rerank_mapper ./data/webap/index/PID_Mapping_Reverse.json
  python3 main.py --result_output ./result/webap_vector_rerank/RepBERT_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/webap/bm25+bert/bm25_default_bert.res --prf_method avg --prf "${prf}" --query_tsv_file ./data/webap/query/full_queries.tsv --collection_index ./data/webap/index/RepBERT_FlatIP_collection.index --model RepBERT --model_path ./RepBERT_Model --vector_rerank_mapper ./data/webap/index/PID_Mapping_Reverse.json

  # TREC CAsT 2019 AVG PRF
  python3 main.py --result_output ./result/trec_cast_vector_rerank/2019/ANCE_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_cast/2019/bm25+bert/bm25_default_bert.res --prf_method avg --prf "${prf}" --query_tsv_file ./data/trec_cast/2019/query/full_queries.tsv --collection_index ./data/trec_cast/2019/index/ANCE_FlatIP_collection.index --model ANCE --model_path ./ANCE_Model --vector_rerank_mapper ./data/trec_cast/2019/index/PID_Mapping_Reverse.json
  python3 main.py --result_output ./result/trec_cast_vector_rerank/2019/RepBERT_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_cast/2019/bm25+bert/bm25_default_bert.res --prf_method avg --prf "${prf}" --query_tsv_file ./data/trec_cast/2019/query/full_queries.tsv --collection_index ./data/trec_cast/2019/index/RepBERT_FlatIP_collection.index --model RepBERT --model_path ./RepBERT_Model --vector_rerank_mapper ./data/trec_cast/2019/index/PID_Mapping_Reverse.json

  # DL HARD AVG PRF
  python3 main.py --result_output ./result/dl_hard_vector_rerank/ANCE_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/dl_hard/bm25+bert/bm25_default_bert.res --prf_method avg --prf "${prf}" --query_tsv_file ./data/dl_hard/query/topics.tsv --collection_index ./data/trec_deep_2019/index/ANCE_FlatIP_collection.index --model ANCE --model_path ./ANCE_Model
  python3 main.py --result_output ./result/dl_hard_vector_rerank/RepBERT_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/dl_hard/bm25+bert/bm25_default_bert.res --prf_method avg --prf "${prf}" --query_tsv_file ./data/dl_hard/query/topics.tsv --collection_index ./data/trec_deep_2019/index/RepBERT_FlatIP_collection.index --model RepBERT --model_path ./RepBERT_Model

  for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
  do
    # TREC DL 2019 Rocchio PRF
    python3 main.py --result_output ./result/trec_deep_2019_vector_rerank/ANCE_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_deep_2019/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --prf "${prf}" --query_tsv_file ./data/trec_deep_2019/query/msmarco-2019test-queries.tsv --collection_index ./data/trec_deep_2019/index/ANCE_FlatIP_collection.index --model ANCE --model_path ./ANCE_Model
    python3 main.py --result_output ./result/trec_deep_2019_vector_rerank/ANCE_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_deep_2019/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --has_alpha --prf "${prf}" --query_tsv_file ./data/trec_deep_2019/query/msmarco-2019test-queries.tsv --collection_index ./data/trec_deep_2019/index/ANCE_FlatIP_collection.index --model ANCE --model_path ./ANCE_Model
    python3 main.py --result_output ./result/trec_deep_2019_vector_rerank/RepBERT_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_deep_2019/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --prf "${prf}" --query_tsv_file ./data/trec_deep_2019/query/msmarco-2019test-queries.tsv --collection_index ./data/trec_deep_2019/index/RepBERT_FlatIP_collection.index --model RepBERT --model_path ./RepBERT_Model
    python3 main.py --result_output ./result/trec_deep_2019_vector_rerank/RepBERT_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_deep_2019/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --has_alpha --prf "${prf}" --query_tsv_file ./data/trec_deep_2019/query/msmarco-2019test-queries.tsv --collection_index ./data/trec_deep_2019/index/RepBERT_FlatIP_collection.index --model RepBERT --model_path ./RepBERT_Model

    # TREC DL 2020 Rocchio PRF
    python3 main.py --result_output ./result/trec_deep_2020_vector_rerank/ANCE_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_deep_2020/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --prf "${prf}" --query_tsv_file ./data/trec_deep_2020/query/msmarco-test2020-queries.tsv --collection_index ./data/trec_deep_2019/index/ANCE_FlatIP_collection.index --model ANCE --model_path ./ANCE_Model
    python3 main.py --result_output ./result/trec_deep_2020_vector_rerank/ANCE_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_deep_2020/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --has_alpha --prf "${prf}" --query_tsv_file ./data/trec_deep_2020/query/msmarco-test2020-queries.tsv --collection_index ./data/trec_deep_2019/index/ANCE_FlatIP_collection.index --model ANCE --model_path ./ANCE_Model
    python3 main.py --result_output ./result/trec_deep_2020_vector_rerank/RepBERT_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_deep_2020/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --prf "${prf}" --query_tsv_file ./data/trec_deep_2020/query/msmarco-test2020-queries.tsv --collection_index ./data/trec_deep_2019/index/RepBERT_FlatIP_collection.index --model RepBERT --model_path ./RepBERT_Model
    python3 main.py --result_output ./result/trec_deep_2020_vector_rerank/RepBERT_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_deep_2020/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --has_alpha --prf "${prf}" --query_tsv_file ./data/trec_deep_2020/query/msmarco-test2020-queries.tsv --collection_index ./data/trec_deep_2019/index/RepBERT_FlatIP_collection.index --model RepBERT --model_path ./RepBERT_Model

    # WebAP Rocchio PRF
    python3 main.py --result_output ./result/webap_vector_rerank/ANCE_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/webap/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --prf "${prf}" --query_tsv_file ./data/webap/query/full_queries.tsv --collection_index ./data/webap/index/ANCE_FlatIP_collection.index --model ANCE --model_path ./ANCE_Model --vector_rerank_mapper ./data/webap/index/PID_Mapping_Reverse.json
    python3 main.py --result_output ./result/webap_vector_rerank/ANCE_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/webap/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --has_alpha --prf "${prf}" --query_tsv_file ./data/webap/query/full_queries.tsv --collection_index ./data/webap/index/ANCE_FlatIP_collection.index --model ANCE --model_path ./ANCE_Model --vector_rerank_mapper ./data/webap/index/PID_Mapping_Reverse.json
    python3 main.py --result_output ./result/webap_vector_rerank/RepBERT_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/webap/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --prf "${prf}" --query_tsv_file ./data/webap/query/full_queries.tsv --collection_index ./data/webap/index/RepBERT_FlatIP_collection.index --model RepBERT --model_path ./RepBERT_Model --vector_rerank_mapper ./data/webap/index/PID_Mapping_Reverse.json
    python3 main.py --result_output ./result/webap_vector_rerank/RepBERT_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/webap/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --has_alpha --prf "${prf}" --query_tsv_file ./data/webap/query/full_queries.tsv --collection_index ./data/webap/index/RepBERT_FlatIP_collection.index --model RepBERT --model_path ./RepBERT_Model --vector_rerank_mapper ./data/webap/index/PID_Mapping_Reverse.json

    # TREC CAsT 2019 Rocchio PRF
    python3 main.py --result_output ./result/trec_cast_vector_rerank/2019/ANCE_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_cast/2019/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --prf "${prf}" --query_tsv_file ./data/trec_cast/2019/query/full_queries.tsv --collection_index ./data/trec_cast/2019/index/ANCE_FlatIP_collection.index --model ANCE --model_path ./ANCE_Model --vector_rerank_mapper ./data/trec_cast/2019/index/PID_Mapping_Reverse.json
    python3 main.py --result_output ./result/trec_cast_vector_rerank/2019/ANCE_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_cast/2019/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --has_alpha --prf "${prf}" --query_tsv_file ./data/trec_cast/2019/query/full_queries.tsv --collection_index ./data/trec_cast/2019/index/ANCE_FlatIP_collection.index --model ANCE --model_path ./ANCE_Model --vector_rerank_mapper ./data/trec_cast/2019/index/PID_Mapping_Reverse.json
    python3 main.py --result_output ./result/trec_cast_vector_rerank/2019/RepBERT_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_cast/2019/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --prf "${prf}" --query_tsv_file ./data/trec_cast/2019/query/full_queries.tsv --collection_index ./data/trec_cast/2019/index/RepBERT_FlatIP_collection.index --model RepBERT --model_path ./RepBERT_Model --vector_rerank_mapper ./data/trec_cast/2019/index/PID_Mapping_Reverse.json
    python3 main.py --result_output ./result/trec_cast_vector_rerank/2019/RepBERT_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/trec_cast/2019/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --has_alpha --prf "${prf}" --query_tsv_file ./data/trec_cast/2019/query/full_queries.tsv --collection_index ./data/trec_cast/2019/index/RepBERT_FlatIP_collection.index --model RepBERT --model_path ./RepBERT_Model --vector_rerank_mapper ./data/trec_cast/2019/index/PID_Mapping_Reverse.json

    # DL HARD Rocchio PRF
    python3 main.py --result_output ./result/dl_hard_vector_rerank/ANCE_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/dl_hard/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --prf "${prf}" --query_tsv_file ./data/dl_hard/query/topics.tsv --collection_index ./data/trec_deep_2019/index/ANCE_FlatIP_collection.index --model ANCE --model_path ./ANCE_Model
    python3 main.py --result_output ./result/dl_hard_vector_rerank/ANCE_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/dl_hard/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --has_alpha --prf "${prf}" --query_tsv_file ./data/dl_hard/query/topics.tsv --collection_index ./data/trec_deep_2019/index/ANCE_FlatIP_collection.index --model ANCE --model_path ./ANCE_Model
    python3 main.py --result_output ./result/dl_hard_vector_rerank/RepBERT_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/dl_hard/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --prf "${prf}" --query_tsv_file ./data/dl_hard/query/topics.tsv --collection_index ./data/trec_deep_2019/index/RepBERT_FlatIP_collection.index --model RepBERT --model_path ./RepBERT_Model
    python3 main.py --result_output ./result/dl_hard_vector_rerank/RepBERT_PRF_Rerank_Runs/prf_retrieval.res --vector_rerank --vector_rerank_basefile Neural-PRF/result/dl_hard/bm25+bert/bm25_default_bert.res --prf_method rocchio --rocchio_beta "${i}" --has_alpha --prf "${prf}" --query_tsv_file ./data/dl_hard/query/topics.tsv --collection_index ./data/trec_deep_2019/index/RepBERT_FlatIP_collection.index --model RepBERT --model_path ./RepBERT_Model

  done
done