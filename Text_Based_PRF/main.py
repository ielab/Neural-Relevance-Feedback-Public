from helper import *
from handler import *
from bert_reranker import *
from t5_reranker import *
# from anserini_retriever import *
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prf_flag', help='Use PRF or doing simple rerank', action='store_true', default=False)
parser.add_argument('--prf', help='PRF number', type=int, default=20)
parser.add_argument('--prf_method',
                    help='Three types of PRF Methods, CT: Concat_Truncate; CA: Concat_Aggregate; SW: Sliding_Window',
                    type=str,
                    required=True)
parser.add_argument('--filetype',
                    help='TREC, CAST, WIKIPASSAGEQA, WEBAP, DLHARD', type=str, required=True)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logging.info("--------------------------------------------------")
logging.info("Loading Config...")
f = open("config.json")
CONF = json.loads(f.read())
f.close()
CONF["PRF"] = args.prf
CONF["PRF_METHOD"] = args.prf_method
CONF["FILETYPE"] = args.filetype
logging.info(f'Using {CONF["MODE"].upper()} Mode')
logging.info(f'Using {CONF["MODEL"].upper()} Model')
logging.info("Loading Query List...")
QUERY = readQueryFile(CONF[CONF["FILETYPE"]]["QUERY"])
if CONF["MODE"] == "file":
    logging.info("Loading BM25 Retrieved List/Collection Dict...")
    RANKED_FILE_CONTENT = readRankFile(CONF)
    COLLECTION_DICT = readCollectionFile(CONF, RANKED_FILE_CONTENT)
# else:
#     RETRIEVER = AnseriniRetriever(CONF)
logging.info(f'Loading {CONF["MODEL"].upper()} Model...')
if CONF["MODEL"] == "bert":
    RERANKER = BERTReranker(CONF)
elif CONF["MODEL"] == "t5":
    RERANKER = T5Reranker(CONF, 24)
logging.info("Loaded")
logging.info("--------------------------------------------------")


def main():
    if args.prf_flag:
        if CONF["MODE"] == "file":
            performPRFFromRankedFile(QUERY, CONF, RANKED_FILE_CONTENT, COLLECTION_DICT, RERANKER)
        # else:
        #     performPRFWithPyserini(QUERY, RETRIEVER, RERANKER, CONF)
        logging.info("Finished")
    else:
        performRerank(QUERY, CONF, RANKED_FILE_CONTENT, COLLECTION_DICT, RERANKER)


if __name__ == '__main__':
    main()
