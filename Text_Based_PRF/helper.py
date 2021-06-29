import json
from tqdm import tqdm
import numpy as np


def readQueryFile(path):
    queryCollection = []
    queryDict = {}
    queryFilePath = path
    with open(queryFilePath, 'r') as f:
        contents = f.readlines()
    for line in contents:
        queryContent = json.loads(line)
        queryCollection.append(queryContent)
    for query in queryCollection:
        queryDict[query["id"]] = query["contents"]
    return queryDict


def readRankFile(CONF):
    FILETYPE = CONF["FILETYPE"]
    rankFilePath = CONF[FILETYPE]["RANK_FILE"]
    with open(rankFilePath, 'r') as f:
        file = f.readlines()
    cleanedCol = []
    for each in tqdm(file, desc='-- Process Ranked List'):
        temp = each.strip().split("\t")
        cleanedCol.append(temp)
    fullCollection = []
    tempCollection = []
    lastCollection = []
    currentID = cleanedCol[0][0]
    for element in cleanedCol:
        if element[0] == currentID:
            tempCollection.append(element)
        else:
            currentID = element[0]
            fullCollection.append(np.array(tempCollection))
            tempCollection = [element]
        if element[0] == cleanedCol[-1][0]:
            lastCollection.append(element)
    fullCollection.append(np.array(lastCollection))
    return fullCollection


def readCollectionFile(CONF, RANKED_FILE_CONTENT):
    dic = {}
    for query in tqdm(RANKED_FILE_CONTENT, desc='-- Create Collection Dict'):
        for document in query:
            if document[1] not in dic:
                dic[document[1]] = ""
            else:
                continue
    collectionDict = getDocumentContent(CONF, dic)
    return collectionDict


def getDocumentContent(CONF, dic):
    collectionPaths = CONF[CONF["FILETYPE"]]["COLLECTION"]
    for path in tqdm(collectionPaths, desc='-- Load Collection'):
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            jsonL = json.loads(line)
            if jsonL["id"] in dic:
                dic[jsonL["id"]] = jsonL["contents"]
            else:
                continue
    return dic
