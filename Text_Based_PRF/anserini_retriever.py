from pyserini.search import SimpleSearcher


class Document:
    def __init__(self, docid, text, score):
        self.id = docid
        self.text = text
        self.score = score


class AnseriniRetriever:
    def __init__(self, CONF):
        self.searcher = SimpleSearcher(CONF[CONF["FILETYPE"]]["INDEX"])
        self.searcher.set_bm25(0.82, 0.68)

    def retrieve(self, query):
        query_results = self.searcher.search(query, 1000)
        results = []
        for item in query_results:
            doc = self.get_doc_from_index(item.docid)
            doc.id = item.docid
            doc.score = item.score
            results.append(doc)
        return results

    def get_doc_from_index(self, doc_id):
        doc = self.searcher.doc(doc_id)
        contents = doc.raw()
        return Document(doc_id, contents, 0.)
