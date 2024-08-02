from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

#创建client
client = OpenSearch(
    'https://localhost:9200',
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
    http_auth=('admin', '114514Aa@')
)
index = 'image-search'

text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

def ann_search(query):
    query_emb = text_model.encode(query)
    hits = client.search(index=index, body={
        "_source": ["name", "text"],
        "size": 2,
        "query": {
            "knn": {
                "img_vector": {
                    "vector": query_emb,
                    "k": 10
                }
            }
        }
    })
    return hits

if __name__ == "__main__":
    hits = ann_search('你好')
    print(hits)
