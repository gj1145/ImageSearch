from opensearchpy import OpenSearch

#创建client
client = OpenSearch(
    'https://localhost:9200',
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
    http_auth=('admin', '114514Aa@')
)

#创建索引
index = 'image-search'
index_body = {
  "settings": {
    "index": {
      "knn": True,
      "knn.algo_param.ef_search": 100
    }
  },
  "mappings": {
    "properties": {
        "name": {
            "type": "text"
        },
        "img_vector": {
          "type": "knn_vector",
          "dimension": 512,
          "method": {
            "name": "hnsw",
            "space_type": "l2",
            "engine": "nmslib",
            "parameters": {
              "ef_construction": 128,
              "m": 24
            }
          }
        },
        "text": {
            "type": "text"
        }
    }
  }
}
client.indices.create(index=index, body=index_body)