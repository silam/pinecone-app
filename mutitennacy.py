# Initialize the Pinecone client with your API key
pc = Pinecone(api_key=os.environ["PINECONE_KEY"])
index = pc.Index('datacamp-index')

# Upsert vector_set1 to namespace1
index.upsert(
  vectors=vector_set1,
  namespace="namespace1"
)

# Upsert vector_set2 to namespace2
index.upsert(
  vectors=vector_set2,
  namespace="namespace2"
)

# Print the index statistics
print(index.describe_index_stats())

# {'dimension': 1536,
#  'index_fullness': 0.0,
#  'namespaces': {'': {'vector_count': 1000}, 'namespace1': {'vector_count': 50}},
#  'total_vector_count': 1050}

# Query namespace1 with the vector provided
query_result = index.query(
    vector=vector,
    top_k = 3,
    namespace="namespace1"
)
print(query_result)

{'matches': [{'id': '1033', 'score': 0.0682480708, 'values': []},
             {'id': '1041', 'score': 0.0521688648, 'values': []},
             {'id': '1013', 'score': 0.0512326546, 'values': []}],
 'namespace': 'namespace1',
 'usage': {'read_units': 1}}


