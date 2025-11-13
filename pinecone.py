# Import ServerlessSpec
from pinecone import ServerlessSpec

# Initialize the Pinecone client with your API key
pc = Pinecone(api_key=os.environ["PINECONE_KEY"])
#Create your Pinecone index
pc.create_index(
    name="my-first-index",
    dimension=1536,
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)



# Connect to your index
index = pc.Index("my-first-index")

# Print the index statistics
print(index.describe_index_stats())


# List your indexes
print(pc.list_indexes())


# Check that each vector has a dimensionality of 1536
vector_dims = [len(vector['values']) == 1536 for vector in vectors]
print(all(vector_dims))

###########################



# Retrieve the MOST similar vector with the year 2024
query_result = index.query(
    vector=vector,
    filter={
        "year": 2024
    },
    top_k = 3
)
print(query_result)








# Ingest the vectors and metadata
index.upsert( 
    vectors=vectors
)

# Print the index statistics
print(index.describe_index_stats())



{
    'namespace': 'namespace1', 
    'usage': 
    {'read_units': 1}, 
    'vectors': 
    {'0': 
     {'id': '0',                   
      'metadata': {"genre": "productivity", "year": 2020},  
    'values': [0.025525547564029694, ...]},             
    '1': {'id': '1',                   
          'metadata': {"genre": "action", "year": 2023},                   
          'values': [-0.0131468913, ...]}}
          }


pc = Pinecone(api_key=os.environ["PINECONE_KEY"])

index = pc.Index('datacamp-index')
ids = ['2', '5', '8']

# Fetch the vectors from the connected Pinecone index
fetched_vectors = index.fetch(
    ids=ids
)
#print(fetched_vectors['vectors']['8']['metadata'])
# Extract the metadata from each result in fetched_vectors
metadatas = [fetched_vectors['vectors'][id]['metadata'] for id in ids]
print(metadatas)


[{'genre': 'comedy', 'year': 2004.0}, {'genre': 'thriller', 'year': 2001.0}, {'genre': 'action', 'year': 2017.0}]



index = pc.Index('datacamp-index')
#############################################################
# Retrieve the top three most similar records
query_result = index.query(
    vector=vector,
    top_k=3
)

print(query_result)

# {'matches': [{'id': '0', 'score': 1.00003695, 'values': []},
#              {'id': '89', 'score': 0.0631115735, 'values': []},
#              {'id': '42', 'score': 0.0616056919, 'values': []}],
#  'namespace': '',
#  'usage': {'read_units': 1}}



# Create an index that uses the dot product distance metric
pc.create_index(
    name="dotproduct-index",
    dimension=1536,
    metric='dotproduct',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

# Print a list of your indexes
print(pc.list_indexes())


# {'indexes': [{'deletion_protection': 'disabled',
#               'dimension': 1536,
#               'host': 'my-first-index-dep9hl4.svc.aped-4627-b74a.pinecone.io',
#               'metric': 'cosine',
#               'name': 'my-first-index',
#               'spec': {'serverless': {'cloud': 'aws', 'region': 'us-east-1'}},
#               'status': {'ready': True, 'state': 'Ready'}},
#              {'deletion_protection': 'disabled',
#               'dimension': 1536,
#               'host': 'dotproduct-index-dep9hl4.svc.aped-4627-b74a.pinecone.io',
#               'metric': 'dotproduct',
#               'name': 'dotproduct-index',
#               'spec': {'serverless': {'cloud': 'aws', 'region': 'us-east-1'}},
#               'status': {'ready': True, 'state': 'Ready'}},
#              {'deletion_protection': 'disabled',
#               'dimension': 1536,
#               'host': 'datacamp-index-dep9hl4.svc.aped-4627-b74a.pinecone.io',
#               'metric': 'cosine',
#               'name': 'datacamp-index',
#               'spec': {'serverless': {'cloud': 'aws', 'region': 'us-east-1'}},
#               'status': {'ready': True, 'state': 'Ready'}}]}





# Retrieve the MOST similar vector with genre and year filters
query_result = index.query(
    vector=vector,
    top_k=1,
    filter={
        "year": {"$lt": 2018},
        "genre" : "thriller"
    },
)
print(query_result)





index = pc.Index('datacamp-index')

# Update the values of vector ID 7
index.update(
    id="7",
    values=vector
)

# Fetch vector ID 7
fetched_vector = index.fetch(
    ids = ["7"]
)
print(fetched_vector)

#######################################

index = pc.Index('datacamp-index')

# Update the metadata of vector ID 7
index.update(
    id="7",
    set_metadata={'genre':'thriller', 'year':2024}
)

# Fetch vector ID 7
fetched_vector = index.fetch(
    ids=["7"]
)
print(fetched_vector)



index = pc.Index('datacamp-index')

# Delete vectors
index.delete(
    ids=["3","4"]
)

# Retrieve metrics of the connected Pinecone index
print(index.describe_index_stats())


# {'dimension': 1536,
#  'index_fullness': 0.0,
#  'namespaces': {'': {'vector_count': 98}},
#  'total_vector_count': 98}