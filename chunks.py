from pinecone import ServerlessSpec

# Initialize the Pinecone client with your API key
pc = Pinecone(api_key=os.environ["PINECONE_KEY"])
#Create your Pinecone index


def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    # Convert the iterable into an iterator
    it = iter(iterable)
    # Slice the iterator into chunks of size batch_size
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        # Yield the chunk
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


# Initialize the Pinecone client with your API key
pc = Pinecone(api_key=os.environ["PINECONE_KEY"])

index = pc.Index('datacamp-index')

# Upsert vectors in batches of 100
for chunk in chunks(vectors):
    index.upsert(vectors=chunk) 

# Retrieve statistics of the connected Pinecone index
print(index.describe_index_stats())


# {'dimension': 1536,
#  'index_fullness': 0.0,
#  'namespaces': {'': {'vector_count': 1000}},
#  'total_vector_count': 1000}



# Initialize the client
pc = Pinecone(api_key=os.environ["PINECONE_KEY"],pool_threads=30)

index = pc.Index('datacamp-index')

# Upsert vectors in batches of 200 vectors
with pc.Index('datacamp-index', pool_threads=30) as index:
    async_results = [index.upsert(vectors=chunk, async_req=True) for chunk in chunks(vectors, batch_size=100)]
    [async_result.get() for async_result in async_results]

# Retrieve statistics of the connected Pinecone index
print(index.describe_index_stats())


# {'dimension': 1536,
#  'index_fullness': 0.0,
#  'namespaces': {'': {'vector_count': 1000}},
#  'total_vector_count': 1000}

from pinecone import ServerlessSpec


# Initialize the Pinecone client with your API key
pc = Pinecone(api_key=os.environ["PINECONE_KEY"])
pc.create_index(
    name="datacamp-index",
    dimension=1536,
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)
index = pc.Index('datacamp-index')

# Upsert vectors in batches of 100
for chunk in chunks(vectors):
    index.upsert(vectors=chunk) 

# Retrieve statistics of the connected Pinecone index
print(index.describe_index_stats())

# Retrieve statistics of the connected Pinecone index
# print(index.describe_index_stats())
# {'dimension': 1536,
#  'index_fullness': 0.0,
#  'namespaces': {'': {'vector_count': 1000}},
#  'total_vector_count': 1000}

