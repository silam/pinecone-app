# Initialize the Pinecone client with your API key
pc = Pinecone(api_key=os.environ["PINECONE_KEY"])


# Create Pinecone index
pc.create_index(
    name='semantic-search-datacamp', 
    dimension=1536,
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)
# Connect to index and print the index statistics
index = pc.Index("semantic-search-datacamp")

print(index.describe_index_stats())


# {'dimension': 1536,
#  'index_fullness': 0.0,
#  'namespaces': {},
#  'total_vector_count': 0}


# Initialize the Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_KEY"])
index = pc.Index('pinecone-datacamp')

batch_limit = 100

for batch in np.array_split(df, len(df) / batch_limit):
    # Extract the metadata from each row
    metadatas = [{
      "text_id": row['id'],
      "text": row['text'],
      "title": row['title']} for _, row in batch.iterrows()]
    texts = batch['text'].tolist()
    
    ids = [str(uuid4()) for _ in range(len(texts))]
    
    # Encode texts using OpenAI
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    embeds = [np.array(x.embedding) for x in response.data]
    
    # Upsert vectors to the correct namespace
    index.upsert(vectors=zip(ids, embeds, metadatas), namespace="squad_dataset")

#  'index_fullness': 0.02, 'namespaces': {'squad_dataset': {'vector_count': 2000}}, 'total_vector_count': 2000}

# Initialize the Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_KEY"])
index = pc.Index('pinecone-datacamp')

query = "What is in front of the Notre Dame Main Building?"

# Create the query vector
query_response = client.embeddings.create(
    input=query,
    model="text-embedding-3-small"
)
query_emb = query_response.data[0].embedding

# Query the index and retrieve the top five most similar vectors
retrieved_docs = index.query(vector=query_emb,
            top_k = 3,
            namespace='squad_dataset',
            include_metadata=True)

for result in retrieved_docs['matches']:
    print(f"{result['id']}: {round(result['score'], 2)}")
    print('\n')


# e26c1aa5-aee3-48d6-984b-1f724999fa1f: 0.46


# 6a337b64-77b8-48e2-8c84-66822d2e8849: 0.46


# b480646e-90cd-4506-8c61-ddc9e0ceed66: 0.29

# Initialize the Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_KEY"])
index = pc.Index('pinecone-datacamp')

batch_limit = 100

for batch in np.array_split(youtube_df, len(youtube_df) / batch_limit):
    # Extract the metadata from each row
    metadatas = [{
      "text_id": row['id'],
      "text": row['text'],
      "title": row['title'],
      "url": row['url'],
      "published": row['published']} for _, row in batch.iterrows()]
    texts = batch['text'].tolist()
    
    ids = [str(uuid4()) for _ in range(len(texts))]
    
    # Encode texts using OpenAI
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    embeds = [np.array(x.embedding) for x in response.data]
    
    # Upsert vectors to the correct namespace
    index.upsert(vectors=zip(ids, embeds, metadatas), namespace='youtube_rag_dataset')
    
print(index.describe_index_stats())

# {'dimension': 1536,
#  'index_fullness': 0.0,
#  'namespaces': {'squad_dataset': {'vector_count': 401},
#                 'youtube_rag_dataset': {'vector_count': 200}},
#  'total_vector_count': 601}

# <script.py> output:
#     {'dimension': 1536,
#      'index_fullness': 0.0,
#      'namespaces': {'squad_dataset': {'vector_count': 401},
#                     'youtube_rag_dataset': {'vector_count': 400}},
#      'total_vector_count': 801}

# Initialize the Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_KEY"])
index = pc.Index('pinecone-datacamp')

# Define a retrieve function that takes four arguments: query, top_k, namespace, and emb_model
def retrieve(query, top_k, namespace, emb_model):
    # Encode the input query using OpenAI
    query_response = client.embeddings.create(
        input=query,
        model=emb_model
    )
    
    query_emb = query_response.data[0].embedding
    
    # Query the index using the query_emb
    docs = index.query(vector=query_emb, top_k=top_k, namespace=namespace, include_metadata=True)
    
    retrieved_docs = []
    sources = []
    for doc in docs['matches']:
        retrieved_docs.append(doc['metadata']['text'])
        sources.append((doc['metadata']['title'], doc['metadata']['url']))
    
    return retrieved_docs, sources

documents, sources = retrieve(
  query="How to build next-level Q&A with OpenAI",
  top_k=3,
  namespace='youtube_rag_dataset',
  emb_model="text-embedding-3-small"
)
print(documents)
print(sources)


print(sources)
# ["and our model into a pipeline, into a Q&A pipeline. So again, we get this pipeline from the Transformers library. So we come down here, do from Transformers, import pipeline. And now what we want to do is just initialize a pipeline object. So to do that, we just write pipeline. And then in here, what we need to add is a model type. So obviously, you can see up here, we have all of these different tasks. So summarization, text generation and so on. The Transformers library needs to understand, or this pipeline object needs to understand which one of those pipelines or functions we are intending to use. So to tell it that we want to do question answering, we just write question answering. And that basically sets the wrapper of the pipeline to handle question answering formats. So we'll see our input and for our input, we will be passing a context and a question. So we'll see that it will convert into the right structure that we need for question answering, which is the CLS context separator, question separator and padding. It will convert into that, feed it into our tokenizer.", "and our model into a pipeline, into a Q&A pipeline. So again, we get this pipeline from the Transformers library. So we come down here, do from Transformers, import pipeline. And now what we want to do is just initialize a pipeline object. So to do that, we just write pipeline. And then in here, what we need to add is a model type. So obviously, you can see up here, we have all of these different tasks. So summarization, text generation and so on. The Transformers library needs to understand, or this pipeline object needs to understand which one of those pipelines or functions we are intending to use. So to tell it that we want to do question answering, we just write question answering. And that basically sets the wrapper of the pipeline to handle question answering formats. So we'll see our input and for our input, we will be passing a context and a question. So we'll see that it will convert into the right structure that we need for question answering, which is the CLS context separator, question separator and padding. It will convert into that, feed it into our tokenizer.", "And on the Hugging Face website, we just want to go over to the Models page. So it's here. Okay and on this Models page, the thing that we want to be looking at is this question and answering task. So here we have all these tasks because when you're working with transformers, they can work with a lot of different things. Text summarization, text classification, generation, loads of different things. But what we want to do is question answering. So we click on here and this filters all of the models that are available to us just purely for question and answering. So this is the sort of power of using the Hugging Face Transformers library. It already has all these pre-trained models that we can just download and start using. Now, when you want to go and apply these to specific use cases, you probably want to fine tune it, which means you want to train it a little bit more than what it is already trained. But for actually getting used to how all of this works, all you need to do is download this model and start asking questions and understanding how everything is actually functioning. So obviously there's a lot of models here. We've got 262 models for question answering,"]
# [('How to Build Q&A Models in Python (Transformers)', 'https://youtu.be/scJsty_DR3o'), ('How to Build Q&A Models in Python (Transformers)', 'https://youtu.be/scJsty_DR3o'), ('How to Build Q&A Models in Python (Transformers)', 'https://youtu.be/scJsty_DR3o')]

# <script.py> output:
#     ["and our model into a pipeline, into a Q&A pipeline. So again, we get this pipeline from the Transformers library. So we come down here, do from Transformers, import pipeline. And now what we want to do is just initialize a pipeline object. So to do that, we just write pipeline. And then in here, what we need to add is a model type. So obviously, you can see up here, we have all of these different tasks. So summarization, text generation and so on. The Transformers library needs to understand, or this pipeline object needs to understand which one of those pipelines or functions we are intending to use. So to tell it that we want to do question answering, we just write question answering. And that basically sets the wrapper of the pipeline to handle question answering formats. So we'll see our input and for our input, we will be passing a context and a question. So we'll see that it will convert into the right structure that we need for question answering, which is the CLS context separator, question separator and padding. It will convert into that, feed it into our tokenizer.", "and our model into a pipeline, into a Q&A pipeline. So again, we get this pipeline from the Transformers library. So we come down here, do from Transformers, import pipeline. And now what we want to do is just initialize a pipeline object. So to do that, we just write pipeline. And then in here, what we need to add is a model type. So obviously, you can see up here, we have all of these different tasks. So summarization, text generation and so on. The Transformers library needs to understand, or this pipeline object needs to understand which one of those pipelines or functions we are intending to use. So to tell it that we want to do question answering, we just write question answering. And that basically sets the wrapper of the pipeline to handle question answering formats. So we'll see our input and for our input, we will be passing a context and a question. So we'll see that it will convert into the right structure that we need for question answering, which is the CLS context separator, question separator and padding. It will convert into that, feed it into our tokenizer.", "And on the Hugging Face website, we just want to go over to the Models page. So it's here. Okay and on this Models page, the thing that we want to be looking at is this question and answering task. So here we have all these tasks because when you're working with transformers, they can work with a lot of different things. Text summarization, text classification, generation, loads of different things. But what we want to do is question answering. So we click on here and this filters all of the models that are available to us just purely for question and answering. So this is the sort of power of using the Hugging Face Transformers library. It already has all these pre-trained models that we can just download and start using. Now, when you want to go and apply these to specific use cases, you probably want to fine tune it, which means you want to train it a little bit more than what it is already trained. But for actually getting used to how all of this works, all you need to do is download this model and start asking questions and understanding how everything is actually functioning. So obviously there's a lot of models here. We've got 262 models for question answering,"]
#     [('How to Build Q&A Models in Python (Transformers)', 'https://youtu.be/scJsty_DR3o'), ('How to Build Q&A Models in Python (Transformers)', 'https://youtu.be/scJsty_DR3o'), ('How to Build Q&A Models in Python (Transformers)', 'https://youtu.be/scJsty_DR3o')]
# In [1]:


# Initialize the Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_KEY"])
index = pc.Index('pinecone-datacamp')

query = "How to build next-level Q&A with OpenAI"

# Retrieve the top three most similar documents and their sources
documents, sources = retrieve(query, top_k=3, namespace='youtube_rag_dataset', emb_model="text-embedding-3-small")

prompt_with_context = prompt_with_context_builder(query, documents)
print(prompt_with_context)

def question_answering(prompt, sources, chat_model):
    sys_prompt = "You are a helpful assistant that always answers questions."
    
    # Use OpenAI chat completions to generate a response
    res = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    answer = res.choices[0].message.content.strip()
    answer += "\n\nSources:"
    for source in sources:
        answer += "\n" + source[0] + ": " + source[1]
    
    return answer

answer = question_answering(
  prompt=prompt_with_context,
  sources=sources,
  chat_model='gpt-4o-mini')
print(answer)


# answer = question_answering(
#   prompt=prompt_with_context,
#   sources=sources,
#   chat_model='gpt-4o-mini')
# print(answer)
# Answer the question based on the context below.

# Context:
# and our model into a pipeline, into a Q&A pipeline. So again, we get this pipeline from the Transformers library. So we come down here, do from Transformers, import pipeline. And now what we want to do is just initialize a pipeline object. So to do that, we just write pipeline. And then in here, what we need to add is a model type. So obviously, you can see up here, we have all of these different tasks. So summarization, text generation and so on. The Transformers library needs to understand, or this pipeline object needs to understand which one of those pipelines or functions we are intending to use. So to tell it that we want to do question answering, we just write question answering. And that basically sets the wrapper of the pipeline to handle question answering formats. So we'll see our input and for our input, we will be passing a context and a question. So we'll see that it will convert into the right structure that we need for question answering, which is the CLS context separator, question separator and padding. It will convert into that, feed it into our tokenizer.

# ---

# and our model into a pipeline, into a Q&A pipeline. So again, we get this pipeline from the Transformers library. So we come down here, do from Transformers, import pipeline. And now what we want to do is just initialize a pipeline object. So to do that, we just write pipeline. And then in here, what we need to add is a model type. So obviously, you can see up here, we have all of these different tasks. So summarization, text generation and so on. The Transformers library needs to understand, or this pipeline object needs to understand which one of those pipelines or functions we are intending to use. So to tell it that we want to do question answering, we just write question answering. And that basically sets the wrapper of the pipeline to handle question answering formats. So we'll see our input and for our input, we will be passing a context and a question. So we'll see that it will convert into the right structure that we need for question answering, which is the CLS context separator, question separator and padding. It will convert into that, feed it into our tokenizer.

# ---

# And on the Hugging Face website, we just want to go over to the Models page. So it's here. Okay and on this Models page, the thing that we want to be looking at is this question and answering task. So here we have all these tasks because when you're working with transformers, they can work with a lot of different things. Text summarization, text classification, generation, loads of different things. But what we want to do is question answering. So we click on here and this filters all of the models that are available to us just purely for question and answering. So this is the sort of power of using the Hugging Face Transformers library. It already has all these pre-trained models that we can just download and start using. Now, when you want to go and apply these to specific use cases, you probably want to fine tune it, which means you want to train it a little bit more than what it is already trained. But for actually getting used to how all of this works, all you need to do is download this model and start asking questions and understanding how everything is actually functioning. So obviously there's a lot of models here. We've got 262 models for question answering,

# Question: How to build next-level Q&A with OpenAI
# Answer:
# To build next-level Q&A with OpenAI, you can follow these steps:

# 1. **Utilize the Transformers Library**: Start by importing the necessary components from the Transformers library. Use the `pipeline` function to create a Q&A pipeline.

# 2. **Initialize the Pipeline**: Set up the pipeline by specifying the model type as "question answering". This will configure the pipeline to handle question answering tasks.

# 3. **Prepare Input**: For the Q&A pipeline, you will need to provide a context (the text from which the answer will be derived) and a question. The pipeline will format this input correctly, including adding necessary tokens like CLS, context separator, and question separator.

# 4. **Access Pre-trained Models**: Visit the Hugging Face Models page to explore and select from a variety of pre-trained models specifically designed for question answering. There are numerous models available that you can download and use directly.

# 5. **Fine-tuning (Optional)**: If you have specific use cases or datasets, consider fine-tuning the selected model to improve its performance on your particular questions.

# 6. **Experiment and Iterate**: Start asking questions using the pipeline and observe how the model responds. This will help you understand its capabilities and limitations, allowing you to refine your approach.

# By following these steps, you can effectively leverage OpenAI's capabilities for advanced question answering applications.

# Sources:
# How to Build Q&A Models in Python (Transformers): https://youtu.be/scJsty_DR3o
# How to Build Q&A Models in Python (Transformers): https://youtu.be/scJsty_DR3o
# How to Build Q&A Models in Python (Transformers): https://youtu.be/scJsty_DR3o

# <script.py> output:
#     Answer the question based on the context below.
    
#     Context:
#     and our model into a pipeline, into a Q&A pipeline. So again, we get this pipeline from the Transformers library. So we come down here, do from Transformers, import pipeline. And now what we want to do is just initialize a pipeline object. So to do that, we just write pipeline. And then in here, what we need to add is a model type. So obviously, you can see up here, we have all of these different tasks. So summarization, text generation and so on. The Transformers library needs to understand, or this pipeline object needs to understand which one of those pipelines or functions we are intending to use. So to tell it that we want to do question answering, we just write question answering. And that basically sets the wrapper of the pipeline to handle question answering formats. So we'll see our input and for our input, we will be passing a context and a question. So we'll see that it will convert into the right structure that we need for question answering, which is the CLS context separator, question separator and padding. It will convert into that, feed it into our tokenizer.
    
#     ---
    
#     and our model into a pipeline, into a Q&A pipeline. So again, we get this pipeline from the Transformers library. So we come down here, do from Transformers, import pipeline. And now what we want to do is just initialize a pipeline object. So to do that, we just write pipeline. And then in here, what we need to add is a model type. So obviously, you can see up here, we have all of these different tasks. So summarization, text generation and so on. The Transformers library needs to understand, or this pipeline object needs to understand which one of those pipelines or functions we are intending to use. So to tell it that we want to do question answering, we just write question answering. And that basically sets the wrapper of the pipeline to handle question answering formats. So we'll see our input and for our input, we will be passing a context and a question. So we'll see that it will convert into the right structure that we need for question answering, which is the CLS context separator, question separator and padding. It will convert into that, feed it into our tokenizer.
    
#     ---
    
#     And on the Hugging Face website, we just want to go over to the Models page. So it's here. Okay and on this Models page, the thing that we want to be looking at is this question and answering task. So here we have all these tasks because when you're working with transformers, they can work with a lot of different things. Text summarization, text classification, generation, loads of different things. But what we want to do is question answering. So we click on here and this filters all of the models that are available to us just purely for question and answering. So this is the sort of power of using the Hugging Face Transformers library. It already has all these pre-trained models that we can just download and start using. Now, when you want to go and apply these to specific use cases, you probably want to fine tune it, which means you want to train it a little bit more than what it is already trained. But for actually getting used to how all of this works, all you need to do is download this model and start asking questions and understanding how everything is actually functioning. So obviously there's a lot of models here. We've got 262 models for question answering,
    
#     Question: How to build next-level Q&A with OpenAI
#     Answer:
#     The context provided does not specifically address building a next-level Q&A system with OpenAI. However, it does explain how to set up a question-answering pipeline using the Transformers library, which can be a foundational step in creating a Q&A system. To build a more advanced Q&A system with OpenAI, you might consider the following steps:
    
#     1. **Choose a Model**: Select a suitable pre-trained model from the Hugging Face Models page that is designed for question answering.
    
#     2. **Initialize the Pipeline**: Use the Transformers library to initialize a question-answering pipeline by importing the necessary components and specifying the model type.
    
#     3. **Fine-tuning**: If needed, fine-tune the model on a specific dataset to improve its performance for your particular use case.
    
#     4. **Input Structure**: Prepare your input in the required format, which includes a context and a question, ensuring it is structured correctly for the model.
    
#     5. **Integration**: Integrate the model into your application, allowing users to input questions and receive answers based on the provided context.
    
#     6. **Enhancements**: Consider adding features such as user feedback loops, context retrieval from databases, or multi-turn conversations to enhance the Q&A experience.
    
#     By following these steps and leveraging the capabilities of the Transformers library and OpenAI models, you can build a sophisticated Q&A system.
    
#     Sources:
#     How to Build Q&A Models in Python (Transformers): https://youtu.be/scJsty_DR3o
#     How to Build Q&A Models in Python (Transformers): https://youtu.be/scJsty_DR3o
#     How to Build Q&A Models in Python (Transformers): https://youtu.be/scJsty_DR3o