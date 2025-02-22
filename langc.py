import httpx, asyncio
import openai
from airtable_functions import fetch_airtable_record, update_airtable_record, update_airtable_match_record
import json
import re
import datetime
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma, InMemoryVectorStore
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


persist_parent_directory = 'docs/chroma/'
#############
# 
# V5 ASSISTANT CREATE: CREATE LANGCHAIN VECTOR STORE
# Returns None if failed, or Assistant_ID if successful
#############  
async def v5_assistant_create(company_record, settings):
    # Collect info of the fields
    attachments = company_record["fields"].get("Attachments")
    one_line_pitch = company_record["fields"].get("One Line Pitch")
    company_name = company_record["fields"].get("Company Name")
    solution = company_record["fields"].get("What is your company's solution to this problem?")
    problem = company_record["fields"].get("What is the problem that your company is addressing?")
    one_line_pitch = company_record["fields"].get("One Line Pitch")
    
    ## Currently only one file is supported: 
    if attachments and len(attachments) > 0:
        fileUrl = attachments[0]['url']  # only one file 
        fileName = attachments[0]['filename']
        print(f"fileUrl: {fileUrl}")
    else:
        print("Could not find Attachment")
        return None
    
    fields_to_add = f"<One Line Pitch>: {one_line_pitch}.\n<Problem that the company is addressing>: {problem}\n.<Company's solution to this problem>: {solution}."
    additional_fields = [fields_to_add]
    pages = await v5_load_document(fileUrl, fileName, additional_fields)
    if not pages:
        print("Could not load document")
        return None
    print(f"Number of pages: {len(pages)}")

    # Print Pages:
    for page in pages:
        if page.metadata:
            print(f"Metadata: {page.metadata}")
        print(f"Page#: {page.metadata.get('page', " No Page Number")}")  # PPTX does not have pagenumber
        if page.page_content:
            pass
            #print(f"Content: {page.page_content}")
    
    await write_content_file(pages, company_name)

    ### SPITTING TEXT INTO CHUNKS: is this required? 
    # text_splitter = RecursiveCharacterTextSplitter(
    # chunk_size = 100,
    # chunk_overlap  = 0,
    # length_function = len,
    # )
    # chunks = text_splitter.split(pages[0].page_content[0])
    # DOES IT CHUNK IN PAGES? 
    # Vector search over PDFs
    embedding = OpenAIEmbeddings(api_key=settings.openai_api_key)
    #vector_store = InMemoryVectorStore.from_documents(pages, embedding)
    persist_directory = persist_parent_directory+company_name

    print(f"Persisting to: {persist_directory}")
    vectordb = Chroma.from_documents(
        documents=pages,
        embedding=embedding,
        persist_directory=persist_directory
        )

    ### Similarity search is done under the hood: REQUIRED??
    # docs = vector_store.similarity_search("Approval for EIS/SEIS", k=2)
    # for doc in docs:
    #     print(f'Page {doc.metadata["page"]}: {doc.page_content[:200]}\n')
    
    # AT THIS POINT, WE HAVE THE VECTOR STORE READY - WE MAY SAVE IT IN THE FILESYSTEM. 
    #vectordb.persist()  
    return vectordb
####
#
# V5 Retrieve all criterias response from vector store (pitch deck)
#
####
async def v5_retrieve_criteria_from_company(investor_criterias_list, vector_db, settings, model):
       
    updated_criterias_list = []
    retrieve_cost = 0
    for criteria in investor_criterias_list:
        # Only extract information if the criteria text is not empty:
        if not criteria['Preference']:
            print("Criteria Text is empty: ", criteria['Question'])
            continue
        completion_answer, cost = await v5_process_question(criteria['Question'], vector_db, model)  #vector_store could be a vector_db
        retrieve_cost += cost
    # Create a new dictionary with the existing data and the new completion_answer
        updated_criteria = {
            "criteria_name": criteria['Question'],
            "investor_preference": criteria['Preference'],
            "weight":  criteria['Weight'],
            "company_retrieve": completion_answer
            }
        # Append the new dictionary to the updated_criterias_list
        updated_criterias_list.append(updated_criteria)

    return updated_criterias_list, retrieve_cost
#####
#
# V5 Process individual Question from vector store
#
#####
async def v5_process_question(criteria_text, vector_db, model):
    client = openai.OpenAI()
    print(f"Starting process_question: {criteria_text}, model: {model}")    
    ### V5 

    system_retrieve_prompt = (
        "You have access to company investment pitch information in the <context> below delimited by triple backtick \
        You answer questions about the company and the investment proposition.\
        You provide answers based both on your provided <context> and general LLM knowledge.\
        Respond in few sentences in a very syntetic way, respond as if you are stating the facts of the company. \
        Round size is the amount of money that the company is looking (seeking) to raise. \
        Pre-money valuation (PMV) is the valuation of the company before the investment.\
        Use factual statements and keep the answer concise.\
        If you don't know the answer, don't make something up: just say you don't know. \
        Output the answer as a valid string, with double quotes. \
        <Context>: ```{context}```"
    )

    # user_prompt = f"Please retrieve the following information from your file about the company: {input}\
    #             respond in few sentences in a very syntetic way, respond as if you are stating the facts of the company about {input}\
    #                 Round size is the amount of money that the company is looking (seeking) to raise. \
    #                 Pre-money valuation (PMV) is the valuation of the company before the investment.\
    #                 It could be calculated from the round size and the percentage of the company that the investor is buying."
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_retrieve_prompt),
            ("human", "{input}"),
#            ("user", user_prompt),
#            ("human", user_prompt),
        ]
    )

    print("=========Prompt: ", prompt)

    #retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 2}) 
    retriever = vector_db.as_retriever() 
    llm = ChatOpenAI(model=model,temperature=0)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    response = chain.invoke({"input": criteria_text})
    print("Context:")
    print(response["context"])
    return response['answer'], 0

    # cost = calculate_cost_tokens(run.usage, model)
    # if run.status == "completed":
    #     messages = client.beta.threads.messages.list(thread_id=thread_id)


########
# LANGCHAIN Load Document
########
async def v5_load_document(fileUrl, fileName, additional_fields=None):
    
    
    from langchain_community.document_loaders import PyPDFLoader,  UnstructuredPowerPointLoader, TextLoader, Docx2txtLoader, PyMuPDFLoader
    # Download the file
    # file_path = download_file(fileUrl)  ???
    

    if fileName.endswith('.pdf'):
        #loader = PyPDFLoader(file_path = fileUrl)
        loader = PyMuPDFLoader(file_path = fileUrl)
    elif fileName.endswith('.pptx'):
        local_file = await download_file(fileUrl)
        loader = UnstructuredPowerPointLoader(local_file)
    elif fileName.endswith('.docx'):
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {fileUrl}')
        loader = Docx2txtLoader(fileUrl)
    elif fileName.endswith('.txt'):
        from langchain.document_loaders import TextLoader
        print(f'Loading {fileUrl}')
        loader = TextLoader(fileUrl)
    else:
        raise ValueError("Unsupported file type")

    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    
    additional_doc = Document(id="999", page_content = additional_fields[0], metadata ={"page": "999"})
    pages.append(additional_doc)

    return pages

########
#
# Download a file from a URL and save it locally.
#
########
async def download_file(url):
    """Download a file from a URL and save it locally."""
    local_filename = "docs/"+url.split('/')[-1]+'.pptx'
    print(f"Downloading file to {local_filename}")
    #response = requests.get(url)
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        file_content = response.content
        response.raise_for_status()
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        return local_filename

########
# 
# Write extracted Document
#
########
async def write_content_file(pages, company_name):

    invalid_chars = '/\\?%*:|"<>'
    translation_table = str.maketrans('', '', invalid_chars) # No replacing, only deleting
    safe_filename = company_name.translate(translation_table)  # Clean filename
    
    address = 'docs/'+'pymu'+safe_filename+'.md'
    if os.path.exists(address):
        print(f"File '{address}' already exists. Skipping creation and writing.")
        return 
    # Open a Markdown file for writing
    invalid_chars = '/\\?%*:|"<>'
    translation_table = str.maketrans('', '', invalid_chars)
    filename = address
    safe_filename = filename.translate(translation_table)
    with open(address, 'w', encoding='utf-8') as file:
        for page in pages:
            # Write page metadata and content to the file using Markdown formatting
            file.write(f"## Metadata\n")
            file.write(f"- Page#: {page.metadata.get('page', " No Page Number")}\n")
            #file.write(f"- Title: {page.metadata['title']}\n")   
            # Additional metadata can be added similarly:
            # file.write(f"- OtherMetadata: {page.metadata['otherKey']}\n")
            file.write(f"## Content\n")
            file.write(f"{page.page_content}\n\n")
