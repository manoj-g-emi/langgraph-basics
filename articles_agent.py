from typing import Sequence, TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader 
from langchain_chroma import Chroma
from langchain_core.documents import Document   


openai_endpoint = 'https://learningoai-si.openai.azure.com/'#os.getenv("openai_endpoint", "https://your-resource-name.openai.azure.com/")
openai_apikey = os.getenv("openai_apikey", "{your-api-key}")
#openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview
print(f"Using OpenAI Endpoint: {openai_endpoint}")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

articles_path = "Articles/"
    
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    azure_endpoint=openai_endpoint,
    api_version="2025-01-01-preview",
    api_key=openai_apikey,
    temperature=0)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    azure_endpoint=openai_endpoint,
    api_version="2023-05-15",
    api_key=openai_apikey)

docs = []
for filename in os.listdir(articles_path):
    if filename.endswith(".txt"):
        print(f"Loading document: {filename}")
        file_path = os.path.join(articles_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            docs.append(Document(page_content=content))


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(docs)

vector_db_directory = "faiss_index"
collection_name = "pdf_collection"

if not os.path.exists(vector_db_directory):
    os.makedirs(vector_db_directory)
    

import time
embedded_docs = []
for i, doc in enumerate(docs):
    # Embed one document at a time, with 500ms delay
    for attempt in range(5):
        try:
            # Chroma expects a list of docs, so wrap doc in a list
            vector_store = Chroma.from_documents([doc], embeddings, persist_directory=vector_db_directory, collection_name=collection_name)
            embedded_docs.append(doc)
            print(f"Embedded doc {i+1}/{len(docs)} successfully.")
            break
        except Exception as e:
            if "RateLimitReached" in str(e):
                wait_time = 60 * (attempt + 1)
                print(f"Rate limit reached. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to create or save vector store: {e}")
    else:
        raise RuntimeError("Exceeded maximum retries due to rate limits.")
    time.sleep(0.1)  # 500ms delay between embeddings


# Now we create our retriever 
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)


# search_results = retriever.invoke("What is the name of the person in the CV?")

# print(search_results)

@tool
def retriever_tool(query: str) -> str:
    """Retrieve relevant document content based on the query."""
    results = retriever.invoke(query)
    
    if not results:
        return "No relevant documents found."
    
    documents = []
    
    for i, doc in enumerate(results):
        print(f"\nDocument {i + 1} Content:\n{doc.page_content}\n")
        documents.append(f"Document {i + 1}:\n{doc.page_content}")
    return "\n\n".join(documents)

tools = [retriever_tool]

llm = llm.bind_tools(tools)


def should_continue(state: AgentState) -> bool:
    """Check if last message was tool call."""
    last_message = state["messages"][-1]
    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0

system_prompt = SystemMessage(content=f"""
                                   You are an intelligent AI assistant who answers questions about news articles based on the documents loaded into your knowledge base.
Use the retriever tool available to answer questions about the news articles. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
                                   """)

tool_dict = {tool.name: tool for tool in tools}

def llm_agent(state: AgentState) -> AgentState:
    """The main agent function to handle user queries about the news articles."""
        
    all_messages = [system_prompt] + list(state["messages"])
    response = llm.invoke(input=all_messages)
    
    # print(f"\nAI Response: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f" Using Tools: {[call['name'] for call in response.tool_calls]}")
        
    return {"messages": [response]}


def take_action(state: AgentState) -> AgentState:
    """Decide whether to call a tool or respond directly."""
    last_message = state["messages"][-1]
    result = []
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            print(f"\nInvoking tool: {tool_call['name']} with input: {tool_call['args'].get('query','No query provided')}")
            
            if not tool_call['name'] in tool_dict:
                tool_response = f"Tool {tool_call['name']} not found."
                print(tool_response)
            else:
                tool = tool_dict[tool_call['name']]
                tool_response = tool.invoke(tool_call['args'].get('query','No query provided')) # Unpack arguments for the tool call eg. query=...
                # print(f"Tool Response: {tool_response}")

            result.append(ToolMessage(content=tool_response, tool_call_id=tool_call['id'], tool_name=tool_call['name']))
    return {"messages": result}


graph = StateGraph(AgentState)
graph.add_node('llm_agent', llm_agent)
graph.add_node('take_action', take_action)
graph.add_edge(START, 'llm_agent')
graph.add_conditional_edges('llm_agent', should_continue, {
    True: 'take_action',
    False: END
})
graph.add_edge('take_action', 'llm_agent')

app = graph.compile()

def running_agent():
    print("============ RAG Agent ============")
    while True:
        user_input = input("\nWhat do you want to know about the articles? ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting the agent. Goodbye!")
            break
        result = app.invoke({"messages": [HumanMessage(content=user_input)]})
        
        
        
        print("===================================")
        print(f"\nAI Response: {result['messages'][-1].content}")
        
        
running_agent()