"""LangGraph Agent"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from supabase.client import Client, create_client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import START, StateGraph, MessagesState
from langchain_community.document_loaders import ArxivLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    
    """
    Multiply two numbers -
    
    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    
    """
    Add two numbers - 
    
    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    
    """
    Subtract two numbers -
    
    Args:
        a: first int
        b: second int
    """
    return a - b

@tool
def divide(a: int, b: int) -> int:
    
    """
    Divide two numbers -
    
    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
        
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    
    """
    Get the modulus of two numbers -
    
    Args:
        a: first int
        b: second int
    """
    return a % b

@tool
def wiki_search(query: str) -> str:
    
    """
    Search Wikipedia for a query and return maximum 2 results -
    
    Args:
        query: The search query.
    """
    
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    
    formatted_search_docs = "\n\n---\n\n".join(
        [f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
        for doc in search_docs])
    
    return {"wiki_results": formatted_search_docs}

@tool
def web_search(query: str) -> str:
    
    """
    Search Tavily for a query and return maximum 3 results -
    
    Args:
        query: The search query.
    """
    
    search_docs = TavilySearchResults(max_results=3).invoke(query=query)
    
    formatted_search_docs = "\n\n---\n\n".join(
        [f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
        for doc in search_docs])
    
    return {"web_results": formatted_search_docs}

@tool
def arvix_search(query: str) -> str:
    
    """
    Search Arxiv for a query and return maximum 3 result -
    
    Args:
        query: The search query.
    """
    
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    
    formatted_search_docs = "\n\n---\n\n".join(
        [f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
        for doc in search_docs])
    
    return {"arvix_results": formatted_search_docs}

with open("PROMPT.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# System message
sys_msg = SystemMessage(content=system_prompt)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

supabase: Client = create_client(os.environ.get("SUPABASE_URL"), 
                                 os.environ.get("SUPABASE_SERVICE_KEY"))

vector_store = SupabaseVectorStore(client=supabase,
                                   embedding= embeddings,
                                   table_name="documents",
                                   query_name="match_documents_langchain_v2")

create_retriever_tool = create_retriever_tool(retriever=vector_store.as_retriever(),
                                              name="Question Search",
                                              description="A tool to retrieve similar questions from a vector store.")

tools = [add,
         divide,
         modulus,
         multiply,
         subtract,
         web_search,
         wiki_search,
         arvix_search]

def build_graph(provider: str = "groq"):
    
    """Build the graph"""

    if provider == "google":
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        
    elif provider == "groq":
        llm = ChatGroq(model="qwen-qwq-32b", temperature=0) 
        
    elif provider == "huggingface":
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                url="https://api-inference.huggingface.co/models/Meta-DeepLearning/llama-2-7b-chat-hf",
                temperature=0))
        
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")
    
    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: MessagesState):
        
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    def retriever(state: MessagesState):
        
        """Retriever node"""
        similar_question = vector_store.similarity_search(state["messages"][0].content)
        
        example_msg = HumanMessage(content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}")
        return {"messages": [sys_msg] + state["messages"] + [example_msg]}

    builder = StateGraph(MessagesState)
    
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()

if __name__ == "__main__":
    
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    
    graph = build_graph(provider="groq")
    
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    
    for m in messages["messages"]:
        m.pretty_print()