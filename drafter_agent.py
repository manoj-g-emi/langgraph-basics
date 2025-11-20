from typing import TypedDict, Sequence, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import os


openai_endpoint = 'https://learningoai-si.openai.azure.com/'#os.getenv("openai_endpoint", "https://your-resource-name.openai.azure.com/")
openai_apikey = os.getenv("openai_apikey", "{your-api-key}")
#openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview
print(f"Using OpenAI Endpoint: {openai_endpoint}")
    
    
document_content = ""
#alternative to injected state

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    
@tool
def update(content: str) -> str:
    """Update the document with new content."""
    global document_content
    document_content= content
    return f"Document updated and content is \n{content}"


@tool
def save(filename: str) -> str:
    """Save the document to a file and finish the process."""
    global document_content
    
    if not filename.endswith(".txt"):
        filename += ".txt"
        
    try:
        with open(filename, "w") as f:
            f.write(document_content)
        print(f"Document saved to {filename}.")
        return f"Document saved to {filename}."
    except Exception as e:
        print(f"Failed to save document: {e}")
        return f"Failed to save document: {e}"


tools = [update, save]


llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    azure_endpoint=openai_endpoint,
    api_version="2025-01-01-preview",
    api_key=openai_apikey).bind_tools(tools)


def our_agent(state: AgentState) -> AgentState:
    system_message = SystemMessage(content=f"""
                                   You are Drafter, a helpful AI writing assistant that helps users draft update and save the documents.
                                   
                                   - If the user wants to update or modify content, use the 'update' tool with the new content.
                                   - If the user wants to save the document, use the 'save' tool with the desired filename.
                                   - Make sure to always show the current document state after modifications.
                                   
                                   The current document content is:
                                   {document_content}
                                   """)
    if not state["messages"]:
        user_input = "Hello, what would you like to do?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"User Input: {user_input}")
        user_message = HumanMessage(content=user_input)
        
    all_messages = [system_message] + list(state["messages"]) + [user_message]
    
    response = llm.invoke(input=all_messages)
    
    print(f"\nAI Response: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f" Using Tools: {[call['name'] for call in response.tool_calls]}")
        
    return {"messages": list(state["messages"]) + [user_message, response]}



def should_continue(state: AgentState) -> str:
    """Decide whether to continue or stop based on the recent AI message."""
    
    messages = state["messages"]
    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end"
    
    return "continue"


def print_messages(messages):
    """Function I made to print messages nicely."""
    if not messages:
        print("No messages yet.")
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"[ToolMessage] {message.content}")
            
            
graph = StateGraph(AgentState)
graph.add_node('our_agent', our_agent)

tool_node = ToolNode(tools)
graph.add_node('tool_node', tool_node)

graph.set_entry_point('our_agent')
graph.add_edge('our_agent', 'tool_node')
graph.add_conditional_edges('tool_node', should_continue, {
    'continue': 'our_agent',
    'end': END
})

app = graph.compile()

def run_document_agent():
    print("\n============DRAFTER AGENT STARTED============\n")
    
    state = {"messages":[]}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
            
    print("\n============DRAFTER AGENT ENDED============\n")
    
    
if __name__ == "__main__":
    run_document_agent()