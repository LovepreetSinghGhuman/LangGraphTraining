import os
import warnings
from typing import Annotated, Sequence, TypedDict

# --- Quiet down noisy library logging/progress bars (must run before the
# relevant libraries are imported / models are loaded) ---
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"     # kills "Loading weights: 100%|..." bars
os.environ["TRANSFORMERS_VERBOSITY"] = "error"        # only show real errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"        # silences the tokenizer fork warning

warnings.filterwarnings("ignore")                     # silences the deprecation / max_new_tokens warnings

import torch
from dotenv import load_dotenv  # Store and load environment variables from a .env file
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

load_dotenv()  # Load environment variables from .env file

# --- ROCm/CUDA device check ---
# ROCm exposes itself to PyTorch through the same torch.cuda API as NVIDIA CUDA,
# so torch.cuda.is_available() / torch.cuda.get_device_name() work as-is on your RX 7900 XT.
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU detected: {device_name} ({total_vram_gb:.1f} GB VRAM)")
else:
    print("WARNING: No GPU detected by torch — falling back to CPU. Check your ROCm/torch install.")

# Global variable to hold the document content
document_content = ""
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # The conversation history, which can include messages from the human, AI, system, and tools.

@tool
def update(content: str) -> str:
    """This is an update function that updates the document content"""
    global document_content
    document_content = content
    return f"Document content updated to: {document_content}"

@tool
def save(filename: str) -> str:
    """This is a save function that saves the document content
    Arguments:
        filename: The name of the file to save the document content.
    """
    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"
    try:
        with open(filename, "w") as f:
            f.write(document_content)
    except Exception as e:
        return f"Error saving document content to {filename}: {str(e)}"
    
    return f"Document content saved to {filename}: {document_content}"

tools = [update, save]

# Build the underlying text-generation pipeline first (same pattern as Agent_Bot.py).
pipeline_llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    device=0 if torch.cuda.is_available() else -1,  # pins the whole model to GPU index 0 (-1 = CPU)
    model_kwargs={
        "torch_dtype": torch.bfloat16,  # ~14GB instead of ~28GB at fp32, should fit in 20GB VRAM
    },
    pipeline_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 512,
        "do_sample": True,
        "return_full_text": False,  # stops the prompt/template being echoed back in the output
    },
)

# ChatHuggingFace wraps that pipeline to give it the chat-model interface, including
# .bind_tools(), which raw transformers model objects don't have.
model = ChatHuggingFace(llm=pipeline_llm).bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end" # goes to the end edge which leads to the endpoint
        
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")


graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()