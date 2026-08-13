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
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

load_dotenv()  # Load environment variables from .env file

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # The conversation history, which can include messages from the human, AI, system, and tools.

# --- ROCm/CUDA device check ---
# ROCm exposes itself to PyTorch through the same torch.cuda API as NVIDIA CUDA,
# so torch.cuda.is_available() / torch.cuda.get_device_name() work as-is on your RX 7900 XT.
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU detected: {device_name} ({total_vram_gb:.1f} GB VRAM)")
else:
    print("WARNING: No GPU detected by torch — falling back to CPU. Check your ROCm/torch install.")

@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a + b

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

tools = [add, subtract, multiply]

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

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("llm_agent", model_call)

tool_node = ToolNode(tools=tools, name="tool_node")
graph.add_node(tool_node)

graph.set_entrypoint("llm_agent")

graph.add_conditional_edges(
    "llm_agent",
    should_continue,
    {
        "continue": "tool_node",
        "end": END
    }
)

graph.add_edge("tool_node", "llm_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))