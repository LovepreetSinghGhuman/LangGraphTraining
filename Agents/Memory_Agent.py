import os
import re
import warnings
from typing import List, TypedDict, Union

# --- Quiet down noisy library logging/progress bars (must run before the
# relevant libraries are imported / models are loaded) ---
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"     # kills "Loading weights: 100%|..." bars
os.environ["TRANSFORMERS_VERBOSITY"] = "error"        # only show real errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"        # silences the tokenizer fork warning

warnings.filterwarnings("ignore")                     # silences the deprecation / max_new_tokens warnings

import torch
from dotenv import load_dotenv # Store and load environment variables from a .env file
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

load_dotenv()  # Load environment variables from .env file

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# --- ROCm/CUDA device check ---
# ROCm exposes itself to PyTorch through the same torch.cuda API as NVIDIA CUDA,
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU detected: {device_name} ({total_vram_gb:.1f} GB VRAM)")
else:
    print("WARNING: No GPU detected by torch — falling back to CPU. Check your ROCm/torch install.")

# Build the underlying text-generation pipeline first.
pipeline_llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-0.6B",
    task="text-generation",
    device=0 if torch.cuda.is_available() else -1,  # pins the whole model to GPU index 0 (-1 = CPU)
    pipeline_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 512,
        "do_sample": True,
        "return_full_text": False,  # stops the prompt/template being echoed back in the output
    },
)

# ChatHuggingFace wraps that pipeline to give it the chat-model interface.
llm = ChatHuggingFace(llm=pipeline_llm)

def strip_thinking(text: str) -> str:
    """Qwen3 emits a <think>...</think> reasoning block before its real answer.
    Remove it (and any stray leading/trailing whitespace it leaves behind)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def process(state: AgentState) -> AgentState:
    # Process the state and generate a response
    response = llm.invoke(state["messages"])

    # Print the AI's response to the console (reasoning block removed)
    print(f"\nAI: {strip_thinking(response.content)}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

print("Type 'exit' or 'quit' to end the conversation.")
print(f"Using device: {'cuda:0 (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'cpu'}")\

user_input = input("Enter: ")
while user_input.lower() not in ["exit", "quit"]:
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]  # Update the conversation history with the latest messages
    
    # Get the next user input
    user_input = input("Enter: ")

with open("conversation_history.txt", "w", encoding="utf-8") as file:
    file.write("Conversation History:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"Human: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")

    file.write("\nEnd of Conversation\n")

print("Conversation history saved to 'conversation_history.txt'.")