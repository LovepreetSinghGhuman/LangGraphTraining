from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
from langgraph import StateGraph, START, END
from dotenv import load_dotenv # Store and load environment variables from a .env file

load_dotenv()  # Load environment variables from .env file

