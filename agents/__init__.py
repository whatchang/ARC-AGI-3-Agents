from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .templates.langgraph import LangGraph
from .templates.langgraph_functional_agent import LangGraphFunc, LangGraphTextOnly
from .templates.langgraph_random_agent import LangGraphRandom
from .templates.llm_agents import LLM, FastLLM, GuidedLLM, ReasoningLLM
from .templates.random_agent import Random
from .templates.reasoning_agent import ReasoningAgent
from .templates.smolagents import SmolCodingAgent, SmolVisionAgent

# Import custom agents
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from custom_agents.action import Action
from custom_agents.naive_cl import NaiveCL
from custom_agents.hope_cl import HopeCL
from custom_agents.meta_cl import MetaCL

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

# update the agent dictionary to include subclasses of LLM class
AVAILABLE_AGENTS["reasoningagent"] = ReasoningAgent

# Add custom CL agents
AVAILABLE_AGENTS["action"] = Action
AVAILABLE_AGENTS["naivecl"] = NaiveCL
AVAILABLE_AGENTS["hopecl"] = HopeCL
AVAILABLE_AGENTS["metacl"] = MetaCL

__all__ = [
    "Swarm",
    "Random",
    "LangGraph",
    "LangGraphFunc",
    "LangGraphTextOnly",
    "LangGraphRandom",
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "ReasoningAgent",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
]
