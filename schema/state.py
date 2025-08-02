from typing import Annotated, Literal, TypedDict, Sequence, Optional, List
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    USER_ID: str
    messages: Annotated[list[AnyMessage], add_messages]
    personal_info_detected: Literal["yes", "no"]
    delete_request: Literal["yes", "no"]
    personal_info_extracted: Optional[List[str]] # this stores entity for either to delete or add
    new_info: Optional[str]
    collected_memories: Optional[str]