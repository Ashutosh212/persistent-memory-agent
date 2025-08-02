from typing import Literal
from schema.state import AgentState

def personal_info_router(state: AgentState) -> Literal["classify_add_or_delete", "retrieve_memories"]:
    """
    If personal info is detected, route to the node that classifies whether to add or delete it.
    Otherwise, route to retrieving memories.
    """
    if state["personal_info_detected"].lower() == "yes":
        return "classify_add_or_delete"
    return "retrieve_memories"

def route_add_or_delete(state: AgentState) -> Literal["extract_delete_entity", "personal_info_extractor"]:
    """
    If user intends to delete info, route to extract_delete_entity.
    Otherwise, route to personal_info_extractor for new additions.
    """
    if state["delete_request"].lower() == "yes":
        return "extract_delete_entity"
    return "personal_info_extractor"

def personal_info_deduper_router(state: AgentState) -> Literal["personal_info_storer", "retrieve_memories"]:
    """
    If 'Yes', store the new info. Otherwise skip storing.
    """
    if state["new_info"].lower() == "yes":
        return "personal_info_storer"
    return "retrieve_memories"