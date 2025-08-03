from dotenv import load_dotenv
import sys
import os
sys.path.append(os.path.abspath(".."))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from utils.memory_manager import personal_info_storer, forget_logic, retrieve_memories
from utils.route import personal_info_deduper_router, personal_info_router, route_add_or_delete
from utils.memory_store import MemoryStore

from schema.state import AgentState

from nodes.call_model import call_model
from nodes.classifiers import classify_add_or_delete, personal_info_duplicate_classifier, personal_info_classifier
from nodes.extract_delete_entity import extract_delete_entity
from nodes.personal_info_extractor import personal_info_extractor

def model(USER_ID: str, text: str):

    workflow = StateGraph(AgentState)
    workflow.add_node("personal_info_classifier", personal_info_classifier)
    workflow.add_node("classify_add_or_delete", classify_add_or_delete)
    workflow.add_node("extract_delete_entity", extract_delete_entity)
    workflow.add_node("forget_logic", forget_logic)
    workflow.add_node("personal_info_extractor", personal_info_extractor)
    workflow.add_node("personal_info_duplicate_classifier", personal_info_duplicate_classifier)
    workflow.add_node("personal_info_storer", personal_info_storer)
    workflow.add_node("retrieve_memories", retrieve_memories)
    # workflow.add_node("log_personal_memory", log_personal_memory)
    workflow.add_node("call_model", call_model)

    workflow.add_edge(START, "personal_info_classifier")
    workflow.add_conditional_edges(
        "personal_info_classifier",
        personal_info_router,
        {
            "classify_add_or_delete": "classify_add_or_delete",
            "retrieve_memories": "retrieve_memories",
        },
    )

    workflow.add_conditional_edges(
        "classify_add_or_delete",
        route_add_or_delete,
        {
            "extract_delete_entity": "extract_delete_entity",
            "personal_info_extractor": "personal_info_extractor"
        }
    )
    workflow.add_edge("personal_info_extractor", "personal_info_duplicate_classifier")
    workflow.add_edge("extract_delete_entity","forget_logic")
    workflow.add_edge("forget_logic", "retrieve_memories")
    workflow.add_conditional_edges(
        "personal_info_duplicate_classifier",
        personal_info_deduper_router,
        {
            "personal_info_storer": "personal_info_storer",
            "retrieve_memories": "retrieve_memories",
        },
    )

    workflow.add_edge("personal_info_storer", "retrieve_memories")
    # workflow.add_edge("retrieve_memories", "log_personal_memory")
    # workflow.add_edge("log_personal_memory", "call_model")
    workflow.add_edge("retrieve_memories", "call_model")
    workflow.add_edge("call_model", END)


    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    # input_messages = [HumanMessage(content="Hey, I hate eating cookies.")]
    input_messages = [HumanMessage(content=text)]

    # config = {"configurable": {"thread_id": "1", "user_id": "1"}}

    config = {"configurable": {"thread_id": "1"}}
    
    final_state = graph.invoke({"USER_ID": USER_ID, "messages": input_messages}, config)
    # print(final_state["messages"][-1].content) 
    return final_state["messages"][-1].content