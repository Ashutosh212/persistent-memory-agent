from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

import json
import sys

sys.path.append(".")
from schema.state import AgentState
from utils.memory_store import MemoryStore
from config import key
from dotenv import load_dotenv
load_dotenv()



from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

store = MemoryStore()
def forget_logic(state: AgentState) -> AgentState:
    entities = [item for item in state['personal_info_extracted']]
    USER_ID = state['USER_ID']
    namespace = ("user", USER_ID)
    # key = "semantic_memory"
    print(f"Deleting these entities: {entities}")
    store.delete(namespace, key, entities)


def personal_info_storer(state: AgentState) -> AgentState:
    """
    Stores the new personal info in memory if it exists.
    """
    extracted = state.get("personal_info_extracted")
    USER_ID = state['USER_ID']
    if extracted:
        namespace = ("user", USER_ID)
        store.put(namespace, key, {"text": extracted})
    return state

def retrieve_memories(state: AgentState) -> AgentState:
    """
    Retrieves all personal info from the store and aggregates into 'collected_memories'.
    """
    USER_ID = state['USER_ID']
    results = store.get(("user", USER_ID), key)
    memory_strs = [doc for doc in results]
    state["collected_memories"] = "\n".join(memory_strs)
    return state

if __name__=="__main__":
    config = {"configurable": {"thread_id": "1", "user_id": "1"}}
    input_messages = [HumanMessage(content="Hi, I want to visit Japan next month.")]
    input = {"messages": input_messages}
    # call_model(input, config)
    # write_memory(input, config)
