from dotenv import load_dotenv

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from schema.state import AgentState

load_dotenv()

def call_model(state: AgentState) -> AgentState:
    """
    Final LLM call that uses the collected memories in a SystemMessage.
    """
    personal_info = state.get("collected_memories", "")
    system_msg = SystemMessage(
        content=f"You are a helpful assistant. The user has shared these personal details:\n{personal_info}"
    )
    all_messages = [system_msg] + list(state["messages"])
    llm = ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=50)
    response = llm.invoke(all_messages)
    state["messages"] = state["messages"] + [response]
    return state