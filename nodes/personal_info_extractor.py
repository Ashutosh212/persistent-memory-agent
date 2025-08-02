from dotenv import load_dotenv

from typing import Literal, List
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from schema.state import AgentState

from utils.prompts import info_extractor_prompt

load_dotenv()

class Entities(BaseModel):
    entities: List[str] = Field(..., description="List of entities the user wants to add to memory")

def personal_info_extractor(state: AgentState) -> AgentState:
    """
    Extracts personal info from the user's message using few-shot examples.
    """
    message = state["messages"][-1].content

    prompt = info_extractor_prompt + message
   
    llm = ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=20)
    structured_llm = llm.with_structured_output(Entities)

    try:
        response = structured_llm.invoke(prompt)
        # print(response)
        extracted = response.entities
    except Exception:
        extracted = []

    # print("Entities to Add:", extracted)


    state["personal_info_extracted"] = extracted
    # print(state["personal_info_extracted"])
    return state