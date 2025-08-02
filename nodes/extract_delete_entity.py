from dotenv import load_dotenv

from typing import Literal, List

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from schema.state import AgentState
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

system_prompt = """You are an assistant that extracts the names of tools, preferences, or technologies the user wants to forget or delete from memory. Only return the list of entities to remove."""

few_shot_examples = [
    {
        "input": "I no longer use Twitter.",
        "output": '["Twitter"]'
    },
    {
        "input": "Forget Notion and Todoist.",
        "output": '["Notion", "Todoist"]'
    },
    {
        "input": "I stopped liking pizza and burgers.",
        "output": '["pizza", "burgers"]'
    },
    {
        "input": "I do not use Figma, Zoom, or Slack anymore.",
        "output": '["Figma", "Zoom", "Slack"]'
    },
]
    
def format_prompt(user_message: str) -> List:
    few_shots = "\n".join(
        [f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in few_shot_examples]
    )
    return [
        SystemMessage(content=system_prompt + "\n\n" + few_shots),
        HumanMessage(content=f"Input: {user_message}\nOutput:")
    ]

class EntitiesToForget(BaseModel):
    entities: List[str] = Field(..., description="List of entities the user wants to delete or forget")

def extract_delete_entity(state: AgentState) -> AgentState:
    messages = state["messages"][-1].content
    user_input = messages # Get last user message content

    # System instructions with few-shot examples
    instruction = """
    You are an intelligent assistant that helps identify which pieces of personal information the user wants to delete from memory.

    Instructions:
    - Analyze the input and extract the names of entities, preferences, or facts the user wants the system to forget.
    - Return only a list of strings, each representing one such item.

    Examples:
    User: "Forget that I live in Delhi and that I work at Microsoft."
    Output: ["Location: Delhi", "Employer: Microsoft"]

    User: "Remove everything about my cat and my love for sushi."
    Output: ["Pet: cat", "Food Preference: sushi"]

    User: "Never store my email or my travel plans to Japan."
    Output: ["Email", "Travel Plan: Japan"]

    User: "Just chatting, nothing to delete."
    Output: []

    Now extract the entities from the following user input:
    """ + user_input

    # Initialize the structured model
    llm = ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=20)
    structured_llm = llm.with_structured_output(EntitiesToForget)

    try:
        response = structured_llm.invoke(instruction)
        extracted = response.entities
    except Exception:
        extracted = []

    # print("Entities to delete:", extracted)
    return {
        "personal_info_extracted": extracted
    }
