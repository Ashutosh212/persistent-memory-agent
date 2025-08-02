from dotenv import load_dotenv
import os
import sys
from typing import Literal
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

sys.path.append(os.path.abspath(".."))

from schema.state import AgentState
from utils.memory_store import MemoryStore
# from config import USER_ID
load_dotenv()

class DeleteRequest(BaseModel):
    delete_request: Literal["yes", "no"] = Field(
        description="Return 'yes' if the user wants to delete or stop using something previously shared, otherwise 'no'."
    )

def classify_add_or_delete(state: AgentState) -> AgentState:
    """
    Classifies whether the personal information is a delete request or an addition.
    """
    last_message = state["messages"][-1].content
    # print(f"last message: {last_message}")

    system_prompt = """You are a classifier that decides whether a user's message indicates a request to delete or stop using something.

    Return "yes" if the user says they:
    - No longer use something
    - Have stopped liking a tool, app, or preference
    - Want to remove or undo a previously stated preference

    Return "no" if the user is sharing a new preference, habit, tool, or hobby.

    Examples:
    User: "I do not use Facebook anymore."
    Classifier: "yes"

    User: "I stopped liking pineapple on pizza."
    Classifier: "yes"

    User: "I enjoy hiking on weekends."
    Classifier: "no"

    User: "I love using Notion for productivity."
    Classifier: "no"
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", last_message),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=50)
    structured_llm = llm.with_structured_output(DeleteRequest)
    chain = prompt | structured_llm

    result = chain.invoke({"message": prompt})
    # print(f"Result: {result}")
    state["delete_request"] = result.delete_request
    return state


class InfoNoveltyGrade(BaseModel):
    score: Literal["yes", "no"] = Field()

def personal_info_duplicate_classifier(state: AgentState) -> AgentState:
    """
    Checks if the newly extracted info is already in the store or not.
    If 'Yes', it's new info. If 'No', it's a duplicate.
    """
    new_info = state.get("personal_info_extracted", "")
    USER_ID = state['USER_ID']
    namespace = ("user", USER_ID)
    key = "semantic_memory"
    store = MemoryStore()
    results = store.get(namespace, key)
    old_info_list = [doc for doc in results]

    system_msg = """You are a classifier that checks if the new personal info is already stored.
        If the new info adds anything new, respond 'Yes'. Otherwise 'No'."""

    old_info_str = "\n".join(old_info_list) if old_info_list else "No stored info so far."
    human_template = """New info:\n{new_info}\n
        Existing memory:\n{old_info}\n
        Answer ONLY 'Yes' if the new info is unique. Otherwise 'No'."""
    human_msg = human_template.format(new_info=new_info, old_info=old_info_str)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", "{human_msg}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=50).with_structured_output(InfoNoveltyGrade)
    chain = prompt | llm
    result = chain.invoke({"human_msg": human_msg})
    state["new_info"] = result.score.strip()
    return state


class ClassifyInformation(BaseModel):
    personal_info: Literal["yes", "no"] = Field(
        description="Indicates whether the information contains any personal user choices or preferences that could help an LLM provide more personalized responses over time for long-term memory use."
    )

def personal_info_classifier(state: AgentState) -> AgentState:
    """
    Classifies if the last user message contains personal info.
    Now also detects if user is forgetting or updating past preferences.
    """
    message = state["messages"][-1].content

    system_prompt = """You are a classifier that checks if a message contains personal info.

        Personal info includes:
        - Names (e.g., "John Smith")
        - Locations (e.g., "Berlin", "123 Main St")
        - Preferences or hobbies (e.g., "I love to code", "I prefer short replies")
        - Tools and Technologies Used  (e.g., "I mostly use PyTorch and FastAPI", "I am using Windows 11 now")
        - Dislikes or updates to previous preferences (e.g., "I do not use Magnet anymore", "I stopped liking Twitter")
        - Occupation

        Respond "yes" if the message reveals or updates any personal preferences, locations, names, etc.
        Respond "no" if it is a general query or lacks any personal information.

        Examples:
        User: "My name is Thomas, I live in Vancouver."
        Classifier: "yes"

        User: "I love pizza with extra cheese."
        Classifier: "yes"

        User: "I no longer use Discord."
        Classifier: "yes"

        User: "I do not like tea anymore."
        Classifier: "yes"

        User: "I still use Notion daily."
        Classifier: "yes"

        User: "What is the capital of France?"
        Classifier: "no"

        User: "This is great weather."
        Classifier: "no"

        User: "Hello, how are you?"
        Classifier: "no"
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{message}"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=50)
    structured_llm = llm.with_structured_output(ClassifyInformation)
    chain = prompt | structured_llm

    result = chain.invoke({"message": message})
    state["personal_info_detected"] = result.personal_info

    return state
