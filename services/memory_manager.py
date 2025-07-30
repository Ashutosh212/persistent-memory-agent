from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

import json
import sys

sys.path.append(".")
from models.memory import MemoryCollection

from dotenv import load_dotenv
load_dotenv()



from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user. 
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

def call_model(state: MessagesState, config: RunnableConfig):
    """Load the memory from the store and use to personlaize the response"""

    user_id = config["configurable"]["user_id"]

    namespace = (user_id, "memory")

    key = "memories"

    # Load the existing memory of the user with the user_id
    with open("memories.json", "r") as f:
        loaded_json = f.read()

    memory_dict = json.loads(loaded_json)
    existing_memory_content = " ".join([mem['content'] for mem in memory_dict["memories"]])

    # if existing_memory:
    #     # Value is a dictionary with a memory key
    #     existing_memory_content = existing_memory.value.get('memory')
    # else:
    #     existing_memory_content = "No existing memory found."

    # print(existing_memory_content)

    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=existing_memory_content)
    
    response = model.invoke([SystemMessage(content=system_msg)]+state["messages"])

    # print(response.content)

    return {"messages": response}

CREATE_MEMORY_INSTRUCTION = """You are collecting information about the user to personalize your responses.

USER INFORMATION FROM PREVIOUS INTERACTIONS:
{memory}

INSTRUCTIONS:
1. Carefully review the chat history below.
2. Identify new factual information about the user that is not already included above, such as:
   - Personal details (e.g., name, location)
   - Food preferences (e.g., likes, dislikes)
   - Interests and hobbies
   - Goals or future plans
3. Combine any new information with the existing memory to create a single, updated memory.
4. Format the updated memory as a clear, concise, bulleted list.
5. If any new information contradicts the existing memory, prefer the most recent version stated by the user.

Important: Only include information explicitly stated by the user. Do not make assumptions or inferences.

Based on the chat history below, please update the user information:
"""

def write_memory(state: MessagesState, config: RunnableConfig):

    """Reflect on the chat history and save a memory to the store."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve existing memory from the store
    namespace = (user_id, "memory")
        
    # Extract the memory
    with open("memories.json", "r") as f:
        loaded_json = f.read()

    memory_dict = json.loads(loaded_json)
    existing_memory_content = " ".join([mem['content'] for mem in memory_dict["memories"]])

    # Format the memory in the system prompt
    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory_content)

    model_structured_output = model.with_structured_output(MemoryCollection)
    new_memory = model_structured_output .invoke([SystemMessage(content=system_msg)]+state['messages'])

    # Overwrite the existing memory in the store 
    key = "user_memory"
    
    with open("memories.json", "w") as f:
        f.write(new_memory.model_dump_json(indent=2))

    # store.put(namespace, key, {"memory": new_memory.content})


if __name__=="__main__":
    config = {"configurable": {"thread_id": "1", "user_id": "1"}}
    input_messages = [HumanMessage(content="Hi, I want to visit Japan next month.")]
    input = {"messages": input_messages}
    # call_model(input, config)
    write_memory(input, config)
