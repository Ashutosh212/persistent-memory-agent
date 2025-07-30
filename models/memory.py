from pydantic import BaseModel, Field
import json

class Memory(BaseModel):
    content: str = Field(description="The main content of the memory. For example: User expressed interest in learning about French.")

class MemoryCollection(BaseModel):
    memories: list[Memory] = Field(description="A list of memories about the user.")

if __name__ == "__main__":
    # Example: LLM's structured output
    structured_output = MemoryCollection(memories=[
        Memory(content="User expressed interest in learning about French."),
    ])

    json_data = structured_output.model_dump_json()

    with open("memories.json", "w") as f:
        f.write(json_data)


    with open("memories.json", "r") as f:
        loaded_json = f.read()

    memory_dict = json.loads(loaded_json)
    memory_collection = MemoryCollection.model_validate_json(loaded_json)

    print(type(memory_dict["memories"][0]))
    print(memory_dict["memories"][1]["content"])

    memory_str = " ".join([mem['content'] for mem in memory_dict["memories"]])
    print(memory_str)