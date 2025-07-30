from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import Image, display

from services.memory_manager import call_model, write_memory

user_id = "1"

thread_id = "1"


builder = StateGraph(MessagesState)

builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)

builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)

# across_memory_thread = InMemoryStore()

# within_memory_thread = MemorySaver()

# graph = builder.compile(checkpointer=within_memory_thread, store=across_memory_thread)
graph = builder.compile()


with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

input_messages = [HumanMessage(content="Also, i want to eat tofu there.")]

config = {"configurable": {"thread_id": "1", "user_id": "1"}}

for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    print(chunk["messages"][-1])
    print("----")