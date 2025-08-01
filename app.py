import streamlit as st
import os
import json

st.set_page_config(layout="wide")  # Make full-screen width available

user_list = ["1", "2", "3"]

user = st.text_input(
    "User ID?",
    placeholder="Example: 1, 2, 3",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_message" not in st.session_state:
    st.session_state.system_message = "You are a helpful assistant."

# Use full width and set equal height for columns
chat_col, sys_col = st.columns([2, 1])

with chat_col:
    st.markdown("## Chat")
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            role = "Human" if msg["role"] == "user" else "AI"
            st.markdown(f"**{role}:** {msg['content']}")

    st.markdown("---")
    user_input = st.text_input("Your prompt:", key="user_input")
    if st.button("Send") and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Simulated AI response (replace with real model call)
        ai_response = f"Echo: {user_input}"
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

data_path = f"/home/ashu/Projects/persistent-memory-agent/memory_store_database/user_{user}/semantic_memory_meta.json"

with sys_col:
    # st.markdown("## System Message")
    
    # st.text_area("System Prompt", value=st.session_state.system_message, height=150, key="system_prompt")

    st.markdown("### Semantic Memory")
    
    if os.path.exists(data_path):
        try:
            with open(data_path, "r") as f:
                memory_data = json.load(f)
            for item in memory_data:
                st.markdown(f"- {item.get('data', 'Missing data')}")
        except Exception as e:
            st.error(f"Failed to load memory: {e}")
    else:
        st.warning("Semantic memory file not found.")


# Optional: Custom CSS to stretch height
# st.markdown("""
#     <style>
#     .block-container {
#         padding-top: 1rem;
#         padding-bottom: 1rem;
#     }
#     textarea, input, button {
#         font-size: 16px !important;
#     }
#     </style>
# """, unsafe_allow_html=True)
