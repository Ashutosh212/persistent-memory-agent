# Initial Interface is creates using ChatGPT, furthur enhaced by me

import streamlit as st
import os
import json
from main_agent import model

st.set_page_config(layout="wide")  

USER_ID = st.text_input(
    "User ID",
    placeholder="Example: 1, 2, 3",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

chat_col, mem_col = st.columns([2, 1])

with chat_col:
    st.markdown("### 💬 Chat")
    
    with st.container(height=300):
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown("#### User")
                st.info(f"{msg['content']}")
            else:
                st.markdown("#### AI")
                st.success(f"{msg['content']}")

    st.divider()
    
    user_input = st.text_input("Your message:", placeholder="Type here..........", key=f"user_input_{st.session_state.input_counter}")
    
    if st.button("📤 Send", type="primary") and user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        history_text = ""
        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

        full_prompt = history_text + f"User: {user_input}"

        with st.spinner("thinking..."):
            response = model(USER_ID, user_input)
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.session_state.input_counter += 1
        st.rerun()


with mem_col:
    st.markdown("#### Semantic Memory")
    
    with st.container(height=300):
        data_path = f"memory_store_database/user_{USER_ID}/semantic_memory_meta.json"
        
        if os.path.exists(data_path):
            try:
                with open(data_path, "r") as f:
                    memory_data = json.load(f)
                
                if memory_data:
                    for i, item in enumerate(memory_data, 1):
                        st.write(f"{i}. {item.get('data', 'Missing data')}")
                else:
                    st.info("No memories stored yet")
                    
            except Exception as e:
                st.error(f"Error loading memory: {e}")
        else:
            st.info("No memory file found for this user")
    
    st.divider()
    st.caption("Memory updates automatically as you chat")