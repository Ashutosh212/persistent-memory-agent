from openai import OpenAI
from dotenv import load_dotenv

def main():
    load_dotenv()
    client = OpenAI()

    with open("chat_history.txt", "r") as file:
        prev_summary = file.read()  # Use .read() to get full text

    query = input("Your Query: ")
    conversation_history = [
        {"role": "system", "content": f"You are a helpful assistant. This is the summary of previous user preferences:\n{prev_summary}\nUse this summary to make a personalized response."},
        {"role": "user", "content": query},
    ]

    response = client.responses.create(
        model="gpt-4o-mini",
        input=conversation_history,
        max_output_tokens=100
    )

    response_text = response.output_text

    summary_input = f"User Query: {query}\nAssistant Response: {response_text}"

    summary = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": "Summarize user preferences based on the query and response below."},
            {"role": "user", "content": summary_input}
        ],
        max_output_tokens=100
    )

    summary_text = summary.output_text

    with open("chat_history.txt", "a") as file:
        file.write(f"{summary_text}\n")

    print(response_text)

if __name__ == "__main__":
    main()
