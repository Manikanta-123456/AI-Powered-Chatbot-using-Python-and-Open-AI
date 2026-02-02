import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

print("AI Chatbot started. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye! Have a great day.")
        break

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_input}
            ]
        )
        reply = response.choices[0].message.content
        print("Bot:", reply)

    except Exception as e:
        print("Error:", e)
