from config import MODEL_NAME, TEMPERATURE, MAX_TOKENS

def generate_chat_response(client, user_input, context,history=[]):    # history=[], extra added for history
    messages = [
        {
            "role": "system",
            "content": f"Answer questions using only this context: {context}. "
                       "If the answer can't be found in the context, say 'This information is not available in the source'. "
                       "Be concise and accurate."
        },
        {
            "role": "user",
            "content": user_input
        }
    ]
    
    # Extra added for history 
    for exchange in history:
        messages.append({"role": "user", "content": exchange['user']})
        if exchange['bot']:
            messages.append({"role": "assistant", "content": exchange['bot']})
    
    messages.append({"role": "user", "content": user_input})
    
    #----------------------------------------------------------------------------------------
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    return chat_completion.choices[0].message.content.strip()





