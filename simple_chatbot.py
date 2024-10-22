# Step 1: Install necessary libraries
# You don't need to include this in the app; it's just for Google Colab or local testing.
# !pip install transformers streamlit

# Step 2: Import necessary libraries
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Step 3: Load the Flan-T5 model and tokenizer
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"  # You can use a larger version if needed
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Step 4: Define chatbot response function
def chatbot_response(user_input):
    input_text = f"Answer the following question: {user_input}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Step 5: Build Streamlit App UI
st.title("Flan-T5 Chatbot")
st.write("This is a simple chatbot powered by the Flan-T5 model from HuggingFace.")

# User input
user_input = st.text_input("You: ", placeholder="Ask me anything...")

# Generate response on button click
if st.button("Send"):
    if user_input:
        response = chatbot_response(user_input)
        st.write(f"Chatbot: {response}")
    else:
        st.write("Please type a question or message.")
