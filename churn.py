import streamlit as st
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration
)
import pickle
import numpy as np

# ---------- Set up device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load Models & Tokenizers ----------
# Sentiment Analysis Model
sentiment_tokenizer = AutoTokenizer.from_pretrained("clapAI/modernBERT-base-multilingual-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("clapAI/modernBERT-base-multilingual-sentiment")
sentiment_model.to(device)
id2label_sentiment = sentiment_model.config.id2label  # e.g., {0: "NEGATIVE", 1: "POSITIVE"}

# T5 Chatbot Model
t5_tokenizer = T5Tokenizer.from_pretrained("cuneytkaya/fintech-chatbot-t5")
t5_model = T5ForConditionalGeneration.from_pretrained("cuneytkaya/fintech-chatbot-t5")
t5_model.to(device)

# Decision Tree Model for Churn Prediction
with open("decision_tree_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Load churn dataset
df_churn = pd.read_csv("churn.csv")
columns_to_drop = ["customerID", "Customer Feedback", "Recommendation", "Churn"]

# ---------- Initialize Session State ----------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = None

# ---------- Streamlit UI ----------
st.title("Customer Churn Analysis & Retention Chatbot")

# Input: Customer ID
customer_id_input = st.text_input("Enter Customer ID:")

# ---------- Analysis Section ----------
if st.button("Analyze"):
    if customer_id_input:
        # Convert customer ID to appropriate type (int or str)
        try:
            cust_id = int(customer_id_input)
        except ValueError:
            cust_id = customer_id_input

        # Retrieve the customer's record using the Customer ID
        customer_record = df_churn[df_churn["customerID"] == cust_id]
        if not customer_record.empty:
            feedback_text = customer_record.iloc[0]["Customer Feedback"]
            st.write("**Customer Feedback:**", feedback_text)
            
            # ---------- Sentiment Analysis ----------
            inputs_sentiment = sentiment_tokenizer(feedback_text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs_sentiment = sentiment_model(**inputs_sentiment)
            sentiment_pred = outputs_sentiment.logits.argmax(dim=-1).item()
            sentiment_label = id2label_sentiment[sentiment_pred]
            st.write("**Sentiment:**", sentiment_label)
            
            # ---------- Churn Prediction ----------
            X_customer = customer_record.drop(columns=columns_to_drop)
            if "Gender" in X_customer.columns:
                mapping = {"Female": 0, "Male": 1}
                X_customer["Gender"] = X_customer["Gender"].map(mapping)
            X_customer = X_customer.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            try:
                churn_prediction = rf_model.predict(X_customer)[0]  # binary: 1 for churn, 0 for non-churn
            except Exception as e:
                st.error(f"Error during churn prediction: {e}")
                churn_prediction = None

            if churn_prediction is not None:
                churn_label = "yes" if churn_prediction == 1 else "no"
                st.write("**Churn Prediction:**", churn_label)
                
                # ---------- Form Prompt for T5 Chatbot ----------
                if churn_label == "yes":
                    prompt = (
                        f"Attention: High churn risk detected. Customer had previous feedback: '{feedback_text}', "
                        f"and sentiment is {sentiment_label}. Consider targeted retention measures."
                    )
                else:
                    prompt = (
                        f"Attention: Low churn risk detected. Customer had previous feedback: '{feedback_text}', "
                        f"and sentiment is {sentiment_label}. Standard retention measures are advised."
                    )
                st.write("**Prompt for Chatbot:**", prompt)
                
                # ---------- Generate Initial Chatbot Response ----------
                input_ids = t5_tokenizer.encode(prompt, return_tensors="pt").to(device)
                outputs_t5 = t5_model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
                chatbot_response = t5_tokenizer.decode(outputs_t5[0], skip_special_tokens=True)
                st.write("**Chatbot Response:**", chatbot_response)
                
                # ---------- Initialize Conversation History (only once) ----------
                if st.session_state.conversation_history is None:
                    st.session_state.conversation_history = (
                        f"Initial Prompt: {prompt}\n"
                        f"Chatbot: {chatbot_response}\n"
                    )
            else:
                st.error("Churn prediction could not be made.")
        else:
            st.error("Customer ID not found in the dataset.")
    else:
        st.warning("Please enter a valid Customer ID.")

# ---------- Multi-turn Conversation Section ----------
if st.session_state.conversation_history is not None:
    st.write("### Conversation History")
    st.text_area("History", st.session_state.conversation_history, height=200, disabled=True)
    
    user_message = st.text_input("Your message:", key="user_message")
    if st.button("Send"):
        if user_message:
            # Append user's message to conversation history
            st.session_state.conversation_history += f"User: {user_message}\n"
            
            # Use the updated conversation history as context for the model
            input_ids = t5_tokenizer.encode(st.session_state.conversation_history, return_tensors="pt").to(device)
            outputs_t5 = t5_model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
            new_chatbot_response = t5_tokenizer.decode(outputs_t5[0], skip_special_tokens=True)
            
            # Append the chatbot's new response to the conversation history
            st.session_state.conversation_history += f"Chatbot: {new_chatbot_response}\n"
            
            # Display the updated conversation history and the new response
            st.write("**Chatbot Response:**", new_chatbot_response)
            st.text_area("History", st.session_state.conversation_history, height=200, disabled=True)
