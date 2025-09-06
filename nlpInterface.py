import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

#loading tokenizer and model
model_path='/Users/lanaa/Downloads/bert_model/'

@st.cache_resource
def load_model():
    tokenizer=BertTokenizer.from_pretrained(model_path)
    model=BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer,model

tokenizer, model =load_model()

#user interface
st.title("Sentiment Analysis with BERT 🤖")
st.write("Enter a sentence to classify it as Positive, Negative, or Neutral.")

user_input=st.text_input("Enter your sentence:")

#prediction
def predict_sentiment(text):
    inputs=tokenizer(text,return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs=model(**inputs)
        logits=outputs.logits
        predicted_class_id= torch.argmax(logits, dim=1).item()
    label_map= {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😊"}
    return label_map[predicted_class_id]

if st.button("Predict Sentiment"):
    if user_input.strip()== "":
        st.warning("Please enter some text.")
    else:
        prediction=predict_sentiment(user_input)
        st.success(f"Sentiment: *{prediction}*")