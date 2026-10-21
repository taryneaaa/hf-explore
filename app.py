import streamlit as st
from transformers import pipeline

CATEGORIES = ["food and drink", "music and arts", "sports and fitness",
              "technology", "cultural", "charity and fundraising"]

@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

st.title("CollabMatch AI: Event Tagger")
text = st.text_input("Describe your event")

if st.button("Tag my event"):
    if not text.strip():
        st.warning("Please describe your event first.")
    else:
        try:
            result = load_model()(text, candidate_labels=CATEGORIES, multi_label=True)
            st.subheader("Suggested tags")
            for label, score in zip(result["labels"][:3], result["scores"][:3]):
                st.write(f"**{label.title()}**: {score:.0%} match")
        except Exception as e:
            st.error(f"Something went wrong running the model: {e}")