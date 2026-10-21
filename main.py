from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

categories = ["food and drink", "music and arts", "sports and fitness",
              "technology", "cultural", "charity and fundraising"]

events = [
    "Culture Night with food, music and dance performances from 12 countries",
    "24-hour hackathon for students building AI apps",
    "5K charity run to raise money for the local food bank",
]

for event in events:
    result = classifier(event, candidate_labels=categories, multi_label=False)
    top = [f"{label} ({score:.0%})"
           for label, score in zip(result["labels"][:3], result["scores"][:3])]
    print(event)
    print("  Tags:", ", ".join(top))