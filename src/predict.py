import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import Config

device = Config.DEVICE

tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained("outputs/model").to(device)

model.eval()


def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=Config.MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        hate_prob = probs[0][1].item()

    return "HATE" if hate_prob > 0.5 else "NOT HATE", hate_prob


texts = [
    # English
    "I hate you",
    "You are amazing",

    # Hindi
    "tum bewakoof ho",
    "tum bahut ache ho",

    # Hinglish
    "tu pagal hai",
    "you are great yaar",

    # Spanish
    "te odio",
    "te quiero",

    # French
    "je te déteste",
    "je t'aime",

    # German
    "ich hasse dich",
    "ich liebe dich"
]

print("\n===== MULTILINGUAL TEST =====\n")

for t in texts:
    label, prob = predict(t)
    print(f"{t} -> {label} ({prob:.2f})")