import torch
from transformers import BertTokenizer, BertForSequenceClassification
import spacy
import re
from collections import Counter

MODEL_PATH = r"C:\Users\polyn\myevn\AI_project\saved_bert_model"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text

def analyze_emotions(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    sentence_emotions = []

    for s in sentences:
        s_clean = preprocess(s)
        encoding = tokenizer(s_clean, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**encoding)
        pred = torch.argmax(outputs.logits, dim=1).item()
        sentence_emotions.append(pred)

    overall_emotion_id = Counter(sentence_emotions).most_common(1)[0][0]
    emotion_counts = Counter(sentence_emotions)

    if hasattr(model.config, "id2label"):
        id2label = model.config.id2label
        overall_emotion = id2label[overall_emotion_id]
        emotion_counts = {id2label[k]: v for k, v in emotion_counts.items()}
        sentence_emotions = [id2label[e] for e in sentence_emotions]
    else:
        overall_emotion = overall_emotion_id

    return sentences, sentence_emotions, overall_emotion, emotion_counts

def summarize_text_bertemb(text, top_n_sentences_per_hero=1):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    words = [token.text for token in doc if token.is_title]
    hero_counts = Counter(words)
    heroes = [hero for hero, count in hero_counts.most_common(5)]

    key_sentences = []
    seen_sentences = set()
    for hero in heroes:
        hero_sents = [s for s in sentences if hero in s]
        for s in hero_sents[:top_n_sentences_per_hero]:
            if s not in seen_sentences:
                key_sentences.append(s)
                seen_sentences.add(s)

    if len(key_sentences) < 3:
        for s in sentences[:3]:
            if s not in seen_sentences:
                key_sentences.append(s)
                seen_sentences.add(s)

    summary = " ".join(key_sentences)
    return summary
