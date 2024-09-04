# Save the fine-tuned model
model.save_pretrained('./fine-tuned-roberta')
tokenizer.save_pretrained('./fine-tuned-roberta')

# Evaluate the model
results = trainer.evaluate()
print("Evaluation results:", results)

# Logistic Regression model on the extracted features
clf = LogisticRegression(max_iter=1000)
train_features = train_encodings['input_ids']
test_features = test_encodings['input_ids']
clf.fit(train_features, train_labels.map({'POSITIVE': 0, 'NEUTRAL': 1, 'NEGATIVE': 2}))

# Predict on the test set
test_predictions = clf.predict(test_features)

# Evaluate the classifier
accuracy = accuracy_score(test_labels.map({'POSITIVE': 0, 'NEUTRAL': 1, 'NEGATIVE': 2}), test_predictions)
print(f"Test Accuracy: {accuracy}")

from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from sklearn.metrics import confusion_matrix, classification_report

# Load the fine-tuned model and tokenizer
model = RobertaForSequenceClassification.from_pretrained('./fine-tuned-roberta')
tokenizer = RobertaTokenizer.from_pretrained('./fine-tuned-roberta')

# Set the model to evaluation mode
model.eval()

# Function to predict sentiment on new text
def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)

    # Get predictions from the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1).squeeze()

    # Get the sentiment with the highest probability
    sentiment = torch.argmax(probabilities).item()

    # Map the sentiment to labels
    sentiment_labels = {0: 'POSITIVE', 1: 'NEUTRAL', 2: 'NEGATIVE'}

    # Print probability of each classification
    print("Probabilities:")
    for i, label in sentiment_labels.items():
        print(f"{label}: {probabilities[i]:.4f}")

    return sentiment_labels[sentiment], probabilities

# Example usage
sample_text = "This drug was very effective and had minimal side effects."
predicted_sentiment, probabilities = predict_sentiment(sample_text)

print(f"Predicted Sentiment: {predicted_sentiment}")

# Test the model on a list of sample texts
test_texts = [
    "This drug was very effective and had minimal side effects.",
    "I felt terrible after taking this medication.",
    "It didn't make much of a difference.",
    "Completely cured my headache.",
    "I experienced severe dizziness and nausea."
]


import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()


# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

# Load sentiment analysis model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Initialize NLTK's VADER sentiment intensity analyzer
vader_analyzer = SentimentIntensityAnalyzer()


aspect_keywords_vectors  = {
    "Effectiveness": [
        "effective", "efficacy", "worked", "helpful", "beneficial", "useful",
        "productive", "successful", "improvement", "remedy", "treatment",
        "potent", "powerful", "impactful", "result", "achieved", "outcome",
        "satisfactory", "performance", "value", "benefit", "success",
        "functionality", "reliable", "consistent", "notable", "significant",
        "remarkable", "help", "assist", "solve", "cure", "restore", "heal",
        "beneficial effects", "therapeutic", "advantageous", "positive results",
        "efficacy rate", "clinical improvement", "successful outcome", "highly effective",
        "fast-acting", "long-lasting", "dependable", "valuable", "worthwhile",
        "therapeutic benefit", "clinical effectiveness", "medical improvement",
        "successful therapy", "efficacy of treatment", "beneficial outcome",
        "remedial effect", "healing properties", "positive impact",
        "high efficacy", "effective dose", "impact level", "improvement rate",
        "positive response rate", "successful treatment rate",
        "remarkably effective", "proven", "validated", "verified", "promising",
        "exceptional", "noteworthy", "substantial improvement", "beneficial result",
        "positive impact", "notable success", "significant gain", "worth the effort",
        "optimal results", "enhanced performance", "strong efficacy", "proven benefit",
        "desired effect", "improved condition", "evident success", "effective remedy",
        "positive change", "successful intervention", "commendable performance",
        "outstanding result", "high performance", "remarkable success", "exemplary effect"
    ],
    "Side Effects": [
        "side effect", "adverse effect", "negative effect", "problem", "reaction",
        "issue", "symptom", "undesirable effect", "discomfort", "nausea",
        "dizziness", "fatigue", "rash", "itching", "allergy", "pain",
        "stomach upset", "headache", "vomiting", "sleepiness", "irritation",
        "confusion", "anxiety", "depression", "mood swings", "cramps",
        "dry mouth", "blurred vision", "tremors", "sweating", "thirst",
        "loss of appetite", "weight gain", "weight loss", "insomnia",
        "sensitivity", "agitation", "restlessness", "weakness",
        "heart palpitations", "shortness of breath", "skin reactions", "nervousness",
        "constipation", "diarrhea", "fever", "chills", "muscle pain", "joint pain",
        "severe reaction", "intense discomfort", "extreme side effect",
        "significant adverse effect", "severe symptom", "high-impact reaction",
        "allergic reaction", "hypersensitivity", "toxic effect", "unwanted effect",
        "intolerance", "withdrawal symptom", "interaction effect",
        "irregular heartbeat", "breathing difficulty", "swelling", "muscle cramps",
        "throat irritation", "chest pain", "drowsiness", "tinnitus", "hot flashes",
        "mental fog", "hair loss", "itchy skin", "blood pressure changes", "swollen ankles",
        "trembling", "shivering", "rashes", "abdominal pain", "leg cramps", "joint stiffness",
        "sudden weight loss", "rapid weight gain", "persistent cough", "unusual fatigue",
        "increased sweating", "sudden mood changes", "short-term memory loss",
        "hallucinations", "increased appetite", "skin dryness", "unusual bruising",
        "cold sweats", "chronic pain", "muscle spasms", "tenderness", "numbness"
    ]
}


# Extract the aspect names and flatten the keyword list
aspect_names = list(aspect_keywords_vectors.keys())
aspect_keywords = [keyword for aspect_list in aspect_keywords_vectors.values() for keyword in aspect_list]

#Fit the vectorizer to your aspect keywords
vectorizer.fit(aspect_keywords)


# Convert aspect keywords to vectors
aspect_keyword_vectors = vectorizer.transform(aspect_keywords)

# Function to match a keyword to an aspect
def match_keyword_to_aspect(sentence, aspect_names):
    # Convert sentence to vector space using the same vectorizer
    sentence_vec = vectorizer.transform([sentence.lower()])
    # Compute cosine similarity between sentence and aspect keywords
    similarities = cosine_similarity(sentence_vec, aspect_keyword_vectors).flatten()
    # Find the aspect with the highest similarity score
    highest_similarity_index = similarities.argmax()
    highest_similarity_score = similarities[highest_similarity_index]

    if highest_similarity_score > 0.1:  # Adjust this threshold if needed
        # The index needs to be mapped to the correct aspect name
        aspect_index = highest_similarity_index // len(aspect_keywords_vectors[aspect_names[0]])
        return aspect_names[aspect_index]
    else:
        return None
      
  
def analyze_aspects(review):
    # Tokenize the review into sentences
    sentences = sent_tokenize(review)
    output = {"aspects": []}

    for sentence in sentences:
        # Split sentence into phrases (to handle multiple aspects)
        phrases = sentence.split(',')  # You can adjust this based on your needs

        for phrase in phrases:
            # Try to match the phrase to a known aspect
            matched_aspect = match_keyword_to_aspect(phrase, aspect_names)
            if matched_aspect:
                # Get sentiment score using the transformer model
                sentiment_result = sentiment_analyzer(phrase)[0]
                sentiment_score = sentiment_result['score']
                sentiment_label = sentiment_result['label'].lower()

                # Map sentiment label to "positive" or "negative"
                aspect_sentiment = "positive" if sentiment_score >= 0.5 else "negative"

                # Append the result
                output["aspects"].append({
                    "aspect": matched_aspect,
                    "sentiment": aspect_sentiment,
                    "sentiment_score": round(sentiment_score, 2),
                    "evidence": phrase.strip()
                })

    return output

# Example review
review = "This drug worked wonders for me, but I did feel some dizziness."

# Analyze aspects in the review
result = analyze_aspects(review)
print(result)