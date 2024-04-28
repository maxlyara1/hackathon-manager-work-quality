import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentAnalyzer:
    def __init__(self, model_checkpoint='cointegrated/rubert-tiny-sentiment-balanced'):
        self.model_initialized = False
        self.model_checkpoint = model_checkpoint
        self.tokenizer = None
        self.model = None

    def initialize_model(self):
        if not self.model_initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint)
            if torch.cuda.is_available():
                self.model.cuda()
            self.model_initialized = True

    def get_sentiment(self, text, return_type=None):
        self.initialize_model()  # Ensure model is initialized
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.model.device)
            proba = torch.sigmoid(self.model(**inputs).logits).cpu().numpy()[0]
        if return_type == 'label':
            return self.model.config.id2label[proba.argmax()]
        elif return_type == 'score':
            return proba.dot([-1, 0, 1])
        return proba

    def analyze_text_sentiment(self, text):
        self.initialize_model()  # Ensure model is initialized
        simple_prediction = self.get_sentiment(text, 'label')
        if any(word in simple_prediction.lower() for word in ['positive', 'neutral']):
            return True
        return False
