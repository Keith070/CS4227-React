import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download required NLTK data
nltk.download('stopwords')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """
    Preprocesses the input text by performing the following steps:
    - Lowercasing
    - Removing non-alphabetic characters (excluding spaces)
    - Removing stopwords
    - Applying stemming
    """
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters except spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stopwords
    words = [word for word in text.split() if word not in stop_words]
    # Apply stemming
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)
