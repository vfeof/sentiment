import nltk

# nltk.download(['names', 'stopwords', 'averaged_perceptron_tagger', 'punkt', 'vader_lexicon'])
stopwords = nltk.corpus.stopwords.words('english')
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()


def hello():
    print('Hi. Enter your text for sentiment analyze')


hello()
text = input()


def analyze_text(text):
    text_scores = sia.polarity_scores(text)
    if text_scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif text_scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return sentiment, text_scores


def analyze():
    sentiment, text_scores = analyze_text(text)
    print('Sentiment:', sentiment)
    print('Positive:', text_scores['pos'])
    print('Negative:', text_scores['neg'])
    print('Neutral:', text_scores['neu'])
    print('Compound:', text_scores['compound'])


analyze()
