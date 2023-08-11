from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    text_scores = sid.polarity_scores(text)
    if text_scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif text_scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return render_template('index.html', text=text, sentiment=sentiment, text_scores=text_scores)


if __name__ == '__main__':
    app.run()
