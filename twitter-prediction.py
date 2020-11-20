import twitter
import nltk
import json
from datetime import date, datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
api = twitter.Api()

# This file won't work because it needs authentification and credentials in order to access tweets

# create a watch list of crypto influencers with a importance weight for each 
# (vitalik buterin, 70) meaning that every tweet is important
# can be changed depending on the influencer's reaction to market, etc ...
# detect bullish and bearish terms
# score the trend using the previous step


word_watchlist_bull = ['bullish', 'up', 'trend', 'trends', 'long', 'good', 'break', 'breaking', 'ATH', 'bear trap', 'FOMO' ]
word_watchlist_bear = ['bearish', 'down', 'trend', 'trends', 'short', 'bad', 'dump', 'pump', 'bull trap', ]

new_words = {
    'not buy': -0.75,
    'not sell': 0.75,
    'long': 1,
    'short': -1,
    'buy': 1,
    'sell': -1,
    'pump': 1,
    'dump': -1,
    'bullish': 1,
    'bearish': -1,
    'up': 1,
    'down': -1,
    'FOMO': 0.5,
    'not good': -0.75,
    'massive': 1,
    "breakout": 0.6,
    }

account_weights = {
'cz_binance': 1, 

}

analyser = SentimentIntensityAnalyzer()
analyser.lexicon.update(new_words)
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return(score)

# print(api.GetUserTimeline(780650911014105089))
sentence="#Bitcoin is close to a massive breakout, long this, do not sell now!"
result = sentiment_analyzer_scores(sentence)
json_result = json.dumps(result, indent = 4)
print(json_result)


texts = api.GetUserTimeline(screen_name="TheMoonCarl")
for text in texts:

    res = sentiment_analyzer_scores(text.text)
    print(res)
    if(res['neg'] > 0.0): 
        print( text.text)

# today = datetime.now()

# date_today = today.strftime("%c")
# print(date_today)
# date_tweet = datetime.strptime(texts[0].created_at, '%c')
# if(date_today > date_tweet):
#     print('after')
# else:
#     print('before')