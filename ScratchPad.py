#Just playing around with corpus movie review dataset to figure out vader a bit better

import nltk
from nltk.sentiment import vader

positiveReviewsFileName = r"C:\Users\L0pht\PycharmProjects\untitled\rt-polaritydata\rt-polarity.pos"
negativeReviewFileName = r"C:\Users\L0pht\PycharmProjects\untitled\rt-polaritydata\rt-polarity.neg"

with open(positiveReviewsFileName, 'r') as f:
    positiveReviews = f.readlines()
    print(len(positiveReviews))

with open(negativeReviewFileName, 'r') as f:
    negativeReviews = f.readlines()
    print(len(negativeReviews))

sia = vader.SentimentIntensityAnalyzer()
def vaderSentiment(review):
    return sia.polarity_scores(review)['compound']

review = "this is the best resturant in the city"
print(vaderSentiment(review))

def getReviewSentiments(sentimentCalculator):
    negReviewResults = [sentimentCalculator(oneNegativeReview) for oneNegativeReview in negativeReviews]
    posReviewResults = [sentimentCalculator(onePositiveReview) for onePositiveReview in positiveReviews]
    return {
        'results-on-positive' : posReviewResults,
        'results-on-negative' : negReviewResults
    }

vaderResults = getReviewSentiments(vaderSentiment)
vaderResults.keys()
print(vaderResults.keys())

def runDiagnostics(reviewResults):
    positiveReviewsResult = reviewResults['results-on-positive']
    negativeReviewsResult = reviewResults['results-on-negative']
    pctTruePositive = float(sum(x > 0 for x in positiveReviewsResult))/len(positiveReviewsResult)
    pctTrueNegative = float(sum(x > 0 for x in negativeReviewsResult))/len(positiveReviewsResult)
    totalAccurate = float(sum(x > 0 for x in positiveReviewsResult)) + float(sum(x < 0 for x in negativeReviewsResult))
    total = len(positiveReviewsResult) + len(negativeReviewsResult)
    print("Accuracy on positive reviews = " +"%.2f" % (pctTruePositive*100) + "%")
    print("Accuracy on negative reviews = " + "%.2f" % (pctTrueNegative * 100) + "%")
    print("Overall accuracy = " + "%.2f" % (totalAccurate*100/total) + "%")