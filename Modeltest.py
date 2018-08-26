import joblib
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import time
start = time.time()


inputfile = open("input11.txt", "r")
test_data_reviews = inputfile.readline()

def cleanText(raw_text):
    '''
    Convert a raw review to a cleaned review
    '''

    letters_only = re.sub("[^a-zA-Z]", " ", raw_text)
    words = letters_only.lower().split()

    return( " ".join(words))

X_test_cleaned = []
X_test_cleaned.append(cleanText(test_data_reviews))
transformer = TfidfTransformer()
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=joblib.load(open("feature.pkl", "rb")))
X_new_tfidf = transformer.fit_transform(loaded_vec.fit_transform(X_test_cleaned))

loaded_model = joblib.load('finalized_model.sav')
predictedNN = loaded_model.predict(X_new_tfidf)
predictedNN = ' '.join(map(str, predictedNN))
print(predictedNN)
end = time.time()
print(end - start)
out = open('output.txt', 'w')
out.write(str(predictedNN))
