from flask import Flask, render_template, request
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("E:\\SpamMsg\\spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Features and Labels
X = df['v2']
y = df['v1']

cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Naive Bayes Classifier

clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        vect=[message]
        data = cv.transform(vect).toarray()
        my_prediction = clf.predict(data)
    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
