from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle

model = pickle.load(open("Spam-Ham_Message_Classifier.pkl", "rb"))
vectorizer = pickle.load(open("CountVectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
@cross_origin()
def home():
    return render_template('WebApp.html')

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        message = request.form['message']
        message_transformed = vectorizer.transform([message])
        prediction = model.predict(message_transformed)

        result = "Spam Message" if prediction[0] else "Ham Message"
        return render_template('WebApp.html', prediction=result)
    
    return render_template('WebApp.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)