import pickle

model = pickle.load(open("D:\\EDA\\Spam-Ham_Message_Classifier v1.0\\Spam-Ham_Message_Classifier.pkl", "rb"))
vectorizer = pickle.load(open("D:\\EDA\\Spam-Ham_Message_Classifier v1.0\\CountVectorizer.pkl", "rb"))

# model = pickle.load(open("D:\\EDA\\Spam-Ham_Message_Classifier v1.0\\test01_mnbc.pkl", "rb"))
# vectorizer = pickle.load(open("D:\\EDA\\Spam-Ham_Message_Classifier v1.0\\cv.pkl", "rb"))

print("\n----------- 📩 Spam-Ham Message Classifier -----------")
print("Type 'q' and press Enter to exit.\n")

while True:
    message = input("Enter a message: ").strip()
    if message.lower() == 'q':
        print("🔚 Exiting the classifier. Goodbye!")
        break
    message_transformed = vectorizer.transform([message])
    prediction = model.predict(message_transformed)
    result = "🚨 Spam Message" if prediction[0] else "✅ Ham Message"
    print(f"Prediction: {result}\n")
