import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset
# Ensure the file 'SMSSpamCollection' is in your working directory.
data = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=["label", "message"])

# 2. Preprocess the data by converting the labels to numerical values (ham=0, spam=1)
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label_num'],
                                                    test_size=0.25, random_state=42)

# 4. Vectorize the text messages using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train the classifier using Multinomial Naive Bayes
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# 6. Make predictions on the test set
y_pred = clf.predict(X_test_vec)

# 7. Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
