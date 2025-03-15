from flask import Flask, request, jsonify, render_template
from model import accuracy, conf_matrix, class_report, vectorizer, clf

app = Flask(__name__)
# print(f"The message '{new_medd}' is a {label} message.")
@app.route('/', methods=['GET', 'POST'])
def main():
    message = ''
    user = ''
    post = ''
    
    if request.method == 'POST':
        new_medd = request.form.get('message')
        new_medd_vec = vectorizer.transform([new_medd])
        prediction = clf.predict(new_medd_vec)
        post = ["Not spam", "spam"][prediction[0]]
    return render_template('index.html', post=post, user=user)

if __name__ == '__main__':
    app.run(debug=True)
