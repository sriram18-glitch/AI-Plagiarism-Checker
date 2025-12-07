
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def check_plagiarism(original, suspect):
    texts = [original, suspect]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    score = None

    if request.method == "POST":
        original = request.form["original"]
        suspect = request.form["suspect"]

        score = check_plagiarism(original, suspect)

        if score > 0.80:
            result = "⚠️ High Plagiarism Detected!"
        elif score > 0.50:
            result = "🟠 Partial Plagiarism Found."
        else:
            result = "✅ No Significant Plagiarism."

    return render_template("index.html", result=result, score=score)

if __name__ == "__main__":
    app.run(debug=True)
