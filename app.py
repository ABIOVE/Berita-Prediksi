from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "pipeline.pkl")
model = joblib.load(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    text_input = ""

    if request.method == "POST":
        text_input = request.form.get("text", "")
        if text_input.strip():
            pred = model.predict([text_input])[0]
            prob = model.predict_proba([text_input]).max()
            confidence = round(prob * 100, 2)
            prediction = "valid" if pred == 1 else "hoax"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        text=text_input
    )

if __name__ == "__main__":
    app.run(debug=True)
