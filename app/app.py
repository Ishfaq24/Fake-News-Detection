from pathlib import Path
import sys

from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import predict_news_with_details  # noqa: E402


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        title = request.form.get("title", "")
        news = request.form.get("news", "")

        details = predict_news_with_details(news, title)
        prediction = details["label"]

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)