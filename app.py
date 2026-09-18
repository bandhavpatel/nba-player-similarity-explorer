from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def home():
    project_features = [
        "Find statistically similar NBA players",
        "Build and analyze a five-player roster",
        "Receive data-driven roster recommendations",
    ]

    return render_template(
        "index.html",
        project_features=project_features,
    )


if __name__ == "__main__":
    app.run(debug=True)