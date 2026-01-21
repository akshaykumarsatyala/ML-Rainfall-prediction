from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get values from form
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)

    prediction = model.predict(final_features)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Rainfall: {prediction[0]:.2f} mm"
    )

if __name__ == "__main__":
    app.run(debug=True)
