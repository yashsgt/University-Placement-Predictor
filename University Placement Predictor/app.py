from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model + scaler
model = pickle.load(open("placement_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_features = [
            float(request.form["cgpa"]),
            float(request.form["internships"]),
            float(request.form["projects"]),
            float(request.form["aptitude"]),
            float(request.form["communication"])
        ]

        # Convert to numpy
        final_features = np.array([input_features])

        # 🔥 MOST IMPORTANT LINE
        final_features = scaler.transform(final_features)

        prediction = model.predict(final_features)

        # result = "YES (Placed)" if prediction[0] == 1 else "NO (Not Placed)"
        # result = "Result: Student is likely to be placed" if prediction[0] == 1 else "Result: Student is likely to be not placed"
        result = "Result: Placed" if prediction[0] == 1 else "Result: Not Placed"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)