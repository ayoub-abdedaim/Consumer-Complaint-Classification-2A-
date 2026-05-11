from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# ====================================
# Charger les modèles
# ====================================

svc_model = joblib.load("LinearSVC_model.pkl")

nb_model = joblib.load("NaiveBayes_model.pkl")

rf_model = joblib.load("RandomForest_model.pkl")

xgb_model = joblib.load("XGBoost_model.pkl")

label_encoder = joblib.load("label_encoder.pkl")

# ====================================
# Route principale
# ====================================

@app.route("/", methods=["GET", "POST"])

def home():

    plainte = ""

    prediction_svc = None
    prediction_nb = None
    prediction_rf = None
    prediction_xgb = None

    if request.method == "POST":

        plainte = request.form["plainte"]

        # LinearSVC
        prediction_svc = svc_model.predict([plainte])[0]

        # Naive Bayes
        prediction_nb = nb_model.predict([plainte])[0]

        # Random Forest
        prediction_rf = rf_model.predict([plainte])[0]

        # XGBoost
        pred_xgb_enc = xgb_model.predict([plainte])

        prediction_xgb = label_encoder.inverse_transform(
            pred_xgb_enc
        )[0]

    return render_template(

        "index.html",

        plainte=plainte,

        prediction_svc=prediction_svc,

        prediction_nb=prediction_nb,

        prediction_rf=prediction_rf,

        prediction_xgb=prediction_xgb
    )

# ====================================
# Run
# ====================================

if __name__ == "__main__":
    app.run(debug=True)