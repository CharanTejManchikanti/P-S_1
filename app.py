from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("xgmodel.pkl")

# Homepage route to render HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    user_data = {
        'year': [2020],
        'gender': [request.form['gender']],
        'age': [int(request.form['age'])],
        'no_of_days_subscribed': [int(request.form['no_of_days_subscribed'])],
        'multi_screen': [request.form['multi_screen']],
        'mail_subscribed': [request.form['mail_subscribed']],
        'weekly_mins_watched': [int(request.form['weekly_mins_watched'])],
        'minimum_daily_mins': [int(request.form['minimum_daily_mins'])],
        'maximum_daily_mins': [int(request.form['maximum_daily_mins'])],
        'weekly_max_night_mins': [int(request.form['weekly_max_night_mins'])],
        'videos_watched': [int(request.form['videos_watched'])],
        'maximum_days_inactive': [int(request.form['maximum_days_inactive'])],
        'customer_support_calls': [int(request.form['customer_support_calls'])]
    }

    # Convert to DataFrame
    user_df = pd.DataFrame(user_data)

    # Apply one-hot encoding (same as training)
    user_df = pd.get_dummies(user_df, columns=['gender', 'multi_screen', 'mail_subscribed'], drop_first=True)

    # Ensure correct columns
    expected_columns = [
        'age', 'no_of_days_subscribed', 'weekly_mins_watched', 'minimum_daily_mins',
        'maximum_daily_mins', 'weekly_max_night_mins', 'videos_watched',
        'maximum_days_inactive', 'customer_support_calls', 'gender_Male',
        'multi_screen_yes', 'mail_subscribed_yes'
    ]
    for col in expected_columns:
        if col not in user_df.columns:
            user_df[col] = 0

    # Reorder columns
    user_df = user_df[expected_columns]

    # Make prediction
    prediction = model.predict(user_df)
    result = "Leaving" if prediction[0] == 1 else "Not Leaving"

    # Return result to the frontend
    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == '__main__':
    app.run(debug=True)
