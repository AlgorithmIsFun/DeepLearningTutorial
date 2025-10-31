import requests
import json

# Prepare test data
test_wine = {
    "dataframe_split": {
        "columns": [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
        ],
        "data": [[7.0, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3.0, 0.45, 8.8]],
    }
}

# Make prediction request
response = requests.post(
    "http://localhost:5002/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(test_wine),
)

prediction = response.json()
print(prediction)
