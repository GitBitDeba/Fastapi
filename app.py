from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import io
from io import StringIO

app = FastAPI()

# Allow CORS for Vite frontend (adjust origin if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://sentinelx-gamma.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = joblib.load("model.joblib")
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Load model
        model = joblib.load("model.joblib")

        # Select only relevant columns
        feature_columns = ["Type", "Air temperature [K]", "Process temperature [K]",
                           "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
        df = df[feature_columns]

        # Convert categorical 'Type' values to numeric
        type_mapping = {'L': 0, 'M': 1, 'H': 2, 'Low': 0, 'Medium': 1, 'High': 2}
        df["Type"] = df["Type"].map(type_mapping)

        # Make predictions
        predictions = model.predict(df)

        # Add predictions to output
        df["Prediction"] = ["Failure" if pred == 1 else "No Failure" for pred in predictions]

        return df.to_dict(orient="records")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
