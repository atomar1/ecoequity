import { useState } from "react";
import axios from "axios";

interface InputData {
  Poverty: number;
  Unemployment: number;
  PM25: number;
  Ozone: number;
  Diesel_PM: number;
  Drinking_Water: number;
  Asthma: number;
  Low_Birth_Weight: number;
  Traffic: number;
  Linguistic_Isolation: number;
}

export default function Home() {
  const [formData, setFormData] = useState<InputData>({
    Poverty: 0,
    Unemployment: 0,
    PM25: 0,
    Ozone: 0,
    Diesel_PM: 0,
    Drinking_Water: 0,
    Asthma: 0,
    Low_Birth_Weight: 0,
    Traffic: 0,
    Linguistic_Isolation: 0,
  });

  const [result, setResult] = useState<null | { prediction: number; confidence: number }>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: parseFloat(e.target.value) });
  };

  const handleSubmit = async () => {
    try {
      const res = await axios.post("http://127.0.0.1:8000/predict", formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Error making prediction. Is the backend running?");
    }
  };

  return (
    <div className="min-h-screen p-6 bg-gray-100">
      <div className="max-w-xl mx-auto bg-white p-8 rounded-2xl shadow-xl">
        <h1 className="text-2xl font-bold mb-6 text-center">
          CalEnviroScreen Predictor
        </h1>

        {Object.keys(formData).map((key) => (
          <div key={key} className="mb-4">
            <label className="block font-medium mb-1 capitalize">{key.replace(/_/g, " ")}</label>
            <input
              type="number"
              name={key}
              step="0.01"
              value={formData[key as keyof InputData]}
              onChange={handleChange}
              className="w-full p-2 border rounded"
            />
          </div>
        ))}

        <button
          onClick={handleSubmit}
          className="w-full bg-blue-600 text-white p-3 rounded hover:bg-blue-700 mt-4"
        >
          Predict
        </button>

        {result && (
          <div className="mt-6 p-4 border rounded text-center">
            <p className="text-lg font-semibold">
              Prediction:{" "}
              {result.prediction === 1 ? "Disadvantaged Community" : "Not Disadvantaged"}
            </p>
            <p>Confidence: {Math.round(result.confidence * 100)}%</p>
          </div>
        )}
      </div>
    </div>
  );
}
