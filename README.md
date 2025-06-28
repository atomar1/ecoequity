# EcoEquity - CalEnviroScreen Disadvantaged Community Predictor

A machine learning API that predicts whether a California census tract is classified as a disadvantaged community based on environmental and socioeconomic indicators from the CalEnviroScreen 3.0 dataset.

## Overview

This project uses the CalEnviroScreen 3.0 dataset to train a machine learning model that can predict whether a census tract qualifies as a disadvantaged community. The model considers various environmental and socioeconomic factors including:

- **Environmental Indicators**: Ozone, PM2.5, Diesel PM, Drinking Water quality, Traffic exposure
- **Health Indicators**: Asthma rates, Low Birth Weight rates
- **Socioeconomic Indicators**: Poverty rates, Unemployment rates, Linguistic Isolation

## Features

- **RESTful API**: Built with FastAPI for easy integration
- **Web Interface**: Modern Next.js frontend with TypeScript and Tailwind CSS
- **Machine Learning Model**: Logistic Regression with feature engineering
- **Feature Engineering**: Includes interaction terms and polynomial features
- **Scalable**: Can handle multiple concurrent requests
- **Documentation**: Auto-generated API documentation

## Installation

### Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- npm or yarn

### Backend Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd ecoequity
   ```

2. **Install Python dependencies**:
   ```bash
   pip3 install fastapi uvicorn pydantic scikit-learn joblib pandas numpy
   ```

3. **Verify backend installation**:
   ```bash
   python3 -c "import fastapi, uvicorn, pydantic, sklearn, joblib; print('All dependencies installed successfully!')"
   ```

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Verify frontend installation**:
   ```bash
   npm run build
   # or
   yarn build
   ```

## Usage

### Running the Backend API

From the project root directory:

```bash
python3 -m uvicorn backend.main:app --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

### Running the Frontend

From the frontend directory:

```bash
cd frontend
npm run dev
# or
yarn dev
```

The frontend will be available at:
- **Web Application**: http://localhost:3000

### Running Both Services

You can run both services simultaneously:

**Terminal 1 (Backend)**:
```bash
python3 -m uvicorn backend.main:app --reload
```

**Terminal 2 (Frontend)**:
```bash
cd frontend
npm run dev
```

### Making Predictions

#### Using the Web Interface

1. Open http://localhost:3000 in your browser
2. Fill in the form with your census tract data
3. Click "Predict" to get results

#### Using curl

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Poverty": 25.5,
       "Unemployment": 8.2,
       "PM25": 12.1,
       "Ozone": 0.055,
       "Diesel_PM": 35.2,
       "Drinking_Water": 500.0,
       "Asthma": 8.5,
       "Low_Birth_Weight": 6.2,
       "Traffic": 750.0,
       "Linguistic_Isolation": 15.3
     }'
```

#### Using Python

```python
import requests
import json

url = "http://localhost:8000/predict"
data = {
    "Poverty": 25.5,
    "Unemployment": 8.2,
    "PM25": 12.1,
    "Ozone": 0.055,
    "Diesel_PM": 35.2,
    "Drinking_Water": 500.0,
    "Asthma": 8.5,
    "Low_Birth_Weight": 6.2,
    "Traffic": 750.0,
    "Linguistic_Isolation": 15.3
}

response = requests.post(url, json=data)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

### API Response Format

```json
{
  "prediction": 1,
  "confidence": 0.8542
}
```

- **prediction**: 0 (not disadvantaged) or 1 (disadvantaged)
- **confidence**: Probability score between 0 and 1

## Frontend Features

- **Modern UI**: Built with Next.js, TypeScript, and Tailwind CSS
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Form Validation**: Client-side validation for input fields
- **Real-time API Integration**: Connects to the FastAPI backend
- **Error Handling**: User-friendly error messages
- **Loading States**: Visual feedback during API calls

## Model Details

### Input Features

The model accepts the following 10 features:

1. **Poverty** (float): Poverty rate percentage
2. **Unemployment** (float): Unemployment rate percentage
3. **PM25** (float): PM2.5 air pollution levels
4. **Ozone** (float): Ozone concentration levels
5. **Diesel_PM** (float): Diesel particulate matter levels
6. **Drinking_Water** (float): Drinking water quality score
7. **Asthma** (float): Asthma hospitalization rate
8. **Low_Birth_Weight** (float): Low birth weight rate
9. **Traffic** (float): Traffic density/exposure
10. **Linguistic_Isolation** (float): Linguistic isolation percentage

### Feature Engineering

The model uses advanced feature engineering including:

- **Interaction Terms**: All pairwise interactions between features
- **Polynomial Features**: Quadratic terms for Poverty and Unemployment
- **Cross-Product**: Poverty × Unemployment interaction

### Model Architecture

- **Algorithm**: Logistic Regression
- **Preprocessing**: StandardScaler for feature normalization
- **Training**: Uses CalEnviroScreen 3.0 dataset
- **Target**: Binary classification (disadvantaged vs. non-disadvantaged)

## Data Source

This project uses the **CalEnviroScreen 3.0** dataset from the California Office of Environmental Health Hazard Assessment (OEHHA). The dataset contains environmental and socioeconomic indicators for California census tracts.

### Key Dataset Features

- **Environmental Indicators**: Air quality, water quality, traffic exposure
- **Health Indicators**: Disease rates, birth outcomes
- **Socioeconomic Indicators**: Poverty, unemployment, education, housing
- **Geographic Coverage**: All California census tracts
- **Temporal Coverage**: 2018 data

## Development

### Backend Development

To retrain the model with updated data:

1. Place your dataset in the `data/` directory
2. Update the `train_model.py` script if needed
3. Run the training script:
   ```bash
   cd backend
   python3 train_model.py
   ```

### Frontend Development

To modify the frontend:

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Make changes to files in `src/` directory
4. The application will hot-reload automatically

### API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Main prediction endpoint

### Error Handling

The API includes comprehensive error handling for:
- Invalid input data types
- Missing required fields
- Out-of-range values
- Model prediction errors

## Troubleshooting

### Common Issues

1. **"uvicorn: command not found"**
   - Solution: Use `python3 -m uvicorn backend.main:app --reload`

2. **Import errors**
   - Solution: Ensure all dependencies are installed with `pip3 install -r requirements.txt`

3. **Model file not found**
   - Solution: Ensure `model.pkl` and `scaler.pkl` exist in the `backend/` directory

4. **Port already in use**
   - Solution: Use a different port: `python3 -m uvicorn backend.main:app --reload --port 8001`

5. **Frontend build errors**
   - Solution: Clear node_modules and reinstall: `rm -rf node_modules package-lock.json && npm install`

6. **API connection errors in frontend**
   - Solution: Ensure the backend is running on http://localhost:8000

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with data usage agreements for the CalEnviroScreen dataset.

## Acknowledgments

- California Office of Environmental Health Hazard Assessment (OEHHA) for the CalEnviroScreen dataset
- FastAPI for the web framework
- Next.js for the frontend framework
- Scikit-learn for machine learning capabilities

## Contact

For questions or issues, please open an issue in the repository!