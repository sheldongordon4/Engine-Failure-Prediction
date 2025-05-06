# Engine Failure Prediction – RUL Estimation with NASA CMAPSS

This project builds a machine learning model to predict the **Remaining Useful Life (RUL)** of turbofan engines using NASA’s CMAPSS dataset. The system supports predictive maintenance by identifying engines at risk of failure before it happens — reducing downtime, improving safety, and optimizing maintenance scheduling.

---

## Project Structure
.
├── data/ # CMAPSS dataset files
│ ├── train_FD001.txt
│ ├── test_FD001.txt
│ └── RUL_FD001.txt
├── cmapss.ipynb # Main notebook: preprocessing, modeling, evaluation
├── streamlit_app.py # Interactive Streamlit dashboard
└── README.md # Project overview and instructions

---

## Dataset Description

- **Train Set**: Full run-to-failure history for 100 engines
- **Test Set**: Partial lifecycle up to a cut-off point
- **RUL File**: Ground truth Remaining Useful Life for test engines

Each entry contains:
- Engine ID
- Cycle number
- 3 operational settings
- 21 sensor readings

📎 [Dataset on Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

---

## Feature Engineering

- Constructed RUL as `RUL = EOL - cycle_time`
- Dropped low-variance sensors
- Selected sensors showing degradation trends
- Standardized features using `StandardScaler`
- Ensured consistency across train/test sets

---

## Model Development

Tested three regression models:
| Model                  | RMSE |
|------------------------|------|
| Support Vector Regressor | 33.31 |
| Random Forest Regressor  | 31.74 |
| **LightGBM**              | **31.32** (selected) |

Final performance (on evaluation set):
- **MAE**: 18.01  
- **RMSE**: 24.31  

✅ LightGBM was selected for its speed, accuracy, and built-in feature importance.

---

## Maintenance Use Case

The system classifies engine health based on RUL:

| Predicted RUL  | Maintenance Action       |
|----------------|--------------------------|
| ≤ 15 cycles    | Immediate Repair         |
| 16–47 cycles   | Schedule Inspection      |
| > 47 cycles    | Continue Normal Operation|

This enables:
- Proactive failure prevention  
- Optimized maintenance planning  
- Reduced cost and downtime

---

## Streamlit Dashboard

Launch a real-time dashboard to monitor engine health:

```bash
streamlit run rul_app.py

Features:

Engine-wise health visualization
Sensor trend plots
Risk classification and RUL alerts
CSV batch upload for predictions

---

## License

MIT License. See LICENSE for details.

---

## Acknowledgments

NASA Prognostics Center
CMAPSS Dataset (2008)
scikit-learn, LightGBM, Streamlit

---
