# **Enhancing S2S Predictive Skill of Indian Heatwave Warnings using Adaptive AI**

## **Overview**
This project focuses on enhancing the predictive skill of Subseasonal-to-Seasonal (S2S) forecasts for heatwave warnings in India by using Adaptive AI (AAI) techniques. It involves downloading S2S data, generating parameter sets for model inputs, applying adaptive AI models to improve the forecast, and validating heatwave warnings.

## **File Structure**
```
.
├── Downloading_S2S_Files.py
├── Generation_of_Parameter_Sets.py
├── Adaptive_Artificial_Intelligence_Code_for_S2S.py
├── Validation_of_Heat_Wave_Warnings.py
└── README.md
```

## **Code Files Description**

### **1. Downloading_S2S_Files.py**
This script downloads S2S forecast data from NCEP CFSv2 for further processing.

- **Usage**: Run this script to retrieve the S2S forecast data files.
- **Dependencies**: `requests`, `pandas`, `numpy`.

### **2. Generation_of_Parameter_Sets.py**
This script generates the necessary parameter sets, collating required data (e.g., year, month, latitude, longitude, forecast data) for model training.

- **Usage**: After downloading the S2S forecast data, run this script to prepare input for the models.
- **Dependencies**: `pandas`, `numpy`.

### **3. Adaptive_Artificial_Intelligence_Code_for_S2S.py**
This script builds and trains adaptive AI models (RF, SVM, XGBoost, LSTM, CNN), iteratively refining them to select the best-performing model for heatwave prediction.

- **Usage**: Run this script to train the models and identify the best-performing model.
- **Dependencies**: `scikit-learn`, `xgboost`, `tensorflow`, `keras`, `pandas`, `numpy`.

### **4. Validation_of_Heat_Wave_Warnings.py**
This script validates the trained AI model by comparing forecasted Tmax values with observed data, classifying them according to IMD heatwave thresholds, and assessing the model's prediction accuracy.

- **Usage**: After training the model, run this script to evaluate the heatwave warning forecasts.
- **Dependencies**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`.

---

## **How to Run the Code**

### **Step 1: Install Dependencies**
Ensure all required Python libraries are installed by running:
```bash
pip install numpy pandas scikit-learn xgboost tensorflow keras matplotlib requests scipy
```

### **Step 2: Download S2S Data**
Run the `Downloading_S2S_Files.py` script to download the necessary S2S forecast data from NCEP:
```bash
python Downloading_S2S_Files.py
```

### **Step 3: Generate Parameter Sets**
Once the S2S data is downloaded, run the `Generation_of_Parameter_Sets.py` script to generate parameter sets for the model:
```bash
python Generation_of_Parameter_Sets.py
```

### **Step 4: Train the Adaptive AI Model**
Run the `Adaptive_Artificial_Intelligence_Code_for_S2S.py` script to train the AI models:
```bash
python Adaptive_Artificial_Intelligence_Code_for_S2S.py
```

### **Step 5: Validate Heatwave Warning Forecasts**
After training the model, run the `Validation_of_Heat_Wave_Warnings.py` script to evaluate the forecast accuracy:
```bash
python Validation_of_Heat_Wave_Warnings.py
```

---

## **Validation**
The model will be validated using an independent dataset to assess its generalization ability, with key metrics like RMSE and R² score used to evaluate performance.

---

## **Notes**
- Update paths and directories to match your system's configuration.
- Ensure that all input data is available and properly formatted before running the scripts.

This README should help you set up and run the project efficiently. 
