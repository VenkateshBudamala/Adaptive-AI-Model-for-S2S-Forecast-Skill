README: Adaptive AI (AAI) Model for Enhancing S2S Forecast Skill

**Overview**
-------------
This project enhances Subseasonal-to-Seasonal (S2S) forecast skill using an Adaptive Artificial Intelligence (AAI) model. The approach improves the accuracy of temperature forecasts (Tmax and Tmin) by dynamically adjusting machine learning and deep learning models.

**Project Files**
-----------------
1. Downloading_S2S_Files.py: Downloads the raw NCEP S2S forecast data.
2. Generation_of_Parameter_Sets.py: Generates parameter vectors for training the AAI model.
3. Adaptive_Artificial_Intelligence_Code_for_S2S.py: Implements the AAI model, trains it, and validates its performance.

**Installation**
----------------
Ensure Python 3.x is installed on your system. Then, install the required libraries by running:

pip install -r requirements.txt

**Setup Instructions**
-----------------------
Before running the scripts, make sure to update the data and working directories in the following files:
- Downloading_S2S_Files.py: Update the directory for storing the raw forecast data.
- Generation_of_Parameter_Sets.py: Specify the correct paths for input and output files.
- Adaptive_Artificial_Intelligence_Code_for_S2S.py: Ensure correct paths for training and validation datasets.

**Running the Code**
--------------------
1. **Download S2S Forecast Data**
   Run the script to download the raw NCEP S2S forecast data:
   python Downloading_S2S_Files.py

2. **Generate Parameter Sets**
   Generate the parameter sets required for training the AAI model:
   python Generation_of_Parameter_Sets.py

3. **Train and Validate AAI Model**
   Run the script to train the AAI model and perform validation:
   python Adaptive_Artificial_Intelligence_Code_for_S2S.py

**Validation**
--------------
The model will be validated using an independent dataset to assess its generalization ability, with key metrics like RMSE and R² score used to evaluate performance.

**Notes**
---------
- Make sure to update paths and directories to match your system's configuration.
- Ensure all required input data is available and correctly formatted before running the scripts.

This README is intended to help you set up and run the project efficiently.
