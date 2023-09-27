import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your dataset, assuming it's in a CSV file
csv_path = 'c:\\Users\\DYlmaz\\Desktop\\Confidence Score\\data_for_test.csv'
data = pd.read_csv(csv_path)

# Load the trained model from the saved file
model_filename = 'confidence_score_model.pkl'
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

# Define the features and target variable
features = ['greenRrFromIbiList', 'irBaselineRrList', 'stdIbiMSecList', 'avgHrFeatureBpmList',
            'irCardiacRespRmsRatioList', 'irRangeRmsRatioList', 'irGreenCorrCoefficientList', 'pred_RR']

X = data[features]

# Use the loaded model to make predictions
predictions = model.predict(X)

# Add the predictions to myDataFrame
data['predicted_conf_score'] = predictions

# Initialize lists to store coverage and RMSE values for different thresholds
thresholds = np.arange(min(predictions), max(predictions) + 1)  # Thresholds from min to max
coverages = []
rmse_values = []

# Calculate coverage and RMSE for each threshold
for threshold in thresholds:
    # Filter data based on the threshold
    filtered_data = data[data['predicted_conf_score'] > threshold]

    # Check if filtered_data is not empty before calculating RMSE
    if not filtered_data.empty:
        # Calculate RMSE for the filtered data
        rmse = np.sqrt(mean_squared_error(filtered_data['ref_RR'], filtered_data['pred_RR']))

        # Calculate coverage as a percentage
        coverage = (len(filtered_data) / len(data)) * 100

        # Append coverage and RMSE values to the lists
        coverages.append(coverage)
        rmse_values.append(rmse)

# Create a line plot of coverage vs. RMSE
plt.figure(figsize=(8, 6))
plt.plot(coverages, rmse_values, marker='o', linestyle='-')
plt.xlabel('Coverage (%)')
plt.ylabel('RMSE')
plt.title('Coverage vs. RMSE')
plt.grid(True)

# Save the plot as an image file (e.g., PNG)
plt.savefig('c:\\Users\\DYlmaz\\Desktop\\Confidence Score\\coverage_vs_rmse_plot.png')

# Show the plot
plt.show()

# Save the updated DataFrame back to Excel
data.to_csv('c:\\Users\\DYlmaz\\Desktop\\Confidence Score\\data_for_test_with_predicted_confScore.csv', index=False)
