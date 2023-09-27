import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
csv_path = 'c:\\Users\\DYlmaz\\Desktop\\Confidence Score\\stLouisPredictionsWithFeatures_refRR_smallerThan_40.csv'
data = pd.read_csv(csv_path)

data['Differences'] = abs(data["ref_RR"] - data["pred_RR"])

min_val = data['Differences'].min()
max_val = data['Differences'].max()

# Calculate the threshold using a percentile (e.g., 95th percentile)
threshold_percentile = 95
threshold_value = np.percentile(data['Differences'], threshold_percentile)

# Filter out differences that are both above the threshold
filtered_data = data[data['Differences'] <= threshold_value] 

# Calculate the logarithmic transformation of differences
log_differences = np.log(filtered_data['Differences'] + 1)

# Calculate scaled logarithmic differences
scaled_log_differences = (log_differences - np.log(min_val + 1)) / (np.log(max_val + 1) - np.log(min_val + 1))

# Calculate confidence scores using the scaled values
confidence_scores = (1 - scaled_log_differences) * 100

# Calculate confidence scores1 using logarithmic transformation
# The purpose of adding 1 is to handle cases where values may be zero or negative.

confidence_scores1 = 100 - (np.log(filtered_data['Differences'] + 1) / np.log(max_val + 1)) * 100

# Assign the calculated confidence scores to the DataFrame
filtered_data['conf_score'] = confidence_scores
filtered_data['conf_score1'] = confidence_scores1
#At the end, i have two methods for calculation of confidence score. The results are very similar.

# Save the DataFrame back to a CSV file if desired
filtered_data.to_csv('c:\\Users\\DYlmaz\\Desktop\\Confidence Score\\stLouisPredictions_with_confidenceScore.csv', index=False)
