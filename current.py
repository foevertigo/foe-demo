import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor 
import numpy as np  
import pandas as pd  
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from sklearn.compose import ColumnTransformer



# Load the dataset
rainfall = pd.read_csv(r"C:\Users\blohs\OneDrive\Desktop\mini project sem 3\rainfall in india 1901-2015.csv")
soil = pd.read_csv(r"C:\Users\blohs\OneDrive\Desktop\mini project sem 3\Daily_data_of_Soil_Moisture_during_August_2023.csv", encoding = 'latin1')

# Drop multiple columns at once
soil.drop(['District', 'Date','Year','Month','Agency_name'], axis=1, inplace=True)


# Inspect original column names
print("Original soil moisture columns:", soil.columns.tolist())

# Check for hidden characters
for char in soil.columns[0]:  # Assuming 'State' is the first column
    print(repr(char), ord(char))  # This will show the character and its unicode

# Remove unwanted characters, including BOM and any non-printing characters
soil.columns = soil.columns.str.replace('[^\x20-\x7E]', '', regex=True)  # Removes non-ASCII characters
soil.columns = soil.columns.str.strip()  # Remove any leading/trailing whitespace

# Confirm updated column names
print("Updated soil moisture columns:", soil.columns.tolist())

# Rename the state column to 'subdivision'
if 'State' in soil.columns:
    soil.rename(columns={'State': 'SUBDIVISION'}, inplace=True)
    
else:
    print("Column 'State' does not exist in the soil moisture dataset.")

# Final check of column names after renaming
print("Final soil moisture columns:", soil.columns.tolist())





# Group by the 'subdivision' column and calculate the mean of the soil moisture level
average_soil_moisture = soil.groupby('SUBDIVISION', as_index=False)['Avg_smlvl_at15cm'].mean()


# Display result
print("\nAverage Soil Moisture by Subdivision:")
print(average_soil_moisture)

columns_to_convert = ['JAN', 'FEB', 'MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL','Jan-Feb','Mar-May','Jun-Sep','Oct-Dec'] 

# Convert specified columns from mm to cm
rainfall[columns_to_convert] = rainfall[columns_to_convert] / 10  # Divide by 10 to convert mm to cm




# Define replacement values
andaman_value = 25.48292
lakshadweep_value = 18.10392

# Replace NA values specifically for Andaman and Nicobar Islands
soil.loc[(soil['SUBDIVISION'] == 'Andaman & Nicobar') & 
            (soil['Avg_smlvl_at15cm'].isna()), 'Avg_smlvl_at15cm'] = andaman_value

# Replace NA values specifically for Lakshadweep Islands
soil.loc[(soil['SUBDIVISION'] == 'Lakshadweep') & (soil['Avg_smlvl_at15cm'].isna()), 'Avg_smlvl_at15cm'] = lakshadweep_value

# Check rows with the updated subdivisions to verify the changes
print("\nUpdated Soil Moisture DataFrame:")
print(soil[soil['SUBDIVISION'].isin(['Andaman & Nicobar', 'Lakshadweep'])])

# Handle missing values
# List of columns to fill with 0
columns_to_fill = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']

# Fill missing values with 0 for specified columns
rainfall[columns_to_fill] = rainfall[columns_to_fill].fillna(0)

# Show the missing values after filling
print("\nMissing values after filling:")
print(rainfall.isnull().sum())

#MERGING ROWS IN RAINFALL
# Define the subdivisions you want to merge and their new name
subdivisions_to_merge = {'Uttar Pradesh': ['EAST UTTAR PRADESH', 'WEST UTTAR PRADESH'],'Rajasthan': ['WEST RAJASTHAN', 'EAST RAJASTHAN'],'Madhya Pradesh': ['WEST MADHYA PRADESH', 'EAST MADHYA PRADESH'],'Karnataka': [ 'NORTH INTERIOR KARNATAKA','SOUTH INTERIOR KARNATAKA']}

# Specify the columns for which you want to calculate the mean
columns_to_average = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL','Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']  # Adjust as necessary



# Initialize a list to store new rows
new_rows = []

# Loop through the subdivisions and calculate averages
for new_name, old_names in subdivisions_to_merge.items():
    # Select the rows to merge
    rows_to_merge = rainfall[rainfall['SUBDIVISION'].isin(old_names)]

    # Check the rows being averaged
    print(f"\nRows to merge for {new_name}:")
    print(rows_to_merge)

    # Calculate average values for the specified columns
    average_row = rows_to_merge[columns_to_average].mean().to_frame().T

    # Handle the year column
    if 'YEAR' in rows_to_merge.columns:
        # Option 1: Use a constant year
        average_row['YEAR'] = 2015  # Replace with the actual year you want

       
    average_row['SUBDIVISION'] = new_name  # Set the new name for the merged entry

    # Add the new average row to the list
    new_rows.append(average_row)

# Create a DataFrame from the new rows
new_rows_df = pd.concat(new_rows, ignore_index=True)

# Drop the old rows from the original DataFrame
rainfall = rainfall[~rainfall['SUBDIVISION'].isin([name for names in subdivisions_to_merge.values() for name in names])]

# Append the new averaged rows to the DataFrame
rainfall = pd.concat([rainfall, new_rows_df], ignore_index=True)

# Step 1: Define the mappings for renaming
rename_map = {
    'ANDAMAN & NICOBAR ISLANDS': 'Andaman & Nicobar',
    'GUJARAT REGION': 'Gujarat',
    'ORISSA': 'Odisha',
    'MADHYA MAHARASHTRA' : 'Maharashtra',
    'COASTAL ANDHRA PRADESH' : 'Andhra Pradesh',
    'ARUNACHAL PRADESH' : 'Arunachal Pradesh',
    'NAGA MANI MIZO TRIPURA':'Nagaland Manipur Mizoram Tripura',
    'JHARKHAND' : 'Jharkhand',
    'BIHAR':'Bihar',
    'UTTARAKHAND':'Uttarakhand',
    'HARYANA DELHI & CHANDIGARH':'Delhi Haryana Chandigarh',
    'PUNJAB':'Punjab',
    'HIMACHAL PRADESH':'Himachal Pradesh',
    'JAMMU & KASHMIR':'Jammu & Kashmir',
    'KONKAN & GOA':'Goa',
    'CHHATTISGARH':'Chhattisgarh',
    'TELANGANA' : 'Telangana',
    'TAMIL NADU':'Tamil Nadu',
    'KERALA':'Kerala',
    'LAKSHADWEEP':'Lakshadweep',
    'GANGETIC WEST BENGAL':'West Bengal',
    'SUB HIMALAYAN WEST BENGAL & SIKKIM':'Sikkim',
    'ASSAM & MEGHALAYA' : 'Assam Meghalaya'


}

# Step 2: Use replace() to rename values in the subdivision column
rainfall['SUBDIVISION'].replace(rename_map, inplace=True)

subdivisions_to_delete = ['SAURASHTRA & KUTCH', 'MATATHWADA','VIDARBHA','RAYALSEEMA','COASTAL KARNATAKA']  # Add any other subdivisions you want to delete

#Use boolean indexing to exclude rows with specified subdivisions
rainfall = rainfall[~rainfall['SUBDIVISION'].isin(subdivisions_to_delete)]


# Step 3: Verify the changes
print("\nUpdated Rainfall DataFrame (after renaming):")
print(rainfall['SUBDIVISION'].unique())  # Show unique values to verify changes


# Step 1: Define the subdivisions to merge and their new names
subdivision_groups = {
    'Nagaland Manipur Mizoram Tripura': ['Nagaland', 'Manipur', 'Mizoram', 'Tripura'],
    'Delhi Haryana Chandigarh': ['Delhi', 'Haryana', 'Chandigarh'],'Assam Meghalaya':['Assam','Meghalaya']
}

# Step 2: Specify the soil moisture level columns to average
average_fields = ['Avg_smlvl_at15cm']  # Update 'another_field' with your actual column names

# Step 3: Create a list to store new averaged rows
new_averaged_rows = []

# Loop through the subdivisions and calculate average for each group
for new_name, old_names in subdivision_groups.items():
    # Select the rows to merge
    rows_to_merge = soil[soil['SUBDIVISION'].isin(old_names)]

    # Check the rows being averaged
    print(f"\nRows to merge for {new_name}:")
    print(rows_to_merge)

    # Calculate average values for the specified columns
    average_row = rows_to_merge[average_fields].mean().to_frame().T
    average_row['SUBDIVISION'] = new_name  # Set the new name for the merged entry

    # Add the new average row to the list
    new_averaged_rows.append(average_row)

# Step 4: Create a DataFrame from the new averaged rows
new_rows_df = pd.concat(new_averaged_rows, ignore_index=True)

# Step 5: Drop old rows from the original DataFrame
soil = soil[~soil['SUBDIVISION'].isin([name for names in subdivision_groups.values() for name in names])]

# Step 6: Append the new averaged rows to the DataFrame
soil = pd.concat([soil, new_rows_df], ignore_index=True)

# Display the updated DataFrame
print("\nUpdated Soil Moisture DataFrame:")

# Step 1: Define the subdivisions you want to delete
subdivision_to_delete = ['Dadra & Nagar Haveli', 'Daman & Diu', 'Puducherry','Ladakh']

# Step 2: Filter the DataFrame to remove the specified subdivisions
soil = soil[~soil['SUBDIVISION'].isin(subdivision_to_delete)]

# Step 3: Verify that the specified entries have been removed
print("\nUpdated Soil Moisture DataFrame (after deletion):")
print(soil['SUBDIVISION'].unique())
print(rainfall['SUBDIVISION'].unique())

# Merge the datasets
data = pd.merge(soil, rainfall, on='SUBDIVISION', how='inner')
# Change 'on' to your key column names, and adjust 'how' as needed.

# Step 4: Handle missing values (if necessary)
print("\nMissing values before handling:")
print(data.isnull().sum())




# Define thresholds (can be modified)
default_rainfall_threshold = 25
default_soil_moisture_threshold = 70

# Calculate the flood_occurrence column
def calculate_flood_occurrence(row, month_threshold=default_rainfall_threshold, soil_threshold=default_soil_moisture_threshold):
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for month in months:
        if row[month] > month_threshold and row['Avg_smlvl_at15cm'] > soil_threshold:
            return 1
    return 0

data['flood_occurrence'] = data.apply(calculate_flood_occurrence, axis=1)

# Target variable
TARGET_VARIABLE = 'flood_occurrence'
X = data.drop(columns=[TARGET_VARIABLE])  # Features
y = data[TARGET_VARIABLE]  # Target

# Ensure all columns in X are numeric, converting non-numeric to NaN and filling them
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)  # Replace NaNs with 0

# Identify columns that need encoding and scaling
numeric_features = ['Avg_smlvl_at15cm'] + [col for col in X.columns if col not in ['SUBDIVISION']]
categorical_features = [col for col in X.columns if col.startswith('SUBDIVISION')]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit the preprocessor on the training data
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Model training
model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and preprocessor
with open('flood_prediction_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('preprocessor.pkl', 'wb') as preprocessor_file:
    pickle.dump(preprocessor, preprocessor_file)

# Preparing for Deployment
# Define a function to make predictions
def predict_flood(month, subdivision):
    # Create an input DataFrame
    input_data = pd.DataFrame({
        'Avg_smlvl_at15cm': [80],  # Assume average soil moisture for the region, can be customized
        month: [30]  # Assume average rainfall value for the given month, can be customized
    })

    # Convert input data to numeric and handle missing values
    input_data = input_data.apply(pd.to_numeric, errors='coerce')
    input_data = input_data.fillna(0)  # Replace NaNs with 0

    # Create one-hot encoded columns for the subdivision
    for col in X.columns:
        if col.startswith('SUBDIVISION_'):
            input_data[col] = 1 if col == f'SUBDIVISION_{subdivision}' else 0
        elif col not in input_data:
            input_data[col] = 0  # Ensure all columns in the training data exist

    # Match the order of columns in the model's training data
    input_data = input_data[X.columns]

    # Standardize the input data using the trained preprocessor
    input_scaled = preprocessor.transform(input_data)
    
    # Predict using the trained model
    prediction = model.predict(input_scaled)
    return 'Flood' if prediction[0] == 1 else 'No Flood'

# Example Usage
print(predict_flood('JAN', 'Andaman & Nicobar'))

print(data['SUBDIVISION'].unique())

