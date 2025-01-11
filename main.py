import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

# Read the data from your txt file (assuming it's already in the correct format)
data = pd.read_csv("Toto658.txt")

# Rename the columns to match the desired format
data.rename(columns={
    "DrawNo": "Draw_Number",
    "DrawnNo1": "Winning_Number_1",
    "DrawnNo2": "Winning_Number_2",
    "DrawnNo3": "Winning_Number_3",
    "DrawnNo4": "Winning_Number_4",
    "DrawnNo5": "Winning_Number_5",
    "DrawnNo6": "Winning_Number_6"
}, inplace=True)
data['DrawNo'] = range(1, len(data) + 1)

# Handle all-zero rows by removing them
data = data[(data != 0).any(axis=1)]  # Remove rows where all columns are zero

# Prepare features (e.g., draw numbers) and target (winning numbers)
X = data[['DrawNo']].values  # Input features
y = data[['Winning_Number_1', 'Winning_Number_2', 'Winning_Number_3', 
          'Winning_Number_4', 'Winning_Number_5', 'Winning_Number_6']].values  # Target

# One-hot encode target values (for numbers, since these are categorical)
encoder = OneHotEncoder(sparse=False, categories='auto')
y_encoded = np.hstack([encoder.fit_transform(y[:, i].reshape(-1, 1)) for i in range(6)])

# Step 2: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 3: Define Neural Network Model (Independent Outputs for Each Number)
input_layer = Input(shape=(X_train.shape[1],))

# Define independent softmax layers for each of the 6 numbers
output_layers = []
for i in range(6):
    output_layers.append(Dense(len(np.arange(1, 46)), activation='softmax', name=f'output_{i+1}')(input_layer))

# Create the model
model = Model(inputs=input_layer, outputs=output_layers)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'] * 6)  # 6 metrics, one for each output

# Step 4: Train the Model
history = model.fit(X_train, [y_train[:, i*45:(i+1)*45] for i in range(6)], 
                    epochs=50, batch_size=16, 
                    validation_data=(X_test, [y_test[:, i*45:(i+1)*45] for i in range(6)]))

# Step 5: Predict Future Numbers
future_draw = np.array([[2497]])  # Draw number you want to predict

# Get the model's predictions (probabilities for each of the 6 numbers)
predictions = model.predict(future_draw)

# Print the shape of the predictions to debug
print("Predictions shape:", [p.shape for p in predictions])

# Extract the predicted numbers (the ones with the highest probability)
predicted_numbers = []
possible_numbers = np.arange(1, 46)  # Adjusted to 45 numbers instead of 59

for i in range(6):
    # Find the index of the highest probability (most likely number)
    predicted_number_index = np.argmax(predictions[i])
    
    # Map the index back to the actual lottery number
    predicted_number = possible_numbers[predicted_number_index]
    
    predicted_numbers.append(predicted_number)

# Print the predicted winning numbers for draw 2497
print("Predicted Winning Numbers for Draw 2497:", predicted_numbers)




# # Handle missing values by replacing zeros or NaNs (if any) with column means
# df = data.fillna(df.mean())  # Replace NaNs with column means
# df = data.replace(0, df.mean())  # Replace zeros with column means

# # Separate features (X) and target variable (y)
# X = data.drop(columns=["Draw_Number"])  # Dropping 'Draw_Number' as it's not part of the prediction
# y = data["Draw_Number"]  # Assuming we are predicting the draw number

# # One-hot encode the features (if necessary, adjust this based on your dataset)
# encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
# X_encoded = encoder.fit_transform(X)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# # Scale the data (pass with_mean=False to handle sparse matrix)
# scaler = StandardScaler(with_mean=False)  # Pass with_mean=False to handle sparse matrices
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Train a machine learning model (RandomForestClassifier as an example)
# model = RandomForestClassifier()
# model.fit(X_train_scaled, y_train)

# # Evaluate the model
# accuracy = model.score(X_test_scaled, y_test)
# print(f"Model Accuracy: {accuracy * 100:.2f}%")

# print(f"X_train_scaled shape: {X_train_scaled.shape}")
# print(f"X_test_scaled shape: {X_test_scaled.shape}")

# train_accuracy = model.score(X_train_scaled, y_train)
# print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# # Initialize RandomForest with hyperparameters
# model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
# model.fit(X_train_scaled, y_train)

# # Evaluate the model
# accuracy = model.score(X_test_scaled, y_test)
# print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")