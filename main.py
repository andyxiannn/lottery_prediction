import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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
print(data)


# Handle all-zero rows by removing them
data = data[(data != 0).any(axis=1)]  # Remove rows where all columns are zero


# Prepare features (e.g., draw numbers) and target (winning numbers)
X = data[['DrawNo']].values  # Input features
y = data[['Winning_Number_1', 'Winning_Number_2', 'Winning_Number_3', 
          'Winning_Number_4', 'Winning_Number_5', 'Winning_Number_6']].values  # Target

# One-hot encode target values (for numbers, since these are categorical)
# Initialize OneHotEncoder (encode each number separately)
encoder = OneHotEncoder(sparse=False, categories='auto')

# We fit and transform the encoder on each number in y
y_encoded = np.hstack([encoder.fit_transform(y[:, i].reshape(-1, 1)) for i in range(6)])


# Step 2: Train/Test Split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Print the shape to confirm everything is correct
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Step 3: Define Neural Network Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(y_train.shape[1], activation='softmax')  # 6 * number_of_possible values (each number)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Step 5: Predict Future Numbers
# Example: Predict numbers for draw 2497
future_draw = np.array([[2497]])  # Draw number you want to predict

# Get the model's predictions (probabilities for each of the 6 numbers)
predictions = model.predict(future_draw)

# Since predictions are in one-hot encoding format, use np.argmax() to get the index with highest probability
predicted_numbers = np.argmax(predictions, axis=1)

# Print the predicted numbers for draw 2497
print("Predicted Winning Numbers for Draw 2497:", predicted_numbers)
# Print the shape of the predictions to debug
print("Predictions shape:", predictions.shape)
predicted_numbers = []
possible_numbers = np.arange(1, 46)  # assuming lottery numbers are from 1 to 59
# We expect predictions to have shape (1, 354) because 6 * 59 = 354 possible outcomes
for i in range(6):
    # Get the slice corresponding to the probabilities for the i-th number
    start_idx = i * len(possible_numbers)  # Start index for the i-th number
    end_idx = (i + 1) * len(possible_numbers)  # End index for the i-th number
    
    # Get the slice of probabilities for the i-th number
    number_probabilities = predictions[0, start_idx:end_idx]
    
    # Find the index of the highest probability (most likely number)
    predicted_number_index = np.argmax(number_probabilities)
    
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