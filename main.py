import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Step 1: Load and Prepare Data
# Replace 'lottery_data.csv' with your actual dataset file
# Dataset should have columns like: 'Draw_Number', 'Winning_Number_1', 'Winning_Number_2', ...
data = pd.read_csv('lottery_data.csv')
print(data)

# Prepare features (e.g., draw numbers) and target (winning numbers)
X = data[['Draw_Number']].values  # Input features
y = data[['Winning_Number_1', 'Winning_Number_2', 'Winning_Number_3', 
          'Winning_Number_4', 'Winning_Number_5', 'Winning_Number_6']].values  # Target

# One-hot encode target values (for numbers, since these are categorical)
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y)

# Step 2: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 3: Define Neural Network Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(y_train.shape[1], activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Step 5: Predict Future Numbers
# Example: Predict numbers for draw 1001
future_draw = np.array([[1001]])
predictions = model.predict(future_draw)
predicted_numbers = encoder.inverse_transform(predictions)

print("Predicted Winning Numbers for Draw 1001:", predicted_numbers[0])