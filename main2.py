import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

# Read the data from txt file
data = pd.read_csv("Toto658.txt")

# Extract only the columns we need
columns_to_use = ['DrawNo', 'DrawnNo1', 'DrawnNo2', ' DrawnNo3', ' DrawnNo4', ' DrawnNo5', ' DrawnNo6']
data = data[columns_to_use]

# Remove any spaces from column names
data.columns = data.columns.str.strip()

# Convert DrawNo to sequential numbers
data['DrawNo'] = range(1, len(data) + 1)

# Prepare features and target
X = data[['DrawNo']].values
y = data[['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']].values

# One-hot encode target values for each number separately
y_encoded_list = []
for i in range(6):
    encoder = OneHotEncoder(sparse=False)
    # Get unique values in current column to determine categories
    unique_values = sorted(np.unique(y[:, i]))
    encoder.fit([[x] for x in unique_values])
    y_encoded_list.append(encoder.transform(y[:, i].reshape(-1, 1)))

# Combine all encoded arrays
y_encoded = np.hstack(y_encoded_list)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define Neural Network Model
input_layer = Input(shape=(X_train.shape[1],))
hidden1 = Dense(128, activation='relu')(input_layer)
hidden2 = Dense(64, activation='relu')(hidden1)
dropout = Dropout(0.2)(hidden2)

# Get the number of unique values for each position
output_dims = [y_encoded_list[i].shape[1] for i in range(6)]

# Define independent softmax layers for each number
output_layers = []
for i, dim in enumerate(output_dims):
    output_layers.append(Dense(dim, activation='softmax', name=f'output_{i+1}')(dropout))

# Create the model
model = Model(inputs=input_layer, outputs=output_layers)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'] * 6)

# Split y_train and y_test into separate outputs for each number
y_train_split = []
y_test_split = []
current_pos = 0
for dim in output_dims:
    y_train_split.append(y_train[:, current_pos:current_pos + dim])
    y_test_split.append(y_test[:, current_pos:current_pos + dim])
    current_pos += dim

# Train the Model
history = model.fit(X_train, y_train_split,
                   epochs=50, batch_size=16,
                   validation_data=(X_test, y_test_split))

def predict_unique_numbers(model, draw_number, encoders, max_attempts=100):
    """
    Predict unique lottery numbers with no duplicates.
    
    Args:
        model: Trained Keras model
        draw_number: Draw number to predict for
        encoders: List of fitted OneHotEncoders
        max_attempts: Maximum number of attempts to find unique numbers
    
    Returns:
        List of unique predicted numbers
    """
    future_draw = np.array([[draw_number]])
    predictions = model.predict(future_draw)
    
    predicted_numbers = []
    
    for i, pred in enumerate(predictions):
        attempt = 0
        while attempt < max_attempts:
            # Get current prediction probabilities
            current_probs = pred.flatten()
            
            # Set previously used numbers to 0 probability
            for used_num in predicted_numbers:
                idx = np.where(encoders[i].categories_[0] == used_num)[0]
                if len(idx) > 0:
                    current_probs[idx[0]] = 0
            
            # Find the highest probability number that hasn't been used
            predicted_index = np.argmax(current_probs)
            predicted_number = encoders[i].categories_[0][predicted_index]
            
            if predicted_number not in predicted_numbers:
                predicted_numbers.append(predicted_number)
                break
                
            attempt += 1
            
        if attempt >= max_attempts:
            # If we couldn't find a unique number, select randomly from remaining numbers
            all_numbers = set(encoders[i].categories_[0])
            remaining_numbers = list(all_numbers - set(predicted_numbers))
            if remaining_numbers:
                predicted_number = np.random.choice(remaining_numbers)
                predicted_numbers.append(predicted_number)
    
    return sorted(predicted_numbers)

# Store the encoders for later use
encoders = []
for i in range(6):
    encoder = OneHotEncoder(sparse=False)
    unique_values = sorted(np.unique(y[:, i]))
    encoder.fit([[x] for x in unique_values])
    encoders.append(encoder)

# Predict future numbers
future_draw_number = len(data) + 1  # Next draw number
predicted_numbers = predict_unique_numbers(model, future_draw_number, encoders)
print(f"Predicted Winning Numbers for Draw {future_draw_number}:", predicted_numbers)