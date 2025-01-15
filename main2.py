import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score

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
encoders = []
for i in range(6):
    encoder = OneHotEncoder(sparse=False)
    unique_values = sorted(np.unique(y[:, i]))
    encoder.fit([[x] for x in unique_values])
    encoders.append(encoder)
    y_encoded_list.append(encoder.transform(y[:, i].reshape(-1, 1)))

# Combine all encoded arrays
y_encoded = np.hstack(y_encoded_list)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
_, _, y_orig_train, y_orig_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

def evaluate_prediction_accuracy(actual, predicted):
    """
    Calculate various accuracy metrics for the prediction.
    """
    actual_set = set(actual)
    predicted_set = set(predicted)
    
    # Exact matches (same number in same position)
    exact_matches = sum(a == p for a, p in zip(actual, predicted))
    
    # Numbers that appear in both sets regardless of position
    correct_numbers = len(actual_set.intersection(predicted_set))
    
    metrics = {
        'exact_position_matches': exact_matches,
        'exact_position_accuracy': (exact_matches / 6) * 100,
        'numbers_matched': correct_numbers,
        'numbers_matched_accuracy': (correct_numbers / 6) * 100
    }
    
    return metrics

def predict_unique_numbers(model, draw_number, encoders, max_attempts=100):
    """
    Predict unique lottery numbers with no duplicates.
    """
    future_draw = np.array([[draw_number]])
    predictions = model.predict(future_draw, verbose=0)
    
    predicted_numbers = []
    
    for i, pred in enumerate(predictions):
        attempt = 0
        while attempt < max_attempts:
            current_probs = pred.flatten()
            
            for used_num in predicted_numbers:
                idx = np.where(encoders[i].categories_[0] == used_num)[0]
                if len(idx) > 0:
                    current_probs[idx[0]] = 0
            
            predicted_index = np.argmax(current_probs)
            predicted_number = encoders[i].categories_[0][predicted_index]
            
            if predicted_number not in predicted_numbers:
                predicted_numbers.append(predicted_number)
                break
                
            attempt += 1
            
        if attempt >= max_attempts:
            all_numbers = set(encoders[i].categories_[0])
            remaining_numbers = list(all_numbers - set(predicted_numbers))
            if remaining_numbers:
                predicted_number = np.random.choice(remaining_numbers)
                predicted_numbers.append(predicted_number)
    
    return sorted(predicted_numbers)

# Evaluate model accuracy on test set
test_accuracies = []
position_accuracies = []
number_accuracies = []

print("\n=== Model Accuracy Evaluation ===")

# Calculate accuracy for each test sample
for i in range(len(X_test)):
    predicted = predict_unique_numbers(model, X_test[i][0], encoders)
    actual = y_orig_test[i]
    metrics = evaluate_prediction_accuracy(actual, predicted)
    position_accuracies.append(metrics['exact_position_accuracy'])
    number_accuracies.append(metrics['numbers_matched_accuracy'])

# Calculate average accuracies
avg_position_accuracy = np.mean(position_accuracies)
avg_number_accuracy = np.mean(number_accuracies)

print(f"\nAverage Accuracy Metrics on Test Set:")
print(f"Position Accuracy: {avg_position_accuracy:.2f}% (exact position matches)")
print(f"Number Accuracy: {avg_number_accuracy:.2f}% (numbers matched regardless of position)")

# Predict future numbers
future_draw_number = len(data) + 1
predicted_numbers = predict_unique_numbers(model, future_draw_number, encoders)
print(f"\nPredicted Winning Numbers for Draw {future_draw_number}:", predicted_numbers)

# Calculate validation accuracies from training history
print("\nTraining History Accuracy:")
for i in range(6):
    val_accuracy = history.history[f'val_output_{i+1}_accuracy'][-1] * 100
    print(f"Position {i+1} Validation Accuracy: {val_accuracy:.2f}%")

# Save prediction probabilities
predictions = model.predict(np.array([[future_draw_number]]), verbose=0)
print("\nPrediction Probabilities for Each Position:")
for i, pred in enumerate(predictions):
    top_probs = np.sort(pred.flatten())[-5:]  # Get top 5 probabilities
    top_numbers = encoders[i].categories_[0][np.argsort(pred.flatten())[-5:]]
    print(f"\nPosition {i+1} Top 5 Numbers and Their Probabilities:")
    for num, prob in zip(top_numbers, top_probs):
        print(f"Number {num}: {prob*100:.2f}%")