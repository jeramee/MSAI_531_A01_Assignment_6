# assig6_test.py

# Name: Jeramee
# Assignment: 6
# Course: MSAI 531 A01 
# Title: Neural Networks Deep Learning
# Date: 9/22/24

import tensorflow as tf
import numpy as np

# Function to calculate the square of the difference (squared error loss)
def squared_difference_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2

# Function to calculate the probability of the correct digit (not a true loss function)
def probability_loss(y_true, probabilities):
    return probabilities[y_true]

# Function to calculate the negative log-probability (cross-entropy loss)
def negative_log_probability_loss(y_true, probabilities):
    return -tf.math.log(probabilities[y_true])

# Test data for all 3 examples
test_data = [
    {"correct_digit": tf.constant(7), "predicted_digit": tf.constant(5), 
     "predicted_probabilities": tf.Variable([0.02, 0.03, 0.1, 0.15, 0.05, 0.05, 0.1, 0.4, 0.05, 0.15], dtype=tf.float32)},
    {"correct_digit": tf.constant(3), "predicted_digit": tf.constant(2), 
     "predicted_probabilities": tf.Variable([0.1, 0.05, 0.2, 0.3, 0.1, 0.1, 0.05, 0.05, 0.02, 0.03], dtype=tf.float32)},
    {"correct_digit": tf.constant(5), "predicted_digit": tf.constant(4), 
     "predicted_probabilities": tf.Variable([0.05, 0.1, 0.2, 0.1, 0.1, 0.3, 0.1, 0.05, 0.05, 0.05], dtype=tf.float32)}
]

# Loop through each test case and calculate the different losses and gradients
for i, data in enumerate(test_data):
    correct_digit = data["correct_digit"]
    predicted_digit = data["predicted_digit"]
    predicted_probabilities = data["predicted_probabilities"]
    
    # Define an optimizer for each test case
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Using TensorFlow's GradientTape
    with tf.GradientTape() as tape:
        # Watch the variables to compute gradients with respect to predictions
        tape.watch(predicted_probabilities)

        # Calculate the losses
        squared_loss = squared_difference_loss(correct_digit, predicted_digit)
        neg_log_prob_loss = negative_log_probability_loss(correct_digit, predicted_probabilities)

    # Compute gradients for the negative log-probability loss
    grad_neg_log_prob_loss = tape.gradient(neg_log_prob_loss, predicted_probabilities)

    # Update the predicted probabilities using the optimizer
    optimizer.apply_gradients(zip([grad_neg_log_prob_loss], [predicted_probabilities]))

    # Clean up the tape to release memory
    del tape

    # Print the results in the desired format
    print()  # Blank line for separation between examples    
    print(f"Example Data {i + 1}:")
    print(f"Predicted Probabilities: \n{np.array(predicted_probabilities).flatten()}")
    print(f"Squared Difference Loss: {squared_loss.numpy()}")
    print(f"Negative Log-Probability Loss: {neg_log_prob_loss.numpy()}")

    print("Gradient of Negative Log-Probability Loss w.r.t. probabilities:")
    print(np.array(grad_neg_log_prob_loss).flatten())
