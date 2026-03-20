import pickle
import numpy as np

# Load the saved model
print("Loading model...")
with open('digit_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model loaded!")
print("\nThis model predicts handwritten digits (0-9)")
print("Each digit is represented as 64 numbers (8x8 pixels)")

# Example: predict a handwritten 5
example_digit = np.array([
    0., 0., 5., 13., 9., 1., 0., 0.,
    0., 0., 13., 15., 10., 15., 5., 0.,
    0., 3., 15., 2., 0., 11., 8., 0.,
    0., 4., 12., 0., 0., 8., 8., 0.,
    0., 5., 8., 0., 0., 9., 8., 0.,
    0., 4., 11., 0., 1., 12., 7., 0.,
    0., 2., 14., 5., 10., 12., 0., 0.,
    0., 0., 6., 13., 10., 0., 0., 0.
]).reshape(1, -1)

prediction = model.predict(example_digit)
print(f"\nPrediction: {prediction[0]}")
print("Expected: 5")