import numpy as np
import torch
from gtt import GTTModel, GTTTokenizer

# Load pretrained GTT model and tokenizer
# modelchoice = 'small'
# modelpath = f'./checkpoints/GTT-{modelchoice}'
model = GTTModel.from_pretrained("gtt-large")
tokenizer = GTTTokenizer.from_pretrained("gtt-large")

# Sample multivariate time series data (engine parameters)
# Shape: (num_channels, sequence_length)
engine_data = np.array([
    [100, 102, 105, 103, 101],  # Temperature
    [500, 510, 505, 515, 520],  # RPM
    [50, 52, 51, 53, 52],       # Oil Pressure
    [10, 11, 10, 12, 11]        # Vibration
])

# Tokenize the input data
input_ids = tokenizer.encode(engine_data)

# Make zero-shot prediction
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=10)

# Decode the predictions
predicted_values = tokenizer.decode(outputs[0])

print("Predicted engine parameters:")
print(predicted_values)

# Analyze predictions for anomalies or potential failures
# (This part would involve domain-specific thresholds and logic)