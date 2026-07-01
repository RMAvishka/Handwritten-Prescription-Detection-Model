import matplotlib.pyplot as plt

# Your training loss values (from logs)
epochs = list(range(1, 16))

loss_values = [
    0.3063,
    0.1837,
    0.1455,
    0.1203,
    0.1028,
    0.0929,
    0.0840,
    0.0723,
    0.0623,
    0.0541,
    0.0475,
    0.0446,
    0.0405,
    0.0395,
    0.0351
]

# Create the plot
plt.figure()
plt.plot(epochs, loss_values, marker='o')

# Labels and title
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")

# Grid for better readability
plt.grid()

# Show plot
plt.show()