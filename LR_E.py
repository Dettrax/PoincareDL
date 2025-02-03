
import torch
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        # Define the linear layer
        self.linear = nn.Linear(input_dim, 32)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, x):
        # Implement the forward pass
        x = self.linear(x)
        x = self.silu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.silu(x)
        x = self.dropout(x)
        return self.linear3(x)


from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

# Load the diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Split the data first to prevent data leakage
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features using only the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)

# Convert the numpy arrays to PyTorch tensors and move them to the GPU
X_train = torch.FloatTensor(X_train_scaled).cuda()
X_test = torch.FloatTensor(X_test_scaled).cuda()
y_train = torch.FloatTensor(y_train_np).reshape(-1, 1).cuda()
y_test = torch.FloatTensor(y_test_np).reshape(-1, 1).cuda()



# Initialize the model
model = LinearRegression(input_dim=X.shape[1]).cuda()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0.001)

# Training loop
n_epochs = 5000
batch_size = 32
n_batches = len(X_train) // batch_size
losses = []

for epoch in range(n_epochs):
    epoch_loss = 0
    for i in range(n_batches):
        # Get batch
        start = i * batch_size
        end = start + batch_size
        x_batch = X_train[start:end]
        y_batch = y_train[start:end]

        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / n_batches
    losses.append(avg_loss)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {avg_loss:.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    train_predictions = model(X_train)
    test_predictions = model(X_test)

    train_mse = criterion(train_predictions, y_train).item()
    test_mse = criterion(test_predictions, y_test).item()

    # Calculate R² scores
    def r2_score(y_true, y_pred):
        ss_total = ((y_true - y_true.mean()) ** 2).sum()
        ss_residual = ((y_true - y_pred) ** 2).sum()
        return 1 - (ss_residual / ss_total)

    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)

print("\nModel Performance:")
print(f"Training MSE: {train_mse:.4f}")
print(f"Testing MSE: {test_mse:.4f}")
print(f"Training R² Score: {train_r2:.3f}")
print(f"Testing R² Score: {test_r2:.3f}")

