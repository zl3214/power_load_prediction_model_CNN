import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Step 1: Data preprocessing
data = pd.read_csv('.csv')
data.drop(columns=['日期', '时间'], inplace=True)

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Step 2: Generate the hotspot graph
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.savefig('hotspot_graph.png')

# Step 3: Create a CNN model
def create_cnn(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Step 4: Train and validate the model
X = scaled_data[:, :-1]
y = scaled_data[:, -1]

# Reshape the input data to be used in CNN
X = X.reshape(X.shape[0], X.shape[1], 1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
model = create_cnn(input_shape)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Step 5: Make predictions
predictions = model.predict(X_test)


from grid_search_cv_with_cnn import GridSearchCVWithCNN

# Data preprocessing, generate the hotspot graph, and create the CNN model
# (Use the code provided in the previous responses)

# Prepare the data for grid search
X_gs = X_train
y_gs = y_train

# Initialize the GridSearchCVWithCNN class
grid_search_cv = GridSearchCVWithCNN(X_gs, y_gs, create_cnn_with_params)

# Define the parameter grid
param_grid = {
    'num_filters': [(16, 32), (32, 64)],
    'dense_units': [32, 64],
    'dropout_rate': [0.2, 0.3]
}

# Perform grid search with cross-validation
grid_result = grid_search_cv.grid_search(param_grid)

# Print the best hyperparameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Train the optimized model with the best hyperparameters
best_params = grid_result.best_params_
optimal_model = create_cnn_with_params(input_shape, **best_params)
optimal_history = optimal_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Make predictions with the optimized model
predictions = optimal_model.predict(X_test)
