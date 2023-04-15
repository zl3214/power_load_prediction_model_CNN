import itertools
from sklearn.model_selection import KFold
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

class GridSearchCVWithCNN:

    def __init__(self, X, y, create_model_func):
        self.X = X
        self.y = y
        self.create_model_func = create_model_func

    def grid_search(self, param_grid, cv=5, scoring='neg_mean_absolute_error'):
        model = KerasRegressor(build_fn=self.create_model_func, verbose=0)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        grid_result = grid.fit(self.X, self.y)
        return grid_result

# Define a modified create_cnn function with additional parameters for grid search
def create_cnn_with_params(input_shape, num_filters=(32, 64), kernel_size=(3, 3), pool_size=(2, 2),
                           dense_units=64, dropout_rate=0.2):
    model = Sequential()
    model.add(Conv2D(num_filters[0], kernel_size, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size))
    model.add(Conv2D(num_filters[1], kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size))
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

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
