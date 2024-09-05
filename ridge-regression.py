from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import numpy as np

def ridge_regression(given_dataset, alpha):
    x = given_dataset.data
    y = given_dataset.target
    
    x_with_bias = np.hstack([np.ones((x.shape[0], 1)), x])
    x_with_bias_transposed = np.transpose(x_with_bias)
    
    y_results = x_with_bias_transposed @ y
    
    x_to_be_regularized = x_with_bias_transposed @ x_with_bias
    
    regularization_matrix = np.identity(x_with_bias.shape[1]) * alpha
    regularization_matrix[0, 0] = 0
    x_regularized = x_to_be_regularized + regularization_matrix
    
    x_regularized_inverse = np.linalg.inv(x_regularized)
    coefficients = x_regularized_inverse @ y_results
    
    y_predicted = x_with_bias @ coefficients
    
    model = Ridge(alpha=alpha)
    model.fit(x, y)
    y_predicted_by_library_model = model.predict(x)

    mse = mean_squared_error(y, y_predicted)
    mse_sklearn = mean_squared_error(y, y_predicted_by_library_model)
    
    print(f"my mean square error: {mse}")
    print(f"sklearn model mean square error: {mse_sklearn}")

def find_alpha(given_dataset):
    x = given_dataset.data
    y = given_dataset.target
    
    model = Ridge()
    alpha_values = np.linspace(0, 100, 1000)
    grid_search = GridSearchCV(estimator=model, param_grid={'alpha': alpha_values}, 
                           scoring='neg_mean_squared_error', cv=10)
    grid_search.fit(x, y)
    
    best_alpha_found = grid_search.best_params_['alpha']
    
    return best_alpha_found