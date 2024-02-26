# WIP ###
def lasso():
    # Define the range of exponents for alpha
    alphas = [2**i for i in range(-5, 4)]

    train_errors = []
    val_errors = []

    # Iterate over alphas and fit Lasso models
    for alpha in alphas:
        # Initialize Lasso model with the current alpha
        lasso = Lasso(alpha=alpha)
        
        # Fit the model on the training data
        lasso.fit(X_train, y_train)
        
        # Predictions on training and validation sets
        y_train_pred = lasso.predict(X_train)
        y_val_pred = lasso.predict(X_val)
        
        # Calculate mean squared errors
        train_error = mean_squared_error(y_train, y_train_pred)
        val_error = mean_squared_error(y_val, y_val_pred)
        
        # Append errors to lists
        train_errors.append(train_error)
        val_errors.append(val_error)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(alphas, train_errors, label='Training error', marker='o')
plt.plot(alphas, val_errors, label='Validation error', marker='o')
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve for Lasso Regression')
plt.xscale('log')  # Use a logarithmic scale for alpha values
plt.xticks(alphas, labels=[f'2^{i}' for i in range(-5, 4)])  # Set x-axis ticks to display as 2^i
plt.legend()
plt.grid(True)
plt.show()