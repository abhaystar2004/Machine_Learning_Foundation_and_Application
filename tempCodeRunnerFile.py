Descent()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
print(y_pred)