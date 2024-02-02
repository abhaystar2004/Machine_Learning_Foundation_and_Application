# Name: Abhay Sharma
# Roll No.: 22CH10001

import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('sales_data.csv')

# Printing first 8 rows of the dataset
print(dataset.head(8))

# Tabulate missing values for each group
missing_values = dataset.isnull().sum()
print(missing_values)

# Fixing missing values by replacing them with the mean value of the respective group
dataset['Price'].fillna(dataset['Price'].mean(), inplace=True)

# Plotting the data
plt.figure(figsize=(16, 8))
plt.plot('Date', 'Price', data=dataset, color='blue', marker='o', linestyle='solid',label="Price")
plt.xlabel('Price')
plt.ylabel('Date')
plt.legend()
plt.title('Price vs Date')
plt.show()

# Calculate the total number of orders
total_orders = dataset['OrderID'].count()
print(f"Total number of orders: {total_orders}")

# Calculate the total revenue generated from the sales
total_revenue = (dataset['Quantity'] * dataset['Price']).sum()
print(f"Total revenue generated from the sales: {total_revenue}")

# Calculate the average price of each product
avg_price = dataset.groupby('Product')['Price'].mean()

# Print the average price of each product
print(avg_price)

# Create a bar plot to visualize the average price of each product
avg_price.plot(kind='bar')
plt.xlabel('Product')
plt.ylabel('Average Price')
plt.title('Average Price of Each Product')
plt.show()

# Identify and print the top most sold products
top_sold_products = dataset.groupby('Product')['Quantity'].sum().sort_values(ascending=False)
print(f"Top most sold products: {top_sold_products}")
print(f"top sold product: {top_sold_products.iloc[0:1]}")