import numpy as np

import pandas as pd

series = pd.Series(['apple', 'banana', 'cherry', 'date'])
print(series)

temperatures = pd.Series([25, 28, 30, 26], index=['Mon', 'Tue', 'Wed', 'Thu'])
print(temperatures)

print(temperatures.at['Wed'])
weekend_temperatures = pd.Series(temperatures)
weekend_temperatures.loc['Sat'] = 28
weekend_temperatures.loc['Sun'] = 25
print(weekend_temperatures)
print(temperatures)
fahrenheit_temperatures = weekend_temperatures.transform(lambda x: x * 9/5 + 32)
print(fahrenheit_temperatures)

student_data = pd.DataFrame(data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 22, 28],
    'City': ['New York', 'London', 'Paris', 'Tokyo'],
    'Grade': [85, 92, 78, 95]
})
print(student_data)

employee_data = pd.DataFrame(data = [
    [1, 'Emily', 'Marketing', 60000],
    [2, 'Frank', 'Sales', 75000],
    [3, 'Grace', 'IT', 80000]
], columns = ['ID', 'Name', 'Department', 'Salary'])
print(employee_data)

random_data = pd.DataFrame(np.random.randn(5,3), columns=['A', 'B', 'C'])
print(random_data)

print("3.1 - \n", student_data.head(2), "\n", student_data.tail(1))
print("3.2 - \n")
print(student_data.info())
print("3.3 - \n")
print(student_data.describe())
print("3.4 - \n")
print(student_data.loc[:,['Name', 'Age']])
print("3.5 - \n")
print(student_data[student_data['Age'] > 25])
print("3.6 - \n")
student_data.loc[:,['Passed']] = [x >= 70 for x in student_data.loc[:,'Grade']]
print(student_data)
print("3.7 - \n")
student_data = student_data.reindex(columns=['Name', 'Age', 'Grade', 'Passed'])
print(student_data)

print("4.1 - \n")
sales_data = pd.DataFrame(data = {
    'Product': ['A', 'B', 'C', 'D', 'E'],
    'Sales': [100, 150, None, 200, 120],
    'Region': ['North', 'South', 'East', None, 'West']
})
print(sales_data)
print("4.2 - \n")
print(sales_data.isnull().sum())
print("4.3 - \n")
sales_data['Sales'] = sales_data['Sales'].fillna(sales_data['Sales'].mean())
print(sales_data)
print("4.4 - \n")
sales_data.dropna(inplace=True)


print("5.1 - \n")

