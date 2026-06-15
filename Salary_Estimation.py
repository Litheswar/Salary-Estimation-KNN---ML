import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

# FIX 1: Use relative path instead of hardcoded absolute path
path = "salary.csv"
df = pd.read_csv(path)

print(df.shape)
print(df.head())

income_set = set(df['income'])

# FIX 2: Strip whitespace before mapping to avoid NaN from leading/trailing spaces
df['income'] = df['income'].str.strip().map({'<=50K': 0, '>50K': 1}).astype(int)
print(df.head())

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

std = StandardScaler()
x_strain = std.fit_transform(x_train)
x_stest = std.transform(x_test)

error = []

# Calculating error for K-value between 1 to 40
for i in range(1, 40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(x_strain, y_train)
    pred = model.predict(x_stest)
    error.append(np.mean(pred != y_test))  # Mean Error

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K value')
plt.xlabel('K value')
plt.ylabel('Mean Error')
plt.show()


# Training
model = KNeighborsClassifier(n_neighbors=8, metric='minkowski', p=2)
model.fit(x_strain, y_train)

# FIX 3: Use scaled test data (x_stest) instead of raw x_test for prediction
y_pred = model.predict(x_stest)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


# Evaluation
print("Accuracy: {0}%".format(accuracy_score(y_test, y_pred) * 100))


# Predicting
# FIX 4: Added descriptive prompts so the user knows what to enter
age = int(input("Enter age: "))
edu = int(input("Enter education level (num): "))
cg  = int(input("Enter capital gain: "))
wh  = int(input("Enter hours per week: "))

new_Emp = [[age, edu, cg, wh]]
result = model.predict(std.transform(new_Emp))
print(result)

# FIX 5 & 6: Index result[0] for proper scalar comparison; fix "Customer" -> "Employee"
if result[0] == 1:
    print("Employee might get salary above 50K")
else:
    print("Employee might not get salary above 50K")
