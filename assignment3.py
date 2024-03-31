def solve_n_queens(n):
    def is_safe(board, row, col):
        for i in range(col):
            if board[row][i] == 1:
                return False
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        for i, j in zip(range(row, n), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        
        return True

    def solve(board, col):
        if col == n:
            solutions.append(["".join("Q" if cell == 1 else "." for cell in row) for row in board])
            return

        for i in range(n):
            if is_safe(board, i, col):
                board[i][col] = 1
                solve(board, col + 1)
                board[i][col] = 0  

    solutions = []
    chessboard = [[0] * n for _ in range(n)]
    solve(chessboard, 0)
    return solutions
  
n = 4
queens_solutions = solve_n_queens(n)


for i, solution in enumerate(queens_solutions):
    print(f"Solution {i + 1}:")
    for row in solution:
        print(row)
    print("\n")

'''
Decision Tree algorithm
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
'''

'''
Random Forest Classification
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest classifier
clf = RandomForestClassifier(n_estimators=100)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

'''

'''
simple Neural Network model
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate synthetic dataset
np.random.seed(0)
X = np.random.randn(100, 2)  # 100 samples, 2 features
y = np.random.randint(2, size=100)  # Binary target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build neural network model
model = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),  # Input layer with 4 neurons and ReLU activation
    Dense(1, activation='sigmoid')  # Output layer with 1 neuron and Sigmoid activation
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
'''

'''
Linear Regression model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic dataset
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relationship with noise

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Calculate R-Squared value
r2 = r2_score(y_test, y_pred)
print("R-Squared value:", r2)

# Plot actual vs predicted values
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title('Actual vs Predicted Values')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
'''

'''
k-nearest neighbour algorithm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the k-NN classifier
k = 3  # Number of neighbors
clf = KNeighborsClassifier(n_neighbors=k)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
'''

'''
SVM Classifier model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Sample dataset with features and labels (0 for orange, 1 for mango)
# Features: [Weight in grams, Color (0 for orange, 1 for yellow)]
X = [[150, 0], [180, 0], [200, 0], [120, 1], [140, 1], [160, 1], [170, 1]]
y = [0, 0, 0, 1, 1, 1, 1]  # Labels: 0 for orange, 1 for mango

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build SVM classifier model
clf = SVC(kernel='linear')

# Train the model
clf.fit(X_train, y_train)

# Make predictions on test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Test with a new fruit (weight = 190 grams, color = yellow)
new_fruit = [[190, 1]]
prediction = clf.predict(new_fruit)
if prediction[0] == 0:
    print("Predicted fruit: Orange")
else:
    print("Predicted fruit: Mango")
'''

'''
deep learning neural network
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate synthetic dataset
np.random.seed(0)
X = np.random.randn(1000, 10)  # 1000 samples, 10 features
y = np.random.randint(2, size=1000)  # Binary target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build deep learning neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),  # First hidden layer with 64 neurons and ReLU activation
    Dense(32, activation='relu'),  # Second hidden layer with 32 neurons and ReLU activation
    Dense(1, activation='sigmoid')  # Output layer with 1 neuron and Sigmoid activation
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
'''
'''
Naive Bayes Model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Sample dataset with features and labels (0 for not buying, 1 for buying)
# Features: [Age, Income, Gender (0 for female, 1 for male)]
X = np.array([[25, 50000, 1], [35, 75000, 0], [30, 60000, 1], [20, 40000, 1], [40, 80000, 0], [45, 90000, 1]])
y = np.array([1, 1, 1, 0, 0, 1])  # Labels: 1 for buying, 0 for not buying

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
clf = GaussianNB()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
'''
'''
BFS and DFS
from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def bfs(self, start):
        visited = set()
        queue = [start]
        bfs_traversal = []

        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                bfs_traversal.append(vertex)
                visited.add(vertex)
                for neighbor in self.graph[vertex]:
                    if neighbor not in visited:
                        queue.append(neighbor)

        return bfs_traversal

    def dfs_util(self, vertex, visited, dfs_traversal):
        visited.add(vertex)
        dfs_traversal.append(vertex)

        for neighbor in self.graph[vertex]:
            if neighbor not in visited:
                self.dfs_util(neighbor, visited, dfs_traversal)

    def dfs(self, start):
        visited = set()
        dfs_traversal = []
        self.dfs_util(start, visited, dfs_traversal)
        return dfs_traversal

# Example usage:
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)

print("BFS Traversal:", g.bfs(2))
print("DFS Traversal:", g.dfs(2))
'''
'''
10.
import pandas as pd

s= "he is a good person"
df_string = pd.DataFrame({"text":[s]})
print(df_string)

car_data = {"car_name":["toyota", "kia" ,"hyundai"] , "price": [200000,250000,400000]}
car_data_df = pd.DataFrame(car_data) 
print(car_data_df)
print(car_data_df["car_name"].tolist())
print(car_data_df.iloc[1])

new_car_data  = {"car_name": ["BMW","Astanmartin"], "price": [600000,1000000]}
car_data_df = pd.concat([car_data_df, new_car_data_df], ignore_index=True)
print(car_data_df)

car_data_df.at[2,'price'] = 500000
print(car_data_df)

car_data_df.to_csv("car_data.csv",index=False)

df_read = pd.read_csv("car_data.csv")
print(df_read)
'''
'''
11.
import numpy as np
np_array = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(np_array)
np_array.size()
np_array.shape
np_array.itemsize
np_array.dtype
np_array_reshape=np_array.reshape(4,3)
print(np_array_reshape)

profit = np.array([100,200,150,300])
sales =np.array([20,40,50,60])
profit_margin_ratio = profit/sales
print(profit_margin_ratio)

plt.plot(profit,sales)
[<matplotlib.lines.Line2D object at 0x0000024D6F095F10>]
plt.title('profit vs sales')
Text(0.5, 1.0, 'profit vs sales')
plt.xlabel('profit')
Text(0.5, 0, 'profit')
plt.ylabel('sales')
Text(0, 0.5, 'sales')
plt.show()

df = pd.read_csv('example.csv')  # Replace 'example.csv' with your CSV file
print("\nDataFrame from CSV file:")
print(df)

profit = np.array([100,200,150,300])
sales =np.array([20,40,50,60])
profit1 = np.array([200,300,100,500])                     
sales1 = np.array([50,100,10,25])
plt.scatter(profit,sales,color='red',label='class1')
plt.scatter(profit1,sales1,color='blue',label='class2')
plt.title('scatter plot with 2 bounds')
plt.xlabel('profit')
plt.ylabel('sales')
plt.legend()
plt.show()
'''
'''
12.
calories_Data = {'day':['monday','tuesday','wednesday','thursday','friday'],'calories_consumed':[20,10,40,50,60],'calories_burnt':[5,10,15,20,40]}
calories_Data_df = pd.DataFrame(calories_date)
print(calories_Data_df)

calories_Data_df['calories_remaining']=calories_Data_df['calories_consumed']-calories_Data_df['calories_burnt']
print(calories_Data_df)

calories_Data_df.set_index('day',inplace=True)
print(calories_Data_df)

calories_Data_df.to_csv('D:\caloresData.csv')

print(pd._version_)
'''
