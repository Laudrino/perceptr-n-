import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def activation_function(self, x):
        return np.where(x >= 0, 1, -1)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Inicialización aleatoria de pesos y sesgo
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn()

        y_ = np.array([1 if i > 0 else -1 for i in y])

        for iteration in range(self.n_iterations):
            errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                error = y_[idx] - y_predicted
                print(f"Iteración {iteration+1}, Muestra {idx+1}")
                print(f"Fórmula: {linear_output} = np.dot({x_i}, {self.weights}) + {self.bias}")
                print(f"Salida esperada: {y_[idx]}, Salida predicha: {y_predicted}, Error: {error}")

                if error != 0:
                    print(f"Pesos antes de actualizar: {self.weights}, Sesgo antes de actualizar: {self.bias}")
                    self.weights += self.learning_rate * error * x_i
                    self.bias += self.learning_rate * error
                    errors += 1
                    print(f"Pesos actualizados: {self.weights}, Sesgo actualizado: {self.bias}\n")

            if errors == 0:
                print("Entrenamiento completado sin errores.")
                break
            print("\n")

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted

# Datos para XOR con tres variables
X = np.array([
    [0, 0, 0*0],
    [0, 1, 0*1],
    [1, 0, 1*0],
    [1, 1, 1*1]
])
y = np.array([0, 1, 1, 0])  # Salidas XOR

perceptron = Perceptron(learning_rate=0.1, n_iterations=1000)
perceptron.fit(X, y)
predictions = perceptron.predict(X)
print("Predicciones finales:", predictions)
