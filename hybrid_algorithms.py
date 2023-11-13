import numpy as np
import matplotlib.pyplot as plt

# Оптимизацияға арналған функция
def objective_function(x):
    return x**2 + 10 * np.sin(x)

# Генетикалық алгоритм
def genetic_algorithm(population_size, generations):
    population = np.random.uniform(-5, 5, size=(population_size, 1))

    for generation in range(generations):
        fitness = -objective_function(population.flatten())
        idx = np.argsort(fitness)
        population = population[idx[:population_size]]

        # Генетикалық алгоритмнің әртүрлі операторлары (кроссовер, мутация)

    return population[0]

# Локальдық оптимизация
def local_optimization(initial_solution):
    from scipy.optimize import minimize

    result = minimize(objective_function, initial_solution, method='BFGS')
    return result.x[0]

# Гибридті алгоритм
def hybrid_algorithm(population_size, generations):
    # Шаг 1: Генетикалық алгоритмді іске қосу
    initial_solution = genetic_algorithm(population_size, generations)

    # Шаг 2: Локальді оптимизация
    final_solution = local_optimization(initial_solution)

    return final_solution

# Гибридті алгоритмді іске қосу
population_size = 50
generations = 50
result = hybrid_algorithm(population_size, generations)

# Нәтижелерді визуализациялау
x_values = np.linspace(-5, 5, 100)
y_values = objective_function(x_values)
plt.plot(x_values, y_values, label='Мақсатты функция')
plt.scatter(result, objective_function(result), color='red', marker='x', label='Оңтайлы шешім')
plt.title('Гибридті алгоритмнің мысалы')
plt.xlabel('x')
plt.ylabel('Мақсатты функцияның мәні')
plt.legend()
plt.show()
