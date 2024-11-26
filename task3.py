import numpy as np

accuracy_order = 5  
eps = 1e-4  

A = np.array([
    [8, 4, 2, 1],
    [4, 16, 7, 2],
    [2, 7, 16, 4],
    [1, 2, 4, 8]
])
b = np.array([7, 13, 17, 3])

def norm_inf(vector):
    """Обчислює нескінченну норму вектора."""
    return np.round(np.max(np.abs(vector)), accuracy_order)

# Реалізація методу Зейделя
def zeidel_method(A, b, x0=None):
    """
    Розв'язання системи лінійних рівнянь методом Зейделя.

    Параметри:
        A (ndarray): Квадратна матриця системи.
        b (ndarray): Вектор вільних членів.
        x0 (ndarray): Початкове наближення (опціонально).

    Повертає:
        ndarray: Розв'язок системи.
    """
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else np.copy(x0)
    k = 0  
    norm = float('inf')  

    while norm > eps or k == 0:
        x_prev = np.copy(x)
        k += 1
        for i in range(n):
            sum_ = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sum_) / A[i, i]

        x = np.round(x, accuracy_order)
        norm = norm_inf(x - x_prev)
        
        print(f"Ітерація k = {k}, x^{k} = {x}, ||x^k - x^(k-1)|| = {norm}")

    return x

def get_initial_approximation(n):
    """
    Запитує початкове наближення у користувача.

    Параметри:
        n (int): Розмірність вектора.

    Повертає:
        ndarray: Початкове наближення.
    """
    print("Введіть початкове наближення для кожного елемента вектора:")
    x0 = []
    for i in range(n):
        value = float(input(f"x[{i}] = "))
        x0.append(value)
    return np.array(x0)

n = A.shape[0]
use_custom_approximation = input("Використовувати початкове наближення? (так/ні): ").strip().lower()
if use_custom_approximation == "так":
    x0 = get_initial_approximation(n)
else:
    x0 = None

x_solution = zeidel_method(A, b, x0)

# Виведення результатів
print("\nРезультати обчислень:")
print(f"Розв'язок системи x = {x_solution}")
print(f"Перевірка Ax = {A @ x_solution}")  
print(f"Різниця Ax - b = {A @ x_solution - b}")  
