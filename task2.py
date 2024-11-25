import numpy as np

# Метод квадратних коренів
def square_root_method(A, b):
    """
    Розв’язує систему методом квадратних коренів.
    """
    n = len(A)
    # Перевірка симетричності матриці
    if not np.allclose(A, A.T):
        raise ValueError("Матриця A повинна бути симетричною для методу квадратних коренів.")

    # Розкладання на L та L^T
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            if i == j:  # Діагональні елементи
                L[i, j] = np.sqrt(A[i, i] - np.sum(L[i, :j] ** 2))
            else:  # Недіагональні елементи
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    
    # Розв’язок системи Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    
    # Розв’язок системи L^T x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(L[i + 1:, i], x[i + 1:])) / L[i, i]
    
    return x

# Введена користувачем матриця та вектор
A = np.array([
    [10, 2, -1, 3],
    [2, 8, 1, -4],
    [-1, 1, 7, 2],
    [3, -4, 2, 9]
], dtype=float)

b = np.array([7, -3, 5, 2], dtype=float)

# Перевірка симетричності
if not np.allclose(A, A.T):
    print("Матриця не є симетричною! Автоматично робимо її симетричною.")
    A = (A + A.T) / 2

# Розв’язання системи методом квадратних коренів
sqrt_solution = square_root_method(A, b)

# Вивід результатів
print("\nМатриця A (симетрична):")
print(A)
print("\nВектор b:")
print(b)
print("\nРозв’язок методом квадратних коренів:")
print(sqrt_solution)
