import numpy as np

# Метод Зейделя
def seidel_method(A, b, tol=1e-4, max_iter=1000):
    """
    Розв’язує систему методом Зейделя.
    """
    n = len(b)
    x = np.zeros(n)
    iter_count = 0
    
    for _ in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])  # Використовуємо вже оновлені значення
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])  # Використовуємо старі значення
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        
        # Перевірка умови зупинки
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new, iter_count + 1
        x = x_new
        iter_count += 1
    
    return x, iter_count

# Введена користувачем матриця та вектор
A = np.array([
    [10, 2, -1, 3],
    [2, 8, 1, -4],
    [-1, 1, 7, 2],
    [3, -4, 2, 9]
], dtype=float)

b = np.array([7, -3, 5, 2], dtype=float)

# Перевірка діагонального домінування
def ensure_diagonal_dominance(A):
    n = len(A)
    for i in range(n):
        if abs(A[i, i]) < np.sum(np.abs(A[i])) - abs(A[i, i]):
            raise ValueError("Матриця не має діагонального домінування, метод Зейделя може не збігатися!")

ensure_diagonal_dominance(A)

# Запит точності у користувача
tolerance = float(input("Введіть бажану точність для методу Зейделя: "))

# Розв’язання системи методом Зейделя
seidel_solution, iterations = seidel_method(A, b, tol=tolerance)

# Вивід результатів
print("\nМатриця A:")
print(A)
print("\nВектор b:")
print(b)
print("\nРозв’язок методом Зейделя:")
print(seidel_solution)
print(f"Кількість ітерацій: {iterations}")
