import numpy as np

# Метод Гаусса з вибором головного елемента по рядку
def gauss_method_row_pivoting(A, b):
    n = len(b)
    M = A.copy().astype(float)
    B = b.copy().astype(float)
    
    for k in range(n):
        # Пошук головного елемента в поточному рядку
        max_col = np.argmax(np.abs(M[k, k:])) + k
        
        # Перестановка стовпців
        if k != max_col:
            M[:, [k, max_col]] = M[:, [max_col, k]]
        
        # Прямий хід
        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k:] -= factor * M[k, k:]
            B[i] -= factor * B[k]

    # Зворотній хід
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (B[i] - np.dot(M[i, i + 1:], x[i + 1:])) / M[i, i]
    
    return x

# Обчислення оберненої матриці
def calculate_inverse(A):
    n = len(A)
    augmented = np.hstack((A, np.eye(n)))
    for k in range(n):
        max_row = np.argmax(np.abs(augmented[k:, k])) + k
        augmented[[k, max_row]] = augmented[[max_row, k]]
        augmented[k] /= augmented[k, k]
        for i in range(k + 1, n):
            augmented[i] -= augmented[k] * augmented[i, k]
    for k in range(n - 1, -1, -1):
        for i in range(k - 1, -1, -1):
            augmented[i] -= augmented[k] * augmented[i, k]
    return augmented[:, n:]

# Введена користувачем матриця та вектор
A = np.array([
    [10, 2, -1, 3],
    [2, 8, 1, -4],
    [-1, 1, 7, 2],
    [3, -4, 2, 9]
], dtype=float)

b = np.array([7, -3, 5, 2], dtype=float)

# Обчислення визначника
det_A = np.linalg.det(A)

# Обчислення оберненої матриці
inv_A = calculate_inverse(A) if det_A != 0 else None

# Розв’язання системи методом Гаусса
gauss_solution = gauss_method_row_pivoting(A, b)

# Вивід результатів
print("\nМатриця A:")
print(A)
print("\nВектор b:")
print(b)
print("\nВизначник матриці A:")
print(det_A)
if inv_A is not None:
    print("\nОбернена матриця A:")
    print(inv_A)
else:
    print("\nМатриця A є виродженою (визначник дорівнює 0).")
print("\nРозв’язок методом Гаусса з вибором головного елемента по рядку:")
print(gauss_solution)
