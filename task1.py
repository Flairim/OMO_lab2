import numpy as np

def print_step(A, b, step, message=""):
    """Функція для друку матриці A і вектора b на кожному кроці."""
    print(f"Крок {step}: {message}")
    print("Матриця A:")
    print(A)
    if b is not None:
        print("Вектор b:")
        print(b)
    print("-" * 40)

def gaussian_elimination_with_pivoting(A, b):
    """
    Розв'язання системи лінійних рівнянь методом Гауса з вибором головного елемента.

    Параметри:
    A (ndarray): Матриця коефіцієнтів (розмір n x n)
    b (ndarray): Вектор правої частини (розмір n)

    Повертає:
    x (ndarray): Розв'язок системи (вектор розмірності n)
    """
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    
    step = 1  
    
    for i in range(n):
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if i != max_row:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
            print_step(A, b, step, f"Після перестановки рядків {i+1} і {max_row+1}")
            step += 1
        
        pivot = A[i, i]
        if pivot == 0:
            raise ValueError("Матриця вироджена, розв'язок неможливий.")
        A[i] = A[i] / pivot
        b[i] = b[i] / pivot
        print_step(A, b, step, f"Після нормалізації рядка {i+1}")
        step += 1
        
        for j in range(i + 1, n):
            factor = A[j, i]
            A[j, :] -= factor * A[i, :]
            b[j] -= factor * b[i]
            A[j, i] = 0.0  
            print_step(A, b, step, f"Після обнулення елемента у рядку {j+1}, стовпці {i+1}")
            step += 1
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - np.sum(A[i, i + 1:] * x[i + 1:])
    
    print("Розв'язок системи:")
    print(x)
    return x

def gaussian_determinant(A):
    """
    Обчислення визначника методом Гауса з покроковим виведенням.

    Параметри:
    A (ndarray): Квадратна матриця (розмір n x n)

    Повертає:
    determinant (float): Визначник матриці
    """
    n = len(A)
    A = A.astype(float)
    determinant = 1
    step = 1

    for i in range(n):
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if i != max_row:
            A[[i, max_row]] = A[[max_row, i]]
            determinant *= -1  
            print_step(A, None, step, f"Після перестановки рядків {i+1} і {max_row+1}")
            step += 1
        
        # Нормалізація діагонального елемента
        pivot = A[i, i]
        if pivot == 0:
            print("Матриця вироджена, визначник дорівнює 0.")
            return 0
        determinant *= pivot
        for j in range(i + 1, n):
            factor = A[j, i] / pivot
            A[j, :] -= factor * A[i, :]
            A[j, i] = 0.0  
        
        print_step(A, None, step, f"Після обнулення елементів під діагоналлю у стовпці {i+1}")
        step += 1
    
    print("Матриця після зведення до верхньотрикутної форми:")
    print(A)
    print("-" * 40)
    
    print("Обчислення визначника:")
    print(f"Добуток діагональних елементів: {np.diag(A)}")
    print("-" * 40)
    
    return determinant

def gaussian_inverse_and_determinant(A):
    """
    Обчислення оберненої матриці методом Гауса з вибором головного елемента.

    Параметри:
    A (ndarray): Матриця коефіцієнтів (розмір n x n)

    Повертає:
    inverse (ndarray): Обернена матриця (n x n)
    """
    n = len(A)
    A = A.astype(float)
    
    I = np.eye(n)  
    extended_matrix = np.hstack((A, I))
    step = 1

    for i in range(n):
        max_row = i + np.argmax(np.abs(extended_matrix[i:, i]))
        if i != max_row:
            extended_matrix[[i, max_row]] = extended_matrix[[max_row, i]]
            print_step(extended_matrix[:, :n], extended_matrix[:, n:], step, f"Після перестановки рядків {i+1} і {max_row+1}")
            step += 1
        
        pivot = extended_matrix[i, i]
        if pivot == 0:
            raise ValueError("Матриця вироджена, обернена матриця не існує.")
        extended_matrix[i] = extended_matrix[i] / pivot
        print_step(extended_matrix[:, :n], extended_matrix[:, n:], step, f"Після нормалізації рядка {i+1}")
        step += 1
        
        for j in range(i + 1, n):
            factor = extended_matrix[j, i]
            extended_matrix[j] -= factor * extended_matrix[i]
            print_step(extended_matrix[:, :n], extended_matrix[:, n:], step, f"Після обнулення елемента у рядку {j+1}, стовпці {i+1}")
            step += 1

    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            factor = extended_matrix[j, i]
            extended_matrix[j] -= factor * extended_matrix[i]
            print_step(extended_matrix[:, :n], extended_matrix[:, n:], step, f"Після обнулення елемента над головним у рядку {j+1}, стовпці {i+1}")
            step += 1
    
    inverse = extended_matrix[:, n:]
    return inverse

A = np.array([[-10, -3, -3, -4],
              [-3, 9, -7, -5],
              [-9, 10, 1, 6],
              [10, 8, -1, 1]], dtype=float)
b = np.array([6, 5, 6, -3], dtype=float)

print("\nРозв'язок системи рівнянь:")
solution = gaussian_elimination_with_pivoting(A.copy(), b)

print("\nОбчислення визначника матриці:")
determinant = gaussian_determinant(A.copy())
print("\nВизначник матриці:", determinant)

print("\nОбчислення оберненої матриці:")
inverse_matrix = gaussian_inverse_and_determinant(A.copy())
print("\nОбернена матриця:")
print(inverse_matrix)
