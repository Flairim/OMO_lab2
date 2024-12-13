import numpy as np

accuracy_order = 5  

def Ashift(i, j, A, D, S):
    """Обчислення суми для S і D."""
    Sum = 0
    for k in range(i):
        Sum += S[k, i] * D[k, k] * S[k, j]
    return np.round(A[i, j] - Sum, accuracy_order)

def SDSdecomposition(A, comment=1):
    """Розклад матриці A у вигляді S^T * D * S."""
    n = A.shape[0]
    D = np.eye(n)
    S = np.zeros((n, n))
    step = 1
    if comment:
        print("Розклад матриці A у вигляді S^T * D * S:")
    for i in range(n):
        A0 = Ashift(i, i, A, D, S)
        D[i, i] = np.sign(A0)
        S[i, i] = np.sqrt(np.abs(A0))
        S[i, i] = np.round(S[i, i], accuracy_order)
        if comment:
            print(f"\nКрок {step}: Обчислення D[{i+1},{i+1}] і S[{i+1},{i+1}]")
            print(f"D[{i+1},{i+1}] = {D[i,i]}, S[{i+1},{i+1}] = {S[i,i]}")
            step += 1
        for j in range(i + 1, n):
            A0 = Ashift(i, j, A, D, S)
            S[i, j] = A0 / (S[i, i] * D[i, i])
            S[i, j] = np.round(S[i, j], accuracy_order)
            if comment:
                print(f"Крок {step}: Обчислення S[{i+1},{j+1}] = {S[i,j]}")
                step += 1
    return D, S

def ReverseGauss(D, S, A, b, comment=1):
    """Розв'язання системи лінійних рівнянь методом зворотного ходу."""
    n = A.shape[0]
    y = np.zeros(n)
    x = np.zeros(n)
    step = 1
    if comment:
        print("\nПрямий хід для обчислення проміжного вектора y:")
    for i in range(n):
        Sum = 0
        for k in range(i):
            Sum += S[k, i] * D[k, k] * y[k]
        y[i] = (b[i] - Sum) / (S[i, i] * D[i, i])
        y[i] = np.round(y[i], accuracy_order)
        if comment:
            print(f"Крок {step}: y[{i+1}] = {y[i]}")
            step += 1
    if comment:
        print("\nЗворотний хід для обчислення кінцевого розв'язку x:")
    for i in range(n - 1, -1, -1):
        Sum = 0
        for k in range(i + 1, n):
            Sum += S[i, k] * x[k]
        x[i] = (y[i] - Sum) / S[i, i]
        x[i] = np.round(x[i], accuracy_order)
        if comment:
            print(f"Крок {step}: x[{i+1}] = {x[i]}")
            step += 1
    return x

def Quadratic(A, b):
    """Розв'язання системи лінійних рівнянь методом квадратних коренів."""
    print("\n=== Розв'язання системи лінійних рівнянь методом квадратних коренів ===")
    D, S = SDSdecomposition(A)
    print("\nМатриці D і S після розкладу:")
    print(f"D = \n{D}")
    print(f"S = \n{S}")
    x = ReverseGauss(D, S, A, b)
    print("\nКінцевий розв'язок системи:")
    print(f"x = {x}")
    return x

def DeT(A):
    """Обчислення визначника матриці через розклад SDS."""
    print("\n=== Обчислення визначника матриці ===")
    D, S = SDSdecomposition(A, comment=1)
    n = A.shape[0]
    Det = 1
    for i in range(n):
        Det *= D[i, i] * S[i, i] * S[i, i]
    return np.round(Det, accuracy_order)

def InvA(A):
    """Обчислення оберненої матриці."""
    print("\n=== Обчислення оберненої матриці ===")
    D, S = SDSdecomposition(A, comment=1)
    n = A.shape[0]
    invA = np.eye(n)
    for i in range(n):
        invA[:, i] = ReverseGauss(D, S, A, invA[:, i], comment=1)
        print(f"\nСтовпець {i+1} оберненої матриці: {invA[:, i]}")
    return invA

A = np.array([[8, 4, 2, 1],
              [4, 16, 7, 2],
              [2, 7, 16, 4],
              [1, 2, 4, 8]])
b = np.array([7, 13, 17, 3])

print("\nРозв'язання системи лінійних рівнянь:")
x = Quadratic(A, b)

print(f"\nВизначник матриці A: {DeT(A)}")

print(f"\nОбернена матриця A:\n{InvA(A)}")
