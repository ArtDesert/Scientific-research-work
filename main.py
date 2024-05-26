import math
from math import cos, sin, sqrt
import matplotlib.pyplot as plt

global a
global d
global T
global h
global H
global c
global sigma
global n

# Аналитическое решение для sigma(x) = c = const
def varphi(x, _lambda):
    p = sqrt(_lambda)
    return cos(p * x) + (h + c) * sin(p * x) / p

def der_varphi(x, _lambda):
    p = sqrt(_lambda)
    return -p * sin(p * x) + (h + c) * cos(p * x)

def psi(x, _lambda):
    p = sqrt(_lambda)
    A = cos(p * T) - (c - H) * sin(p * T) / p
    B = (c - H) * cos(p * T) / p + sin(p * T)
    return A * cos(p * x) + B * sin(p * x)

def der_psi(x, _lambda):
    p = sqrt(_lambda)
    A = cos(p * T) - (c - H) * sin(p * T) / p
    B = (c - H) * cos(p * T) / p + sin(p * T)
    return - A * p * sin(p * x) + B * p * cos(p * x)

def quasi_der_varphi(x, _lambda):
    return der_varphi(x, _lambda) - sigma(x) * varphi(x, _lambda)

def quasi_der_psi(x, _lambda):
    return der_psi(x, _lambda) - sigma(x) * psi(x, _lambda)

def Delta(_lambda):
    '''Возвращает значение характеристической функции, вычисленной аналитически для случай sigma(x) = c'''
    return pow(a, -1) * quasi_der_varphi(d, _lambda) * psi(d, _lambda) - a * varphi(d, _lambda) * quasi_der_psi(d, _lambda)

# Задача с нулевым потенциалом
def Delta0(_lambda):
    '''Возвращает значение характеристической функции для задачи с нулевым потенциалом'''
    p = sqrt(_lambda)
    return -p * (sin(p * T) - sin(p * (2 * d - T)))

# Численное решение
def get_arr(left, right, n):
    '''Возвращет массив из значений от left до right, дискретизированных с шагом h = (right - left) / n
    left: левая граница отрезка,
    right: правая граница отрезка,
    n: число отрезков в разбиении (число элементов на 1 больше)'''
    h = (right - left) / n
    arr = [0] * (n + 1)
    arr[0] = left
    for i in range(1, n + 1):
        arr[i] = arr[i - 1] + h
    return arr

def f(u1, u2, sigma, _lambda):
    '''Возвращает вектор правых частей для метода Рунге-Кутты:
    1-ое значение для функции u1,
    2-ое значение для функции u2'''
    return [u2 + sigma * u1,
            -(sigma * u2 + sigma ** 2 * u1 + _lambda * u1)]

def get_solution_by_runge_kutta_4th_order_accuracy_left(left, right, init_conditions, _lambda, n):
    """Вычисляет решение задачи Коши на левом отрезке методом Рунге-Кутты 4-го порядка точности при фиксированном значении спектрального параметра _lambda"""
    stride = (right - left) / n # Шаг разбиения
    x_arr = get_arr(left, right, n)
    u1_arr = [0] * (n + 1) # Массив для решения
    u2_arr = [0] * (n + 1) # Массив для квазипроизводной решения

    # Инициализация начальных условий
    u1_arr[0] = init_conditions[0]
    u2_arr[0] = init_conditions[1]

    for i in range(1, n + 1):
        # Вспомогательные переменные
        x = x_arr[i - 1]
        x1 = x + stride / 2
        x2 = x_arr[i]
        u1 = u1_arr[i - 1]
        u2 = u2_arr[i - 1]

        # Вычисление вспомогательных коэффициентов
        k1 = f(u1, u2, sigma(x), _lambda)
        k2 = f(u1 + stride * k1[0] / 2, u2 + stride * k1[1] / 2, sigma(x1), _lambda)
        k3 = f(u1 + stride * k2[0] / 2, u2 + stride * k2[1] / 2, sigma(x1), _lambda)
        k4 = f(u1 + stride * k3[0], u2 + stride * k3[1], sigma(x2), _lambda)

        # Вычисление значений решений в текущем узле
        u1_arr[i] = u1 + stride * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        u2_arr[i] = u2 + stride * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6

    return (u1_arr, u2_arr)

def get_solution_by_runge_kutta_4th_order_accuracy_right(left, right, init_conditions, _lambda, n):
    """Вычисляет решение задачи Коши на правом отрезке методом Рунге-Кутты 4-го порядка точности при фиксированном значении спектрального параметра _lambda"""
    stride = (right - left) / n # Шаг разбиения
    x_arr = get_arr(left, right, n)
    u1_arr = [0] * (n + 1) # Массив для значений решения
    u2_arr = [0] * (n + 1) # Массив для значений квазипроизводной решений

    # Инициализация начальных условий
    u1_arr[-1] = init_conditions[0]
    u2_arr[-1] = init_conditions[1]

    for i in range(n - 1, -1, -1):
        # Вспомогательные переменные
        x = x_arr[i + 1]
        x1 = x - stride / 2
        x2 = x_arr[i]
        u1 = u1_arr[i + 1]
        u2 = u2_arr[i + 1]

        # Вычисление вспомогательных коэффициентов
        k1 = f(u1, u2, sigma(x), _lambda)
        k2 = f(u1 - stride * k1[0] / 2, u2 - stride * k1[1] / 2, sigma(x1), _lambda)
        k3 = f(u1 - stride * k2[0] / 2, u2 - stride * k2[1] / 2, sigma(x1), _lambda)
        k4 = f(u1 - stride * k3[0], u2 - stride * k3[1], sigma(x2), _lambda)

        # Вычисление значений решений в текущем узле
        u1_arr[i] = u1 - stride * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        u2_arr[i] = u2 - stride * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6

    return (u1_arr, u2_arr)


def calc_Delta(_lambda):
    '''Возвращает значение характеристической функции для задачи с ненулевым потенциалом в точке _lambda за O(n),
    где n - число отрезков в разбиении отрезков [0,d] и [d,T] в методе Рунге-Кутты 4-го порядка.'''
    (varphi, quasi_der_varphi) = get_solution_by_runge_kutta_4th_order_accuracy_left(0, d, [1, h], _lambda, n)
    (psi, quasi_der_psi) = get_solution_by_runge_kutta_4th_order_accuracy_right(d, T, [1, -H], _lambda, n)
    return math.pow(a, -1) * quasi_der_varphi[-1] * psi[0] - a * varphi[-1] * quasi_der_psi[0]

def get_zero_of_function_by_dichotomy_method(func, left, right, eps):
    '''Возвращает нуль функции func на отрезке [left, right] c точность eps методом дихотомии'''
    prev_m = (left + right) / 2
    if func(left) > 0:
        if func(prev_m) < 0:
            right = prev_m
        else:
            left = prev_m
    else:
        if func(prev_m) < 0:
            left = prev_m
        else:
            right = prev_m
    while True:
        cur_m = (left + right) / 2
        if func(left) > 0:
            if func(cur_m) < 0:
                right = cur_m
            else:
                left = cur_m
        else:
            if func(cur_m) < 0:
                left = cur_m
            else:
                right = cur_m
        if abs(cur_m - prev_m) < eps:
            break
        prev_m = cur_m
    return round(cur_m, 3)

def get_eigenvalues(func, left=0.0, right=20.0, eps=10**-6):
    '''Вовзращает собственные значения функции func на отрезке [left, right] c точностю eps'''
    eigenvalues = [0.0]
    x_arr = get_arr(left, right, n)
    y_arr = [func(x) for x in x_arr]
    for i in range(n):
        if y_arr[i] * y_arr[i + 1] < 0:
            eigenvalues.append(get_zero_of_function_by_dichotomy_method(func, x_arr[i], x_arr[i + 1], eps))
    return eigenvalues

def draw(x_arr, y_arr, label="", color='orange'):
    '''Строит график функции по массивам значений x_arr и y_arr'''
    plt.figure(1)
    plt.plot(x_arr, y_arr, label=label, color=color)
    ax = plt.gca()
    ax.axhline(y=0, color='black')
    plt.legend()
    plt.show()

#Глобальные переменные
a = 1
d = 1
T = 5
h = 1
H = 2
c = 5
sigma = lambda x : c
n = 1000 # Число отрезков разбиения для метода Рунге-Кутты

# Локальные переменные
_lambda = 3
x_arr_left = get_arr(0, d, n)
x_arr_right = get_arr(d, T, n)
lambda_arr = get_arr(1, 50, n)

# Массивы значений
#y_varphi = [varphi(x, _lambda) for x in x_arr_left]
#y_quasi_der_varphi = [quasi_der_varphi(x, _lambda) for x in x_arr_left]
#y_psi = [psi(x, _lambda) for x in x_arr_right]
#y_quasi_der_psi = [quasi_der_psi(x, _lambda) for x in x_arr_right]
y_Delta = [Delta(_lambda) for _lambda in lambda_arr]
y_Delta0 = [Delta0(_lambda) for _lambda in lambda_arr]

#[numeric_varphi, numeric_quasi_der_varphi] = get_solution_by_runge_kutta_4th_order_accuracy_left(0, d, [1, h], _lambda, n)
#[numeric_psi, numeric_quasi_der_psi] = get_solution_by_runge_kutta_4th_order_accuracy_right(d, T, [1, -H], _lambda, n)
numerical_Delta = [calc_Delta(_lambda) for _lambda in lambda_arr]

#draw(x_arr_left, y_varphi, label='Phi')
#draw(x_arr_left, y_quasi_der_varphi, label='Phi^[1]')
#draw(x_arr_right, y_psi, label='Psi')
#draw(x_arr_right, y_quasi_der_psi, label='Psi^[1]')
draw(lambda_arr, y_Delta, label='Характеристическая функция (аналит.)', color='green')

#draw(x_arr_left, numeric_varphi, label='Численное решение Phi', color='blue')
#draw(x_arr_left, numeric_quasi_der_varphi, label='Численное решение Phi^[1]', color='blue')
#draw(x_arr_right, numeric_psi, label='Численное решение Psi', color='blue')
#draw(x_arr_right, numeric_quasi_der_psi, label='Численное решение Psi^[1]', color='blue')
draw(lambda_arr, numerical_Delta, label='Характеристическая функция (числ.)', color='blue')

draw(lambda_arr, y_Delta0, label='Характеристическая функция задачи с нулевым потенциалом', color='green')

# Получение собственных значений
#eigenvalues = get_eigenvalues(calc_Delta, right=2000)
#eigenvalues0 = get_eigenvalues(Delta0, right=2000)

#Вывод последних 20 значений
#print(eigenvalues[-20:])
#rint(eigenvalues0[-20:])