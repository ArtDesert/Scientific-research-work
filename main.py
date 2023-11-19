import math
from math import cos, sin, sqrt
import matplotlib.pyplot as plt


def f(p, a, d, T):
    b_1 = (a + 1 / a) / 2.0
    b_2 = (a - 1 / a) / 2.0
    return -p * (b_1 * sin(p * T) - b_2 * sin(p * (2 * d - T)))


def get_solve_by_dichotomy_method(f, l, r, eps, a, d, T):
    prev_m = (l + r) / 2
    if f(l, a, d, T) > 0:
        if f(prev_m, a, d, T) < 0:
            r = prev_m
        else:
            l = prev_m
    else:
        if f(prev_m, a, d, T) < 0:
            l = prev_m
        else:
            r = prev_m
    while True:
        cur_m = (l + r) / 2
        if f(l,a,d,T) > 0:
            if f(cur_m,a,d,T) < 0:
                r = cur_m
            else:
                l = cur_m
        else:
            if f(cur_m,a,d,T) < 0:
                l = cur_m
            else:
                r = cur_m
        if abs(cur_m - prev_m) < eps:
            break
        prev_m = cur_m
    return round(cur_m, 3)


def get_all_solutions(f, l, r, eps, a, d, T):
    solutions = [0.0]
    n = 100
    x_arr = get_arr(l, n + 1, (r - l) / n)
    for i in range(n):
        cur_left = x_arr[i]
        cur_right = x_arr[i + 1]
        if f(cur_left, a, d, T) * f(cur_right, a, d, T) < 0:
            solutions.append(get_solve_by_dichotomy_method(f, cur_left, cur_right, eps, a, d, T))
    return solutions


def get_arr(left, n, h):
    arr = [0] * n
    arr[0] = left
    for i in range(1, n):
        arr[i] = arr[i - 1] + h
    return arr

#n - число отрезков разбиения
#[left, right] - область определения
# a, d, T - параметры краевой задачи соответственно
def draw_graphic(left, right, n, a, d, T):
    p_arr = get_arr(left, n + 1, (right - left) / n)
    plt.figure(1)
    plt.plot(p_arr, [f(p, a, d, T) for p in p_arr], label=f"a={round(a, 6)}, d={round(d, 6)}, T={round(T, 6)}", color="green")
    ax = plt.gca()
    ax.axhline(y=0, color='black')
    plt.legend()
    plt.show()


eps = 10 ** (-6)
a = 5
d = 2
T = 3

print(get_all_solutions(f, 0, 20, eps, a, d, T))
draw_graphic(0, 20, 500, a, d, T)

