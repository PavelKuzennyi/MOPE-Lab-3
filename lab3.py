import numpy as np
from math import *
from prettytable import PrettyTable
import scipy.stats


# Variant №208

x1_min = -30
x1_max = 0
x2_min = 10
x2_max = 60
x3_min = 10
x3_max = 35

#x1_min = -25
#x1_max = 75
#x2_min = 5
#x2_max = 40
#x3_min = 15
#x3_max = 25

m = 3

y_min = 200 + int((x1_min + x2_min + x3_min) / 3)
y_max = 200 + int((x1_max + x2_max + x3_max) / 3)

x_norm = np.array([
    [1, -1, -1, -1],
    [1, -1, 1, 1],
    [1, 1, -1, 1],
    [1, 1, 1, -1]])

x_matrix = np.array([
    [x1_min, x2_min, x3_min],
    [x1_min, x2_max, x3_max],
    [x1_max, x2_min, x3_max],
    [x1_max, x2_max, x3_min]
])

matrix_plan = np.random.randint(y_min, y_max, size=(4, m))

#matrix_plan = np.array([
#    [15, 18, 16],
#    [10, 19, 13],
#    [11, 14, 12],
#    [16, 19, 16]])

y1_av = sum(matrix_plan[0, :] / 3)
y2_av = sum(matrix_plan[1, :] / 3)
y3_av = sum(matrix_plan[2, :] / 3)
y4_av = sum(matrix_plan[3, :] / 3)

# print(matrix_plan[0, :])
# print(matrix_plan)

mx1 = sum(x_matrix[:, 0] / 4)
mx2 = sum(x_matrix[:, 1] / 4)
mx3 = sum(x_matrix[:, 2] / 4)

my = (y1_av + y2_av + y3_av + y4_av) / 4

a1 = (x_matrix[0][0] * y1_av + x_matrix[1][0] * y2_av + x_matrix[2][0] * y3_av + x_matrix[3][0] * y4_av) / 4
a2 = (x_matrix[0][1] * y1_av + x_matrix[1][1] * y2_av + x_matrix[2][1] * y3_av + x_matrix[3][1] * y4_av) / 4
a3 = (x_matrix[0][2] * y1_av + x_matrix[1][2] * y2_av + x_matrix[2][2] * y3_av + x_matrix[3][2] * y4_av) / 4
a11 = (x_matrix[0][0] ** 2 + x_matrix[1][0] ** 2 + x_matrix[2][0] ** 2 + x_matrix[3][0] ** 2) / 4
a22 = (x_matrix[0][1] ** 2 + x_matrix[1][1] ** 2 + x_matrix[2][1] ** 2 + x_matrix[3][1] ** 2) / 4
a33 = (x_matrix[0][2] ** 2 + x_matrix[1][2] ** 2 + x_matrix[2][2] ** 2 + x_matrix[3][2] ** 2) / 4
a12 = a21 = (x_matrix[0][0] * x_matrix[0][1] + x_matrix[1][0] * x_matrix[1][1] +
             x_matrix[2][0] * x_matrix[2][1] + x_matrix[3][0] * x_matrix[3][1]) / 4
a13 = a31 = (x_matrix[0][0] * x_matrix[0][2] + x_matrix[1][0] * x_matrix[1][2] +
             x_matrix[2][0] * x_matrix[2][2] + x_matrix[3][0] * x_matrix[3][2]) / 4
a23 = a32 = (x_matrix[0][1] * x_matrix[0][2] + x_matrix[1][1] * x_matrix[1][2] +
             x_matrix[2][1] * x_matrix[2][2] + x_matrix[3][1] * x_matrix[3][2]) / 4

znam_matrix = [
    [1, mx1, mx2, mx3],
    [mx1, a11, a12, a13],
    [mx2, a12, a22, a32],
    [mx3, a13, a23, a33]
]

b0_matrix = [
    [my, mx1, mx2, mx3],
    [a1, a11, a12, a13],
    [a2, a12, a22, a32],
    [a3, a13, a23, a33]
]

b1_matrix = [
    [1, my, mx2, mx3],
    [mx1, a1, a12, a13],
    [mx2, a2, a22, a32],
    [mx3, a3, a23, a33]
]

b2_matrix = [
    [1, mx1, my, mx3],
    [mx1, a11, a1, a13],
    [mx2, a12, a2, a32],
    [mx3, a13, a3, a33]
]

b3_matrix = [
    [1, mx1, mx2, my],
    [mx1, a11, a12, a1],
    [mx2, a12, a22, a2],
    [mx3, a13, a23, a3]
]

b0 = np.linalg.det(b0_matrix) / np.linalg.det(znam_matrix)
b1 = np.linalg.det(b1_matrix) / np.linalg.det(znam_matrix)
b2 = np.linalg.det(b2_matrix) / np.linalg.det(znam_matrix)
b3 = np.linalg.det(b3_matrix) / np.linalg.det(znam_matrix)

table = PrettyTable()
my_table = np.hstack((x_matrix, matrix_plan))
table.field_names = ["X1", "X2", "X3", "Y1", "Y2", "Y3"]
for i in range(len(my_table)):
    table.add_row(my_table[i])

print(table)

print("\nb0:", "%.3f " % b0, "\nb1:", "%.3f" % b1, "\nb2:", "%.3f" % b2, "\nb3:", "%.3f\n" % b3)
print(f"Рівняння регресії: y = {b0:.3f}{b1:+.3f}*x1{b2:+.3f}*x2{b3:+.3f}*x3")

print("b0 + b1*X11 + b2*X12 + b3*X13 =",
      "%.2f" % (b0 + b1 * x_matrix[0][0] + b2 * x_matrix[0][1] + b3 * x_matrix[0][2]),
      "| y1 =", "%.2f" % y1_av)
print("b0 + b1*X21 + b2*X22 + b3*X23 =",
      "%.2f" % (b0 + b1 * x_matrix[1][0] + b2 * x_matrix[1][1] + b3 * x_matrix[1][2]),
      "| y2 =", "%.2f" % y2_av)
print("b0 + b1*X31 + b2*X32 + b3*X33 =",
      "%.2f" % (b0 + b1 * x_matrix[2][0] + b2 * x_matrix[2][1] + b3 * x_matrix[2][2]),
      "| y3 =", "%.2f" % y3_av)
print("b0 + b1*X41 + b2*X42 + b3*X43 =",
      "%.2f" % (b0 + b1 * x_matrix[3][0] + b2 * x_matrix[3][1] + b3 * x_matrix[3][2]),
      "| y4 =", "%.2f" % y4_av)

d1 = ((matrix_plan[0][0] - y1_av) ** 2 + (matrix_plan[0][1] - y1_av) ** 2 + (matrix_plan[0][2] - y1_av) ** 2) / 3
d2 = ((matrix_plan[1][0] - y2_av) ** 2 + (matrix_plan[1][1] - y2_av) ** 2 + (matrix_plan[1][2] - y2_av) ** 2) / 3
d3 = ((matrix_plan[2][0] - y3_av) ** 2 + (matrix_plan[2][1] - y3_av) ** 2 + (matrix_plan[2][2] - y3_av) ** 2) / 3
d4 = ((matrix_plan[3][0] - y4_av) ** 2 + (matrix_plan[3][1] - y4_av) ** 2 + (matrix_plan[3][2] - y4_av) ** 2) / 3

d_matrix = [d1, d2, d3, d4]

Gp = max(d_matrix) / sum(d_matrix)

m = len(matrix_plan[0])
f1 = m - 1
f2 = N = len(x_matrix)
q = 0.05
Gt = 0.7679

print("\n-------------------")

print("\nКритерій Фішера")

print("\nGp = %.4f" % Gp)
print("Gt =", Gt, "\n")

if Gp < Gt:
    print("%.4f < %.4f " % (Gp, Gt))
    print("Дисперсія однорідна\n")
else:
    print("%.4f > %.4f " % (Gp, Gt))
    print("Дисперсія не однорідна\n")

S2 = sum(d_matrix) / N
S2b = S2 / (N * m)
Sb = sqrt(S2b)

y_list = [y1_av, y2_av, y3_av, y4_av]

B0 = sum(y_list * x_norm[:, 0]) / N
B1 = sum(y_list * x_norm[:, 1]) / N
B2 = sum(y_list * x_norm[:, 2]) / N
B3 = sum(y_list * x_norm[:, 3]) / N

t0 = fabs(B0) / Sb
t1 = fabs(B1) / Sb
t2 = fabs(B2) / Sb
t3 = fabs(B3) / Sb

print("-------------------")

print("\nКритерій Стьюдента\n")

p = 0.95
f3 = f1 * f2
t_tab = scipy.stats.t.ppf((1 + p) / 2, f3)
print("t0:", "%.3f " % t0, "\nt1:", "%.3f" % t1, "\nt2:", "%.3f" % t2, "\nt3:", "%.3f\n" % t3)
if t0 < t_tab:
    b0 = 0
    print("t0 < t_таб; отже b0=0")

if t1 < t_tab:
    b1 = 0
    print("t1 < t_таб; отже b1=0")

if t2 < t_tab:
    b2 = 0
    print("t2 < t_таб; отже b2=0")

if t3 < t_tab:
    b3 = 0
    print("t3 < t_таб; отже b3=0")


y1_cov = b0 + b1 * x_matrix[0][0] + b2 * x_matrix[0][1] + b3 * x_matrix[0][2]
y2_cov = b0 + b1 * x_matrix[1][0] + b2 * x_matrix[1][1] + b3 * x_matrix[1][2]
y3_cov = b0 + b1 * x_matrix[2][0] + b2 * x_matrix[2][1] + b3 * x_matrix[2][2]
y4_cov = b0 + b1 * x_matrix[3][0] + b2 * x_matrix[3][1] + b3 * x_matrix[3][2]

print("\ny1:", "%.3f " % y1_cov, "\ny2:", "%.3f" % y2_cov, "\ny3:", "%.3f" % y3_cov, "\ny4:", "%.3f\n" % y4_cov)

print("-------------------\n")

print("Критерій Фішера")


d = 2
f4 = N - d


S2_ad = (m / (N - d)) * ((y1_cov - y1_av) ** 2 + (y2_cov - y2_av) ** 2 + (y3_cov - y3_av) ** 2 + (y4_cov - y4_av) ** 2)
Fp = S2_ad / S2b
Ft = scipy.stats.f.ppf(p, f4, f3)
print("\nFt =", Ft)
print("Fp = %.2f" % Fp)
if Fp > Ft:
    print("Fp > Ft")
    print("Рівняння регресії не адекватно оригіналу при рівні значимості 0,05")
else:
    print("Fp < Ft")
    print("Рівняння регресії адекватно оригіналу при рівні значимості 0,05")

