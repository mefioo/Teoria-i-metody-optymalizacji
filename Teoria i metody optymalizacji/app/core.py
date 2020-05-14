import sympy as sym
from sympy import log, sin, pi
from sympy.parsing import sympy_parser
from sympy.parsing.mathematica import mathematica
import numpy as np
from math import pi, e, sqrt
import random
from mpl_toolkits import mplot3d
import base64
from io import BytesIO
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axes3d

x1, x2, x3, x4, x5, x6 = sym.symbols('x1 x2 x3 x4 x5 x6')
Pi = sym.symbols('Pi')
x_vector = [x1, x2, x3, x4, x5, x6]
FUNCTION_STRINGS = ["x1**4+x2**4-0.62*x2**2-0.62*x1**2", #suggested functions
                "100*(x2-x1**2)**2+(1-x1)**2",
              "(x1-x2+x3)**2 + (-x1+x2+x3)**2+(x1+x2-x3)**2",
              "(1+(x1+x2+1)**2*(19-14*x1+3*x1**2-14*x2+6*x1*x2+3*x2**2))*(30+(2*x1-3*x2)**2*(18-32*x1+12*x1**2+48*x2-36*x1*x2+27*x2**2))",
               "Exp[-21*Log[2]*(x1-0.08)**2/(0.854**2)]*Sin[5*Pi*(x1**(3/4)-0.05)]",
              "(x1**2+x2-11)**2+(x1+x2**2-7)**2-200"]

POINTS = [[1, 1], #suggested started points
          [-1.2, 1.0],
          [100.0, -1.0, 2.5],
          [-0.4, -0.6],
          [random.random()],
          [-5, 5],]

GOLDEN_VALUE = (1+sqrt(5))/2 #Golden value

class Steepest_descent():
    """
    :param
    equation - input function in R^n, on which the Steepest Descent algorithm will work
    starting point - coordinates of each "X" - default starting point is vector of zeors
    function accuracy - the diffrence between equation value in two next iterations
    x_accuracy - the diffrence betweem coordinates in two next iterations
    gradient_accuracy - minimal value of gradient that we are working for (vector length)
    number_of_iterations - maximum number of iterations that our algoritm will go  through

    :returns
    None, it's instance of class.
    """

    def __init__(self, equation, starting_point=0, function_accuracy=1e-3, x_accuracy=1e-3, gradient_accuracy=1e-3, number_of_iterations=5000, plot_size=6, test_param=0.4):

        self.function = self.parse_equation(equation)
        self.point = np.array(starting_point) if starting_point else np.zeros((len(self.function.free_symbols), 1))
        self.grad = np.array([sym.diff(self.function, x) for x in x_vector if x in self.function.free_symbols])

        self.path_x = [self.point[0]]
        self.path_y = [self.point[-1]]
        self.path_z = [np.float(self.calculate_function_value_in_point(self.function, self.point))]
        self.plot_size = plot_size
        self.test_param = test_param
        ##Data for TABLE:
        self.x_points = [self.point]
        self.function_values = [self.calculate_function_value_in_point(self.function, self.point)]
        self.gradient_values = []

        self.function_accuracy = float(function_accuracy)
        self.x_accuracy = float(x_accuracy)
        self.gradient_accuracy = float(gradient_accuracy)
        self.max_iter = int(number_of_iterations)

    def parse_equation(self, equation):
        """
        :param equation: equation from string input, to be converted
        :return: equation in form understand by Sympy library
        """
        equation.replace("^", "**")
        return mathematica(equation)

    def calculate_function_value_in_point(self, function, point):
        """
        :param function: any equation with vector of undefined variables
        :param point: coordinates to fill the functions with values
        :return: value of the function in point
        """
        return function.subs([(x, value) for x, value in zip(x_vector, point)])

    def get_vector_length(self, vector):
        return np.sqrt(np.float((vector.dot(vector))))

    def get_direction(self):
        return np.array([-self.calculate_function_value_in_point(self.grad[i], self.point) for i in range(len(self.grad))])

    def get_starting_TR(self, direction):
        for i in [89, 55, 34, 21, 13, 8, 5, 3, 2, 1, 0.8, 0.5, 0.3, 0.2, 0.1, 0.02, 0.01]:
            if self.calculate_function_value_in_point(self.function, self.point) > self.calculate_function_value_in_point(self.function, self.point+(direction*i)):
                return i
        return self.x_accuracy*2

    def get_hesjan(self, matrix_size):
        hesjan = np.zeros((matrix_size, matrix_size))
        for f, i in zip(self.grad, range(matrix_size)):
            for symbol, j in zip(x_vector, range(matrix_size)):
                hesjan[i, j] = self.calculate_function_value_in_point(sym.diff(f, symbol), self.point)
        return hesjan

    def check_if_in_min(self):
        matrix_size = len(self.grad)
        hesjan = self.get_hesjan(matrix_size)
        for i in range(1, matrix_size+1):
            if np.linalg.det(hesjan[0:i, 0:i]) <= 0:
                return "Nie"
        return "Tak"

    def get_best_step_distance(self, direction, p, old_value, lower_bound=0, upper_bound=10, test_param=0.4, test_acc =1e-5, depth=50):
        mid_bound = (lower_bound+upper_bound)/2
        lower = old_value + (1-test_param)*p*mid_bound
        upper = old_value + test_param * p * mid_bound
        new_val = self.calculate_function_value_in_point(self.function, self.point + mid_bound * direction)
        if depth < 5:
            if depth < 0:
                return mid_bound
        if new_val < lower and abs(new_val - lower) > test_acc:
            return self.get_best_step_distance(direction, p, old_value, mid_bound, upper_bound, test_param, test_acc, depth - 1)
        if new_val > upper and abs(new_val - upper) > test_acc:
            return self.get_best_step_distance(direction, p, old_value, lower_bound, mid_bound, test_param, test_acc, depth-1)

        return mid_bound

    def find_minimum(self):
        direction = self.get_direction()
        direction_magnitude = self.get_vector_length(direction)
        self.gradient_values.append(direction_magnitude)
        inf_loop_guardian = 0

        while direction_magnitude > self.gradient_accuracy and inf_loop_guardian < self.max_iter:
            inf_loop_guardian=inf_loop_guardian+1
            upper_bound = self.get_starting_TR(direction)
            p = -np.array(direction) @ np.array(direction).T
            staring_f_value = self.calculate_function_value_in_point(self.function, self.point)
            step_distance = self.get_best_step_distance(direction, p, staring_f_value, 0, upper_bound, self.test_param)
            self.point = self.point + direction * step_distance
            if len(self.point) == 2:
                self.path_x.append(self.point[0])
                self.path_y.append(self.point[-1])
                self.path_z.append(np.float(self.calculate_function_value_in_point(self.function, self.point)))

            direction = self.get_direction()
            direction_magnitude = self.get_vector_length(direction)

            self.x_points.append(self.point)
            self.gradient_values.append(direction_magnitude)
            self.function_values.append(self.path_z[-1])

        print(direction_magnitude)
        print(self.point)
        return self.point
    
    def generate_plot(self, plot_size=6):
        plot_size = int(self.plot_size) if int(self.plot_size) else plot_size
        if len(self.point) == 2:
            My_X, My_Y = np.meshgrid(range(10 * plot_size), range(10 * plot_size))
            My_X = (My_X - 5 * plot_size) / 5
            My_Y = (My_Y - 5 * plot_size) / 5
            My_Z = np.ones((10 * plot_size, 10 * plot_size))
            for i in range(10 * plot_size):
                for j in range(10 * plot_size):
                    My_Z[i, j] = self.calculate_function_value_in_point(self.function, [My_X[i, j], My_Y[i][0]])

            fig = Figure()
            ax = Axes3D(fig)

            # 1stplot

            X, Y, Z = My_X, My_Y, My_Z
            ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
            ax.plot3D(np.array(self.path_x), np.array(self.path_y), np.array(self.path_z), c='r', marker='o')
            cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
            cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
            cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

            buf = BytesIO()
            fig.savefig(buf, format="png")
            img1 = base64.b64encode(buf.getbuffer()).decode("ascii")

            # 2nd plot

            ax.view_init(azim=-90, elev=90)

            buf = BytesIO()
            fig.savefig(buf, format="png")
            img2 = base64.b64encode(buf.getbuffer()).decode("ascii")

            # 3rd plot
            surf = ax.plot_surface(My_X, My_Y, My_Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
            ax.plot(np.array(self.path_x), np.array(self.path_y), np.array(self.path_z), 'ro', alpha=0.5)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.view_init(azim=None, elev=None)

            buf = BytesIO()
            fig.savefig(buf, format="png")
            img3 = base64.b64encode(buf.getbuffer()).decode("ascii")




            img = [img1, img2, img3]

            return img
            #return plt
            #plt.show()
        else:
            return ["It's not 3D object", "It's not 3D object", "It's not 3D object"]