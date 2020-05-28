from app import app
from flask import flash, url_for, redirect, render_template
from app.forms import InputData
from app.core import Steepest_descent
import numpy as np


data = {}


def format_float(num):
    return np.format_float_positional(num, trim='-')


@app.route('/', methods=['GET', 'POST'])
@app.route('/main', methods=['GET', 'POST'])
def main():
    form = InputData()
    if form.validate_on_submit():
        try:
            data["gradient_accuracy"] = form.gradient_accuracy.data
            data["x_accuracy"] = form.x_accuracy.data
            data["function_accuracy"] = form.function_accuracy.data
            data["number_of_iterations"] = form.number_of_iterations.data
            data["equation"] = form.equation.data
            data["plot_size"] = form.plot_size.data
            form.point.data = list(form.point.data.split(" "))
            numbers = []
            for val in form.point.data:
                numbers.append(float(val))
            data["point"] = numbers
            data["min_factor"] = form.min_factor.data

            algorithm = Steepest_descent(form.equation.data, data["point"], data["function_accuracy"],data["x_accuracy"], data["gradient_accuracy"],data["number_of_iterations"], data["plot_size"], data["min_factor"])
            data["parsed_equation"] = algorithm.function
            data["parsed_point"] = algorithm.point
            data["gradient"] = algorithm.grad
            data["value_in_point"] = algorithm.calculate_function_value_in_point(algorithm.function, algorithm.point)
            data["minimum"] = algorithm.find_minimum()
            data["is_minimum"] = algorithm.check_if_in_min() + ', sprawdź hesjan'

            plots = algorithm.generate_plot(6)
            data["plot1"] = plots[0]
            data["plot2"] = plots[1]
            data["plot3"] = plots[2]

            data["iterations"] = [i+1 for i in range(len(algorithm.x_points)-1)]
            data["path"] = algorithm.x_points
            data["gradients"] = algorithm.gradient_values
            data["function_values"] = algorithm.function_values
            flash('Pomyślnie wykonano obliczenia!', 'success')
            return redirect(url_for('result'))
        except Exception as e:
            flash('Nie wykonano obliczeń: '+str(e))

    return render_template('main.html', form=form, title='Strona główna projektu')


@app.route('/result')
def result():
    table_data = []


    ### Setting gradient accuracy
    list = []
    acc = len(str(format_float(data["gradient_accuracy"]))) - 2
    for num in data["gradients"]:
        list.append(str(format_float(round(num, acc))))
    data["gradients"] = list
    data["gradient_accuracy"] = format_float(data["gradient_accuracy"])
    item = ['Dokładność gradientu', data["gradient_accuracy"]]
    table_data.append(item)


    ### Setting minimum accuracy
    acc = len(str(format_float(data["x_accuracy"]))) - 2
    nums = ''
    for num in data["minimum"]:
        nums = nums + str(round(num, acc)) + ", "
    nums = nums[0:-2]

    ### Setting x accuracy
    list = []
    acc = len(str(format_float(data["x_accuracy"]))) - 2
    for line in data["path"]:
        point = ''
        for my_x in line:
            point = point + str(round(my_x, acc)) + ', '
        list.append(point[:-2])
    data["path"] = list
    data["x_accuracy"] = format_float(data["x_accuracy"])
    item = ['Dokładność x', data["x_accuracy"]]
    table_data.append(item)

    ### Setting function accuracy
    list = []
    acc = len(str(format_float(data["function_accuracy"]))) - 2
    for num in data["function_values"]:
        list.append(str(format_float(round(num, acc))))
    data["function_values"] = list
    data["function_accuracy"] = format_float(data["function_accuracy"])
    item = ['Dokładność funkcji', data["function_accuracy"]]
    table_data.append(item)

    item = ['Liczba iteracji', data["number_of_iterations"]]
    table_data.append(item)
    item = ['Funkcja', data["equation"]]
    table_data.append(item)
    data["min_factor"] = format_float(data["min_factor"])
    item = ['Parametr dla metody minimum', data["min_factor"]]
    table_data.append(item)

    point = ""
    for num in data["point"]:
        point = point + str(num) + ", "
    point = point[0:-2]
    item = ['Punkt startowy', point]
    table_data.append(item)


    item = ['Minimum znalezione przez algorytm', nums]
    table_data.append(item)
    item = ['Czy znaleziony punkt to minimum?', data["is_minimum"]]
    table_data.append(item)

    img1 = data["plot1"]
    img2 = data["plot2"]
    img3 = data["plot3"]

    return render_template('result.html', table_data=table_data, img1=img1, img2=img2, img3=img3, title='Wynik', iterations = data["iterations"],
                           points=data["path"], gradients = data["gradients"], functions = data["function_values"])

