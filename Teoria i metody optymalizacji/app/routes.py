from app import app
from flask import flash, url_for, redirect, render_template
from app.forms import InputData
from app.core import Steepest_descent


data = {}


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
            form.point.data = list(form.point.data.split(" "))
            numbers = []
            for str in form.point.data:
                numbers.append(float(str))
            data["point"] = numbers
            data["min_factor"] = form.min_factor.data

            algorithm = Steepest_descent(form.equation.data, data["point"])
            data["parsed_equation"] = algorithm.function
            data["parsed_point"] = algorithm.point
            data["gradient"] = algorithm.grad
            data["value_in_point"] = algorithm.calculate_function_value_in_point(algorithm.function, algorithm.point)
            data["minimum"] = algorithm.find_minimum()

            plots = algorithm.generate_plot(6)
            data["plot1"] = plots[0]
            data["plot2"] = plots[1]
            data["plot3"] = plots[2]
            
            flash('Pomyślnie wykonano obliczenia!', 'success')
            return redirect(url_for('result'))
        except Exception as e:
            flash('Nie wykonano obliczeń: '+str(e))

    return render_template('main.html', form=form, title='Strona główna projektu')


@app.route('/result')
def result():
    table_data = []
    item = ['Dokładność gradientu', data["gradient_accuracy"]]
    table_data.append(item)
    item = ['Dokładność x', data["x_accuracy"]]
    table_data.append(item)
    item = ['Dokładność funkcji', data["function_accuracy"]]
    table_data.append(item)
    item = ['Liczba iteracji', data["number_of_iterations"]]
    table_data.append(item)
    item = ['Funkcja', data["equation"]]
    table_data.append(item)
    item = ['Parametr dla metody minimum', data["min_factor"]]
    table_data.append(item)

    ### Setting minimum accuracy
    acc = len(str(data["x_accuracy"])) - 2
    point = ""
    for num in data["point"]:
        point = point+str(num)+", "
    point = point[0:-2]
    item = ['Punkt startowy', point]
    table_data.append(item)
    nums = ''
    for num in data["minimum"]:
        nums = nums + str(round(num, acc)) + ", " #if
    nums = nums[0:-2]


    item = ['Minimum znalezione przez algorytm', nums]
    table_data.append(item)

    img1 = data["plot1"]
    img2 = data["plot2"]
    img3 = data["plot3"]

    return render_template('result.html', table_data=table_data, img1=img1, img2=img2, img3=img3, title='Wynik')

