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
            data["point"] = form.point.data.split(" ")
            data["min_factor"] = form.min_factor.data

            algorithm = Steepest_descent(form.equation.data,)
            data["parsed_equation"] = algorithm.function
            data["parsed_point"] = algorithm.point
            data["gradient"] = algorithm.grad
            data["value_in_point"] = algorithm.calculate_function_value_in_point(algorithm.function, algorithm.point)
            data["minimum"] = algorithm.find_minimum()
            #algorithm.generate_plot(6)

            flash('Pomyślnie wykonano obliczenia!', 'success')
            return redirect(url_for('result'))
        except Exception as e:
            flash('Nie wykonano obliczeń: '+str(e))

    return render_template('main.html', form=form, title='Strona główna projektu')


@app.route('/result')
def result():
    return render_template('result.html', data=data, title='Wynik')
