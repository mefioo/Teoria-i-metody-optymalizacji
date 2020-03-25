from app import app
from flask import flash, url_for, redirect, render_template
from app.forms import InputData





@app.route('/', methods=['GET', 'POST'])
@app.route('/main', methods=['GET', 'POST'])
def main():
    form = InputData()
    if form.validate_on_submit():
        flash('Pomyślnie wykonano obliczenia!', 'success')
        return redirect(url_for('main'))
    else:
        flash('Nie wykonano obliczeń')
    return render_template('main.html', form=form, title='Strona główna projektu')