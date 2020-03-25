from app import app
from flask import render_template





@app.route('/')
@app.route('/main')
def main():
    return render_template('main.html', title='Strona główna projektu')