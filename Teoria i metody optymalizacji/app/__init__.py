from flask import Flask


app = Flask(__name__)
app.config['SECRET_KEY'] = 'kondziu_zabij_mnie'


from app import routes