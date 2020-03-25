from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, SubmitField, IntegerField
from wtforms.validators import DataRequired, Length, NumberRange

class InputData(FlaskForm):
    gradient_accuracy = FloatField('ε1 - dokładność gradientu', validators=[DataRequired(), NumberRange(min=0)])
    x_accuracy = FloatField('ε2 - dokładność x', validators=[DataRequired(), NumberRange(min=0)])
    function_accuracy = FloatField('ε3 - dokładność funkcji', validators=[DataRequired(), NumberRange(min=0)])
    number_of_iterations = IntegerField('L - liczba iteracji', validators=[DataRequired(), NumberRange(min=1,
                                                                                                       max=1000000)])
    function_ = StringField('f(x) - funkcja', validators=[DataRequired()])
    point = StringField('P - punkt startowy', validators=[DataRequired()])
    calculate = SubmitField('Znajdź minimum')
