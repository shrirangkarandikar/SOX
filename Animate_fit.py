import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np



fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_xlim(-1, 1)
ax.set_ylim(-100, 500)
initial_poly, = ax.scatter([], [], lw=2, label='Initial Polynomial', color='red')
fitting_poly, = ax.scatter([], [], lw=2, label='Fitting Polynomial', color='blue')
ax.legend()


def init():
    y = polynomial(x_exp)
    y = y[x_random1]
    line_initial_BGD.set_data(x_random[x_random1], y)
    line_fitting_BGD.set_data([], [])
    line_err_BGD.set_data([], [])
    iteration_text_BGD.set_text('')
    line_initial_SGD.set_data(x_random[x_random1], y)
    line_fitting_SGD.set_data([], [])
    line_err_SGD.set_data([], [])
    iteration_text_SGD.set_text('')
    return line_initial_BGD, line_fitting_BGD, line_err_BGD, iteration_text_BGD, line_initial_SGD, line_fitting_SGD, line_err_SGD, iteration_text_SGD


