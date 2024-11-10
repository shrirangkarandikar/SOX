import Sine_Decay
import Sine_RMSProp
import Sine_Adagrad
import Sine_Adam
import Cos_Decay
import Cos_RMSProp
import Cos_AdaGrad
import Cos_Adam
import Log_Decay
import Log_RMSProp
import Log_Adagrad
import Log_Adam
import Geometric_Decay
import Geometric_RMSProp
import Geometric_AdaGrad
import Geometric_Adam
import Exponential_Decay
import Exponential_RMSProp
import Exponential_AdaGrad
import Exponential_Adam
import numpy as np
import matplotlib.pyplot as plt


def reshape_for_plotting(x, y_orig, y_new, decay=False):
    if decay:
        x = np.array(x).ravel()
        y_orig = np.array(y_orig).ravel()
        y_new = np.array(y_new).ravel()
    else:
        x = np.asarray(x).reshape(-1)
        y_orig = np.asarray(y_orig).reshape(-1)
        y_new = np.asarray(y_new).reshape(-1)
    return x, y_orig, y_new

# Sine Functions
sine_decay_error, sine_decay_y, sine_decay_x, sine_decay_y_orig, sine_decay_iter = Sine_Decay.fit(4, 80)
sine_decay_x, sine_decay_y_orig, sine_decay_y = reshape_for_plotting(sine_decay_x, sine_decay_y_orig, sine_decay_y, decay=True)

sine_rmsprop_error, sine_rmsprop_y, sine_rmsprop_x, sine_rmsprop_y_orig, sine_rmsprop_iter = Sine_RMSProp.fit(4, 80)
sine_rmsprop_x, sine_rmsprop_y_orig, sine_rmsprop_y = reshape_for_plotting(sine_rmsprop_x, sine_rmsprop_y_orig, sine_rmsprop_y)

sine_adagrad_error, sine_adagrad_y, sine_adagrad_x, sine_adagrad_y_orig, sine_adagrad_iter = Sine_Adagrad.fit(4, 80)
sine_adagrad_x, sine_adagrad_y_orig, sine_adagrad_y = reshape_for_plotting(sine_adagrad_x, sine_adagrad_y_orig, sine_adagrad_y)

sine_adam_error, sine_adam_y, sine_adam_x, sine_adam_y_orig, sine_adam_iter = Sine_Adam.fit(4, 80)
sine_adam_x, sine_adam_y_orig, sine_adam_y = reshape_for_plotting(sine_adam_x, sine_adam_y_orig, sine_adam_y)

# Cosine Functions
cos_decay_error, cos_decay_y, cos_decay_x, cos_decay_y_orig, cos_decay_iter = Cos_Decay.fit(4, 80)
cos_decay_x, cos_decay_y_orig, cos_decay_y = reshape_for_plotting(cos_decay_x, cos_decay_y_orig, cos_decay_y, decay=True)

cos_rmsprop_error, cos_rmsprop_y, cos_rmsprop_x, cos_rmsprop_y_orig, cos_rmsprop_iter = Cos_RMSProp.fit(4, 80)
cos_rmsprop_x, cos_rmsprop_y_orig, cos_rmsprop_y = reshape_for_plotting(cos_rmsprop_x, cos_rmsprop_y_orig, cos_rmsprop_y)

cos_adagrad_error, cos_adagrad_y, cos_adagrad_x, cos_adagrad_y_orig, cos_adagrad_iter = Cos_AdaGrad.fit(4, 80)
cos_adagrad_x, cos_adagrad_y_orig, cos_adagrad_y = reshape_for_plotting(cos_adagrad_x, cos_adagrad_y_orig, cos_adagrad_y)

cos_adam_error, cos_adam_y, cos_adam_x, cos_adam_y_orig, cos_adam_iter = Cos_Adam.fit(4, 80)
cos_adam_x, cos_adam_y_orig, cos_adam_y = reshape_for_plotting(cos_adam_x, cos_adam_y_orig, cos_adam_y)

# Logarithmic Functions
log_decay_error, log_decay_y, log_decay_x, log_decay_y_orig, log_decay_iter = Log_Decay.fit(4, 80)
log_decay_x, log_decay_y_orig, log_decay_y = reshape_for_plotting(log_decay_x, log_decay_y_orig, log_decay_y, decay=True)

log_rmsprop_error, log_rmsprop_y, log_rmsprop_x, log_rmsprop_y_orig, log_rmsprop_iter = Log_RMSProp.fit(4, 80)
log_rmsprop_x, log_rmsprop_y_orig, log_rmsprop_y = reshape_for_plotting(log_rmsprop_x, log_rmsprop_y_orig, log_rmsprop_y)

log_adagrad_error, log_adagrad_y, log_adagrad_x, log_adagrad_y_orig, log_adagrad_iter = Log_Adagrad.fit(4, 80)
log_adagrad_x, log_adagrad_y_orig, log_adagrad_y = reshape_for_plotting(log_adagrad_x, log_adagrad_y_orig, log_adagrad_y)

log_adam_error, log_adam_y, log_adam_x, log_adam_y_orig, log_adam_iter = Log_Adam.fit(4, 80)
log_adam_x, log_adam_y_orig, log_adam_y = reshape_for_plotting(log_adam_x, log_adam_y_orig, log_adam_y)

# Geometric Functions
geometric_decay_error, geometric_decay_y, geometric_decay_x, geometric_decay_y_orig, geometric_decay_iter = Geometric_Decay.fit(4, 80)
geometric_decay_x, geometric_decay_y_orig, geometric_decay_y = reshape_for_plotting(geometric_decay_x, geometric_decay_y_orig, geometric_decay_y, decay=True)

geometric_rmsprop_error, geometric_rmsprop_y, geometric_rmsprop_x, geometric_rmsprop_y_orig, geometric_rmsprop_iter = Geometric_RMSProp.fit(4, 80)
geometric_rmsprop_x, geometric_rmsprop_y_orig, geometric_rmsprop_y = reshape_for_plotting(geometric_rmsprop_x, geometric_rmsprop_y_orig, geometric_rmsprop_y)

geometric_adagrad_error, geometric_adagrad_y, geometric_adagrad_x, geometric_adagrad_y_orig, geometric_adagrad_iter = Geometric_AdaGrad.fit(4, 80)
geometric_adagrad_x, geometric_adagrad_y_orig, geometric_adagrad_y = reshape_for_plotting(geometric_adagrad_x, geometric_adagrad_y_orig, geometric_adagrad_y)

geometric_adam_error, geometric_adam_y, geometric_adam_x, geometric_adam_y_orig, geometric_adam_iter = Geometric_Adam.fit(4, 80)
geometric_adam_x, geometric_adam_y_orig, geometric_adam_y = reshape_for_plotting(geometric_adam_x, geometric_adam_y_orig, geometric_adam_y)

# Exponential Functions
exponential_decay_error, exponential_decay_y, exponential_decay_x, exponential_decay_y_orig, exponential_decay_iter = Exponential_Decay.fit(4, 80)
exponential_decay_x, exponential_decay_y_orig, exponential_decay_y = reshape_for_plotting(exponential_decay_x, exponential_decay_y_orig, exponential_decay_y, decay=True)

exponential_rmsprop_error, exponential_rmsprop_y, exponential_rmsprop_x, exponential_rmsprop_y_orig, exponential_rmsprop_iter = Exponential_RMSProp.fit(4, 80)
exponential_rmsprop_x, exponential_rmsprop_y_orig, exponential_rmsprop_y = reshape_for_plotting(exponential_rmsprop_x, exponential_rmsprop_y_orig, exponential_rmsprop_y)

exponential_adagrad_error, exponential_adagrad_y, exponential_adagrad_x, exponential_adagrad_y_orig, exponential_adagrad_iter = Exponential_AdaGrad.fit(4, 80)
exponential_adagrad_x, exponential_adagrad_y_orig, exponential_adagrad_y = reshape_for_plotting(exponential_adagrad_x, exponential_adagrad_y_orig, exponential_adagrad_y)

exponential_adam_error, exponential_adam_y, exponential_adam_x, exponential_adam_y_orig, exponential_adam_iter = Exponential_Adam.fit(4, 80)
exponential_adam_x, exponential_adam_y_orig, exponential_adam_y = reshape_for_plotting(exponential_adam_x, exponential_adam_y_orig, exponential_adam_y)



# Define a list of all optimizers and their data for easy iteration
optimizer_data = [
    # Sine optimizers
    ("Sine RMSProp", sine_rmsprop_x, sine_rmsprop_y_orig, sine_rmsprop_y, sine_rmsprop_error, sine_rmsprop_iter),
    ("Sine AdaGrad", sine_adagrad_x, sine_adagrad_y_orig, sine_adagrad_y, sine_adagrad_error, sine_adagrad_iter),
    ("Sine Adam", sine_adam_x, sine_adam_y_orig, sine_adam_y, sine_adam_error, sine_adam_iter),
    ("Sine Decay", sine_decay_x, sine_decay_y_orig, sine_decay_y, sine_decay_error, sine_decay_iter),

    # Cosine optimizers
    ("Cos RMSProp", cos_rmsprop_x, cos_rmsprop_y_orig, cos_rmsprop_y, cos_rmsprop_error, cos_rmsprop_iter),
    ("Cos AdaGrad", cos_adagrad_x, cos_adagrad_y_orig, cos_adagrad_y, cos_adagrad_error, cos_adagrad_iter),
    ("Cos Adam", cos_adam_x, cos_adam_y_orig, cos_adam_y, cos_adam_error, cos_adam_iter),
    ("Cos Decay", cos_decay_x, cos_decay_y_orig, cos_decay_y, cos_decay_error, cos_decay_iter),

    # Logarithmic optimizers
    ("Log RMSProp", log_rmsprop_x, log_rmsprop_y_orig, log_rmsprop_y, log_rmsprop_error, log_rmsprop_iter),
    ("Log AdaGrad", log_adagrad_x, log_adagrad_y_orig, log_adagrad_y, log_adagrad_error, log_adagrad_iter),
    ("Log Adam", log_adam_x, log_adam_y_orig, log_adam_y, log_adam_error, log_adam_iter),
    ("Log Decay", log_decay_x, log_decay_y_orig, log_decay_y, log_decay_error, log_decay_iter),

    # Geometric optimizers
    ("Geometric RMSProp", geometric_rmsprop_x, geometric_rmsprop_y_orig, geometric_rmsprop_y, geometric_rmsprop_error, geometric_rmsprop_iter),
    ("Geometric AdaGrad", geometric_adagrad_x, geometric_adagrad_y_orig, geometric_adagrad_y, geometric_adagrad_error, geometric_adagrad_iter),
    ("Geometric Adam", geometric_adam_x, geometric_adam_y_orig, geometric_adam_y, geometric_adam_error, geometric_adam_iter),
    ("Geometric Decay", geometric_decay_x, geometric_decay_y_orig, geometric_decay_y, geometric_decay_error, geometric_decay_iter),

    # Exponential optimizers
    ("Exponential RMSProp", exponential_rmsprop_x, exponential_rmsprop_y_orig, exponential_rmsprop_y, exponential_rmsprop_error, exponential_rmsprop_iter),
    ("Exponential AdaGrad", exponential_adagrad_x, exponential_adagrad_y_orig, exponential_adagrad_y, exponential_adagrad_error, exponential_adagrad_iter),
    ("Exponential Adam", exponential_adam_x, exponential_adam_y_orig, exponential_adam_y, exponential_adam_error, exponential_adam_iter),
    ("Exponential Decay", exponential_decay_x, exponential_decay_y_orig, exponential_decay_y, exponential_decay_error, exponential_decay_iter)
]

fig1, axes1 = plt.subplots(nrows=4, ncols=5, figsize=(20, 15))
fig1.suptitle("Original vs Predicted Values")

axes1[0, 0].scatter(sine_rmsprop_x, sine_rmsprop_y_orig, label='Original')
axes1[0, 0].scatter(sine_rmsprop_x, sine_rmsprop_y, label='Predicted')
axes1[0, 0].set_title("Sine RMSProp")
axes1[0, 0].legend()

axes1[0, 1].scatter(cos_rmsprop_x, cos_rmsprop_y_orig, label='Original')
axes1[0, 1].scatter(cos_rmsprop_x, cos_rmsprop_y, label='Predicted')
axes1[0, 1].set_title("Cos RMSProp")
axes1[0, 1].legend()

axes1[0, 2].scatter(log_rmsprop_x,log_rmsprop_y_orig , label='Original')
axes1[0, 2].scatter(log_rmsprop_x, log_rmsprop_y, label='Predicted')
axes1[0, 2].set_title("Log RMSProp")
axes1[0, 2].legend()

axes1[0, 3].scatter(geometric_rmsprop_x, geometric_rmsprop_y_orig, label='Original')
axes1[0, 3].scatter(geometric_rmsprop_x, geometric_decay_y, label='Predicted')
axes1[0, 3].set_title("Geometric RMSProp")
axes1[0, 3].legend()

axes1[0, 4].scatter(exponential_rmsprop_x, exponential_rmsprop_y_orig, label='Original')
axes1[0, 4].scatter(exponential_rmsprop_x, exponential_rmsprop_y, label='Predicted')
axes1[0, 4].set_title("Exponential RMSProp")
axes1[0, 4].legend()

axes1[1, 0].scatter(sine_adagrad_x, sine_adagrad_y_orig, label='Original')
axes1[1, 0].scatter(sine_adagrad_x, sine_adagrad_y, label='Predicted')
axes1[1, 0].set_title("Sine AdaGrad")
axes1[1, 0].legend()

axes1[1, 1].scatter(cos_adagrad_x, cos_adagrad_y_orig, label='Original')
axes1[1, 1].scatter(cos_adagrad_x, cos_adagrad_y, label='Predicted')
axes1[1, 1].set_title("Cos AdaGrad")
axes1[1, 1].legend()

axes1[1, 2].scatter(log_adagrad_x, log_adagrad_y_orig, label='Original')
axes1[1, 2].scatter(log_adagrad_x, log_adagrad_y, label='Predicted')
axes1[1, 2].set_title("Log AdaGrad")
axes1[1, 2].legend()

axes1[1, 3].scatter(geometric_adagrad_x, geometric_adagrad_y_orig, label='Original')
axes1[1, 3].scatter(geometric_adagrad_x, geometric_adagrad_y, label='Predicted')
axes1[1, 3].set_title("Geometric AdaGrad")
axes1[1, 3].legend()

axes1[1, 4].scatter(exponential_adagrad_x, exponential_adagrad_y_orig, label='Original')
axes1[1, 4].scatter(exponential_adagrad_x, exponential_adagrad_y, label='Predicted')
axes1[1, 4].set_title("Exponential AdaGrad")
axes1[1, 4].legend()

axes1[2, 0].scatter(sine_adam_x, sine_adam_y_orig, label='Original')
axes1[2, 0].scatter(sine_adam_x, sine_adam_y, label='Predicted')
axes1[2, 0].set_title("Sine Adam")
axes1[2, 0].legend()

axes1[2, 1].scatter(cos_adam_x, cos_adam_y_orig, label='Original')
axes1[2, 1].scatter(cos_adam_x, cos_adam_y, label='Predicted')
axes1[2, 1].set_title("Cos Adam")
axes1[2, 1].legend()

axes1[2, 2].scatter(log_adam_x, log_adam_y_orig, label='Original')
axes1[2, 2].scatter(log_adam_x, log_adam_y, label='Predicted')
axes1[2, 2].set_title("Log Adam")
axes1[2, 2].legend()

axes1[2, 3].scatter(geometric_adam_x, geometric_adam_y_orig, label='Original')
axes1[2, 3].scatter(geometric_adam_x, geometric_adam_y, label='Predicted')
axes1[2, 3].set_title("Geometric Adam")
axes1[2, 3].legend()

axes1[2, 4].scatter(exponential_adam_x, exponential_adam_y_orig, label='Original')
axes1[2, 4].scatter(exponential_adam_x, exponential_adam_y, label='Predicted')
axes1[2, 4].set_title("Exponential Adam")
axes1[2, 4].legend()

axes1[3, 0].scatter(sine_decay_x, sine_decay_y_orig, label='Original')
axes1[3, 0].scatter(sine_decay_x, sine_decay_y, label='Predicted')
axes1[3, 0].set_title("Sine Decay")
axes1[3, 0].legend()

axes1[3, 1].scatter(cos_decay_x, cos_decay_y_orig, label='Original')
axes1[3, 1].scatter(cos_decay_x, cos_decay_y, label='Predicted')
axes1[3, 1].set_title("Cos Decay")
axes1[3, 1].legend()

axes1[3, 2].scatter(log_decay_x, log_decay_y_orig, label='Original')
axes1[3, 2].scatter(log_decay_x, log_decay_y, label='Predicted')
axes1[3, 2].set_title("Log Decay")
axes1[3, 2].legend()

axes1[3, 3].scatter(geometric_decay_x, geometric_decay_y_orig, label='Original')
axes1[3, 3].scatter(geometric_decay_x, geometric_decay_y, label='Predicted')
axes1[3, 3].set_title("Geometric Decay")
axes1[3, 3].legend()

axes1[3, 4].scatter(exponential_decay_x, exponential_decay_y_orig, label='Original')
axes1[3, 4].scatter(exponential_decay_x, exponential_decay_y, label='Predicted')
axes1[3, 4].set_title("Exponential Decay")
axes1[3, 4].legend()
#----------------------------------------------------------------------------------------------------
fig2, axes2 = plt.subplots(nrows=4, ncols=5, figsize=(20, 15))
fig2.suptitle("Error Convergence vs Iterations")

axes2[0, 0].plot(range(len(sine_rmsprop_error)), sine_rmsprop_error, label='Error')
axes2[0, 0].set_title("Sine RMSProp")
axes2[0, 0].set_xlabel("Iterations")
axes2[0, 0].set_ylabel("Error")
axes2[0, 0].legend()

axes2[0, 1].plot(range(len(cos_rmsprop_error)), cos_rmsprop_error, label='Error')
axes2[0, 1].set_title("Cos RMSProp")
axes2[0, 1].set_xlabel("Iterations")
axes2[0, 1].set_ylabel("Error")
axes2[0, 1].legend()

axes2[0, 2].plot(range(len(log_rmsprop_error)), log_rmsprop_error, label='Error')
axes2[0, 2].set_title("Log RMSProp")
axes2[0, 2].set_xlabel("Iterations")
axes2[0, 2].set_ylabel("Error")
axes2[0, 2].legend()

axes2[0, 3].plot(range(len(geometric_rmsprop_error)), geometric_rmsprop_error, label='Error')
axes2[0, 3].set_title("Geometric RMSProp")
axes2[0, 3].set_xlabel("Iterations")
axes2[0, 3].set_ylabel("Error")
axes2[0, 3].legend()

axes2[0, 4].plot(range(len(exponential_rmsprop_error)), exponential_rmsprop_error, label='Error')
axes2[0, 4].set_title("Exponential RMSProp")
axes2[0, 4].set_xlabel("Iterations")
axes2[0, 4].set_ylabel("Error")
axes2[0, 4].legend()

axes2[1, 0].plot(range(len(sine_adagrad_error)), sine_adagrad_error, label='Error')
axes2[1, 0].set_title("Sine AdaGrad")
axes2[1, 0].set_xlabel("Iterations")
axes2[1, 0].set_ylabel("Error")
axes2[1, 0].legend()

axes2[1, 1].plot(range(len(cos_adagrad_error)), cos_adagrad_error, label='Error')
axes2[1, 1].set_title("Cos Adagrad")
axes2[1, 1].set_xlabel("Iterations")
axes2[1, 1].set_ylabel("Error")
axes2[1, 1].legend()

axes2[1, 2].plot(range(len(log_adagrad_error)), log_adagrad_error, label='Error')
axes2[1, 2].set_title("Log Adagrad")
axes2[1, 2].set_xlabel("Iterations")
axes2[1, 2].set_ylabel("Error")
axes2[1, 2].legend()

axes2[1, 3].plot(range(len(geometric_adagrad_error)), geometric_adagrad_error, label='Error')
axes2[1, 3].set_title("Geometric Adagrad")
axes2[1, 3].set_xlabel("Iterations")
axes2[1, 3].set_ylabel("Error")
axes2[1, 3].legend()

axes2[1, 4].plot(range(len(exponential_adagrad_error)), exponential_adagrad_error, label='Error')
axes2[1, 4].set_title("Exponential Adagrad")
axes2[1, 4].set_xlabel("Iterations")
axes2[1, 4].set_ylabel("Error")
axes2[1, 4].legend()

axes2[2, 0].plot(range(len(sine_adam_error)), sine_adam_error, label='Error')
axes2[2, 0].set_title("Sine Adam")
axes2[2, 0].set_xlabel("Iterations")
axes2[2, 0].set_ylabel("Error")
axes2[2, 0].legend()

axes2[2, 1].plot(range(len(cos_adam_error)), cos_adam_error, label='Error')
axes2[2, 1].set_title("Cos Adam")
axes2[2, 1].set_xlabel("Iterations")
axes2[2, 1].set_ylabel("Error")
axes2[2, 1].legend()

axes2[2, 2].plot(range(len(log_adam_error)), log_adam_error, label='Error')
axes2[2, 2].set_title("Log Adam")
axes2[2, 2].set_xlabel("Iterations")
axes2[2, 2].set_ylabel("Error")
axes2[2, 2].legend()

axes2[2, 3].plot(range(len(geometric_adam_error)), geometric_adam_error, label='Error')
axes2[2, 3].set_title("Geometric Adam")
axes2[2, 3].set_xlabel("Iterations")
axes2[2, 3].set_ylabel("Error")
axes2[2, 3].legend()

axes2[2, 4].plot(range(len(exponential_adam_error)), exponential_adam_error, label='Error')
axes2[2, 4].set_title("Exponential Adam")
axes2[2, 4].set_xlabel("Iterations")
axes2[2, 4].set_ylabel("Error")
axes2[2, 4].legend()

axes2[3, 0].plot(range(len(sine_decay_error)), sine_decay_error, label='Error')
axes2[3, 0].set_title("Sine Decay")
axes2[3, 0].set_xlabel("Iterations")
axes2[3, 0].set_ylabel("Error")
axes2[3, 0].legend()

axes2[3, 1].plot(range(len(cos_decay_error)), cos_decay_error, label='Error')
axes2[3, 1].set_title("Cos Decay")
axes2[3, 1].set_xlabel("Iterations")
axes2[3, 1].set_ylabel("Error")
axes2[3, 1].legend()

axes2[3, 2].plot(range(len(log_decay_error)), log_decay_error, label='Error')
axes2[3, 2].set_title("Log Decay")
axes2[3, 2].set_xlabel("Iterations")
axes2[3, 2].set_ylabel("Error")
axes2[3, 2].legend()

axes2[3, 3].plot(range(len(geometric_decay_error)), geometric_decay_error, label='Error')
axes2[3, 3].set_title("Geometric Decay")
axes2[3, 3].set_xlabel("Iterations")
axes2[3, 3].set_ylabel("Error")
axes2[3, 3].legend()

axes2[3, 4].plot(range(len(exponential_decay_error)), exponential_decay_error, label='Error')
axes2[3, 4].set_title("Exponential Decay")
axes2[3, 4].set_xlabel("Iterations")
axes2[3, 4].set_ylabel("Error")
axes2[3, 4].legend()

# Adjust layout and show plots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
