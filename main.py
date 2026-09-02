from numpy import sin
from trainingFunction import *
import matplotlib.pyplot as plt
from basicFunction import * 

def predict(x_test, activation, weight1, weight2):
    hiddenvals = activation(np.dot(x_test, weight1))
    hiddenvals = np.hstack([hiddenvals, np.ones((hiddenvals.shape[0], 1))])
    return np.dot(hiddenvals, weight2)

def draw_plot(x, y_true, y_predict, plot_title):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.scatter(x, y_true, color="blue", marker='o', label="y_true", alpha=0.6)
    ax.scatter(x, y_predict, color="red", marker='x', label="y_predict", alpha=0.6)
    
    ax.set_title(plot_title)
    ax.set_xlabel("X‑axis")
    ax.set_ylabel("Y‑axis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    x = np.random.randn(99999, 1)

#    y = x ** 2
    y = np.sin(x)

    x_test = np.random.randn(999, 1)
#    y_test = x_test ** 2 
    y_test = np.sin(x_test)

    # add i colum for bias
    x = np.hstack([x, np.ones((x.shape[0], 1))])
    x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))])

    # args: (acti, acti_d, loss, loss_d, weight_init, x, y, epoches, lr, hidden_size)
#    weight1, weight2 = bgd(relu_leaky, relu_leaky_d, mse, mse_d, weight_init_relu, x, y, 999, 0.001, 64)
#    bgd_predict = predict(x_test, relu_leaky, weight1, weight2)
#    print("---------up is bdg-----------")
#    print(f"test samples' loss = {mse(y_test, bgd_predict):.11f}")
#    draw_plot(x_test[:, 0], y_test[:, 0], bgd_predict[:, 0], "bdg")

    # args: (acti, acti_d, loss, loss_d, weight_init, x, y, epoches, lr, hidden_size, batch)
#    weight1, weight2 = mbgd(relu_leaky, relu_leaky_d, mse, mse_d, weight_init_relu, x, y, 999, 0.001, 64, 64)
#    mbgd_predict = predict(x_test, relu_leaky, weight1, weight2)
#    print("---------up is mbdg-----------")
#    print(f"test samples' loss = {mse(y_test, mbgd_predict):.11f}")
#    draw_plot(x_test[:, 0], y_test[:, 0], mbgd_predict[:, 0], "mbdg")

    # args: (acti, acti_d, loss, loss_d, weight_init, x, y, epoches, lr, hidden_size, batch, beta1, beta2)
    weight1, weight2 = adam_mbgd(relu_leaky, relu_leaky_d, mse, mse_d, weight_init_relu, x, y, 999, 0.001, 64, 64, 0.9, 0.999)
    adam_mbgd_predict = predict(x_test, relu_leaky, weight1, weight2)
    print("---------up is adam-----------")
    print(f"test samples' loss = {mse(y_test, adam_mbgd_predict):.11f}")
    draw_plot(x_test[:, 0], y_test[:, 0], adam_mbgd_predict[:, 0], "adam")
    


