import numpy as np
import matplotlib.pyplot as plot


def phi(x, nargout=1):
    x1, x2, x3 = x[0], x[1], x[2]
    mul_x = float(x1 * x2 * x3)
    phi_x = np.sin(mul_x)
    if nargout == 1:
        return phi_x
    elif nargout == 2:
        grad_phi_x = np.transpose(
            np.array([[np.cos(mul_x) * x2 * x3, np.cos(mul_x) * x1 * x3, np.cos(mul_x) * x1 * x2]]))
        grad_phi_x = grad_phi_x.reshape((3, 1))
        return phi_x, grad_phi_x
    elif nargout == 3:
        grad_phi_x = np.transpose(
            np.array([[np.cos(mul_x) * x2 * x3, np.cos(mul_x) * x1 * x3, np.cos(mul_x) * x1 * x2]]))
        grad_phi_x = grad_phi_x.reshape((3, 1))
        hes_phi_x = np.array([[-np.sin(mul_x) * (x2 ** 2) * (x3 ** 2), x3 * (-mul_x * np.sin(mul_x) + np.cos(mul_x)),
                               x2 * (-mul_x * np.sin(mul_x) + np.cos(mul_x))],
                              [x3 * (-mul_x * np.sin(mul_x) + np.cos(mul_x)), -np.sin(mul_x) * (x1 ** 2) * (x3 ** 2),
                               x1 * (-mul_x * np.sin(mul_x) + np.cos(mul_x))],
                              [x2 * (-mul_x * np.sin(mul_x) + np.cos(mul_x)),
                               x1 * (-mul_x * np.sin(mul_x) + np.cos(mul_x)), -np.sin(mul_x) * (x1 ** 2) * (x2 ** 2)]])
        hes_phi_x = hes_phi_x.reshape((3, 3))
        return phi_x, grad_phi_x, hes_phi_x
    else:
        raise ValueError("Invalid nargout")


def f1(x, *kwargs, nargout=1):
    A = np.array(kwargs[0])
    Ax = np.matmul(A, x)
    if nargout == 1:
        return kwargs[1](Ax, nargout)
    elif nargout == 2:
        phi_x, grad_x = kwargs[1](Ax, nargout)
        gradient_x = np.matmul(np.transpose(A), grad_x)
        return phi_x, gradient_x
    elif nargout == 3:
        phi_x, grad_x, hes_x = kwargs[1](Ax, nargout)
        gradient_x = np.matmul(np.transpose(A), grad_x)
        hessian_x = np.matmul(np.transpose(A), np.matmul(hes_x, A))
        return phi_x, gradient_x, hessian_x
    else:
        raise ValueError("Invalid nargout")


def h(x, nargout=1):
    h_x = np.exp(x)
    if nargout == 1:
        return h_x
    elif nargout == 2:
        return h_x, h_x
    elif nargout == 3:
        return h_x, h_x, h_x
    raise ValueError("Invalid nargout")


def f2(x, *kwargs, nargout=1):
    phi_in = kwargs[0]
    h_in = kwargs[1]
    if nargout == 1:
        phi_x = phi_in(x)
        h_phi_x = h_in(phi_x)
        return h_phi_x
    elif nargout == 2:
        phi_x, grad_phi_x = phi_in(x, nargout)
        h_phi_x, h_derivative = h_in(phi_x, nargout)
        grad_h_phi_x = h_derivative * grad_phi_x
        return h_phi_x, grad_h_phi_x
    elif nargout == 3:
        phi_x, grad_phi_x, hes_phi_x = phi_in(x, nargout)
        h_phi_x, h_derivative, h_second_derivative = h_in(phi_x, nargout)
        grad_h_phi_x = h_derivative * grad_phi_x
        hes_h_phi_x = np.multiply(np.matmul(np.transpose(grad_phi_x), grad_phi_x), h_second_derivative) + np.multiply(
            hes_phi_x, h_derivative)
        return h_phi_x, grad_h_phi_x, hes_h_phi_x
    else:
        raise ValueError("Invalid nargout")


def numdiff(func, x, *kwargs):
    identity = np.identity(len(x))
    eps = kwargs[0]
    id_eps = identity * eps
    grad = np.array([[np.multiply((func(x + id_eps[:, i].reshape((3, 1)), *kwargs[1:]) -
                                   func(x - id_eps[:, i].reshape((3, 1)), *kwargs[1:])), 1 / (2 * eps))
                      for i in range(len(id_eps))]]).T
    hes = np.array([np.multiply((func(x + id_eps[:, i].reshape((3, 1)), *kwargs[1:], nargout=2)[1] -
                                 func(x - id_eps[:, i].reshape((3, 1)), *kwargs[1:], nargout=2)[1]), 1 / (2 * eps))
                    for i in range(len(id_eps))]).reshape((3, 3))
    return grad, hes


# TODO make sure return matches input.

def plot_graphs(x, A):
    hw_epsilon = ((2 * (10 ** -16)) ** (1 / 3)) * x_vector.max()
    f1_analytic = f1(x, A, phi, nargout=3)
    f2_analytic = f2(x, phi, h, nargout=3)
    epsilons_list = [((2 * (10 ** -16)) ** (1 / i)) * x_vector.max() for i in range(1, 1000)]

    f1_grad_diff_infinity_norm, f1_hes_diff_infinity_norm, f2_grad_diff_infinity_norm, f2_hes_diff_infinity_norm \
        = [], [], [], []

    """ Best epsilon holder:
    best_f1_grad_diff, best_f1_hes_diff, best_f2_grad_diff, best_f2_hes_diff = \
        [np.inf, 0], [np.inf, 0], [np.inf, 0], [np.inf, 0]"""

    for epsilon in epsilons_list:

        f1_grad_numeric, f1_hes_numeric = numdiff(f1, x, epsilon, A, phi)
        f2_grad_numeric, f2_hes_numeric = numdiff(f2, x, epsilon, phi, h)
        f1_grad_diff = f1_analytic[1] - f1_grad_numeric
        f1_hes_diff = f1_analytic[2] - f1_hes_numeric
        f2_grad_diff = f2_analytic[1] - f2_grad_numeric
        f2_hes_diff = f2_analytic[2] - f2_hes_numeric

        if epsilon == hw_epsilon:  # plotting graphs for HW
            plot.plot(f1_grad_diff, 'o')
            plot.xlabel('Coordinate')
            plot.ylabel('Value')
            plot.show()
            plot.imshow(f1_hes_diff)
            plot.colorbar()
            plot.show()
            plot.plot(f2_grad_diff, 'o')
            plot.xlabel('Coordinate')
            plot.ylabel('Value')
            plot.show()
            plot.imshow(f2_hes_diff)
            plot.colorbar()
            plot.show()

        f1_grad_diff_infinity_norm += [np.abs(np.amax(f1_grad_diff))]
        f1_hes_diff_infinity_norm += [np.abs(np.amax(f1_hes_diff))]
        f2_grad_diff_infinity_norm += [np.abs(np.amax(f2_grad_diff))]
        f2_hes_diff_infinity_norm += [np.abs(np.amax(f2_hes_diff))]

        """ Best epsilon calculations:
        if np.abs(np.amax(f1_grad_diff)) < best_f1_grad_diff[0]:
            best_f1_grad_diff = [np.abs(np.amax(f1_grad_diff)), epsilon]
        if np.abs(np.amax(f1_hes_diff)) < best_f1_hes_diff[0]:
            best_f1_hes_diff = [np.abs(np.amax(f1_hes_diff)), epsilon]
        if np.abs(np.amax(f2_grad_diff)) < best_f2_grad_diff[0]:
            best_f2_grad_diff = [np.abs(np.amax(f2_grad_diff)), epsilon]
        if np.abs(np.amax(f2_hes_diff)) < best_f2_hes_diff[0]:
            best_f2_hes_diff = [np.abs(np.amax(f2_hes_diff)), epsilon]"""

    for y, y_name in zip([f1_grad_diff_infinity_norm, f1_hes_diff_infinity_norm, f2_grad_diff_infinity_norm,
                          f2_hes_diff_infinity_norm],
                         [r'$f_1$ grad diff infinity norm', r'$f_1$ hes diff infinity norm',
                          r'$f_2$ grad diff infinity norm', r'$f_2$ hes diff infinity norm']):
        plot.plot(epsilons_list, y)
        plot.xlabel('epsilon')
        plot.loglog()
        plot.ylabel(y_name)
        plot.show()

    """Printing best epsilon:
    print(f"best_f1_grad_diff: {best_f1_grad_diff}")
    print(f"best_f1_hes_diff: {best_f1_hes_diff}")
    print(f"best_f2_grad_diff: {best_f2_grad_diff}")
    print(f"best_f2_hes_diff: {best_f2_hes_diff}")"""


def magic():
    p = np.arange(1, 3 + 1)
    return 3 * np.mod(p[:, None] + p - (3 + 3) // 2, 3) + np.mod(p[:, None] + 2 * p - 2, 3) + 1


if __name__ == "__main__":
    x_vector = np.random.rand(3, 1)
    """x_vector = np.array([[0.06892114230312663, 0.24130053450481037, 0.4669045325053164]]).transpose()
    - the x_vector used for the dry graphs"""
    A_mat = magic()
    # A_mat = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    plot_graphs(x_vector, A_mat)
