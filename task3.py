import numpy as np
import matplotlib.pyplot as plot


def phi(x):
    mul_x = x[0]*x[1]*x[2]
    phi_x = np.sin(mul_x)
    grad_phi_x = np.transpose(np.array([[np.cos(mul_x)*x[1]*x[2], np.cos(mul_x)*x[0]*x[2], np.cos(mul_x)*x[0]*x[1]]]))
    hes_phi_x = np.array([[-np.sin(mul_x)*(x[1]**2)*(x[2]**2), -np.sin(mul_x)*x[0]*(x[2]**2), -np.sin(mul_x)*x[0]*(x[1]**2)],
                         [-np.sin(mul_x)*x[0]*(x[2]**2), -np.sin(mul_x)*(x[0]**2)*(x[2]**2), -np.sin(mul_x)*(x[0]**2)*x[1]],
                         [-np.sin(mul_x)*x[0]*(x[1]**2), -np.sin(mul_x)*(x[0]**2)*x[1], -np.sin(mul_x)*(x[0]**2)*(x[1]**2)]])

    return phi_x, grad_phi_x, hes_phi_x


def f1(x, *kwargs):
    A = np.array(kwargs[0])
    Ax = np.matmul(A, x)
    phi_x, grad_x, hes_x = kwargs[1](Ax)
    gradient_x = np.matmul(np.transpose(A), grad_x)
    hessian_x = np.matmul(np.transpose(A), np.matmul(hes_x, A))
    return phi_x, gradient_x, hessian_x


def h(x):
    h_x = np.exp(x)
    grad_h_x = h_x
    hes_h_x = grad_h_x
    return h_x, grad_h_x, hes_h_x


def f2(x, *kwargs):
    phi_in = kwargs[0]
    h_in = kwargs[1]
    phi_x, grad_phi_x, hes_phi_x = phi_in(x)
    h_phi_x, h_derivative, h_second_derivative = h_in(phi_x)
    grad_h_phi_x = h_derivative*grad_phi_x
    hes_h_phi_x = np.multiply(np.matmul(np.transpose(grad_phi_x), grad_phi_x), h_second_derivative) + np.multiply(hes_phi_x, h_derivative)
    return h_phi_x, grad_h_phi_x, hes_h_phi_x


def numdiff(func, x, *kwargs):
    id = np.identity(len(x))
    eps = kwargs[0]
    grad = np.array([np.multiply((func(x + eps*id_vec, *kwargs[1:])[0] - func(x - eps*id_vec, *kwargs[1:])[0]), 1/(2*eps)) for id_vec in id])
    hes = np.array([np.multiply((func(x + eps*id_vec, *kwargs[1:])[1] - func(x - eps*id_vec, *kwargs[1:])[1]), 1/(2*eps)) for id_vec in id])
    return grad, hes

# TODO make sure return matches input.

if __name__ == "__main__":
    x = np.array([1, 2, 3])
    x = np.transpose(x)
    A = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    eps = ((2 * (10 ** -15))**(1/3))*x.max()
    f1_analytic = f1(x, A, phi)
    f2_analytic = f2(x, phi, h)
    f1_grad_numeric, f1_hes_numeric = numdiff(f1, x, eps, A, phi)
    f2_grad_numeric, f2_hes_numeric = numdiff(f2, x, eps, phi, h)
    f1_grad_numeric = f1_grad_numeric.reshape((3, ))
    f1_hes_numeric = f1_hes_numeric.reshape((3, -1))
    f2_grad_numeric = f2_grad_numeric.reshape((3,))
    f2_hes_numeric = f2_hes_numeric.reshape((3, -1))
    f1_grad_diff = f1_analytic[1]-f1_grad_numeric
    f1_hes_diff = f1_analytic[2]-f1_hes_numeric
    f2_grad_diff = f2_analytic[1] - f2_grad_numeric
    f2_hes_diff = f2_analytic[2] - f2_hes_numeric
    f1_grad_diff_infinity_norm = np.amax(f1_grad_diff)
    f1_hes_diff_infinity_norm = np.amax(f1_hes_diff)
    f2_grad_diff_infinity_norm = np.amax(f2_grad_diff)
    f2_hes_diff_infinity_norm = np.amax(f2_hes_diff)
    plot.plot(f1_grad_diff)
    plot.show()
    plot.imshow(f1_hes_diff)
    print(f1_hes_diff)
    plot.show()
    plot.plot(f2_grad_diff)
    plot.show()
    plot.imshow(f2_hes_diff)
    print(f2_hes_diff)
    plot.show()
    arg = f2_grad_numeric


