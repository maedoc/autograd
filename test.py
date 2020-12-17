from autograd import numpy as np, grad, jacobian

def f(x):
    return x**2 + np.sum(x)

def rk4(f,dt=0.1):
    def _(x):
        k1 = f(x)
        k2 = f(x + dt/2*k1)
        k3 = f(x + dt/2*k2)
        k4 = f(x + dt*k3)
        return x + dt/6*(k1 + 2*(k2 + k3) + k4)
    return _


F = jacobian(rk4(f))
x = np.r_[:1.0]
Fx = F(x)

print(Fx)
