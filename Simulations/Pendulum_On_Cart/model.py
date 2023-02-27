import math
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from torch import autograd
from parameters import m_c, m_p, l, g, n_x, n_u, dt, d

# python libraries
import numpy as np

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

# Cartpole (nonlinear) dynamics
def f(x, u):
    '''
        The cartpole dynamics is a nonlinear dynamics, and is very popular "toy example" in model-based as well as data-driven algorithms.
    '''
    theta = x[2]
    s = torch.sin(theta)
    c = torch.cos(theta)

    theta_dot = x[-1]
    x_dot = x[1]
    theta_dot_2 = ( (m_p+m_c)*g*s-m_p*l*(theta_dot**2)*s*c-u[0]*c +d*c*x_dot )/( l*(m_c+m_p*(s**2)) )
    x_dot_2 = (u[0]-m_p*g*s*c+m_p*l*(theta_dot**2)*s-d*x_dot)/(m_c+m_p*(s**2)) 

    return torch.stack([x_dot, x_dot_2, theta_dot, theta_dot_2], dim=0)

def rk4_single_step(x0, u0):
    """
        This function does a single step 4th-order Runge-Kutta (a.k.a RK4)
    """
    f1 = f(x0, u0)
    f2 = f(x0 + (dt/2) * f1, u0)
    f3 = f(x0 + (dt/2) * f2, u0)
    f4 = f(x0 + dt * f3, u0)
    x_next = x0 + (dt/6) * (f1 + 2 * f2 + 2 * f3 + f4)
    return x_next


def euler_single_step(dt, x, u):
    '''
        Euler method for integration is a first order integration.
        Usually sufficient for most toy examples, however, for some nonlinear scenarios with rapid changes, we 
        will prefer high order integration methods such as RK45.
    '''
    return x + dt * f(x, u) 
