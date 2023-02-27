import torch
import math
import numpy as np

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    dev = torch.device("cpu")
    print("Running on the CPU")
    
# 
# torch type: setting to double precision to align to PyDrake
dtype_ = torch.float64
    
##########################
### Problem Parameters ###
##########################
n_x = 4
n_y = 4
n_u = 1

m_p = 0.3     # pendulum (bob) mass [kg]
m_c = 1.0     # cart mass [kg]
g = 9.81      # gravity [m/s^2]
l = 1.0       # length of a rod [m]
d = 1         # damping 

## Integration solver parameters
dt = 0.1                    # resolution time step
T = 5                       # Time window [sec]
num_time_pts = int(T / dt)  # Time window in samples

t = torch.linspace(0, T, num_time_pts, dtype=dtype_)

# Target/Final state
x_goal = torch.tensor([0.0, 0.0, np.pi, 0.0], dtype=dtype_)

# Initial state and covariance
m1x_0 = torch.tensor([1.0, 0.0, 0.0*np.pi, 0], dtype=dtype_) #torch.ones(m, 1) * 10
m2x_0 = torch.zeros(n_x, n_x, dtype=dtype_)

# Max iLQR iterations
max_iter=5
# regularization 
regu_init=100

T_test = num_time_pts


# Set noise mean and covariances for dynamics and measurement
sigma_x = 0.1
sigma_y = 0.1
W = (sigma_x**2)*torch.eye(n_x, dtype=dtype_)
V = (sigma_y**2)*torch.eye(n_y, dtype=dtype_)

mean_x = torch.zeros([n_x], dtype=dtype_)
mean_y = torch.zeros([n_x], dtype=dtype_)

# Regulator weights and matrices (LQR Cost)
x_weight = 500.0
v_weight = 1.0
theta_weight = 500.0
omega_weight = 1.0

Q = torch.tensor([[x_weight, 0.0, 0.0, 0.0],[0.0, v_weight, 0.0, 0.0],[0.0, 0.0, theta_weight, 0.0], [0.0, 0.0, 0.0, omega_weight]], dtype=dtype_).to(dev)
QT = torch.tensor([[x_weight, 0.0, 0.0, 0.0],[0.0, v_weight, 0.0, 0.0],[0.0, 0.0, theta_weight, 0.0], [0.0, 0.0, 0.0, omega_weight]], dtype=dtype_).to(dev) # final weight matrix
R = torch.squeeze(torch.eye(n_u, dtype=dtype_)).to(dev) # Control input weight 


