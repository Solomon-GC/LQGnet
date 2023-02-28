import numpy as np
import torch 
import torch.nn as nn
from torch.autograd.functional import jacobian, hessian
from torch.autograd import grad

from torch.distributions.multivariate_normal import MultivariateNormal

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU")


from model import rk4singlestep, f
from parameters import x_goal, n_x, n_u, dtype_, Q, R, QT, rk4_single_step


def rollout(x0, u_trj):
    '''
        Rollout refers to open-loop instance
    '''
    x_trj = torch.zeros((u_trj.shape[0]+1, x0.shape[0]), dtype=dtype_)
    # Define the rollout here and return the state trajectory x_trj: [N, number of states]
    N = u_trj.shape[0]+1
    x_trj[0] = x0 
    for n in range(N-1):
        # x_trj[n+1] = rk4singlestep(pendulum_on_cart_continuous_dynamics, dt, t[n], x_trj[n], u_trj[n])
        x_trj[n+1] = rk4singlestep(x_trj[n], u_trj[n])
        
    return x_trj


'''
    Quadratic Cost
    1. cost stage (a.k.a. running cost)
    2. cost final (refers to the cost of calculated in the final state)
    3. cost_trj - is the total cost, i.e. cost stage + cost final.
'''
cost_stage = lambda x, u: (x-x_goal).matmul(Q).matmul(x-x_goal) + R*(u**2)
cost_final = lambda x: (x-x_goal).matmul(QT).matmul(x-x_goal)

def cost_trj(x_trj, u_trj):
    '''
        Total cost over a full trajectory
    '''
    total = 0.0
    # Sum up all costs
    N = u_trj.shape[0]+1
    for n in range(N-1):
        total += cost_stage(x_trj[n], u_trj[n]) 
    
    total += cost_final(x_trj[-1])

    return total

'''
    iLQR class packages all necessary function to execute open loop iLQR.
'''
class iLQR:
    ''' 
      Init iLQR Class
    '''
    # def __init__(self, next_state, running_cost, final_cost, umax, state_dim, pred_time=50, is_control_constraint=False):
    def __init__(self, next_state, running_cost, final_cost, umax=20, pred_time=50, is_control_constraint=False):
      self.is_control_constraint = is_control_constraint
      # self.pred_time = pred_time
      self.umax = umax
      self.f = next_state
      self.lf = final_cost
      self.lr = running_cost

    ''' 
      Running cost derivatives
    '''
    def derivatives_running_cost(self, x, u):
      lf_x = jacobian(self.lf, x).reshape([n_x])
      lf_xx = hessian(self.lf, x)
      l_x, l_u = jacobian(self.lr, (x, u)) 
      l_x = l_x.reshape([n_x])
      l_u = l_u.reshape([n_u])

      D2x, D2u = hessian(self.lr, (x, u))
      l_xx = D2x[0]
      l_uu = D2u[1]
      l_ux = D2u[0] 
      
      f_x, f_u = jacobian(self.f, (x, u))

      return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u

    ''' 
      Final cost derivatives
    '''
    def derivative_final_cost(self, x):
      # The 1st and 2nd derivatives are 
      l_x = jacobian(self.lf, x).reshape([n_x])
      l_xx = hessian(self.lf, x)
      
      return l_x, l_xx

    ''' 
      Q function terms 
    '''
    def Q_terms(self, l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
      # Define the Q-terms here
      Q_x = l_x + f_x.transpose(0, 1).matmul(V_x)
      Q_u = l_u + f_u.transpose(0, 1).matmul(V_x)
      Q_xx = l_xx + (f_x.transpose(0, 1).matmul(V_xx)).matmul(f_x)
      Q_uu = l_uu + (f_u.transpose(0, 1).matmul(V_xx)).matmul(f_u) 
      Q_ux = l_ux + (f_u.transpose(0, 1).matmul(V_xx)).matmul(f_x)
    
      return Q_x, Q_u, Q_xx, Q_ux, Q_uu

    ''' 
      Compute control gains
    '''
    def gains(self, Q_uu, Q_u, Q_ux):
      Q_uu_inv = torch.linalg.inv(Q_uu)
      # Implement the feedforward gain k and feedback gain K.
      k = -Q_uu_inv.matmul(Q_u)
      K = -Q_uu_inv.matmul(Q_ux)
      
      return k, K
    
    ''' 
      Compute value function terms (based on Bellman equation)
    '''
    def V_terms(self, Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
      # Implement V_x and V_xx, hint: use the A.dot(B) function for matrix multiplcation.

      V_x = Q_x + K.transpose(0, 1).matmul(Q_u) + Q_ux.transpose(0, 1).matmul(k) + (k.transpose(0, -1).matmul(Q_uu)).matmul(K) 
      V_xx = Q_xx + Q_ux.transpose(0, 1).matmul(K) + K.transpose(0, 1).matmul(Q_ux) + (K.transpose(0, 1).matmul(Q_uu)).matmul(K) 
      
      return V_x, V_xx

    '''
        Expected cost reduction: estimate by how much we expect to reduce the cost by applying the optimal controls.
    '''
    def expected_cost_reduction(self, Q_u, Q_uu, k):
      return -Q_u.matmul(k) - 0.5 * (k.transpose(0, -1).matmul(Q_uu)).matmul(k)
    
    '''
      Backward pass
    '''
    def backward_pass(self, x_trj, u_trj, regu):
      k_trj = torch.zeros([u_trj.shape[0], u_trj.shape[1]], dtype=dtype_)
      K_trj = torch.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]], dtype=dtype_)
      expected_cost_redu = 0
      # Set terminal boundary condition here (V_x, V_xx)
      V_x, V_xx = self.derivative_final_cost(x_trj[-1])

      for n in range(u_trj.shape[0]-1, -1, -1):
          # First compute derivatives, then the Q-terms 
          l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = self.derivatives_running_cost(x_trj[n], u_trj[n])
          Q_x, Q_u, Q_xx, Q_ux, Q_uu = self.Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)
          # We add regularization to ensure that Q_uu is invertible and nicely conditioned
          # i.e., we apply regularization term to the local cost to account for the non-positive-definite second order 
          # Hessian's computed on the cost:
          Q_uu_regu = Q_uu + torch.eye(Q_uu.shape[0])*regu
          k, K = self.gains(Q_uu_regu, Q_u, Q_ux)
          k_trj[n,:] = k
          K_trj[n,:,:] = K
          V_x, V_xx = self.V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
          expected_cost_redu += self.expected_cost_reduction(Q_u, Q_uu, k)
      return k_trj, K_trj, expected_cost_redu
    ''' 
      Run Forward pass
    '''
    def forward_pass(self, x_trj, u_trj, k_trj, K_trj):
      x_trj_new = torch.zeros(x_trj.shape, dtype=dtype_)
      x_trj_new[0,:] = x_trj[0,:]
      u_trj_new = torch.zeros(u_trj.shape, dtype=dtype_)

      for n in range(u_trj.shape[0]):
          
          # Apply feedback law
          if self.is_control_constraint:
            u_trj_new[n,:] = torch.clip(u_trj[n] + k_trj[n] + K_trj[n].matmul(x_trj_new[n]-x_trj[n]), -self.umax, self.umax)
          else:
            u_trj_new[n,:] = u_trj[n] + k_trj[n] + K_trj[n].matmul(x_trj_new[n]-x_trj[n])# Apply feedback law
          
          x_trj_new[n+1,:] = self.f(x_trj_new[n], u_trj_new[n])
      return x_trj_new, u_trj_new

def run_ilqr(x0, N, max_iter=50, regu_init=100):
    # First forward rollout
    torch.manual_seed(0)

    ilqr = iLQR(rk4_single_step,  # x(i+1) = f(x(i), u)
            cost_stage,  # l(x, u)
            cost_final)  # lf(x)
            # max_force,
            # env.observation_space.shape[0])
    
    u_trj = torch.randn(N-1, n_u, requires_grad=True, dtype=dtype_)*0.0001
    x_trj = rollout(x0, u_trj)
    total_cost = cost_trj(x_trj, u_trj)
    regu = regu_init
    max_regu = 10000
    min_regu = 0.01
    
    # Setup traces
    cost_trace = [total_cost.cpu().detach().numpy()]
    expected_cost_redu_trace = []
    redu_ratio_trace = [1]
    redu_trace = []
    regu_trace = [regu]
    
    # Run main loop
    for it in range(max_iter):
        # Backward and forward pass
        k_trj, K_trj, expected_cost_redu = ilqr.backward_pass(x_trj, u_trj, regu)
        x_trj_new, u_trj_new = ilqr.forward_pass(x_trj, u_trj, k_trj, K_trj)

        # Evaluate new trajectory
        total_cost = cost_trj(x_trj_new, u_trj_new)
        cost_redu = cost_trace[-1] - total_cost.cpu().detach().numpy()
        redu_ratio = cost_redu / abs(expected_cost_redu.cpu().detach().numpy())
        # Accept or reject iteration
        if cost_redu > 0:
            # Improvement! Accept new trajectories and lower regularization
            redu_ratio_trace.append(redu_ratio)
            cost_trace.append(total_cost.cpu().detach().numpy())
            x_trj = x_trj_new
            u_trj = u_trj_new
            regu *= 0.7
        else:
            # Reject new trajectories and increase regularization
            regu *= 2.0
            cost_trace.append(cost_trace[-1])
            redu_ratio_trace.append(0)
        regu = min(max(regu, min_regu), max_regu)
        regu_trace.append(regu)
        redu_trace.append(cost_redu)

        # Early termination if expected improvement is small
        if expected_cost_redu <= 1e-6:
            break


    return x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace