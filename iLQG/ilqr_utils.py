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
from parameters import x_goal, n_x, n_u, dtype_, Q, R, QT


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


cost_stage = lambda x, u: (x-x_goal).matmul(Q).matmul(x-x_goal) + R*(u**2)
cost_final = lambda x: (x-x_goal).matmul(QT).matmul(x-x_goal)

# def cost_stage(x, u):
#     x_diff = x-x_goal
#     return (x_diff.matmul(Q)).matmul(x_diff) + R*(u**2)

# def cost_final(x):
#     x_diff = x-x_goal
#     return (x_diff.matmul(QT)).matmul(x_diff)


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


# Q function of the Bellman equation
def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
    # Define the Q-terms here
    Q_x = l_x + f_x.transpose(0, 1).matmul(V_x)
    Q_u = l_u + f_u.transpose(0, 1).matmul(V_x)
    Q_xx = l_xx + (f_x.transpose(0, 1).matmul(V_xx)).matmul(f_x)
    Q_uu = l_uu + (f_u.transpose(0, 1).matmul(V_xx)).matmul(f_u) 
    Q_ux = l_ux + (f_u.transpose(0, 1).matmul(V_xx)).matmul(f_x)
    
    return Q_x, Q_u, Q_xx, Q_ux, Q_uu

# Feedforward gains
def gains(Q_uu, Q_u, Q_ux):
    Q_uu_inv = torch.linalg.inv(Q_uu)
    k = -Q_uu_inv.matmul(Q_u)
    K = -Q_uu_inv.matmul(Q_ux)
    
    return k, K


'''
    Derive the backwards update equation for the value function.
'''
def V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
    V_x = Q_x + K.transpose(0, 1).matmul(Q_u) + Q_ux.transpose(0, 1).matmul(k) + (k.transpose(0, -1).matmul(Q_uu)).matmul(K) 
    V_xx = Q_xx + Q_ux.transpose(0, 1).matmul(K) + K.transpose(0, 1).matmul(Q_ux) + (K.transpose(0, 1).matmul(Q_uu)).matmul(K) 
    
    return V_x, V_xx



'''
    Expected Cost Reduction:
    We can also estimate by how much we expect to reduce the cost by applying the optimal controls. 
    Simply subtract the previous nominal Q-value ($\delta \mathbf{x}[n] = 0$ and $\delta \mathbf{u}[n]=0$) from the value function.  
    The result is implemented below and is a useful aid in checking how accurate the quadratic approximation is during convergence of iLQR and adapting stepsize and regularization.
'''
def expected_cost_reduction(Q_u, Q_uu, k):
    return -Q_u.T.dot(k) - 0.5 * k.T.dot(Q_uu.dot(k))


'''
    Forward Pass:
    In the forward pass, at each timestep the new updated control $\mathbf{u}' =  \bar{\mathbf{u}} + k + K (x' - \bar{\mathbf{x}})$ is applied and the dynamis propagated based on the updated control. 
    The nominal control and state trajectory $\bar{\mathbf{u}}, \bar{\mathbf{x}}$ with which we computed $k$ and $K$ are then updated and we receive a new set of state and control trajectories.
'''
def forward_pass(x_trj, u_trj, k_trj, K_trj):
    x_trj_new = np.zeros(x_trj.shape)
    x_trj_new[0,:] = x_trj[0,:]
    u_trj_new = np.zeros(u_trj.shape)

    # Implement the forward pass
    for n in range(u_trj.shape[0]):
        u_trj_new[n,:] = u_trj[n] + k_trj[n] + K_trj[n].dot(x_trj_new[n]-x_trj[n])# Apply feedback law
        # u_trj_new[n,:] = u_trj[n] + k_trj[n] + K_trj[n].dot(x_trj_new[n]-x_goal)# Apply feedback law
        # x_trj_new[n+1,:] = discrete_dynamics(x_trj_new[n], u_trj_new[n])# Apply dynamics
        x_trj_new[n+1,:] = rk4singlestep(x_trj_new[n], u_trj_new[n])
    return x_trj_new, u_trj_new

'''
Backward Pass
'''
def backward_pass(x_trj, u_trj, regu):
    k_trj = np.zeros([u_trj.shape[0], u_trj.shape[1]])
    K_trj = np.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])
    expected_cost_redu = 0
    # Set terminal boundary condition here (V_x, V_xx)
    V_xx = np.zeros((n_x, n_x))
    V_x = np.zeros((n_x))

    derivs = derivatives(rk4singlestep, f, cost_stage, cost_final, n_x, n_u)

    for n in range(u_trj.shape[0]-1, -1, -1):
        # First compute derivatives, then the Q-terms 
        l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = derivs.stage(x_trj[n], u_trj[n])
        F = np.hstack((f_x, f_u))
        c = np.vstack((l_x.reshape(n_x,1), l_u.reshape(n_u,1))).reshape(-1)
        C = np.vstack((np.hstack((l_xx, l_ux.T)), np.hstack((l_ux, l_uu))))
        # Q_x, Q_u, Q_xx, Q_ux, Q_uu = Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)
        Q_xx, Q_ux, Q_uu, q_x, q_u = Q_terms(F, C, c, V_xx, V_x, x_trj[n], u_trj[n])
        # We add regularization to ensure that Q_uu is invertible and nicely conditioned
        # i.e., we apply regularization term to the local cost to account for the non-positive-definite second order 
        # Hessian's computed on the cost:
        Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0])*regu
        k, K = gains(Q_uu_regu, q_u, Q_ux)
        k_trj[n,:] = k
        K_trj[n,:,:] = K
        V_x, V_xx = V_terms(q_x, q_u, Q_xx, Q_ux, Q_uu, K, k)
        expected_cost_redu += expected_cost_reduction(q_u, Q_uu, k)
        
    return k_trj, K_trj, expected_cost_redu


class ILqr:
    def __init__(self, next_state, running_cost, final_cost,
                 umax, state_dim, pred_time=10):
        self.pred_time = pred_time
        self.umax = umax
        self.v = [0.0 for _ in range(pred_time + 1)]
        self.v_x = [np.zeros(state_dim) for _ in range(pred_time + 1)]
        self.v_xx = [np.zeros((state_dim, state_dim)) for _ in range(pred_time + 1)]
        self.f = next_state
        self.lf = final_cost
        self.lf_x = grad(self.lf)
        self.lf_xx = jacobian(self.lf_x)
        self.l_x = grad(running_cost, 0)
        self.l_u = grad(running_cost, 1)
        self.l_xx = jacobian(self.l_x, 0)
        self.l_uu = jacobian(self.l_u, 1)
        self.l_ux = jacobian(self.l_u, 0)
        self.f_x = jacobian(self.f, 0)
        self.f_u = jacobian(self.f, 1)
        self.f_xx = jacobian(self.f_x, 0)
        self.f_uu = jacobian(self.f_u, 1)
        self.f_ux = jacobian(self.f_u, 0)

    def backward(self, x_seq, u_seq):
        self.v[-1] = self.lf(x_seq[-1])
        self.v_x[-1] = self.lf_x(x_seq[-1])
        self.v_xx[-1] = self.lf_xx(x_seq[-1])
        k_seq = []
        kk_seq = []
        for t in range(self.pred_time - 1, -1, -1):
            f_x_t = self.f_x(x_seq[t], u_seq[t])
            f_u_t = self.f_u(x_seq[t], u_seq[t])
            q_x = self.l_x(x_seq[t], u_seq[t]) + np.matmul(f_x_t.T, self.v_x[t + 1])
            q_u = self.l_u(x_seq[t], u_seq[t]) + np.matmul(f_u_t.T, self.v_x[t + 1])
            q_xx = self.l_xx(x_seq[t], u_seq[t]) + \
              np.matmul(np.matmul(f_x_t.T, self.v_xx[t + 1]), f_x_t) + \
              np.dot(self.v_x[t + 1], np.squeeze(self.f_xx(x_seq[t], u_seq[t])))
            tmp = np.matmul(f_u_t.T, self.v_xx[t + 1])
            q_uu = self.l_uu(x_seq[t], u_seq[t]) + np.matmul(tmp, f_u_t) + \
              np.dot(self.v_x[t + 1], np.squeeze(self.f_uu(x_seq[t], u_seq[t])))
            q_ux = self.l_ux(x_seq[t], u_seq[t]) + np.matmul(tmp, f_x_t) + \
              np.dot(self.v_x[t + 1], np.squeeze(self.f_ux(x_seq[t], u_seq[t])))
            inv_q_uu = np.linalg.inv(q_uu)
            k = -np.matmul(inv_q_uu, q_u)
            kk = -np.matmul(inv_q_uu, q_ux)
            dv = 0.5 * np.matmul(q_u, k)
            self.v[t] += dv
            self.v_x[t] = q_x - np.matmul(np.matmul(q_u, inv_q_uu), q_ux) #q_x + np.matmul(kk.T, q_u) + np.matmul(q_ux.T, k) + np.matmul(np.matmul(k.T,q_uu), kk)
            self.v_xx[t] = q_xx + np.matmul(q_ux.T, kk) #q_xx + np.matmul(q_ux.T, kk) + np.matmul(kk.T, q_ux) + np.matmul(np.matmul(kk.T, q_uu), kk) 
            k_seq.append(k)
            kk_seq.append(kk)
        k_seq.reverse()
        kk_seq.reverse()
        return k_seq, kk_seq

    def forward(self, x_seq, u_seq, k_seq, kk_seq):
        x_seq_hat = np.array(x_seq)
        u_seq_hat = np.array(u_seq)
        for t in range(len(u_seq)):
            control = k_seq[t] + np.matmul(kk_seq[t], (x_seq_hat[t] - x_seq[t]))
            u_seq_hat[t] = np.clip(u_seq[t] + control, -self.umax, self.umax)
            x_seq_hat[t + 1] = self.f(x_seq_hat[t], u_seq_hat[t])
        return x_seq_hat, u_seq_hat

def run_ilqr(x0, N, max_iter=50, regu_init=100):
    # First forward rollout
    u_trj = np.random.randn(N-1, n_u)*0.0001
    x_trj = rollout(x0, u_trj)
    total_cost = cost_trj(x_trj, u_trj)
    regu = regu_init
    max_regu = 10000
    min_regu = 0.01
    
    # Setup traces
    cost_trace = [total_cost]
    expected_cost_redu_trace = []
    redu_ratio_trace = [1]
    redu_trace = []
    regu_trace = [regu]
    
    # Run main loop
    for it in range(max_iter):
        # Backward and forward pass
        k_trj, K_trj, expected_cost_redu = backward_pass(x_trj, u_trj, regu)
        x_trj_new, u_trj_new = forward_pass(x_trj, u_trj, k_trj, K_trj)

        # Evaluate new trajectory
        total_cost = cost_trj(x_trj_new, u_trj_new)
        cost_redu = cost_trace[-1] - total_cost
        redu_ratio = cost_redu / abs(expected_cost_redu)
        # Accept or reject iteration
        if cost_redu > 0:
            # Improvement! Accept new trajectories and lower regularization
            redu_ratio_trace.append(redu_ratio)
            cost_trace.append(total_cost)
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