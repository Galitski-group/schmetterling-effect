#Import Libraries
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import json
from pathlib import Path

############################## Define Functions ########################
def TM_Eigvals(num_qudits:int, bc_type:str, pars:list,local_hs_dim:int, corr_sites:bool):
    """
    Calculates eigenvalues of the transfer matrix. It constructs a reduced transfer
    matrix which is 2^(N/2) by 2^(N/2) dimensional with the same nonzero eigenvalues
    as the original transfer matrix and diagonalizes it.
    Inputs
    ------
    num_qudits (int): number of qudits in the circuit
    bc_type (int): boundary condition type either 'open' or 'periodic'
    pars (int): either [p] (measurement probability) or [alpha,beta] (see paper definition)
    local_hs_dim (int): qudit dimension (q=2 for qubits)
    corr_sites (bool): whether to use correlated sites of not. They just differ by p->(2-p)p
    Outputs
    -------
    eigenvalues (np.ndarray) sorted array of all eigenvalues of the reduced transfer matrix.
    
    """
    q=local_hs_dim
    #check if it is dephasing/measurement non-unitaries specifically
    if len(pars) == 1:
        p_in = pars[0]
        if corr_sites == True:
            p = p_in
        else:
            p = (2-p_in)*p_in
        alpha = q*q*(1-p) + q*p
        beta = q
    #if the local non-unitaries are the same
    elif len(pars) == 2:
        alpha = pars[0]
        beta = pars[1]
    #invalid input
    else:
        raise ValueError("Invalid number of parameters in pars")
    #calculate the coefficients
    A = 1/(q*q*(q**4-1))
    a = A * (q**3 * beta - alpha)
    b = A * (q**4 * alpha - q**3 * beta)
    c = A * (beta**2 - 1/(q**2) * alpha**2)
    d = A * (q**2*alpha**2 - beta**2)
    U = np.array([[1.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,1.0]])
    V = np.array([[1.0, a, a, c],[0.0, b, b, d]])
    M = U @ V
    if bc_type == "open":
        A  = np.eye(2)
        I2 = np.eye(2)
        for n in range(1,num_qudits//2):
            I_temp = np.eye(2**(n-1))
            V_temp = np.kron(np.kron(I_temp,V), I2)
            U_temp = np.kron(np.kron(I_temp, U), I2)
            A = V_temp @ (np.kron(A,M))@ U_temp
        I_temp = np.eye(2**(num_qudits//2 -1))
        V_temp = np.kron(I_temp,V)
        U_temp = np.kron(I_temp,U)
        K  = V_temp @ (np.kron(A,I2))@ U_temp
    elif bc_type == "periodic":
        def cyclic_shift_matrix(L, dtype=float):
            d = 2**L
            j = np.arange(d)
            shifted = ((j & 1) << (L - 1)) | (j >> 1)
            return np.eye(d, dtype=dtype)[shifted].T
        A = M
        I2 = np.eye(2)
        Q = cyclic_shift_matrix(num_qudits // 2 + 1, dtype=A.dtype)
        for n in range(1,num_qudits//2):
            I_temp = np.eye(2**n)
            V_temp = np.kron(np.kron(I_temp,V), I2)
            U_temp = np.kron(np.kron(I_temp, U), I2)
            A = V_temp @ (np.kron(A,M))@ U_temp
        I_temp = np.eye(2**(num_qudits//2 -1))
        V_temp = np.kron(V,I_temp)
        U_temp = np.kron(U,I_temp)
        K  = V_temp @ Q @ A @ Q.T @ U_temp
    else:
        raise ValueError("Invalid boundary condition")
    #Diagonalize K
    eigenvalues = np.sort(np.linalg.eigvals(K).real)[::-1]
    return eigenvalues

def Island_Lower_Bound(L:int, p_in:float,q:int, corr_sites:bool):
    """
    Calculates a Lower bound on the Transfer Matrix eigenvalue controlling decay. 
    The ansatz is a sum of island states, states with a contiguous region of 1s
    in a sea of zeros. Here L is the maximum island size (in the reduced 2^(N/2)
    dimensional space). The resulting lower bound is valid for any N>= 4(L-1) and
    hence is valid in the thermodynamic limit. This bound is obtained by finding
    the larges eigenvalue of a generalized eigenvalue problem with L by L
    dimensional matrices.
    Inputs
    ------
    L (int): max island size for ansatz. The lower bound improves monatonically with L.
    p_in (float): measurement probability
    q (int):  qudit dimension (q-2 for qubits)
    corr_sites (int):  whether to define p as the measurement induced OTOC
        corresponding to the experiment with correlated or uncorrelated sites.
        They just differ by p->(2-p)p
    Outputs
    -------
    eta (float): lower bound on the decay eigenvalue.
    """
    #check which averaging convention is used
    if corr_sites == True:
        p = p_in
    else:
        p = (2-p_in)*p_in
    alpha = q*q*(1-p) + q*p
    beta = q
    #calculate the coefficients
    A = 1/(q*q*(q**4-1))
    a = A * (q**3 * beta - alpha)
    b = A * (q**4 * alpha - q**3 * beta)
    c = A * (beta**2 - 1/(q**2) * alpha**2)
    d = A * (q**2*alpha**2 - beta**2)
    #the coefficients u,v,w used in the H matrix
    u = a*q**2 + b*(a+b)
    v = q**2*(a+(a+b)**2 *b)
    w = q**4*(c+(a+b)**2 *d)
    w0 = q**(-4)*w
    #Define the G matrix
    #define function for the right hand side of G
    def G_func(l,l_prime,q):
        return q**(-2*(l-l_prime)) *( l-l_prime +1 +2*((1-q**(-4*(l_prime -1)))/(q**4-1)) +(1-l-l_prime) *q**(-4*l_prime))
    G = np.fromfunction(lambda l, l_prime: G_func(np.maximum(l,l_prime) + 1, np.minimum(l,l_prime) +1, float(q)), (L,L),dtype=float)
    #define the H matrix off diagonal
    def H_func(l,l_prime,q):
        return q**(-2*(l-l_prime))*(2*u*v*q**(-4)*w0**(l_prime -1)+(l - l_prime - 1)*v*v*q**(-4)*w0**(l_prime -1)
        +2*v*v*((w0**(l_prime -1)*q**(-4)-q**(-4*l_prime))/(w-1))+(2*u-l-l_prime-1)*q**(-4*l_prime))
    H = np.fromfunction(lambda l, l_prime: H_func(np.maximum(l,l_prime) + 1, np.minimum(l,l_prime) +1, float(q)), (L,L),dtype=float)
    #redefine the H matrix diagonal
    def H_diag_func(l,q):
        return (q**(-4)*w0**(l-1)*u*u+2*v*v*(q**(-4)*w0**(l-1)-q**(-4*l))/(w-1)+(2*u-2*l-1)*q**(-4*l))
    l_array = np.arange(1,L+1)
    np.fill_diagonal(H,H_diag_func(l_array,float(q)))
    #solve the generalized eigenvalue problem
    eta = eigh(H, G, eigvals_only=True, subset_by_index=[L-1, L-1])[0]
    return eta


#########Calculate eigenvalues and variational bound,plot and export results###
#calculate variational lower bound
#Max island size for ansatz
L=1000
#qudit dimension
q=2
#whether to correlate measurement sites in average
corr_sites = True
#number of p values to sample
num_points = 500
#calculate lower bound at different values of p
p_vals = np.linspace(0,1,num_points)
eta_vals = np.zeros_like(p_vals)
for n in range(num_points):
    p = p_vals[n]
    eta_vals[n] = Island_Lower_Bound(L,p,q,corr_sites)
    
#Diagonalize the tranfer matrix
num_qudits = 20
#bc_type can be 'periodic' or 'open'
bc_type = "periodic"
eigvals_list = []
for n in range(num_points):
    pars = [p_vals[n]]
    eigenvalues = TM_Eigvals(num_qudits, bc_type, pars,q, corr_sites)
    eigvals_list.append(eigenvalues)
#eigenvalues array is a 2d numpy array 
#where first index is over p and second over eigenvalues
eigvals_array = np.array(eigvals_list).T
lambda1 = eigvals_array[0,:]
lambda2 = eigvals_array[1,:]

#plot the result
plt.figure()
plt.plot(p_vals,eigvals_array[0,:], label = f"lambda_ {1}", color ="blue")
plt.plot(p_vals,eigvals_array[1,:], label = f"lambda_ {2}",color = "red")
plt.plot(p_vals, eta_vals, label = "Large N lambda_2 lower bound",linestyle ='--',color = "green")
plt.xlabel("Measurement Probability p")
plt.ylabel("Eigenvalues")
plt.legend()
plt.show()

#export results as json file

data = {
        "qudit_dimension": q,
        "num_qudits": num_qudits,
        "bc_type ": bc_type,
        "island_max_size": L, 
        "corr_sites": corr_sites,
        "p_vals":  p_vals.tolist(),
        "lambda1": lambda1.tolist(),
        "lambda2": lambda2.tolist(),
        "lower_bound": eta_vals.tolist()
        }

file_path = Path.home() / "Desktop" / "eigvals_data.json"
with file_path.open("w") as f:
    json.dump(data, f, indent=4)

print("Saved here:", file_path)
print("File exists:", file_path.exists())

