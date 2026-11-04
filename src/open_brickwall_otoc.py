#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


class Open_Brickwall_OTOC:
    def __init__(self, num_qubits:int, bc_type:str = "periodic", local_hs_dim:int  = 2, corr_sites = False):
        """
        Attributes
        ----------
        num_qubits: (int) [Assumed to be even]
        bc_type: (str) Options are "periodic" or "open"
        local_hs_dim: (int) Dimension of the local Hilbert space, default is 2 for qubits.
        corr_sites: (bool) Whether to use same sites in epsilon and epislon^dagger in each realization or make them uncorrelated (for dephasing non-unitaries).

        Methods
        -------
        Local_Transfer_Matrix: Returns 4 by 4 transfer matrix given parameters p or alpha and beta.
        TM_OTOC: Calculates the OTOC exactly via direct application of the transfer matrix.
        Monte_Carlo_OTOC: Calculates the OTOC using Monte Carlo simulation.
        First_Order_OTOC: Calculates the OTOC to first order in p exactly by updating propagators (dephasing non-unitaries).
        Truncated_HS_OTOC: Calculates the OTOC using an effective truncated Fock space approximation.
        Front_Position: Given the OTOC C(x,t) calculates the contour x_h(t) of the normalized OTOC C(x_h(t),t)/Max_{x_prime}[C(x_prime,t)] = h.
        """
        #check that the number of qubits is even
        if not isinstance(num_qubits,int) or num_qubits % 2 != 0:
            raise ValueError("Number of qubits must be even integer")
        #check local hilbert space dimension is greater or equal to 2
        if not isinstance(local_hs_dim,int) or local_hs_dim < 2:
            raise ValueError("invalid loca hilbert space dimension, must be a positive integer >=2")
        if not isinstance(bc_type,str) or bc_type not in ["periodic", "open"]:
            raise ValueError("invalid boundary condition type, must be 'periodic' or 'open'")
        if not isinstance(corr_sites,bool):
            raise ValueError("corr_sites must be a boolean")
        #check that local_hc_dim is an even integer greater or equal to 2
        self.num_qubits = num_qubits
        self.bc_type= bc_type
        self.local_hs_dim = local_hs_dim
        self.corr_sites = corr_sites

    def Local_Transfer_Matrix(self, pars):
        """
        Calculates the local transfer matrix given the parameter values pars.
        The input pars should be a python list. It can either be pars =[p] where
        p is the measurement/dephasing probability or it can be pars =[alpha,beta]
        where  alpha equiv Tr(E^{otimes 2} (S) S),
        beta equiv Tr(mathcal{E}(I_q)^2) where math_cal{E} is the single
        qudit local non-unitary channel and S is the qudit SWAP matrix, or it
        can take pars =[alpha0,beta0,alpha1,beta1] (corresponding to different local non-unitarie
        at different sites). It is up to the user to choose parameter values
        for p or alpha/beta within the range physically allowed by the CPTP condition.

        Inputs
        ------
        pars:(python list of floats) can be pars = [p], pars = [alpha,beta], or pars = [alpha0, beta0, alpha1, beta1]
        """
        q = self.local_hs_dim
        #check if it is dephasing/measurement non-unitaries specifically
        if len(pars) == 1:
            p_in = pars[0]
            if self.corr_sites == True:
                p = p_in
            else:
                p = (2-p_in)*p_in
            alpha0 = q*q*(1-p) + q*p
            beta0 = q
            alpha1 = alpha0
            beta1 = beta0
        #if the local non-unitaries are the same
        elif len(pars) == 2:
            alpha0 = pars[0]
            beta0 = pars[1]
            alpha1 = alpha0
            beta1 = beta0
        #most general case
        elif len(pars) == 4:
            alpha0 = pars[0]
            beta0 = pars[1]
            alpha1 = pars[2]
            beta1 = pars[3]
        #invalid input
        else:
            raise ValueError("Invalid number of parameters in pars")
        #calculate the coefficients
        A = 1/(q*q*(q**4-1))
        a0 = A * (q**3 * beta0 - alpha0)
        a1 = A * (q**3 * beta1 - alpha1)
        b0 = A * (q**4 * alpha0 - q**3 * beta0)
        b1 = A * (q**4 * alpha1 - q**3 * beta1)
        c = A * (beta0*beta1 - 1/(q**2) * alpha0*alpha1)
        d = A * (q**2*alpha0*alpha1 - beta0*beta1)
        #calculate the local transfer matrix
        M = np.array([[1.0, a1, a0, c],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0, b1, b0, d]
                      ])
        return M

    def Monte_Carlo_OTOC(self, num_samples:int, T:int, pars:list, pert_site:int, init_parity:int):
        #define a function for updating the trajectory weight and state
        def Update_State(weight:float, state:np.ndarray, coeffs:list,t:int,rng: np.random.Generator):
            """
            Implements a Monte-Carlo update step for a single layer (even or odd based on t).
            """
            #read off the bulk coefficients for this update.
            a, b, c, d = coeffs
            #define normalization gamma = a + b = sqrt(c + d)
            gamma = a + b
            #define an array with probabilities for transitioning to 11 state
            prob_11 = np.array([
                0.0, # P(11|00)
                b/gamma, #P(11|01)
                b/gamma, #P(11|10)
                d/(gamma**2) #P(11|11)
            ])
            #roll the input state if its an odd time
            if t % 2 == 1:
                state = np.roll(state, 1)
            #Group this into pairs of sites
            pairs = state.reshape(-1,2)
            #now map pairs (00,01,10,11) to states(0,1,2,3)
            #00->0,01->1,10->2,11->3. states is an N//2 length array of integers
            states = 2 * pairs[:,0] + pairs[:,1]
            #bulk transition probabilities
            probs = prob_11[states]
            #get outcomes based on a random array and these probabilities
            outcomes = (rng.random(len(states)) < probs).astype(np.uint8)
            #Next we repeat these to get the new state
            new_state  = np.repeat(outcomes,2)
            #we need to determine the number of each type of transition which occured
            #in order to update the weights, here this is encoded in transition codes
            #with (00->00) = 0, (01->00) = 2, (10->00)=4, (11->00) = 6
            # (00->11,doesn't occur)=1, (01->11)=3, (10->11)=5, (11->11)=7
            transition_codes = 2 * states + outcomes
            #count how many of these occur
            transition_counts = np.bincount(transition_codes, minlength=8)
            #if we are on odd time and open bc we should not update the 1st one
            if t % 2 == 1:
                if self.bc_type == "open":
                    new_state[0] = state[0]
                    new_state[1] = state[1]
                    #make sure to decrease the transition counts to not include it
                    transition0 = 2 * states[0] + outcomes[0]
                    transition_counts[transition0] -= 1
                #roll back
                new_state = np.roll(new_state, -1)
                state = np.roll(state, -1)
            #update the weight.
            na = transition_counts[2] + transition_counts[4]
            nb = transition_counts[3] + transition_counts[5]
            nc = transition_counts[6]
            nd = transition_counts[7]
            new_weight = (gamma**(na+nb+2*nc+2*nd))*weight
            #return the updated state and weight
            return new_weight, new_state
        def Monte_Carlo_Amplitude(init_sites:list, num_samples, coeffs:list, T:int, rng: np.random.Generator):
            """
            Function Calculates monte-carlo estimate of A(x,t) equiv sum_{s_j,j neq x}<s_j,s_x=1|T^t|psi_0>
            for a given initial product state |psi_0> =|00...0110...0>
            """
            #Initial state
            init_site1, init_site2  = init_sites
            init_parity = min(init_site1,init_site2) % 2
            init_state = np.zeros(self.num_qubits, dtype = np.int8)
            init_state[init_site1] = 1
            init_state[init_site2] = 1
            #Initialize the Amplitude A(x,t)
            A = np.zeros((self.num_qubits, T+1))
            A[init_site1,0] = num_samples
            A[init_site2,0] = num_samples
            #Iterate through monte-carlo samples
            for _ in range(num_samples):
                #Initialize wieght (float) and state (int8 np array)
                weight = 1.0
                state = init_state.copy()
                #enter time loop
                for t in range(1,T+1):
                    #sample one trajectory layer
                    weight, state = Update_State(weight, state, coeffs,t+init_parity, rng)
                    #update A(x,t)
                    A[:,t] += weight * state
            A = 1/num_samples * A
            return A
        #The perturbation dependent initial value. C0 below corresponds to this
        #value for a pauli perturbation
        q = self.local_hs_dim
        C0 = (2*q**4)/(q**4-1)
        #if we use periodic boundary conditions we only need to calculate the
        #amplitude once due to translational symmetry
        M = self.Local_Transfer_Matrix(pars)
        a = M[0,1]
        b = M[3,1]
        c = M[0,3]
        d = M[3,3]
        coeffs = [a,b,c,d]
        rng = np.random.default_rng()
        if self.bc_type == "periodic":
            A = Monte_Carlo_Amplitude([0,1], num_samples, coeffs, T, rng)
            #calculate the OTOC from this, taking advantage of translational invariance.
            y = pert_site
            x = np.arange(self.num_qubits)[:,None]
            t = np.arange(T+1)[None,:]
            j = y-x + (x+t-init_parity) % 2
            C = C0*A[j,t]
            return C, x.ravel(), t.ravel()
        if self.bc_type == "open":
            C = np.zeros((self.num_qubits, T+1))
            x_even = np.arange(0,self.num_qubits, 2)
            for xval in x_even:
                init_site1_even =  min(xval, xval + (-1)**(0+init_parity))
                init_site1_odd = min(xval, xval + (-1)**(1+init_parity))
                init_site2_even = init_site1_even + 1
                init_site2_odd = init_site1_odd + 1
                if init_site1_even >=0:
                    A_even = Monte_Carlo_Amplitude([init_site1_even,init_site2_even], num_samples, coeffs, T, rng)
                    C[init_site1_even,::2] = C0*A_even[pert_site,::2]
                    C[init_site2_even,::2] = C0*A_even[pert_site,::2]
                if init_site1_odd >= 0:
                    A_odd = Monte_Carlo_Amplitude([init_site1_odd,init_site2_odd], num_samples, coeffs, T, rng)
                    C[init_site1_odd,1::2] = C0*A_odd[pert_site,1::2]
                    C[init_site2_odd,1::2] = C0*A_odd[pert_site,1::2]
            #manually define edge sites to have same OTOC as previous layer edge sites
            for tp in range(1,T+1):
                if C[0,tp] !=0:
                    t1 = tp
                    break
            for tp in range(1,T+1):
                if C[-1,tp] !=0:
                    t2 = tp
                    break
            C[0,t1 + 1::2] = C[0,t1:-1:2]
            C[-1,t2 + 1::2] = C[-1,t2:-1:2]
            x = np.arange(self.num_qubits)
            t = np.arange(T+1)
            return C, x, t
        else:
            raise ValueError("invalid boundary condition type, must be 'periodic' or 'open'")

    def Exact_OTOC(self, T:int, pars:list, pert_site:int, init_parity:int ):
        """
        Calculates the exact average OTOC using the transfer matrix. 

        Inputs
        ------
        p: (float) probability for the bernoulli trial to determine whether the site gets measured.
        tf: (int) final time
        pert_site: (int|str) either the integer value of the perturbation site or the string "all".
        probe_site: (int|str) either the integer value of the probe site or the string "all"

        Outputs
        -------
        OTOC: (nd.nparray) Outputs a numpy array of OTOC values. The dimension of the array depends on
             the input probe_site and pert_site. If probe_site and pert_site are both integers then it
             only calculates a 1d array of OTOC time values. If one of pert_site/probe_site is set to all
             it outputs a 2d array of values [probe_sites or perturbation_sites,time]. If both probe_site and
             perturbation_site are set to all it outputs a 3d array of values [probe_sites,perturbation_sites,time].

        time_vals: (nd.nparray) numpy array of time values.
        site_vals: (nd.nparray) numpy array of spacial site values.
        """
        if not isinstance(pert_site,int) and pert_site!= "all":
            raise ValueError("pert_site must be an integer or 'all'")
        #Function definitions:
        #---------------------
        def Apply_Brick(W, M, i, j):
            """
            Apply a 4 x 4 transfer matrix to tensor sites i and j,
            with periodic wrapping.
            """
            N = W.ndim

            # Periodic boundary conditions
            i = i % N
            j = j % N

            input_axes = list(range(N))
            output_axes = input_axes.copy()

            new_i, new_j = N, N + 1

            output_axes[i] = new_i
            output_axes[j] = new_j

            return np.einsum(
                M.reshape(2, 2, 2, 2),
                [new_i, new_j, i, j],
                W,
                input_axes,
                output_axes
            )

        def Adjoint_OTOC(self, p_in:float, tf:int, pert_site:int|str):
            """
            This code is usefull for calculating the OTOC at a fixed perturbation site
            for all probe sites when open boundary conditions are used. It assumes
            probe_site == "all" and self.bc_type == "open" since this is the only
            situation where this function is usefull.
            """
            #define the local transfer matrix
            #00->00, 10 or 01 -> a(00)+ b(11), 11 ->c (00) +d (11)
            if self.corr_sites == True:
                p = p_in
            else:
                p = (2-p_in)*p_in
            q = self.local_hs_dim
            a = 1/((q**4-1)*q)*(q**3 - q + (q-1)*p)
            b = ((q**2)/(q**4-1))*(q**2 -1 -q*(q-1)*p)
            c = 1/((q**4-1)*q**2)*(2*q*(q-1)*p -(q-1)**2 *p**2)
            d = 1/(q**4-1) *(q**4 -1 -2*q**3*(q-1)*p + q**2 *(q-1)**2 *p**2)
            T = np.array([
            [1, a, a, c],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, b, b, d],
            ])
            #take its transpose
            T_T = np.transpose(T)
            # Define an even edge local transfer matrix. This is only used in the niche
            #situation where we use open boundary conditions and we measure all sites so
            #The site at the edges before even layers effectively have a greater probability of
            #being measured, which is (2-p)p.
            #effective probability at the edge
            pe = (2-p)*p
            ae = 1/((q**4-1)*q)*(q**3 - q + (q-1)*pe)
            be = ((q**2)/(q**4-1))*(q**2 -1 -q*(q-1)*pe)
            ce = 1/((q**4-1)*q**2)*((p+pe)*q*(q-1)-p*pe*(q-1)**2)
            de = 1/(q**4 -1)*(-(p+pe)*q**3*(q-1)+p*pe*q**2*(q-1)**2)+1
            #the local transfer matrices for the right and left edge
            Tr = np.array([
            [1, ae, a, ce],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, be, b, de],
            ])
            Tl = np.array([
            [1, a, ae, ce],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, b, be, de],
            ])
            T_lT = np.transpose(Tl)
            T_rT = np.transpose(Tr)
            #Initialize the OTOC
            C = np.zeros((self.num_qubits, tf+1))
            C0 = 2*q**4/(q**4-1)
            init_site = pert_site - pert_site % 2
            C[init_site:init_site + 2, 0] = C0
            #apply the brickwall to the final state in reverse order
            #need to store separate even and odd wieghts if we want data at each
            #individual staggered layer rather than a pair
            W = np.zeros((2,)*self.num_qubits, dtype = np.float64)
            index =[slice(None)]*self.num_qubits
            index[pert_site] = 1
            W[tuple(index)] = 1.0
            for t in range(1, tf+1):
                even_sites = list(range(2, self.num_qubits - 2))
                odd_sites = list(range(1, self.num_qubits - 1))
                #Apply the local transfer matrices for this layer
                if t % 2 ==0:
                    for j in range(len(odd_sites)//2):
                        W = Apply_Brick(W, T_T, odd_sites[2*j], odd_sites[2*j+1])
                elif t % 2 ==1:
                    for j in range(len(even_sites)//2):
                        W = Apply_Brick(W, T_T, even_sites[2*j], even_sites[2*j+1])
                    W = Apply_Brick(W, T_lT, 0,1)
                    W = Apply_Brick(W, T_rT, self.num_qubits -2,self.num_qubits -1)
                #update the OTOC
                for j in range(self.num_qubits//2 -(t%2) ):
                    i1 = 2*j + (t % 2)
                    i2 = i1+1
                    index =[0]*self.num_qubits
                    index[i1] = 1
                    index[i2] = 1
                    C_val = C0*W[tuple(index)]
                    C[i1,t] = C_val
                    C[i2,t] = C_val
                #If no gate hit this site this step (the edges) the OTOC there didn't change
                if t % 2 == 1:
                    C[0,t] = C[0,t-1]
                    C[-1,t] = C[-1,t-1]
            return C
        return Adjoint_OTOC(self, pars[0], T, pert_site)

    def Front_Position(self, C:np.ndarray, pert_site:int, h:float):
        """
        Given the OTOC C(x,t) calculates the contour x_h(t) of the normalized OTOC C(x_h(t),t)/Max_{x_prime}[C(x_prime,t)] = h.
        """
        #calculate the normalized OTOC
        Z = np.max(C, axis=0)
        C_normalized = C / Z
        #distance of each point from h
        d = np.abs(C_normalized - h)
        #calculate x_h(t), define it to be to the right of the perturbation
        x = np.arange(self.num_qubits)
        right_indices = np.flatnonzero(x >= pert_site)
        x_h = right_indices[np.argmin(d[right_indices], axis=0)]
        return x_h


#Call Open_Brickwall_OTOC to do a Monte Carlo OTOC estimate
#define run parameters
num_qubits = 2000
bc_type = "periodic"
local_hs_dim = 2
corr_sites = True
circ = Open_Brickwall_OTOC(num_qubits, bc_type, local_hs_dim, corr_sites)
pert_site = num_qubits//2
num_samples = 10**(4)
init_parity = (num_qubits//2+1)%2
p=10**(-5)
pars = [p]
T = (num_qubits+1) // 2 
#calculate OTOC at p=0 and p!=0
C,x,t= circ.Monte_Carlo_OTOC(num_samples, T, pars, pert_site,init_parity)
C_unitary, x, t = circ.Monte_Carlo_OTOC(num_samples, T, [0.0], pert_site,init_parity)

plt.figure(figsize=(8, 5))
mesh = plt.pcolormesh(x, t, C.T, shading="auto", cmap="viridis")
plt.figure(figsize=(8, 5))
mesh = plt.pcolormesh(x, t, C_unitary.T, shading="auto", cmap="viridis")
plt.xlabel("x")
plt.ylabel("t")
plt.colorbar(mesh, label="C")
plt.show()

#define the analytic gaussian
q = local_hs_dim
kappa = (q*q-1)*(q-1)/(q*(q*q+1))
var = 1/(kappa*p)
gaussian = 2*np.exp(-t**2/var)

#plot OTOC as a function of t at a particular site
plt.figure()
probe_site = pert_site + 50
C_site = np.zeros(len(t))
C_site[:] = C[probe_site,:].copy()
plt.plot(t,C_site, label = "C")
plt.plot(t,gaussian, label = "Gaussian", color = "red", linestyle = "--")
plt.xlabel("t")
plt.ylabel("C")
plt.legend()
plt.show()

plt.figure()
C_site_unitary = np.zeros(len(t))
C_site_unitary[:] = C_unitary[probe_site,:].copy()
plt.plot(t,C_site_unitary, label = "C")
plt.xlabel("t")
plt.ylabel("C")
plt.legend()
plt.show()

#export the data
data = {
        "qudit_dimension": local_hs_dim,
        "num_qudits": num_qubits,
        "bc_type ": bc_type,
        "corr_sites": corr_sites,
        "p":  p,
        "num_samples":num_samples,
        "pert_site": pert_site,
        "probe_site":probe_site,
        "C": C.tolist(),
        "C_unitary":C_unitary.tolist(),
        "x": x.tolist(),
        "t": t.tolist(),
        "C_site":C_site.tolist(),
        "C_site_unitary": C_site_unitary.tolist(),
        "gaussian": gaussian.tolist(),
        
        }

file_path = Path.home() / "Desktop" / "open_otoc.json"
with file_path.open("w") as f:
    json.dump(data, f, indent=4)

print("Saved here:", file_path)
print("File exists:", file_path.exists())
