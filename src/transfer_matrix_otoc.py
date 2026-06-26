import numpy as np


class TransferMatrix:
    def __init__(
        self,
        probe_site: int,
        pert_site: int,
        num_qubits: int,
        bc_type: str = "periodic",
    ):
        """
        Attributes
        ----------
        num_qubits: (int) [Assumed to be even]
        pert_site: (int)
        probe_site: (int)
        W: (numpy array)
        bc_type: (str) Options are "periodic" or "open"

        Methods
        -------
        Layer_Update(W_in,l,p)
        """
        assert num_qubits % 2 == 0, "# Qubits must be even"
        self.num_qubits = num_qubits
        self.pert_site = pert_site
        self.probe_site = probe_site
        self.W = np.zeros((2,) * self.num_qubits)
        self.bc_type = bc_type

    def update_layer(self, W_in: np.ndarray, t: int, p: float):
        """
        Updates the weights for one layer.

        Inputs
        ------
            W_in: (numpy array)
            Input weights tensor. It is in tensor form with N indices each taking on 2 values.

            t: (int)
            Layer number. It is only used to determine the parity of the layer.

            p: (float)
            Measurement strength (p in [0,1]).

        Outputs
        -------
            W (numpy array)
            Weights tensor after the layer update. It is in tensor form with N indices each taking on 2 values.
        """
        # Calculate the transfer matrix coefficients
        a = 1 / 30 * (6 + 2 * p - p**2)
        b = 4 / 15 * (3 - 4 * p + 2 * p**2)
        c = 1 / 60 * p * (8 - 8 * p + 4 * p**2 - p**3)
        d = 1 / 15 * (3 + 8 * (p - 1) ** 2 + 4 * (p - 1) ** 4)
        W = W_in.copy()
        # qubit number
        N = W.ndim
        # Sites for two-qubit gates to be applied to
        # Sites for two-qubit gates to be applied to
        if self.bc_type == "open":
            if t % 2 == 0:
                sites = [(i, i + 1) for i in range(0, N - 1, 2)]
            else:
                sites = [(i, i + 1) for i in range(1, N - 1, 2)]

        elif self.bc_type == "periodic":
            if N % 2 != 0:
                raise ValueError("Periodic brickwall update currently requires even N.")

            if t % 2 == 0:
                sites = [(i, i + 1) for i in range(0, N - 1, 2)]
            else:
                sites = [(i, (i + 1) % N) for i in range(1, N, 2)]

        else:
            raise ValueError("self.bc must be either 'open' or 'periodic'.")
        # apply all gates (i.e. update W)
        for gate in sites:
            i, j = gate
            # Define the local update matrix that does the following
            # 00->00, 10 or 01 -> a(00)+ b(11), 11 ->c (00) +d (11)
            T = np.array(
                [
                    [1, a, a, c],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, b, b, d],
                ]
            )
            # Next apply the local update matrix, first move the i,j indices to the front
            axes = [i, j] + [k for k in range(N) if k not in (i, j)]
            W_perm = np.transpose(W, axes)
            # reshape 1st two axes become size 4 (so we can apply the 4x4 T matrix)
            W_temp1 = W_perm.reshape(4, -1)
            # apply update matrix
            W_temp2 = T @ W_temp1
            # reshape back
            # reshape back
            W = W_temp2.reshape([2, 2] + list(W_perm.shape[2:]))
            # invert permutation
            inv_axes = np.argsort(axes)
            W = np.transpose(W, inv_axes)
        return W

    def get_OTOC_from_weights(self, W, probe_site):
        return W.sum(axis=tuple(i for i in range(W.ndim) if i != probe_site))[1]

    def get_OTOC(self, p, tf, pert_site, probe_site):
        # get variables
        N = self.num_qubits
        W = self.W.copy()
        # List for the OTOC Values
        F = []
        # Apply the 1st perturbation. Assume this first step is an even layer.
        if pert_site % 2 == 0:
            i = pert_site
            j = pert_site + 1
        else:
            i = pert_site - 1
            j = pert_site
        indx = [0] * N
        indx[i] = 1
        indx[j] = 1
        # this particular 32/15 weight is for a pauli perturbation (no 1/2 out front convention)
        W[tuple(indx)] += 32 / 15
        F_val = self.OTOC_From_Weights(W, probe_site)
        F.append(F_val)
        for t in range(1, tf):
            # Update the weights
            W = self.Layer_Update(W, t, p)
            # Calculate the OTOC
            F_val = self.OTOC_From_Weights(W, probe_site)
            F.append(F_val)
        return np.arange(0, tf), np.array(F)
