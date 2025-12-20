import numpy as np
import control as ct
from scipy.linalg import block_diag

def ss_siso(Gu: ct.TransferFunction):
    if Gu.dt is None:
        raise ValueError("As TFs precisam ser discretas (dt definido).")

    # pega num/den (SISO)
    dt = float(Gu.dt)
    num_u = np.array(Gu.num[0][0], dtype=float)#.ravel()
    den_u = np.array(Gu.den[0][0], dtype=float)#.ravel()

    a = den_u[:]            # [1, a1, ..., an]
    b = num_u[:]            # [b0, b1, ..., bn]
    nx = len(a) + len(b) -2 # estados y + estados u

    # matrizes do sistema
    A = np.zeros((nx, nx))
    B = np.zeros((nx, 1))
    C = np.zeros((1, nx))
    D = np.zeros((1, 1))

    # y(k+1)
    A[0, :len(a)-1] = -a[1:] # -a1..-an
    A[0, len(a)-1:] = b[1:]  # b1..bn
    B[0] = b[0]              # u(k)

    # shift y
    for i in range(1, len(a)-1):
        A[i, i-1] = 1.0

    # shift u
    for i in range(len(a)-1, nx-1):
        A[i+1, i] = 1.0

    # adiciona u(k) nos espaços x
    if len(b) > 1:
        B[len(b)] = 1.0

    # saída: y[k] = x1
    C[0, 0] = 1.0

    return ct.ss(A, B, C, D, dt)

def ss_mimo(ss_siso_arr):
    # A = diag(A1, A2, ..., An)
    A_blocks = [np.array(ss.A) for ss in ss_siso_arr]
    A = block_diag(*A_blocks)

    # B = [B1; B2; ...; Bn]
    B_blocks = [np.array(ss.B) for ss in ss_siso_arr]
    B = np.vstack(B_blocks)

    # C = diag(C1, C2, ..., Cn)
    C_blocks = [np.array(ss.C) for ss in ss_siso_arr]
    C = block_diag(*C_blocks)

    # D = [D1; D2; ...; Dn]   (mesma entrada u)
    D = np.vstack([np.asarray(ss.D) for ss in ss_siso_arr])

    dt = ss_siso_arr[0].dt
    return ct.ss(A, B, C, D, dt)

import numpy as np

def get_F_G(ss, N, N_u):
    """
    Predição:
      Y = F x[k] + G U

    onde:
      Y = [y[k+1]; y[k+2]; ...; y[k+N]]  (empilhado, cada y tem ny linhas)
      U = [u[k]; u[k+1]; ...; u[k+N_u-1]]
    """
    A = np.asarray(ss.A, dtype=float)
    B = np.asarray(ss.B, dtype=float)   # (nx, 1) para MISO
    C = np.asarray(ss.C, dtype=float)

    nx = A.shape[0]
    ny = C.shape[0]

    F = np.zeros((N * ny, nx))
    G = np.zeros((N * ny, N_u))

    for i in range(N):
        # y[k+i+1] = C A^(i+1) x[k] + ...
        A_ip1 = np.linalg.matrix_power(A, i + 1)
        F[i*ny:(i+1)*ny, :] = C @ A_ip1

        # termos de entrada: somatório em j = 0..min(i, N_u-1)
        for j in range(min(i + 1, N_u)):
            # contribuição de u[k+j] em y[k+i+1] é: C A^(i-j) B
            A_ij = np.linalg.matrix_power(A, i - j)
            G[i*ny:(i+1)*ny, j] = (C @ A_ij @ B).reshape(ny,)

    return F, G

def separete_prediction(Y_pred, ny, N):
    """
    Separa a predição Y empilhada em uma lista de vetores y[k+i]
    """
    y_list = np.zeros((N, ny))
    for i in range(N):
        y_list[i, :] = Y_pred[i*ny:(i+1)*ny].ravel()

    return y_list