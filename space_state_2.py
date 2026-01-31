import numpy as np
import control as ct
from scipy.linalg import block_diag


def ss_siso(Gu: ct.TransferFunction):
    if Gu.dt is None:
        raise ValueError("As TFs precisam ser discretas (dt definido).")

    dt = float(Gu.dt)
    num_u = np.array(Gu.num[0][0], dtype=float)
    den_u = np.array(Gu.den[0][0], dtype=float)

    a = den_u[:]            # [1, a1, ..., an]
    b = num_u[:]            # [b0, b1, ..., bm]
    nx = len(a) + len(b) - 2  # estados y + estados u

    A = np.zeros((nx, nx))
    B = np.zeros((nx, 1))
    C = np.zeros((1, nx))
    D = np.zeros((1, 1))

    # y(k+1)
    A[0, :len(a)-1] = -a[1:]   # -a1..-an
    A[0, len(a)-1:] = b[1:]    # b1..bn (se existir)
    B[0, 0] = b[0]             # u(k)

    # shift y
    for i in range(1, len(a)-1):
        A[i, i-1] = 1.0

    # shift u
    for i in range(len(a)-1, nx-1):
        A[i+1, i] = 1.0

    # adiciona u(k) nos espaços x (como você tinha)
    if len(b) > 1:
        idx = len(b)  # mantém sua lógica original
        if 0 <= idx < nx:
            B[idx, 0] = 1.0

    # saída: y[k] = x1
    C[0, 0] = 1.0

    return ct.ss(A, B, C, D, dt)


def ss_mimo(ss_siso_arr):
    A_blocks = [np.array(ss.A) for ss in ss_siso_arr]
    A = block_diag(*A_blocks)

    B_blocks = [np.array(ss.B) for ss in ss_siso_arr]
    B = np.vstack(B_blocks)

    C_blocks = [np.array(ss.C) for ss in ss_siso_arr]
    C = block_diag(*C_blocks)

    D = np.vstack([np.asarray(ss.D) for ss in ss_siso_arr])

    dt = ss_siso_arr[0].dt
    return ct.ss(A, B, C, D, dt)


def get_F_G(ss, N, N_u):
    """
    Predição (EMPILHAMENTO POR SAÍDA):
      Y = F x[k] + G U

    onde:
      Y = [ y1(k+1)..y1(k+N), y2(k+1)..y2(k+N), ..., y_ny(k+1)..y_ny(k+N) ]^T
      U = [ u(k); u(k+1); ...; u(k+N_u-1) ]
    """
    A = np.asarray(ss.A, dtype=float)
    B = np.asarray(ss.B, dtype=float)   # (nx, 1) para MISO (1 entrada)
    C = np.asarray(ss.C, dtype=float)   # (ny, nx)

    nx = A.shape[0]
    ny = C.shape[0]

    # agora as linhas são "blocos por saída": (ny*N, nx) e (ny*N, N_u)
    F = np.zeros((ny * N, nx))
    G = np.zeros((ny * N, N_u))

    # helper: índice de linha para (output o, passo i)
    # o = 0..ny-1, i = 0..N-1 (i=0 -> k+1)
    def row(o, i):
        return o * N + i

    for i in range(N):
        # y[k+i+1] = C A^(i+1) x[k] + ...
        A_ip1 = np.linalg.matrix_power(A, i + 1)
        CA_ip1 = C @ A_ip1  # (ny, nx)

        # Preenche F por saída (cada saída vira um bloco contínuo)
        for o in range(ny):
            r = row(o, i)
            F[r, :] = CA_ip1[o, :]

        # termos de entrada
        for j in range(min(i + 1, N_u)):
            # contribuição de u[k+j] em y[k+i+1]: C A^(i-j) B
            A_ij = np.linalg.matrix_power(A, i - j)
            CAB = (C @ A_ij @ B).reshape(ny,)  # (ny,)

            for o in range(ny):
                r = row(o, i)
                G[r, j] = CAB[o]

    return F, G


def separete_prediction(Y_pred, ny, N):
    """
    Separa Y empilhado (por saída) em uma matriz (ny, N):
      out[o, i] = y_o(k+i+1)

    Ou seja:
      out[0,:] = y1(k+1..k+N)
      out[1,:] = y2(k+1..k+N)
      ...
    """
    Y_pred = np.asarray(Y_pred).reshape(-1)
    if Y_pred.size != ny * N:
        raise ValueError(f"Y_pred tem tamanho {Y_pred.size}, esperado {ny*N}.")

    y_mat = np.zeros((ny, N))
    for o in range(ny):
        start = o * N
        end = (o + 1) * N
        y_mat[o, :] = Y_pred[start:end]

    return y_mat
