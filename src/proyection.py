import quadprog
import numpy as np

def proyect_into_linear(x, A, b):
    """
    This function projects the vector
    x into the set Ay \geq b

    """
    N = A.shape[1]
    G = np.eye(N).astype('double')
    x_ = x.astype('double')
    A_ = A.T.astype('double')
    b_= b.astype('double')

    sol = quadprog.solve_qp(G, x_, A_, b_, 0)
    return sol
     



if __name__ == '__main__':
    A1 = np.array([
        [1, 0],
        [0, 1]
    ])
    b1 = np.array([0, 0])
    x1 = np.array([3, -2])

    proyect_into_linear(x1, A1, b1)
