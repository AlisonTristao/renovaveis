import control as ctrl
import numpy as np

class Plant:
    def __init__(self, transfer_function, q_transfer_function=None, saturation=None):
        self.system = ctrl.tf(transfer_function)
        q_system = ctrl.tf(q_transfer_function) if q_transfer_function is not None else self.system
        self.__Ts = self.system.dt  # sampling time

        # get polynomial coefficients
        alpha = self.system.den[0][0].tolist()[1:]  # exclude leading 1
        alpha = [-a for a in alpha]  # change sign for difference equation
        beta = self.system.num[0][0].tolist()
        gama = q_system.num[0][0].tolist()

        if saturation is None:
            saturation = float('inf')  # no saturation by default
    
        self.__set_params(alpha, beta, gama, saturation)

    def __set_params(self, alpha, beta, gama, saturation):
        # store coefficients in reverse order
        self.__alpha = [alpha[len(alpha) - 1 - i] for i in range(len(alpha))]
        self.__beta = [beta[len(beta) - 1 - i] for i in range(len(beta))]
        self.__gama = [gama[len(gama) - 1 - i] for i in range(len(gama))]
        # initialize past values
        self.__y = [0 for _ in range(len(alpha))]
        self.__u = [0 for _ in range(len(beta))]
        self.__q = [0 for _ in range(len(gama))]
        # saturation limit
        self.__saturation = saturation

        # debug print
        #print(f"Plant initialized with alpha: {self.__alpha}, beta: {self.__beta}, gama: {self.__gama}, saturation: {self.__saturation}")
    def y_past(self):
        return np.array(self.__y.copy()[::-1])
    
    def u_past(self):
        # the first element is u(k), not the past value - return values inversed without the first element
        return np.array((self.__u.copy()[1:])[::-1])
    
    def q_past(self):
        return np.array((self.__q.copy()[1:])[::-1])

    def step(self, u, q=0):
        # store u and q
        self.__u.append(u)
        self.__u = self.__u[1:]
        self.__q.append(q)
        self.__q = self.__q[1:]

        # calculate y response
        y = 0
        y += sum(self.__y[i] * self.__alpha[i] for i in range(len(self.__y)))
        y += sum(self.__u[i]/self.__Ts * self.__beta[i] for i in range(len(self.__u)))
        y += sum(self.__q[i]/self.__Ts * self.__gama[i] for i in range(len(self.__q)))

        # saturate y
        y = min(y, self.__saturation)
        y = max(y, -self.__saturation)

        self.__y.append(y)
        self.__y = self.__y[1:]

        # return latest output
        return self.__y[-1]
        