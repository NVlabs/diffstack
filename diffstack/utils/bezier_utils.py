import numpy as np
import math

from scipy.special import comb, gamma

def bezier_base(n,i,s):
    b=math.factorial(n)/math.factorial(i)/math.factorial(n-i)*s**i*(1-s)**(n-i)
    return b


def bezier_integral(alpha):
    n=len(alpha)-1
    m=0
    for i in range(n+1):
        m = m + alpha(1+i) * comb(n,i)*gamma(1+i)*gamma(n+1-i)/gamma(n+2)
    return m

def symbolic_dbezier(n):
# alpha_df=A*alpha_f
    A = np.zeros([n,n+1])
    for i in range(0,n):
        A[i,i]=-n
        A[i,i+1]=n
    return A

class bezier_regressor(object):
    def __init__(self, N=6, Nmin=8, Nmax=200,Gamma=1e-2):
        self.N = N
        self.dB = symbolic_dbezier(N)
        self.ddB = symbolic_dbezier(N-1)
        self.Nmin = Nmin
        self.Nmax = Nmax
        self.Bez_matr = dict()
        self.bez_reg = dict()
        self.Bez_matr_der = dict()
        self.Bez_matr_dder = dict()
        for i in range(Nmin,Nmax+1):
            t=np.linspace(0,1,i)
            self.Bez_matr[i]=np.zeros([i,self.N+1])
            self.Bez_matr_der[i]=np.zeros([i,self.N])
            self.Bez_matr_dder[i]=np.zeros([i,self.N-1])
            for j in range(i):
                for k in range(self.N+1):
                    self.Bez_matr[i][j,k]=comb(self.N,k)*(1-t[j])**(self.N-k)*t[j]**k
                    if k<self.N:
                        self.Bez_matr_der[i][j,k]=comb(self.N-1,k)*(1-t[j])**(self.N-1-k)*t[j]**k
                    if k<self.N-1:
                        self.Bez_matr_dder[i][j,k]=comb(self.N-2,k)*(1-t[j])**(self.N-2-k)*t[j]**k

            self.bez_reg[i]=np.linalg.solve((self.Bez_matr[i].T@self.Bez_matr[i])+Gamma*np.eye(self.N+1),self.Bez_matr[i].T)



    def bezier_regression(self,t,x,n=9):
    # use bezier polynomial to fit x locally, and generate filtered x and dxdt.
    # x: input signal
    # n: regression window size
        L = x.shape[0]
        Ts = (t[-1]-t[0])/(L-1)
        x_bar = np.zeros_like(x)
        dx_bar = np.zeros_like(x)
        for i in range(L):
            idx_low  = max(0,i-n)
            idx_high = min(L-1,i+n)
            idx = np.arange(idx_low,idx_high+1)   # regression window
            M = idx_high-idx_low+1                # number of data points in the window
            T=(M-1)*Ts
            if len(idx)< self.N+2:
                raise ValueError('Not enough data points for regression')

            reg=self.bez_reg[M]
            
            alpha=(reg@x[idx])                # get the bezier coefficient with pre-computed regressor
            
            dalpha=self.dB@alpha                 # get the bezier coefficient for the derivative
            x_temp = self.Bez_matr[M]@alpha
            dx_temp = self.Bez_matr_der[M]@dalpha/T
            x_bar[i]=x_temp[i-idx_low]
            dx_bar[i]=dx_temp[i-idx_low]

        return x_bar, dx_bar


def test():
    t = np.linspace(0,1,100)
    x = np.sin(2*np.pi*t)[:,None]+np.random.randn(100,2)*0.05
    x[50:]+=1
    regressor = bezier_regressor(Gamma=0.2)
    x_bar, dx_bar = regressor.bezier_regression(t,x,n=20)
    import matplotlib.pyplot as plt
    plt.plot(t,x,'b')
    plt.plot(t,x_bar,'r')
    plt.plot(t,dx_bar,'g')
    plt.show()
    
if __name__ == '__main__':
    test()