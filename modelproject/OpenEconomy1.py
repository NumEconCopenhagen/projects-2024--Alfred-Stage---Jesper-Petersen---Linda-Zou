import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from ipywidgets import interact, FloatSlider


class OpenEconomy:
    def __init__(self, alpha=0.3, r=0.02, delta=0.05, n=0.02, s=0.2, A=1, V0=1, L0=1):
        self.base_params = {'alpha': alpha, 'r': r, 'delta': delta, 'n': n, 's': s, 'A': A, 'V0': V0, 'L0': L0}
        self.reset()

    def transition_calculations(self, v_range=(0, 20), num_points=100): 
        num_points = int(num_points)  
        v_t = np.linspace(v_range[0], v_range[1], num_points)
        vt1 = (self.s * self.w / (1 + self.n)) + \
            ((1 - self.delta + self.s * self.r) / (1 + self.n)) * v_t
        return v_t, vt1


    def vstar(self):
        self.K = ((self.alpha*self.A)/self.r)**(1/(1-self.alpha)) * self.L
        self.Y = self.A * self.K**self.alpha * self.L**(1 - self.alpha)
        self.w = (1 - self.alpha) * self.Y / self.L
        savings = self.s * self.w    
        discount_rate = 1 + self.n - (1 - self.delta + self.s * self.r)
        v_ss = savings / discount_rate
        return v_ss

    def simulation_economy(self, periods, shock=None, new_A=None, new_s=None, new_r=None):
        Lt, Kt, Yt, kt, yt, Vt, Yn, St, vt = np.zeros(periods), np.zeros(periods), np.zeros(periods), np.zeros(periods), np.zeros(periods), np.zeros(periods), np.zeros(periods), np.zeros(periods), np.zeros(periods)
    
        Lt[0], Kt[0], Vt[0] = self.L, self.K, self.V
        Yt[0] = self.A * Kt[0]**self.alpha * Lt[0]**(1 - self.alpha)
        kt[0] = Kt[0] / Lt[0]
        yt[0] = Yt[0] / Lt[0]
        Yn[0] = Yt[0] + self.r * (Vt[0] - Kt[0])
        St[0] = self.s * Yn[0]
        vt[0] = Vt[0] / Lt[0]
    
        for t in range(1, periods):
            if t == shock:
                if new_A is not None:
                    self.A = new_A
                if new_s is not None:
                    self.s = new_s
                if new_r is not None:
                    self.r = new_r
    
                self.k = (self.r / (self.alpha * self.A)) ** (1 / (self.alpha - 1))
    
            Lt[t] = Lt[t - 1] * (1 + self.n)
            Kt[t] = self.k * Lt[t]
            Yt[t] = self.A * Kt[t]**self.alpha * Lt[t]**(1 - self.alpha)
            Vt[t] = St[t - 1] + (1 - self.delta) * Vt[t - 1]  
            Yn[t] = Yt[t] + self.r * (Vt[t] - Kt[t])  
            St[t] = self.s * Yn[t]

            # Per Capita
            kt[t] = Kt[t] / Lt[t]
            yt[t] = Yt[t] / Lt[t]
            vt[t] = Vt[t] / Lt[t]

        return Lt, Kt, Yt, kt, yt, vt

    def underlying_variables(self, periods=100, shock=None, s=None, n=None, alpha=None, A=None, delta=None): 
        if s is not None:
            self.s = s
        if n is not None:
            self.n = n
        if alpha is not None:
            self.alpha = alpha
        if A is not None:
            self.A = A
        if delta is not None:
            self.delta = delta

        Lt, Kt, Yt, kt, yt, vt = self.simulation_economy(periods, shock=shock)

        plt.figure(figsize=(18, 12))
        titles = ['$L_t$', '$K_t$', '$Y_t$', '$k_t$', '$y_t$', '$v_t$']
        data = [Lt, Kt, Yt, kt, yt, vt]
        
        for i in range(6):
            plt.subplot(3, 2, i + 1)
            plt.plot(data[i], label=titles[i])
            plt.title(titles[i])
            plt.xlabel('t')
            plt.legend()
        plt.tight_layout()
        plt.show()

    def phase_diagram(self, s=None, n=None, alpha=None, A=None, delta=None): 
        if s is not None:
            self.s = s
        if n is not None:
            self.n = n
        if alpha is not None:
            self.alpha = alpha
        if A is not None:
            self.A = A
        if delta is not None:
            self.delta = delta
        
        v_t, vt1 = self.transition_calculations((0, 20), 100) 
        v_ss = self.vstar()
        
        plt.figure(figsize=(10, 6))
        plt.plot(v_t, vt1, label='$v_{t+1}$')
        plt.plot(v_t, v_t, 'r--', label='$v_t = v_{t+1}$')
        plt.scatter([v_ss], [v_ss], color='orange', zorder=5)
        plt.annotate(f'Steady State ($v^* = {v_ss:.2f}$)', (v_ss, v_ss), textcoords="offset points", xytext=(10, 10), ha='center', c='blue')
        plt.title("Phase Diagram: $v_t$")
        plt.xlabel("$v_t$")
        plt.ylabel("$v_{t+1}$")
        plt.legend()
        plt.grid(True)
        plt.show()

  
    def reset(self): 
        self.alpha = self.base_params['alpha']
        self.r = self.base_params['r']
        self.delta = self.base_params['delta']
        self.n = self.base_params['n']
        self.s = self.base_params['s']
        self.A = self.base_params['A']
        self.L = self.base_params['L0']
        self.V = self.base_params['V0']

        self.k = (self.r / (self.alpha * self.A)) ** (1 / (self.alpha - 1))
        self.K = self.k * self.L
        self.F = self.V - self.K

        self.Y = self.A * self.K**self.alpha * self.L**(1 - self.alpha)
        self.w = (1 - self.alpha) * self.Y / self.L

    def convergence_ss(self, periods=100, shock=None, s=None, n=None, alpha=None, A=None, delta=None):
        if s is not None:
            self.s = s
        if n is not None:
            self.n = n
        if alpha is not None:
            self.alpha = alpha
        if A is not None:
            self.A = A
        if delta is not None:
            self.delta = delta

        Lt, Kt, Yt, kt, yt, vt = self.simulation_economy(periods, shock=shock)
        v_ss = self.vstar()
        threshold = 0.01

        time_ss = None
        for t in range(periods):
            v_diff = abs(vt[t] - v_ss)
            if v_diff < threshold:
                time_ss = t
                break

        halfway_value = (vt[0] + v_ss) / 2
        halfway_time = None
        for t in range(periods):
            if abs(vt[t] - halfway_value) < threshold:
                halfway_time = t
                break

        plt.figure(figsize=(10, 6))
        plt.plot(vt, label="$v_t$")
        plt.axhline(y=v_ss, c='black', linestyle='--', label=f'($v^* = {v_ss:.2f}$)')
        plt.axhline(y=halfway_value, c='g', linestyle='--', label='Halfway to Steady State')
        plt.xlabel("t")
        plt.ylabel("$v_t$")
        plt.title("Convergence to Steady State")
        plt.legend()
        plt.grid(True)
        plt.show()

        return time_ss, halfway_time

