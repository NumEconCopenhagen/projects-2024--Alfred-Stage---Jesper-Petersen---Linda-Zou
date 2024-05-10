import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

class OpenEconomyNX:
    def __init__(self, alpha=0.3, r=0.02, delta=0.05, n=0.02, s=0.2, A=1, V0=1, L0=1,
                 exchange_rate=1.0, foreign_inflation_rate=0.02, domestic_inflation_rate=0.02, target_inflation_rate=0.02):
        self.base_params = {'alpha': alpha, 'r': r, 'delta': delta, 'n': n, 's': s, 'A': A, 'V0': V0, 'L0': L0}
        self.exchange_rate = exchange_rate
        self.foreign_inflation_rate = foreign_inflation_rate
        self.domestic_inflation_rate = domestic_inflation_rate
        self.target_inflation_rate = target_inflation_rate
        self.reset()
    
    def nx_trans_calc(self, v_range=(0, 20), num_points=100):
        num_points = int(num_points)  
        v_t = np.linspace(v_range[0], v_range[1], num_points)
        vt1 = (self.s * self.w / (1 + self.n)) + \
            ((1 - self.delta + self.s * self.r) / (1 + self.n)) * v_t
        return v_t, vt1
    

    def update_exchange_rate(self):
        inflation_differential = self.domestic_inflation_rate - self.foreign_inflation_rate
        self.exchange_rate *= np.exp(-inflation_differential)

    def update_interest_rates(self):
        interest_rate_adjustment = 0.5 * (self.domestic_inflation_rate - self.target_inflation_rate)
        self.r += interest_rate_adjustment

    def calculate_nx(self):
        base_nx = 0.05
        influence_factor = (self.exchange_rate * self.foreign_inflation_rate) / (self.r * self.domestic_inflation_rate)
        return base_nx * influence_factor
    
    def vstar(self): 
        self.K = ((self.alpha*self.A)/self.r)**(1/(1-self.alpha)) * self.L
        self.Y = self.A * self.K**self.alpha * self.L**(1 - self.alpha)+self.calculate_nx()
        self.w = (1 - self.alpha) * self.Y / self.L
        savings = self.s * self.w    
        discount_rate = 1 + self.n - (1 - self.delta + self.s * self.r)
        v_ss = savings / discount_rate
        return v_ss

    def nx_simulation_economy(self, periods, shock=None, new_A=None, new_s=None, new_r=None, new_foreign_inflation_rate=None, new_domestic_inflation_rate=None, new_n=None, new_alpha=None, new_delta=None):
        Lt, Kt, Yt, kt, yt, Vt, Yn, St, vt, NXt = np.zeros(periods), np.zeros(periods), np.zeros(periods), np.zeros(periods), np.zeros(periods), np.zeros(periods), np.zeros(periods), np.zeros(periods), np.zeros(periods), np.zeros(periods)
    
        Lt[0], Kt[0], Vt[0] = self.L, self.K, self.V
        Yt[0] = self.A * Kt[0]**self.alpha * Lt[0]**(1 - self.alpha)+self.calculate_nx()
        NXt[0] = self.calculate_nx()
        Yn[0] = Yt[0] + NXt[0] + self.r * (Vt[0] - Kt[0])
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
                if new_foreign_inflation_rate is not None:
                    self.foreign_inflation_rate = new_foreign_inflation_rate
                if new_domestic_inflation_rate is not None:
                    self.domestic_inflation_rate = new_domestic_inflation_rate
                if new_n is not None:
                    self.n = new_n
                if new_alpha is not None:
                    self.alpha = new_alpha
                if new_delta is not None:
                     self.delta = new_delta
                
            
            self.update_exchange_rate()
            self.update_interest_rates()
            
            Lt[t] = Lt[t - 1] * (1 + self.n)
            Kt[t] = self.k * Lt[t]
            Yt[t] = self.A * Kt[t]**self.alpha * Lt[t]**(1 - self.alpha)+self.calculate_nx()
            NXt[t] = self.calculate_nx()
            Yn[t] = Yt[t] + NXt[t] + self.r * (Vt[t] - Kt[t])  
            St[t] = self.s * Yn[t]
            vt[t] = Vt[t] / Lt[t]

        return Lt, Kt, Yt, kt, yt, vt
    
    def nx_underlying_variables(self, periods, shock, alpha=None, n=None, s=None, r=None, A=None, delta=None, 
                     foreign_inflation_rate=None, domestic_inflation_rate=None):
        if alpha is not None:
            self.alpha = alpha
        if n is not None:
            self.n = n
        if s is not None:
            self.s = s
        if r is not None:
            self.r = r
        if A is not None:
            self.A = A
        if delta is not None:
            self.delta = delta
        if foreign_inflation_rate is not None:
            self.foreign_inflation_rate = foreign_inflation_rate
        if domestic_inflation_rate is not None:
            self.domestic_inflation_rate = domestic_inflation_rate


        Lt, Kt, Yt, kt, yt, vt = self.nx_simulation_economy(periods, shock)

        plt.figure(figsize=(18, 12))
        titles = ['$L_t$', '$K_t$', '$Y_t$', '$k_t$', '$y_t$', '$v_t$']
        data = [Lt, Kt, Yt, kt, yt, vt]
        
        for i in range(6):
            plt.subplot(3, 2, i + 1)
            plt.plot(data[i], label=titles[i])
            plt.title(titles[i])
            plt.xlabel('Time (t)')
            plt.legend()
        plt.tight_layout()
        plt.show()
    
    def nx_phase_diagram(self, alpha=None, n=None, s=None, r=None, foreign_inflation_rate=None, domestic_inflation_rate=None, delta=None):

        if alpha is not None:
            self.alpha = alpha
        if n is not None:
            self.n = n
        if s is not None:
            self.s = s
        if r is not None:
            self.r = r
        if foreign_inflation_rate is not None:
            self.foreign_inflation_rate = foreign_inflation_rate
        if domestic_inflation_rate is not None:
            self.domestic_inflation_rate = domestic_inflation_rate
        if delta is not None:
            self.delta = delta

        self.reset()  
        v_t, vt1 = self.nx_trans_calc()
        v_ss = self.vstar()

        plt.figure(figsize=(10, 6))
        plt.plot(v_t, vt1, label='Dynamics $v_{t+1}$')
        plt.plot(v_t, v_t, 'r--', label='$v_t = v_{t+1}$')
        plt.scatter([v_ss], [v_ss], color='orange', zorder=5)
        plt.annotate(f'Steady State ($v^* = {v_ss:.2f}$)', (v_ss, v_ss), textcoords="offset points", xytext=(10, 10), ha='center')
        plt.title("Phase Diagram of $v_t$")
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

        self.Y = self.A * self.K**self.alpha * self.L**(1 - self.alpha)+self.calculate_nx()
        self.w = (1 - self.alpha) * self.Y / self.L
     
        self.update_exchange_rate()
        self.update_interest_rates()

    
    