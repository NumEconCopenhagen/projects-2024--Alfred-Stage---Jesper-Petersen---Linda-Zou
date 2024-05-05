import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

class SmallOpenEconomyModel:
    def __init__(self, alpha=0.3, r=0.02, delta=0.05, n=0.02, s=0.2, A=1, V0=1, L0=1):
        self.base_params = {'alpha': alpha, 'r': r, 'delta': delta, 'n': n, 's': s, 'A': A, 'V0': V0, 'L0': L0}
        self.reset()

    def calculate_phase_dynamics(self, wealth_range=(0, 20), num_points=100):
        v_t = np.linspace(*wealth_range, num_points)
        v_t1 = (self.s * self.w / (1 + self.n)) + \
               ((1 - self.delta + self.s * self.r) / (1 + self.n)) * v_t
        return v_t, v_t1

    def steady_state_wealth_per_worker(self):
        # Debugging: print parameter values
        print(f"Calculating steady-state with A={self.A:.2f}, s={self.s:.2f}, r={self.r:.2f}, delta={self.delta:.2f}, n={self.n:.2f}, alpha={self.alpha:.2f}")
        
        # Calculate intermediate values in a readable format
        self.K = (self.r / (self.alpha * self.A)) ** (1 / (self.alpha - 1)) * self.L
        print(f"Calculated capital K: {self.K:.2f}")
    
        self.Y = self.A * self.K**self.alpha * self.L**(1 - self.alpha)
        print(f"Calculated output Y: {self.Y:.2f}")
    
        self.w = (1 - self.alpha) * self.Y / self.L
        print(f"Calculated wage rate w: {self.w:.2f}")
    
        numerator = self.s * self.w    
        denominator = 1 + self.n - (1 - self.delta + self.s * self.r)
    
        v_ss = numerator / denominator
        print(f"Calculated steady-state wealth per worker: v_ss = \\frac{{s \\cdot w}}{{1 + n - (1 - delta + s \\cdot r)}} = \\frac{{{numerator:.2f}}}{{{denominator:.2f}}} = {v_ss:.2f}")
    
        return v_ss



    def simulate(self, periods, t_shock=None, new_A=None, new_s=None, new_r=None):
        Lt = np.zeros(periods)
        Kt = np.zeros(periods)
        Yt = np.zeros(periods)
        kt = np.zeros(periods)
        yt = np.zeros(periods)
        Vt = np.zeros(periods)
        Yn = np.zeros(periods)
        St = np.zeros(periods)
        vt = np.zeros(periods)
    
        Lt[0], Kt[0], Vt[0] = self.L, self.K, self.V
        Yt[0] = self.A * Kt[0]**self.alpha * Lt[0]**(1 - self.alpha)
        kt[0] = Kt[0] / Lt[0]
        yt[0] = Yt[0] / Lt[0]
        Yn[0] = Yt[0] + self.r * (Vt[0] - Kt[0])
        St[0] = self.s * Yn[0]
        vt[0] = Vt[0] / Lt[0]
    
        for t in range(1, periods):
            if t == t_shock:
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
            Vt[t] = St[t - 1] + (1 - self.delta) * Vt[t - 1]  # Corrected formula for Vt
            Yn[t] = Yt[t] + self.r * (Vt[t] - Kt[t])  # Corrected formula for Yn
            St[t] = self.s * Yn[t]
            kt[t] = Kt[t] / Lt[t]
            yt[t] = Yt[t] / Lt[t]
            vt[t] = Vt[t] / Lt[t]

        return Lt, Kt, Yt, kt, yt, vt

    def plot_results(self, Lt, Kt, Yt, kt, yt, vt):
        plt.figure(figsize=(18, 12))
        titles = ['Labor L(t)', 'Capital K(t)', 'Output Y(t)', 'Capital/Worker k(t)', 'Output/Worker y(t)', 'Wealth/Worker v(t)']
        data = [Lt, Kt, Yt, kt, yt, vt]
        
        for i in range(6):
            plt.subplot(3, 3, i + 1)
            plt.plot(data[i], label=titles[i])
            plt.title(titles[i])
            plt.xlabel('Time (t)')
            plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_phase_diagram(self, wealth_range=(0, 20), num_points=100):
        v_t, v_t1 = self.calculate_phase_dynamics(wealth_range, num_points)
        v_ss = self.steady_state_wealth_per_worker()
    
        plt.figure(figsize=(10, 6))
        plt.plot(v_t, v_t1, label='Dynamics $v_{t+1}$')
        plt.plot(v_t, v_t, 'r--', label='45-degree line where $v_t = v_{t+1}$')
        plt.scatter([v_ss], [v_ss], color='blue', zorder=5)  # Mark the steady state
        plt.annotate(f'Steady State ($v^* = {v_ss:.2f}$)', (v_ss, v_ss), textcoords="offset points", xytext=(10, 10), ha='center', color='blue')
        plt.title("Phase Diagram of Wealth per Worker Dynamics")
        plt.xlabel("$v_t$ (Wealth per worker at time t)")
        plt.ylabel("$v_{t+1}$ (Wealth per worker at time t+1)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def print_sim(self, periods, t_shock=None, new_A=None, new_s=None, new_r=None):
        Lt, Kt, Yt, kt, yt, vt = self.simulate(periods, t_shock=t_shock, new_A=new_A, new_s=new_s, new_r=new_r)
        print("t\tLt\t\tKt\t\tYt\t\tkt\t\tyt\t\tvt")
        for t in range(periods):
            print(f"{t}\t{Lt[t]:.2f}\t\t{Kt[t]:.2f}\t\t{Yt[t]:.2f}\t\t{kt[t]:.2f}\t\t{yt[t]:.2f}\t\t{vt[t]:.2f}")

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

    def time_to_steady_state(self, periods, threshold=0.01, t_shock=None, new_A=None, new_s=None, new_r=None):
        Lt, Kt, Yt, kt, yt, vt = self.simulate(periods, t_shock=t_shock, new_A=new_A, new_s=new_s, new_r=new_r)
        v_ss = self.steady_state_wealth_per_worker()
    
        for t in range(t_shock, periods):
            v_diff = abs(vt[t] - v_ss)
            
            if v_diff < threshold:
                print(f"Time to reach steady state: {t - t_shock} periods after shock")
                break
        else:
            print("Steady state not reached within given periods after shock")
    
        # Plot Wealth/Worker dynamics with steady-state annotation
        plt.figure(figsize=(10, 6))
        plt.plot(vt, label="Wealth/Worker v(t)")
        plt.axhline(y=v_ss, color='r', linestyle='--', label=f'Steady State ($v^* = {v_ss:.2f}$)')
        plt.annotate(f'Steady State ($v^* = {v_ss:.2f}$)', (t, v_ss), textcoords="offset points", xytext=(10, 10), ha='center', color='red')
        plt.xlabel("Time (t)")
        plt.ylabel("Wealth/Worker v(t)")
        plt.legend()
        plt.grid(True)
        plt.show()
    
        return t - t_shock if v_diff < threshold else periods

