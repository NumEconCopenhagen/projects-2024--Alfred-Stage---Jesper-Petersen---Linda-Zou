import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class ProductionEcon:
    
    def __init__(self):
        # Define parameters
        par = self.par = SimpleNamespace()

        # firms
        par.A = 1.0
        par.gamma = 0.5

        # households
        par.alpha = 0.3
        par.nu = 1.0
        par.epsilon = 2.0

        # government
        par.tau = 0.0
        par.T = 0.0

        # Question 3
        par.kappa = 0.1

        par.w = 1

        par.p1 = np.linspace(0.1, 2.0, 10)
        par.p2 = np.linspace(0.1, 2.0, 10)
        
    def profit_condition(self, l):
        par = self.par
        return par.A * l**par.gamma

    def optimal_labor(self, p):
        par = self.par
        return (p * par.A / par.w)**(1 / (1 - par.gamma))
    
    def optimal_production(self, p):
        par = self.par
        l_star = self.optimal_labor(p)
        return self.profit_condition(l_star)
    
    def optimal_profit(self, p):
        par = self.par
        l_star = self.optimal_labor(p)
        return (1 - par.gamma) / par.gamma * par.w * l_star
    
    def consumer_utility(self, c1, c2, ell):
        par = self.par
        return np.log(c1**par.alpha * c2**(1 - par.alpha)) - par.nu * ((ell**(1 + par.epsilon)) / (1 + par.epsilon))
    
    def optimal_consumption(self, ell, p1, p2, T):
        par = self.par
        w_ell = par.w * ell
        pi1 = self.optimal_profit(p1)
        pi2 = self.optimal_profit(p2)
        c1 = (par.alpha * (w_ell + T + pi1 + pi2)) / p1
        c2 = ((1 - par.alpha) * (w_ell + T + pi1 + pi2)) / (p2 + par.tau)
        return c1, c2
    
    def optimal_labor_supply(self, p1, p2):
        def negative_utility(ell):
            c1, c2 = self.optimal_consumption(ell, p1, p2, self.par.T)
            return -self.consumer_utility(c1, c2, ell)
        
        result = minimize(negative_utility, x0=1.0, bounds=[(0, None)])
        return result.x[0]
    
    def check_market_clearing(self):
        results = []
        for p1 in self.par.p1:
            for p2 in self.par.p2:
                ell_star = self.optimal_labor_supply(p1, p2)
                c1_star, c2_star = self.optimal_consumption(ell_star, p1, p2, self.par.T)
                y1_star = self.optimal_production(p1)
                y2_star = self.optimal_production(p2)

                labor_market_clear = np.isclose(ell_star, self.optimal_labor(p1) + self.optimal_labor(p2), atol=1e-6)
                goods_market_1_clear = np.isclose(c1_star, y1_star, atol=1e-6)
                goods_market_2_clear = np.isclose(c2_star, y2_star, atol=1e-6)

                if labor_market_clear and goods_market_1_clear and goods_market_2_clear:
                    results.append({
                        'p1': p1,
                        'p2': p2,
                        'ell_star': ell_star,
                        'c1_star': c1_star,
                        'c2_star': c2_star,
                        'y1_star': y1_star,
                        'y2_star': y2_star
                    })
        return results

