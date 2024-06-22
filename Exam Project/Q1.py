import numpy as np
from scipy.optimize import minimize
from types import SimpleNamespace

class ProductionEconomy:
    def __init__(self, par):
        self.par = SimpleNamespace()
        self.A = par.A
        self.gamma = par.gamma
        self.alpha = par.alpha
        self.nu = par.nu
        self.epsilon = par.epsilon
#        self.p1 = par.p1  
#        self.p2 = par.p2  
        self.w = 1.0   # numeraire
        self.tau = par.tau
        self.T = par.T
        self.kappa = par.kappa

    def set_prices(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    
    def optimal_labor(self, pj):
        return (pj * self.A * self.gamma / self.w) ** (1 / (1 - self.gamma))
    
    def optimal_output(self, ell_j):
        return self.A * ell_j ** self.gamma
    
    def optimal_profits(self, pj):
        return (1 - self.gamma) / self.gamma * self.w * (pj * self.A * self.gamma / self.w) ** (1 / (1 - self.gamma))
    
    def optimal_consumption(self, ell):
        income = self.w * ell + self.T + self.optimal_profits(self.p1) + self.optimal_profits(self.p2)
        c1 = self.alpha * income / self.p1
        c2 = (1 - self.alpha) * income / (self.p2 + self.tau)
        return c1, c2
    
    def utility(self, ell):
        c1, c2 = self.optimal_consumption(ell)
        return np.log(c1 ** self.alpha * c2 ** (1 - self.alpha)) - self.nu * ell ** (1 + self.epsilon) / (1 + self.epsilon)
    
    def optimal_labor_supply(self):
        result = minimize(lambda ell: -self.utility(ell), x0=1, bounds=[(1e-8, None)])
        return result.x[0]
    
    def calculate_transfer(self, c2_star):
        self.T = self.tau * c2_star
    
    def market_clearing(self):
        ell_star = self.optimal_labor_supply()
        c1_star, c2_star = self.optimal_consumption(ell_star)
        self.calculate_transfer(c2_star)
        
        ell1_star = self.optimal_labor(self.p1)
        ell2_star = self.optimal_labor(self.p2)
        y1_star = self.optimal_output(ell1_star)
        y2_star = self.optimal_output(ell2_star)
        
        labor_diff = ell_star - (ell1_star + ell2_star)
        good1_diff = c1_star - y1_star
        good2_diff = c2_star - y2_star
        
        return labor_diff, good1_diff, good2_diff

    def check_market_clearing(economy, p1_range, p2_range, threshold=1e-6):
        clearing_solutions = []
        for p1 in p1_range:
            for p2 in p2_range:
                economy.set_prices(p1, p2)
                labor_diff, good1_diff, good2_diff = economy.market_clearing()
                
                # Check if all differences are within the threshold
                if (abs(labor_diff) < threshold and 
                    abs(good1_diff) < threshold and 
                    abs(good2_diff) < threshold):
                    clearing_solutions.append({
                        'p1': p1,
                        'p2': p2,
                        'labor_diff': labor_diff,
                        'good1_diff': good1_diff,
                        'good2_diff': good2_diff
                    })
        return clearing_solutions
