import numpy as np
from scipy.optimize import minimize, minimize_scalar, root
from types import SimpleNamespace
import matplotlib.pyplot as plt

class ProductionEconomy:
    def __init__(self, par):
        """
        Initialize the production economy with given parameters.

        Args:
            par (SimpleNamespace): A namespace containing the parameters A, gamma, alpha, nu, epsilon, tau, T, kappa.
        """
        self.par = SimpleNamespace()
        self.par.A = par.A
        self.par.gamma = par.gamma
        self.par.alpha = par.alpha
        self.par.nu = par.nu
        self.par.epsilon = par.epsilon
        self.par.tau = par.tau
        self.par.T = par.T
        self.par.kappa = par.kappa
        self.w = 1.0   # numeraire

    def set_prices(self, p1, p2):
        """
        Set the prices for the two goods.

        Args:
            p1 (float): Price of the first good.
            p2 (float): Price of the second good.
        """
        self.p1 = p1
        self.p2 = p2
    
    def optimal_labor(self, pj):
        """
        Calculate the optimal labor given the price of a good.

        Args:
            pj (float): Price of the good.

        Returns:
            float: Optimal labor.
        """
        return (pj * self.par.A * self.par.gamma / self.w) ** (1 / (1 - self.par.gamma))
    
    def optimal_output(self, ell_j):
        """
        Calculate the optimal output given labor.

        Args:
            ell_j (float): Amount of labor.

        Returns:
            float: Optimal output.
        """
        return self.par.A * ell_j ** self.par.gamma
    
    def optimal_profits(self, pj):
        """
        Calculate the optimal profits given the price of a good.

        Args:
            pj (float): Price of the good.

        Returns:
            float: Optimal profits.
        """
        return (1 - self.par.gamma) / self.par.gamma * self.w * (pj * self.par.A * self.par.gamma / self.w) ** (1 / (1 - self.par.gamma))
    
    def optimal_consumption(self, ell):
        """
        Calculate the optimal consumption given labor supply.

        Args:
            ell (float): Amount of labor supply.

        Returns:
            tuple: Optimal consumption of goods 1 and 2.
        """
        income = self.w * ell + self.par.T + self.optimal_profits(self.p1) + self.optimal_profits(self.p2)
        c1 = self.par.alpha * income / self.p1
        c2 = (1 - self.par.alpha) * income / (self.p2 + self.par.tau)
        # Clamp c1 and c2 to a small positive value to avoid zero or negative consumption
        c1 = max(c1, 1e-8)
        c2 = max(c2, 1e-8)
        return c1, c2

    def utility(self, ell):
        """
        Calculate the utility given labor supply.

        Args:
            ell (float): Amount of labor supply.

        Returns:
            float: Utility.
        """
        c1, c2 = self.optimal_consumption(ell)
        # Ensure c1 and c2 are positive to avoid invalid logarithm operations
        if c1 <= 0 or c2 <= 0:
            return -np.inf
        return np.log(c1 ** self.par.alpha * c2 ** (1 - self.par.alpha)) - self.par.nu * ell ** (1 + self.par.epsilon) / (1 + self.par.epsilon)

    def optimal_labor_supply(self):
        """
        Calculate the optimal labor supply.

        Returns:
            float: Optimal labor supply.
        """
        result = minimize(lambda ell: -self.utility(ell), x0=1, bounds=[(1e-8, None)])
        if not result.success or np.isnan(result.x[0]):
            return 1.0  # Return a default positive value if optimization fails
        return result.x[0]

    def calculate_transfer(self, c2_star):
        """
        Calculate the transfer payment based on consumption of the second good.

        Args:
            c2_star (float): Consumption of the second good.
        """
        if c2_star <= 0:
            self.par.T = 0.0
        else:
            self.par.T = self.par.tau * c2_star

    def market_clearing(self):
        """
        Calculate market clearing conditions.

        Returns:
            tuple: Differences in labor, good 1, and good 2 markets.
        """
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
        """
        Check for market clearing across a range of prices.

        Args:
            economy (ProductionEconomy): The economy instance.
            p1_range (iterable): Range of prices for the first good.
            p2_range (iterable): Range of prices for the second good.
            threshold (float): Threshold for market clearing.

        Returns:
            list: Market clearing solutions.
        """
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

    def market_clearing_conditions(self, prices):
        """
        Calculate market clearing conditions given a set of prices.

        Args:
            prices (iterable): Prices of the two goods.

        Returns:
            list: Market clearing conditions.
        """
        self.set_prices(*prices)
        
        # Optimal labor supply and consumption calculations
        ell_star = self.optimal_labor_supply()
        c1_star, c2_star = self.optimal_consumption(ell_star)
        
        # Single update of transfer payment based on new c2_star
        if not hasattr(self, '_transfer_updated') or not self._transfer_updated:
            self.calculate_transfer(c2_star)
            self._transfer_updated = True
        
        # Optimal labor for the two goods
        ell1_star = self.optimal_labor(self.p1)
        ell2_star = self.optimal_labor(self.p2)
        
        # Optimal output
        y1_star = self.optimal_output(ell1_star)
        
        # Market clearing conditions
        labor_diff = ell_star - (ell1_star + ell2_star)
        good1_diff = c1_star - y1_star
        
        return [labor_diff, good1_diff]

    def find_equilibrium_prices(self, initial_guess):
        """
        Find equilibrium prices starting from an initial guess.

        Args:
            initial_guess (list): Initial guess for the prices.

        Returns:
            list: Equilibrium prices if found, otherwise None.
        """
        self._transfer_updated = False  # Reset transfer update flag before finding equilibrium
        result = root(self.market_clearing_conditions, initial_guess)
        return result.x if result.success else None

    def calculate_social_welfare(self, tau: float) -> float:
        """
        Calculate social welfare for a given tax rate.

        Args:
            tau (float): Tax rate.

        Returns:
            float: Social welfare.
        """
        self.par.tau = tau
        
        # Reset the transfer T and the update flag
        self.par.T = 0.0
        self._transfer_updated = False
        
        equilibrium_prices = self.find_equilibrium_prices([1.0, 1.0])
        if equilibrium_prices is None:
            return -np.inf  # Return a very low value if no equilibrium is found
        
        self.p1, self.p2 = equilibrium_prices
        ell_star = self.optimal_labor_supply()
        c1_star, c2_star = self.optimal_consumption(ell_star)
        self.calculate_transfer(c2_star)
        
        utility = self.utility(ell_star)
        if utility == -np.inf:
            return -np.inf
    
        y2_star = self.optimal_output(self.optimal_labor(self.p2))
        
        return utility - self.par.kappa * y2_star

    def optimize_social_welfare(self) -> tuple:
        """
        Optimize social welfare by adjusting the tax rate.

        Returns:
            tuple: Optimal tax rate and corresponding transfer.
        """
        result = minimize_scalar(lambda tau: -self.calculate_social_welfare(tau), bounds=(0.001, 0.99), method='bounded')
        
        if result.success:
            optimal_tau = result.x
            self.par.tau = optimal_tau
            self._transfer_updated = False
            optimal_T = self.par.tau * self.optimal_consumption(self.optimal_labor_supply())[1]
            return optimal_tau, optimal_T
        else:
            return None, None

    def calculate_swf(self, tau: float) -> float:
        """
        Calculate social welfare function for a given tax rate.

        Args:
            tau (float): Tax rate.

        Returns:
            float: Social welfare function value.
        """
        self.par.tau = tau
        
        # Reset the transfer T and the update flag
        self.par.T = 0.0
        self._transfer_updated = False
        
        equilibrium_prices = self.find_equilibrium_prices([1.0, 1.0])
        if equilibrium_prices is None:
            return -np.inf  # Return a very low value if no equilibrium is found
        
        self.p1, self.p2 = equilibrium_prices
        ell_star = self.optimal_labor_supply()
        c1_star, c2_star = self.optimal_consumption(ell_star)
        self.calculate_transfer(c2_star)
        
        utility = self.utility(ell_star)
        if utility == -np.inf:
            return -np.inf
        
        y2_star = self.optimal_output(self.optimal_labor(self.p2))
        
        return utility - self.par.kappa * y2_star

    def optimize_tax(self) -> float:
        """
        Optimize the tax rate to maximize social welfare.

        Returns:
            float: Optimal tax rate.
        """
        def negative_swf(tau: float) -> float:
            return -self.calculate_swf(tau)
        
        result = minimize_scalar(negative_swf, bounds=(1e-8, 0.99), method='bounded')
        return result.x if result.success else None

    def plot_tax_rate_impact(self):
        """
        Plot the impact of different tax rates on total consumption, transfer, utility, and social welfare.
        """
        # Generate tax rates from 0 to 1 with higher resolution
        tax_rates = np.linspace(0.001, 0.99, 1000)  # Increase number of points
    
        # Initialize lists to store results
        total_consumptions = []
        transfers = []
        utilities = []
        social_welfares = []
    
        # Calculate values for each tax rate
        for tau in tax_rates:
            self.par.tau = tau
            
            # Reset critical parameters and flags
            self.par.T = 0.0
            self._transfer_updated = False
            
            # Find equilibrium prices
            equilibrium_prices = self.find_equilibrium_prices([1.0, 1.0])
            if equilibrium_prices is not None:
                self.p1, self.p2 = equilibrium_prices
                ell_star = self.optimal_labor_supply()
                c1_star, c2_star = self.optimal_consumption(ell_star)
                self.calculate_transfer(c2_star)
                
                utility = self.utility(ell_star)
                y2_star = self.optimal_output(self.optimal_labor(self.p2))
                social_welfare = utility - self.par.kappa * y2_star
                
                total_consumption = c1_star + c2_star
                
                # Append results to lists
                total_consumptions.append(total_consumption)
                transfers.append(self.par.T)
                utilities.append(utility)
                social_welfares.append(social_welfare)
            else:
                # Append NaN if equilibrium is not found
                total_consumptions.append(np.nan)
                transfers.append(np.nan)
                utilities.append(np.nan)
                social_welfares.append(np.nan)
    
        # Create subplots
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
        # Plot total consumption vs tax rate
        axs[0].plot(tax_rates, total_consumptions, label='Total Consumption', color='b')
        axs[0].set_xlabel('Tax Rate')
        axs[0].set_ylabel('Total Consumption')
        axs[0].set_title('Tax Rate vs Total Consumption')
    
        # Plot transfer T vs tax rate
        axs[1].plot(tax_rates, transfers, label='Transfer T', color='g')
        axs[1].set_xlabel('Tax Rate')
        axs[1].set_ylabel('Transfer T')
        axs[1].set_title('Tax Rate vs Transfer T')
    
        # Plot utility U vs tax rate
        axs[2].plot(tax_rates, utilities, label='Utility U', color='r')
        axs[2].set_xlabel('Tax Rate')
        axs[2].set_ylabel('Utility U')
        axs[2].set_title('Tax Rate vs Utility U')
    
        # Plot social welfare vs tax rate
        axs[3].plot(tax_rates, social_welfares, label='Social Welfare', color='m')
        axs[3].set_xlabel('Tax Rate')
        axs[3].set_ylabel('Social Welfare')
        axs[3].set_title('Tax Rate vs Social Welfare')
    
        # Show the optimal tax rate for reference
        optimal_tau, _ = self.optimize_social_welfare()
        for ax in axs:
            ax.axvline(optimal_tau, color='k', linestyle='--', label=f'Optimal Tax Rate: {optimal_tau:.3f}')
            ax.legend()
    
        # Adjust layout
        plt.tight_layout()
        plt.show()
