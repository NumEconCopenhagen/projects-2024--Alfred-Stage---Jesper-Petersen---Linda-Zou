from types import SimpleNamespace
import numpy as np 
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint  
import warnings
# Suppress the specific UserWarning related to no artists with labels found
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message="No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.")

class ExchangeEconomyClass:
    """
    This class seeks to model an exchange economy with two consumers and two goods. Preferences are 
    defined in the simplest Cobb-Douglas case. Each consumer determines demand based on prices and 
    given initial endowments. Class calculates demand for each good for each consumer and checks 
    if we achieve market clearing.
    """

    def __init__(self):
        """
        Initializes the exchange economy model with preferences, initial endowments,
        and sets up parameters for consumers A and B.
        """
        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3  # Consumer A's preference for good x1
        par.beta = 2/3   # Consumer B's preference for good x1

        # b. endowments
        self.par.w1A = 0.8  # Initial endowment of good x1 for consumer A
        self.par.w2A = 0.3  # Initial endowment of good x2 for consumer A
        
        self.comparison_results = {}
        self.W = None
        
    def utility_A(self, x1A, x2A):
        """
        Calculates CD utility of consumer A given consumption
        
        Parameters:
        - x1A (float): Quantity of good x1 consumed by A.
        - x2A (float): Quantity of good x2 consumed by A.
        
        Returns:
        - Utility (float): The utility value for consumer A.
        """
        par = self.par
        epsilon = 1e-8
        if x1A + epsilon <= 0 or x2A + epsilon <= 0:
            return -np.inf  # or some other indication of invalid utility
        return (x1A + epsilon)**self.par.alpha * (x2A + epsilon)**(1 - self.par.alpha)
        
    def utility_B(self, x1B, x2B):
        """
        Calculates CD utility of consumer B given consumption.
        
        Parameters:
        - x1B (float): Quantity of first good consumed by B.
        - x2B (float): Quantity of second good consumed by B.
        
        Returns:
        - Utility (float): The utility value for consumer B.
        """
        par = self.par
        epsilon = 1e-8
        return (x1B + epsilon)**self.par.beta * (x2B + epsilon)**(1 - self.par.beta)
        
    def demand_A(self, p1):
        """
        Calculates the optimal demand for goods x1 and x2 for consumer A given the price of good x1.
        
        Parameters:
        - p1 (float): Price of good x1.
        
        Returns:
        - x1A_star, x2A_star (tuple): A's optimal choice of x1 and x2.
        """
        par = self.par
        income_A = p1*par.w1A + par.w2A  # Total income for A, with P2 being numeraire
        x1A_star = par.alpha * income_A / p1
        x2A_star = (1 - par.alpha) * income_A
        return x1A_star, x2A_star
        
    def demand_B(self, p1):
        """
        Calculates the optimal demand for goods x1 and x2 for consumer B given the price of good x1.
        
        Parameters:
        - p1 (float): Price of good x1.
        
        Returns:
        - x1B_star, x2B_star (tuple): B's 's optimal choice of x1 and x2.
        """
        par = self.par
        income_B = p1*(1 - par.w1A) + (1 - par.w2A)  # Total income for B
        x1B_star = par.beta * income_B / p1
        x2B_star = (1 - par.beta) * income_B
        return x1B_star, x2B_star
        
    def check_market_clearing(self, p1):
        """
        Checks if the market clears for given price of good x1. We have market clearing when total
        demand equals total endowment for each good.
        
        Parameters:
        - p1 (float): Price of good x1.
        
        Returns:
        - eps1, eps2 (tuple): Epsilons express excess demand.
        """
        par = self.par

        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)

        eps1 = x1A + x1B - (par.w1A + (1 - par.w1A))  # Excess demand for good x1
        eps2 = x2A + x2B - (par.w2A + (1 - par.w2A))  # Excess demand for good x2

        return eps1, eps2
        
    def store_and_print_results(self, scenario_name, utility_A, utility_B, optimal_allocation_A, optimal_allocation_B, other_info=None, print_results=False):
        """
        Stores and optionally prints the results of a scenario in a consistent format.
    
        Parameters:
        - scenario_name (str): The name of the scenario.
        - utility_A (float): Utility for consumer A.
        - utility_B (float): Utility for consumer B.
        - optimal_allocation_A (tuple): Optimal allocation for consumer A.
        - optimal_allocation_B (tuple): Optimal allocation for consumer B.
        - other_info (dict, optional): Additional relevant information.
        - print_results (bool): If True, print the results.
        """
        results = {
            'utilities': {
                'A': utility_A,
                'B': utility_B
            },
            'allocations': {
                'A': optimal_allocation_A,
                'B': optimal_allocation_B
            }
        }
    
        if other_info:
            results['other_info'] = other_info
    
        # Optionally print the results
        if print_results:
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.2f}")
                elif isinstance(value, tuple):
                    formatted_tuple = ", ".join([f"{v:.2f}" for v in value])  # Format each element in the tuple
                    print(f"{key}: ({formatted_tuple})")
                else:
                    # For 'utilities', 'allocations' or 'other_info'
                    print(f"{key}:")
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float)):
                            print(f"  {subkey}: {subvalue:.2f}")
                        elif isinstance(subvalue, tuple):
                            formatted_tuple = ", ".join([f"{v:.2f}" for v in subvalue])
                            print(f"  {subkey}: ({formatted_tuple})")
                        else:
                            print(f"  {subkey}: {subvalue}")
    
        # Store the structured results
        self.comparison_results[scenario_name] = results

        
    def find_market_clearing_price(self, N=75, price_range=(0.01, 5), threshold=0.005, scenario_name='market_clearing', print_results=False):

        """
        Finds the market clearing price within a specified range and threshold, along with
        optimal allocations and utilities for consumers A and B. Results are printed directly
        from within the method and stored for later comparison.

        Parameters:
        - N (int): Number of price points to evaluate.
        - price_range (tuple): The (min, max) range of prices to evaluate.
        - threshold (float): Acceptable error margin for market clearing.
        - scenario_name (str): A name for the scenario to uniquely identify and store results for comparison.

        Returns:
        - None: Directly prints the optimal price, max utility for A and B, and optimal allocations.
        """
        p1_values = np.linspace(price_range[0], price_range[1], N + 1)
        optimal_info = None
        min_epsilon_norm = float('inf')

        for p1 in p1_values:
            eps1, eps2 = self.check_market_clearing(p1)
            epsilon_norm = abs(eps1) + abs(eps2)

            if epsilon_norm < min_epsilon_norm:
                min_epsilon_norm = epsilon_norm
                if epsilon_norm < threshold:
                    x1A_star, x2A_star = self.demand_A(p1)
                    x1B_star, x2B_star = self.demand_B(p1)
                    utility_A = self.utility_A(x1A_star, x2A_star)
                    utility_B = self.utility_B(x1B_star, x2B_star)

                    # Store the allocations for potential use in initial guesses for other optimizations
                    self.market_clearing_allocations = {'A': (x1A_star, x2A_star), 'B': (x1B_star, x2B_star)}

                    self.store_and_print_results(
                        scenario_name=scenario_name, 
                        utility_A=utility_A, 
                        utility_B=utility_B, 
                        optimal_allocation_A=(x1A_star, x2A_star), 
                        optimal_allocation_B=(x1B_star, x2B_star), 
                        other_info={'market_clearing_price': p1},
                        print_results=print_results
                    )
                    return  # Exiting the loop after finding the market clearing price

        if not optimal_info:
            print(f"No market clearing price found within the defined threshold for scenario: {scenario_name}")
            
    def optimize_utility_for_A(self, N=75, price_range=(0.01, 5), scenario_name='price'):


        """
        Optimizes utility for consumer A by selecting the best price from a set or range of prices,
        taking into account the allocation left after consumer B's demand is met. This method is
        designed to work for both discrete sets of prices and continuous ranges by adjusting the
        number of price points (N) and the price range. Results are printed directly from within the method
        and stored for later comparison.

        Parameters:
        - N (int): Number of price points to evaluate within the range, allowing for granularity adjustment.
        - price_range (tuple): The (min, max) range of prices to evaluate, accommodating both discrete and continuous cases.
        - scenario_name (str): A name for the scenario to uniquely identify and store results for comparison.

        Returns:
        - None: Directly prints the optimal price, max utility for A and B, and optimal allocations.
        """
        p1_values = np.linspace(price_range[0], price_range[1], N+1)

        max_utility_A = -np.inf
        optimal_p1 = None
        optimal_allocation_A = None
        optimal_allocation_B = None
        max_utility_B = None

        for p1 in p1_values:
            x1B_star, x2B_star = self.demand_B(p1)
            x1A_left = 1 - x1B_star
            x2A_left = 1 - x2B_star
            utility_A = self.utility_A(x1A_left, x2A_left)
            utility_B = self.utility_B(x1B_star, x2B_star)

            if utility_A > max_utility_A:
                max_utility_A = utility_A
                optimal_p1 = p1
                optimal_allocation_A = (x1A_left, x2A_left)
                optimal_allocation_B = (x1B_star, x2B_star)
                max_utility_B = utility_B

        self.store_and_print_results(
            scenario_name=scenario_name,
            utility_A=max_utility_A,
            utility_B=max_utility_B,
            optimal_allocation_A=optimal_allocation_A,
            optimal_allocation_B=optimal_allocation_B,
            other_info={'optimal_p1': optimal_p1},
            print_results=True
)
        
    def find_optimal_allocation_within_C(self, N=75, scenario_name='choice_set'):
        """
        Finds and prints the optimal allocation for consumer A within the choice set C, ensuring that consumer B is not worse off than in the initial endowment. Results are printed and stored for later comparison.

        Parameters:
        - N (int): The granularity of the allocation space.
        - scenario_name (str): A name for the scenario to store and retrieve results for comparison.

        Returns:
        - None: Directly prints the results and stores them in the class for later access.
        """
        best_utility_A = -float('inf')
        optimal_allocation_A = (None, None)
        utility_B_at_optimal_A = None

        initial_utility_A = self.utility_A(self.par.w1A, self.par.w2A)
        initial_utility_B = self.utility_B(1 - self.par.w1A, 1 - self.par.w2A)

        for x1A in [i / N for i in range(N + 1)]:
            for x2A in [i / N for i in range(N + 1)]:
                x1B = 1 - x1A
                x2B = 1 - x2A

                if (self.utility_A(x1A, x2A) >= initial_utility_A and
                    self.utility_B(x1B, x2B) >= initial_utility_B):

                    current_utility_A = self.utility_A(x1A, x2A)
                    if current_utility_A > best_utility_A:
                        best_utility_A = current_utility_A
                        optimal_allocation_A = (x1A, x2A)
                        utility_B_at_optimal_A = self.utility_B(x1B, x2B)

        # Construct the optimal_info dictionary after the loop and condition checks
        if optimal_allocation_A != (None, None):
            self.store_and_print_results(
                scenario_name=scenario_name,
                utility_A=best_utility_A,
                utility_B=utility_B_at_optimal_A,
                optimal_allocation_A=optimal_allocation_A,
                optimal_allocation_B=(1 - optimal_allocation_A[0], 1 - optimal_allocation_A[1]),
                print_results=True  
            )
        else:
            print("No optimal allocation found within the constraints of C.")   
            
    def optimize_allocation_no_restrictions(self, scenario_name='less_restrictions', print_results=True):
        """
        Optimizes allocation for consumer A without restrictions, ensuring consumer B
        is not worse off than at their initial endowment.

        Parameters:
        - scenario_name (str): Identifier for the optimization scenario.
        - print_results (bool): If True, prints the results after storing.

        Returns:
        - None: Directly prints and stores the results.
        """
        # Objective function for Consumer A's utility (to be maximized)
        def objective_A(x):
            return -self.utility_A(x[0], x[1])  

        # Constraint ensuring Consumer B is not worse off
        def constraint_B(x):
            w1B_initial, w2B_initial = 1 - self.par.w1A, 1 - self.par.w2A
            initial_utility_B = self.utility_B(w1B_initial, w2B_initial)
            return self.utility_B(1 - x[0], 1 - x[1]) - initial_utility_B

        # Constraints and bounds setup
        constraints = ({
            'type': 'ineq',  
            'fun': constraint_B
        })
        bounds = [(0, 1), (0, 1)]

        # Multi-start optimization
        initial_guesses = [
            [0.1, 0.1],
            [0.5, 0.5],
            [0.9, 0.9],
            [0.2, 0.8],
            [0.8, 0.2]
        ]

        best_utility_A = -np.inf
        optimal_allocation_A = None

        for guess in initial_guesses:
            result = minimize(objective_A, guess, method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success and -result.fun > best_utility_A:
                best_utility_A = -result.fun
                optimal_allocation_A = result.x

        if optimal_allocation_A is not None:
            optimal_allocation_B = (1 - optimal_allocation_A[0], 1 - optimal_allocation_A[1])
            utility_B = self.utility_B(*optimal_allocation_B)

            self.store_and_print_results(
                scenario_name=scenario_name,
                utility_A=best_utility_A,
                utility_B=utility_B,
                optimal_allocation_A=optimal_allocation_A,
                optimal_allocation_B=optimal_allocation_B,
                print_results=print_results  
            )
        else:
            print("Optimization failed to find a solution.")       
            
    def find_optimal_allocation_social_planner(self, scenario_name='social_planner', print_results=True):
        """
        Finds the optimal allocation chosen by a utilitarian social planner to maximize aggregate utility.

        Parameters:
        - scenario_name (str): A name for the optimization scenario.
        - print_results (bool): If True, prints the results after storing.

        Returns:
        - None: Directly prints and stores the results.
        """
        # Aggregate utility function to be maximized
        
        def objective_aggregate(x):
            utility_A = self.utility_A(x[0], x[1])
            utility_B = self.utility_B(1 - x[0], 1 - x[1])
            return -(utility_A + utility_B)  # Negate for minimization

        bounds = [(0, 1), (0, 1)]
        initial_guess = self.market_clearing_allocations['A'] if hasattr(self, 'market_clearing_allocations') else [0.5, 0.5]

        result = minimize(objective_aggregate, initial_guess, method='SLSQP', bounds=bounds)

        if result.success:
            optimal_x1A = result.x[0]
            optimal_x2A = result.x[1]
            utility_A = self.utility_A(optimal_x1A, optimal_x2A)
            utility_B = self.utility_B(1 - optimal_x1A, 1 - optimal_x2A)
            optimal_aggregate_utility = utility_A + utility_B

            self.store_and_print_results(
                scenario_name=scenario_name,
                utility_A=utility_A,
                utility_B=utility_B,
                optimal_allocation_A=(optimal_x1A, optimal_x2A),
                optimal_allocation_B=(1 - optimal_x1A, 1 - optimal_x2A),
                other_info={'optimal_aggregate_utility': optimal_aggregate_utility},
                print_results=print_results  
            )
        else:
            print("Optimization failed:", result.message)
            
    def compare_results(self):
        """
        Compares and prints the stored results from various scenarios, formatting numbers to two decimals
        and displaying them in a concise format.
        """
        if not hasattr(self, 'comparison_results') or not self.comparison_results:
            print("No results available for comparison.")
            return

        for scenario_name, results in self.comparison_results.items():
            # Initialize strings to hold the formatted data for each aspect of the result
            allocations_str = ""
            utilities_str = ""
            aggregate_util_str = ""

            if 'allocations' in results:
                allocations = results['allocations']
                allocations_str = f"Allocation A = ({allocations['A'][0]:.2f}, {allocations['A'][1]:.2f}); " \
                                  f"Allocation B = ({allocations['B'][0]:.2f}, {allocations['B'][1]:.2f})"
            if 'utilities' in results:
                utilities = results['utilities']
                utilities_str = f"Utility A = {utilities['A']:.2f}, Utility B = {utilities['B']:.2f}"
            if 'optimal_aggregate_utility' in results:
                aggregate_util_str = f"Agg Util = {results['optimal_aggregate_utility']:.2f}"
            elif 'utilities' in results:  # Fallback if aggregate utility isn't directly provided
                aggregate_util = results['utilities']['A'] + results['utilities']['B']
                aggregate_util_str = f"Agg Util = {aggregate_util:.2f}"

            print(f"{scenario_name}: \n  {allocations_str}\n  {utilities_str}, {aggregate_util_str}\n----------")
            
    def plot_edgeworth_box(self, w1bar=1.0, w2bar=1.0):
        """
        Plots an Edgeworth box for the economy.
        Parameters:
        - w1bar: Total endowment of good 1
        - w2bar: Total endowment of good 2
        """
        # Figure set up
        sns.set_theme(style="ticks")
        sns.color_palette("ch:s=.25,rot=-.25")
        fig = plt.figure(frameon=False, figsize=(6, 6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)
        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")
        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()
        # Plot limits
        ax_A.plot([0, w1bar], [0, 0], lw=2, color='black')
        ax_A.plot([0, w1bar], [w2bar, w2bar], lw=2, color='black')
        ax_A.plot([0, 0], [0, w2bar], lw=2, color='black')
        ax_A.plot([w1bar, w1bar], [0, w2bar], lw=2, color='black')
        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])
        return ax_A, ax_B  # Returning axes might be useful for further customization outside the method
    
    def plot_feasible_set(self, w1bar=1.0, w2bar=1.0, N=75):
        """
        Plots the feasible set within the Edgeworth box that leaves both parties at least as well off as initially.
        """
        initial_utility_A = self.utility_A(self.par.w1A, self.par.w2A)
        initial_utility_B = self.utility_B(1 - self.par.w1A, 1 - self.par.w2A)
    
        ax_A, _ = self.plot_edgeworth_box(w1bar, w2bar)  # Use the class method to set up the box
    
        # Finding combinations that leave both parties better off
        for x1A in np.linspace(0, w1bar, N+1):
            for x2A in np.linspace(0, w2bar, N+1):
                x1B = w1bar - x1A
                x2B = w2bar - x2A
                if self.utility_A(x1A, x2A) >= initial_utility_A and self.utility_B(x1B, x2B) >= initial_utility_B:
                    ax_A.scatter(x1A, x2A, color='blue', s=1, label=None)  # Plotting eligible points on ax_A
            
    def plot_comparisons_with_pareto_improvements(self, w1bar=1.0, w2bar=1.0, N=75):
        """
        Plots the Edgeworth box, overlays the feasible set (Pareto improvements) with a specified alpha, 
        and then overlays allocations from various scenarios for comparison, each in a different color, 
        and includes the scenario names in the legend.
        """
        initial_utility_A = self.utility_A(self.par.w1A, self.par.w2A)
        initial_utility_B = self.utility_B(1 - self.par.w1A, 1 - self.par.w2A)

        # Plot the Edgeworth box
        ax_A, _ = self.plot_edgeworth_box(w1bar, w2bar)

        # Overlay the Pareto improvements with a specified alpha
        for x1A in np.linspace(0, w1bar, N+1):
            for x2A in np.linspace(0, w2bar, N+1):
                x1B = w1bar - x1A
                x2B = w2bar - x2A
                if self.utility_A(x1A, x2A) >= initial_utility_A and self.utility_B(x1B, x2B) >= initial_utility_B:
                    ax_A.scatter(x1A, x2A, color='blue', alpha=0.2, s=1)  

        # Overlay each scenario's allocations from self.comparison_results
        colors = iter(['red', 'green', 'blue', 'purple', 'orange', 'cyan'])  # Define a list of colors for different scenarios
        for scenario_name, results in self.comparison_results.items():
            color = next(colors)  # Get the next color
            # Extract allocations for A using the updated structure
            if 'allocations' in results and 'A' in results['allocations']:
                x1A, x2A = results['allocations']['A']
                ax_A.scatter(x1A, x2A, color=color, label=scenario_name, s=20, edgecolor='black')  

        ax_A.legend(title='Scenarios', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()      
        


    def plot_utility_differences(self):
        """
        Plots differences in aggregate utility relative to the Social Planner scenario,
        using data from comparison_results.
        """
        # Print the keys to debug and find the correct key for the social planner scenario
        print("Available keys in comparison_results:", self.comparison_results.keys())
    
        # Check if 'Social Planner' or its equivalent is in comparison_results
        social_planner_key = 'Social Planner' if 'Social Planner' in self.comparison_results else 'Q6a'
        if social_planner_key not in self.comparison_results:
            print(f"{social_planner_key} data not available for comparison.")
            return
    
        # Retrieve Social Planner's aggregate utility
        sp_utilities = self.comparison_results[social_planner_key]['utilities']
        social_planner_aggregate_utility = sp_utilities['A'] + sp_utilities['B']
    
        differences = [0]  # Difference for the Social Planner itself is zero
        scenarios = [social_planner_key]
    
        for scenario_name, results in self.comparison_results.items():
            if scenario_name == social_planner_key:
                continue  # Skip the social planner in the differences calculation but include in the plot
    
            utilities = results.get('utilities', {})
            utility_A = utilities.get('A', 0)
            utility_B = utilities.get('B', 0)
            aggregate_utility = utility_A + utility_B
            difference = aggregate_utility - social_planner_aggregate_utility
    
            differences.append(difference)
            scenarios.append(scenario_name)
    
        # Create a DataFrame for plotting
        data = pd.DataFrame({
            "Scenario": scenarios,
            "Difference": differences  
        })
    
        # Convert inf values to NaN to avoid warnings
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
        # Drop rows with NaN values
        data.dropna(inplace=True)
    
        # Suppress the specific FutureWarning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message="use_inf_as_na option is deprecated")
            sns.set_theme(style="ticks")
            palette = sns.color_palette("ch:s=.25,rot=-.25", n_colors=len(data))
    
            plt.figure(figsize=(10, 6))
            sns.stripplot(
                x="Difference", 
                y="Scenario", 
                data=data, 
                jitter=False, 
                size=10, 
                marker="D", 
                alpha=.8
            )
    
            plt.title('Differences in Aggregate Utility Relative to Social Planner')
            plt.xlabel('Difference from Social Planner Aggregate Utility')
            plt.ylabel('Scenario')
            plt.axvline(x=0, color='red', linestyle='--')
            plt.show()

        


    def print_comparison_results(self):
        """
        Prints the comparison results stored in the dictionary in a readable format.
        """
        pprint(self.comparison_results) 
        
    def generate_W(self, n_elements=50):
        np.random.seed(42)
        w1A = np.random.uniform(0, 1, n_elements)
        w2A = np.random.uniform(0, 1, n_elements)
        return list(zip(w1A, w2A))  
        
    def draw_random_set(self, n_elements=50, seed=42):
        """
        Draws a random set W consisting of elements (w1A, w2A) and plots it.

        Parameters:
        - n_elements (int): Number of elements to generate for the set.
        - seed (int): Seed for the random number generator for reproducibility.
        """
        W = self.generate_W()

        print("First few elements of set W:")
        for element in list(W)[:5]:  # Convert set to list and then slice
            print(element)
            
        w1A_values = [elem[0] for elem in W]
        w2A_values = [elem[1] for elem in W]

        plt.figure(figsize=(8, 6))
        plt.scatter(w1A_values, w2A_values, color='blue', marker='x', label='Elements of $\mathcal{W}$')
        plt.title('Set $\mathcal{W}$ with 50 Elements')
        plt.xlabel('$\omega_1^A$')
        plt.ylabel('$\omega_2^A$')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def infer_market_clearing_price(self, N=1000, price_range=(0.01, 5), threshold=0.005):
        p1_values = np.linspace(*price_range, N+1)
        market_clearing_price = None
        min_epsilon_norm = float('inf')

        for p1 in p1_values:
            eps1, eps2 = self.check_market_clearing(p1)
            epsilon_norm = abs(eps1) + abs(eps2)

            if epsilon_norm < min_epsilon_norm:
                min_epsilon_norm = epsilon_norm
                if epsilon_norm < threshold:
                    market_clearing_price = p1
                    break  # Stop if a suitable market-clearing price is found

        return market_clearing_price
        
    def find_equilibriums_with_clearing_price_and_check(self, original_endowment, W):
        equilibriums = []
        original_utility_A = self.utility_A(*original_endowment)
        original_utility_B = self.utility_B(1 - original_endowment[0], 1 - original_endowment[1])

        for omega in W:
            self.par.w1A, self.par.w2A = omega
            market_clearing_price = self.infer_market_clearing_price()

            if market_clearing_price:
                x1A_star, x2A_star = self.demand_A(market_clearing_price)
                x1B_star, x2B_star = self.demand_B(market_clearing_price)

                utility_A = self.utility_A(x1A_star, x2A_star)
                utility_B = self.utility_B(x1B_star, x2B_star)

                # Check if this allocation is a Pareto improvement over the original endowment
                if utility_A >= original_utility_A and utility_B >= original_utility_B:
                    equilibriums.append((omega, market_clearing_price, (x1A_star, x2A_star), (x1B_star, x2B_star)))

        return equilibriums
        
    def plot_movement_from_endowment_to_equilibrium(self, equilibriums):
        # Plot the Edgeworth box
        ax_A, ax_B = self.plot_edgeworth_box()
        
        legend_handles = []

        for idx, (omega, market_clearing_price, alloc_A, alloc_B) in enumerate(equilibriums):
      
            start_point_A = omega  
            end_point_A = alloc_A  

            ax_A.annotate('', xy=end_point_A, xycoords='data', xytext=start_point_A, textcoords='data',
                          arrowprops=dict(arrowstyle="->", color="red"))

            # Optionally, mark the equilibrium point
            ax_A.scatter(*end_point_A, color='red', label=f'Equilibrium {idx+1}')
            legend_handles.append(ax_A.scatter(*end_point_A, color='red'))
            

        plt.legend(handles=legend_handles, labels=['Red dots = equilibria'], fontsize='small')

        plt.show()

    def plot_contract_curve(self, initial_utility_A, initial_utility_B, equilibriums):
        """
        Plot the contract curve in an Edgeworth box diagram.

        This method visualizes the initial endowment, all potential Pareto improvements, 
        and the equilibrium allocations in an Edgeworth box for a given economic model.

        Parameters:
        initial_utility_A (float): The initial utility level for agent A.
        initial_utility_B (float): The initial utility level for agent B.
        equilibriums (list): A list of tuples, each containing (omega, p1, alloc_A, alloc_B), 
                             representing equilibrium outcomes.
        """
        sns.set_theme(style="ticks")
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plotting the initial endowment from Q1
        ax.scatter(0.8, 0.3, color='black', label='Initial Endowment from Q1')

        # Plotting all potential Pareto improvements
        N = 75
        for x1A in np.linspace(0, 1, N+1):
            for x2A in np.linspace(0, 1, N+1):
                if self.utility_A(x1A, x2A) >= initial_utility_A and self.utility_B(1 - x1A, 1 - x2A) >= initial_utility_B:
                    ax.scatter(x1A, x2A, color='blue', s=1, alpha=0.2)  # Lighter dots for the background

        # Overlaying the equilibrium outcomes from Q8
        for omega, p1, alloc_A, alloc_B in equilibriums:
            ax.scatter(*alloc_A, color='red', s=10, label='Equilibrium Allocations')  # Mark equilibriums in red

        # Setting the limits and borders for the Edgeworth box
        w1bar, w2bar = 1.0, 1.0
        ax.plot([0, w1bar], [0, 0], lw=2, color='black')
        ax.plot([0, w1bar], [w2bar, w2bar], lw=2, color='black')
        ax.plot([0, 0], [0, w2bar], lw=2, color='black')
        ax.plot([w1bar, w1bar], [0, w2bar], lw=2, color='black')

        ax.set_xlim([-0.1, w1bar + 0.1])
        ax.set_ylim([-0.1, w2bar + 0.1])

        # Labels and Legend
        ax.set_xlabel('$x_1^A$')
        ax.set_ylabel('$x_2^A$')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # Remove duplicate labels
        ax.legend(by_label.values(), by_label.keys(), frameon=True, loc='upper right', bbox_to_anchor=(1.6, 1.0))

        plt.show()
