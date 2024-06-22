from types import SimpleNamespace
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})

class CareerClass:

    def __init__(self):
        """ initialize the Career class with default parameters """

        par = self.par = SimpleNamespace()
        par.J = 3
        par.N = 10
        par.K = 10000
        
        par.F = np.arange(1,par.N+1)
        par.sigma = 2
        par.v = np.array([1,2,3])
        par.c = 1

    def simulate(self):
        """Calculate expected utility and average realized utility"""
        expect_u = np.zeros(self.par.J)
        avg_real_u = np.zeros(self.par.J)

        for j in range(self.par.J):
            v_j = self.par.v[j]
            epsilon = np.random.normal(0, self.par.sigma, self.par.K)
            
            # Expected utility
            expect_u[j] = v_j + np.mean(epsilon)
            
            # Average realized utility
            avg_real_u[j] = np.mean(v_j + epsilon)
            
            print(f"Career track {j + 1}:")
            print(f"Expected utility: {expect_u[j]:.4f}")
            print(f"Average realized utility: {avg_real_u[j]:.4f}")
            print('-----------')

    def simulate_friend(self):

        # Initialize arrays to store results
        chose_car = np.zeros((self.par.N, self.par.K), dtype=int)
        p_expect = np.zeros((self.par.N, self.par.K))
        real_u = np.zeros((self.par.N, self.par.K))

        for i in range(self.par.N):  # Loop over each graduate i
            F_i = i + 1  # Number of friends for graduate i
            
            for k in range(self.par.K):  # Perform K simulations
                # Step 1: draw noise of both friend and graduate
                epsilon_f = np.random.normal(0, self.par.sigma, size=(F_i, self.par.J))
                epsilon_g = np.random.normal(0, self.par.sigma, size=self.par.J)                
                avg_u_f = (self.par.v + epsilon_f).mean(axis=0) # calculate prior expected utility
                
                # Step 2 determine the career track with the highest prior expected utility
                j_star_f = np.argmax(avg_u_f)
                
                # Step 3 Storage
                chose_car[i, k] = j_star_f
                p_expect[i, k] = avg_u_f[j_star_f]
                
                # Calculate realized utility of chosen career track
                real_u[i, k] = self.par.v[j_star_f] + epsilon_g[j_star_f]
                
        return chose_car, p_expect, real_u

    def simulate_switch(self):
        """Simulate career change after one year of working"""

        # Initialize arrays to store results
        chose_car_i = np.zeros((self.par.N, self.par.K), dtype=int)
        p_expect_s = np.zeros((self.par.N, self.par.K))
        real_u_s = np.zeros((self.par.N, self.par.K))
        switched_career = np.zeros((self.par.N, self.par.K), dtype=bool)  # To track if switched

        for i in range(self.par.N):  # Loop over each graduate i
            F_i = i + 1  # Number of friends for graduate i
            
            for k in range(self.par.K):  # Perform K simulations
                # Step 1: draw noise of both friend and graduate
                epsilon_f = np.random.normal(0, self.par.sigma, size=(F_i, self.par.J))
                epsilon_g = np.random.normal(0, self.par.sigma, size=self.par.J)                  
                avg_u_f = (self.par.v + epsilon_f).mean(axis=0) # calculate prior expected utility
                
                # Step 2 determine the career track with the highest prior expected utility
                j_optimal = np.argmax(avg_u_f)

                chose_car_i[i, k] = j_optimal

                avg_u_f_drop = avg_u_f.copy()  # drop the career chosen before
                avg_u_f_drop[j_optimal] = -np.inf  # Set the chosen career's utility to a very low number

                # Determine the new optimal career choice excluding j_star_f
                j_optimal_new = np.argmax(avg_u_f_drop)

                utility_switch = avg_u_f_drop[j_optimal_new] - self.par.c
                utility_stay = self.par.v[j_optimal] + epsilon_g[j_optimal]

                # Determine the optimal choice after considering switching cost
                if utility_switch > utility_stay:
                    p_expect_s[i, k] = utility_switch
                    real_u_s[i, k] = self.par.v[j_optimal_new] + epsilon_g[j_optimal_new]
                    switched_career[i, k] = True
                else:
                    p_expect_s[i, k] = utility_stay
                    real_u_s[i, k] = self.par.v[j_optimal] + epsilon_g[j_optimal]
    
        return chose_car_i, p_expect_s, real_u_s, switched_career

    def calculate_metrics(self, chose_car, p_expect, real_u):
        """Calculate metrics for visualization"""

        shares_g = np.zeros((self.par.N, self.par.J))
        avg_sub_expect_u = np.zeros(self.par.N)
        avg_ex_post_real_u = np.zeros(self.par.N)

        for i in range(self.par.N):
            count_choices = np.bincount(chose_car[i])
            shares_g[i] = count_choices / self.par.K
            avg_sub_expect_u[i] = np.mean(p_expect[i])
            avg_ex_post_real_u[i] = np.mean(real_u[i])

        return shares_g, avg_sub_expect_u, avg_ex_post_real_u

    def plot_shares(self, shares_g):
        """Plot the share of graduates choosing each career track as stacked bars"""

        colors = ['#FD8A8A', '#FFCBCB', '#9EA1D4'] 

        # Initialize figure with subplots
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_aspect('auto')
        fig.suptitle('Share of Graduates Choosing Each Career Track')

        # Plot stacked bars for each career track
        bottom = np.zeros(self.par.N)
        for j in range(self.par.J):
            share_j = shares_g[:, j]
            ax.bar(np.arange(1, self.par.N + 1), share_j, bottom=bottom, label=f'Career {j + 1}', alpha=0.8, color=colors[j])
            bottom += share_j

            # Add exact values as text annotations
            for i, share in enumerate(share_j):
                ax.text(i + 1, bottom[i] - share / 2, f'{share:.2f}', ha='center', va='center', color='white', fontsize=10)

        ax.set_xlabel('Graduate')
        ax.set_ylabel('Share of Graduates')
        ax.legend(loc='right', bbox_to_anchor=(1.5, 0.8))

        plt.tight_layout()
        plt.show()

    def plot_utilities(self, avg_sub_expect_u, avg_ex_post_real_u):
        """Plot the average subjective expected utility and average ex post realized utility"""

        # Initialize figure
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_aspect('auto')
        fig.suptitle('Average Utility Metrics')

        # Plot average subjective expected utility and average ex post realized utility
        ax.plot(np.arange(1, self.par.N + 1), avg_sub_expect_u, marker='o', color = '#F1F7B5', label='Average Subjective Expected Utility')
        ax.plot(np.arange(1, self.par.N + 1), avg_ex_post_real_u, marker='x', color = '#A8D1D1', label='Average Ex Post Realized Utility')

        ax.set_xlabel('Graduate')
        ax.set_ylabel('Utility')

        ax.legend(loc='upper center', bbox_to_anchor=(1.5, 0.8), fontsize = 10)
        plt.xticks(np.arange(1, self.par.N + 1))

        plt.tight_layout()
        plt.show()

    def calculate_switch_metrics(self, chose_car_i, p_expect_s, real_u_s, switched_career):
        """Calculate metrics for visualization after switching careers"""

        avg_sub_expect_u_s = np.zeros(self.par.N)
        avg_ex_post_real_u_s = np.zeros(self.par.N)
        switch_shares = np.zeros((self.par.N, self.par.J))

        for i in range(self.par.N):
            avg_sub_expect_u_s[i] = np.mean(p_expect_s[i])
            avg_ex_post_real_u_s[i] = np.mean(real_u_s[i])

        switch_shares = np.zeros((self.par.N, self.par.J))
        for j in range(self.par.J):
            for i in range(self.par.N):
                initial_choice_mask = (chose_car_i[i, :] == j)
                if np.sum(initial_choice_mask) > 0:
                    switch_shares[i, j] = np.mean(switched_career[i, initial_choice_mask])
        
        return switch_shares, avg_sub_expect_u_s, avg_ex_post_real_u_s
    

    # Combined Plot, only choose one between this one and separate plotting
    def plot_switch_results(self, switch_shares, avg_sub_expect_u_s, avg_ex_post_real_u_s):
        """Plot the switching shares and average utilities"""
        
        colors = ['#FD8A8A', '#FFCBCB', '#9EA1D4']

        fig, ax1 = plt.subplots(figsize=(12, 8))
        fig.suptitle('Switching Shares and Average Utilities by Graduate')

        # Plot switching shares as stacked bars
        bottom = np.zeros(self.par.N)
        for j in range(self.par.J):
            share_j = switch_shares[:, j]
            ax1.bar(np.arange(1, self.par.N + 1), share_j, bottom=bottom, label=f'Career {j + 1}', alpha=0.8, color=colors[j])
            bottom += share_j
            
            # Add exact values as text annotations
            for i, share in enumerate(share_j):
                ax1.text(i + 1, bottom[i] - share / 2, f'{share:.2f}', ha='center', va='center', color='white', fontsize=10, fontweight='bold')

        ax1.set_xlabel('Graduate')
        ax1.set_ylabel('Share of Graduates Switching')
        ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 0.9))
        
        # Plot average utilities on a second y-axis
        ax2 = ax1.twinx()
        ax2.plot(np.arange(1, self.par.N + 1), avg_sub_expect_u_s, marker='o', color='blue', label='Avg. Subjective Expected Utility')
        ax2.plot(np.arange(1, self.par.N + 1), avg_ex_post_real_u_s, marker='x', color='green', label='Avg. Ex Post Realized Utility')
        ax2.set_ylabel('Utility')
        ax2.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8))
        
        plt.tight_layout()
        plt.show()


    def plot_switch_share(self, switch_shares):
            """Plot the switching shares and average utilities"""
            
            colors = ['#FD8A8A', '#FFCBCB', '#9EA1D4']

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle('Switching Shares and Average Utilities by Graduate')

            # Plot switching shares as stacked bars
            bottom = np.zeros(self.par.N)
            for j in range(self.par.J):
                share_j = switch_shares[:, j]
                ax.bar(np.arange(1, self.par.N + 1), share_j, bottom=bottom, label=f'Career {j + 1}', alpha=0.8, color=colors[j])
                bottom += share_j
                
                # Add exact values as text annotations
                for i, share in enumerate(share_j):
                    ax.text(i + 1, bottom[i] - share / 2, f'{share:.2f}', ha='center', va='center', color='white', fontsize=10, fontweight='bold')

            ax.set_xlabel('Graduate')
            ax.set_ylabel('Share of Graduates Switching')
            ax.legend(loc='upper right', bbox_to_anchor=(1.5, 0.8))

            plt.tight_layout()
            plt.show()


    def plot_utilities_switch(self, avg_sub_expect_u_s, avg_ex_post_real_u_s):
        """Plot the average subjective expected utility and average ex post realized utility"""

        # Initialize figure
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_aspect('auto')
        fig.suptitle('Average Utility Metrics')

        # Plot average subjective expected utility and average ex post realized utility
        ax.plot(np.arange(1, self.par.N + 1), avg_sub_expect_u_s, marker='o', color = '#F1F7B5', label='Average Subjective Expected Utility')
        ax.plot(np.arange(1, self.par.N + 1), avg_ex_post_real_u_s, marker='x', color = '#A8D1D1', label='Average Ex Post Realized Utility')

        ax.set_xlabel('Graduate')
        ax.set_ylabel('Utility')

        ax.legend(loc='upper center', bbox_to_anchor=(1.5, 0.8), fontsize = 10)
        plt.xticks(np.arange(1, self.par.N + 1))

        plt.tight_layout()
        plt.show()


        

            
    
