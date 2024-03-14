import numpy as np # numerical computation
import pandas as pd # data manipulation

from matplotlib import style # asthetics
import matplotlib.pyplot as plt # vizualiation
plt.style.use('seaborn-v0_8-ticks')


class Summerizer:
    def __init__(self, results:dict):
        self.results = results

    @staticmethod
    def create_best_obj_matrix(ff:np.array, cc:np.array)->np.array:
        """
        Returns a new matrix in the form [simulation, function_eval].

        :params ff:(np.array) functional value matrix of form [simulation, function_eval].
        :params cc:(np.array) constraint value matrix of form [simulation, function_eval].
        """
        fx = np.copy(ff)
        for i in range(np.shape(fx)[0]):
            idxx = np.where(cc[i]>=0)
            if idxx[0][0]!=0:
                fx[i][0:idxx[0][0]] = -np.inf
            hx = np. zeros(np.shape(fx)[1])
            hx[idxx] = ff[i] [idxx[0]]
            for j in range(np.shape(fx)[1]-1):
                if ff[i][j+1] > fx[i][j] and hx[j+1]!=0:
                    fx[i][j+1] = ff[i][j+1]
                else:
                    fx[i][j+1] = fx[i][j]
        return fx
    
    def get_results(self, evaluate_at:list=[10,20,30])->pd.DataFrame:
        header1 = 3*['5%']+3*['mean']+3*['95%']
        header2 = 3*evaluate_at
        header = [np.array(header1), np.array(header2)]
        Results = []
        index = []
        avg_value = []
        self.best_objectives = {}
        for method, result in self.results.items():
            ff = np.array([x[:,0] for x in result]).T
            cc = np.array([x[:,1] for x in result]).T
            m = self.create_best_obj_matrix(ff, cc)
            mean = np.apply_along_axis(np.mean, 0, m)
            lower = np.apply_along_axis(np.quantile, 0, m, 0.05)
            upper = np.apply_along_axis(np.quantile, 0, m, 0.95)

            Results.append([lower[evaluate_at[0]], lower[evaluate_at[1]], lower[evaluate_at[2]]]+\
            [mean[evaluate_at[0]],mean[evaluate_at[1]], mean[evaluate_at[2]]]+\
            [upper[evaluate_at[0]], upper[evaluate_at[1]], upper[evaluate_at[2]]])

            index.append(method)

            avg_value.append(mean)
            self.best_objectives[method] = np.max(ff, 0)

        Results = pd.DataFrame(Results, index=index, columns=header)

        x = np.arange(1, avg_value[0].shape[0]+1)
        fig, ax = plt.subplots(1, 2, figsize=(14,5)) 

        fig.suptitle('Simulation Results', fontweight="bold")
        for i, avg in enumerate(avg_value):
        # Plotting the data with legends and labels
            ax[0].plot(x, avg, label=index[i])
        ax[0].text(0.5, 1.05, "Avg. Objective Value v/s Evaluation", horizontalalignment='center', 
                       verticalalignment='center', transform=ax[0].transAxes, fontweight='bold')
        # Set legend titles and labels for axes
        ax[0].legend()
        ax[0].set_xlabel('Evaluation (n)')
        ax[0].set_ylabel('Best Objective Value (Sharpe Ratio)')
        self.best_objectives = pd.DataFrame(self.best_objectives)
        ax[1].boxplot(self.best_objectives.values, vert=True, widths=0.3, medianprops=dict(color='orange'))
        ax[1].text(0.5, 1.05, "Distributions of Best Objective Values", horizontalalignment='center', 
                       verticalalignment='center', transform=ax[0].transAxes, fontweight='bold')
        ax[1].set_xticklabels(self.best_objectives.columns)
        #self.best_objectives.plot(kind="box", ax=ax[1])
        # Show the plot
        plt.show()
        return Results

    
