import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from tqdm.notebook import tqdm

from joblib import Parallel, delayed

from bayes_opt import BayesianOptimization

class VectorizedBT(object):
    """
    A class to backtest trading signals 
    using vectorized backtesting method.
    """
    def __init__(self, commission_per_side:float=0.006):
        self.commission_per_side = commission_per_side

        
    def get_signals(self):
        """
        Implement the strategy signals here.

        :return : None
        """
        raise NotImplementedError('Subclass must implement this!')
    
    @staticmethod
    def get_commission(signals:pd.Series, commissions_per_side:float):
        commissions = pd.Series(0, index=signals.index)
        trades_loc = list(map(int, signals[:-1].values != signals.shift(-1)[:-1].values))
        commissions.iloc[1:] = trades_loc
        return abs(commissions)*commissions_per_side

    def run(self)->None:
        """
        Run the vectorized backtest.
        """
        self.signals = self.get_signals()
        commissions_ = self.get_commission(self.signals, self.commission_per_side)
        returns = (1+self.PL).pct_change()[1:]
        pnl = self.PL.diff()[1:]
        self.returns_strat = self.signals * returns
        self.commissions_accum = self.returns_strat * commissions_
        self.returns_strat = self.returns_strat - self.commissions_accum
        self.pnl_strat = self.signals * pnl
  
    def get_summary(self, show:bool=True)->None:
        """
        Calculate the backtest summary results.

        :param show: (bool) whether to print the report 
        :return : None
        """
        #Calculate cumulative P&L commission adjusted
        accum_commissions = self.commissions_accum.cumsum()
        self.equity_pnl = self.pnl_strat.cumsum()
        #Calculate the sharpe ratio
        self.volatility = np.nanstd(self.returns_strat)+1E-10
        self.avg_return = np.nanmean(self.returns_strat)
        self.sharpe = (self.avg_return/(self.volatility))*np.sqrt(252*375)
        #Calculate running maximum
        running_max = self.equity_pnl.cummax()
        #Calculate drawdown
        drawdowns = self.equity_pnl - running_max
        #Maximum Drawdown
        max_dd = drawdowns.min()
        entries = self.signals.diff()[1:]
        if show:
            #Print Metrics
            print("                   Results              ")
            print("-------------------------------------------")
            print("%14s %21s" % ('statistic', 'value'))
            print("-------------------------------------------")
            print("%20s %20.2f" % ("Absolute P&L :", self.equity_pnl.iloc[-1]))
            print("%20s %20.2f" % ("Sharpe Ratio :", self.sharpe))
            print("%20s %20.2f" % ("Volatility :", self.volatility))
            print("%20s %20.2f" % ("Max. Drawdown P&L:", round(max_dd, 2)))
            print("%20s %20.2f" % ("Total % Commission:", round(accum_commissions.iloc[-1], 2)))
            #print("%20s %20.2f" % ("Total Trades :", sum(abs(entries))))
            #Plots
            x = self.equity_pnl.index
            #fig, axs = plt.subplots(4, figsize=(16, 15), height_ratios=[4, 3, 4, 4])
            fig, axs = plt.subplots(3, figsize=(16, 12), height_ratios=[4, 4, 4])
            fig.suptitle('Backtest Report', fontweight="bold")
            axs[0].plot(x, self.spread.values, color='#aec6cf')
            axs[0].title.set_text("P/L")
            axs[0].grid()
            # axs[1].plot(x, self.signals.values, color='#aec6cf')
            # axs[1].title.set_text("Positions")
            # axs[1].grid()
            axs[1].plot(x, self.equity_pnl.values, color='#77dd77')
            axs[1].title.set_text("Strategy Equity Curve : P&L")
            axs[1].grid()
            axs[2].fill_between(x, drawdowns.values, color='#ff6961', alpha=0.5)
            axs[2].title.set_text("Drawdowns : P&L")
            axs[2].grid()
            plt.show()


class Optimizer:
    def __init__(self, strategy:VectorizedBT, df:pd.DataFrame):
        self.strategy = strategy
        self.df = df
        
    def grid_search(self, params:dict={}, category:str='sharpe')->None:
        """
        Run grid search over the parameters.
        Returns the optimal parameters for 
        the strategy that maximize the 
        category.
        
        :param category :(str) Use either 'sharpe' or 'pnl'
        :param params :(dict) search dictionary for the 
                        strategy parameters. Make sure 
                        the keys are the arguments of strategy.
        :return : None
        """
        keys = list(params.keys())
        search_space = [dict(zip(keys, values)) for values in product(*params.values())]
        self.best_category_val = -np.inf
        self.best_param = None
        for search_param in tqdm(search_space):
            this_strat = self.strategy(self.df, **search_param)
            this_strat.run()
            this_strat.get_summary(False)
            val = None
            if category=='sharpe':
                val = this_strat.sharpe
            elif category=='pnl':
                val = this_strat.equity_pnl
            else:
                raise ValueError("Invalid Category.")
                
            if val > self.best_category_val:
                self.best_category_val = val
                self.best_param = search_param
        print(f"Best {category} score achived : {self.best_category_val}\n")
        print(f"Best parameters for the above score : {self.best_param}")

    def randomized_search(self, params:list=[], constraint=None, constraint_value=0, 
                          category:str='sharpe', n_iter:int=20, verbose:bool=True, 
                          random_state:int=4)->None:
        """
        Run grid search over the parameters.
        Returns the optimal parameters for 
        the strategy that maximize the 
        category.
        
        :param category :(str) Use either 'sharpe' or 'pnl'
        :param params :(list) 
                        MAKE SURE THE PARAMETERS ARE IN FOLLOWING
                        FORMAT
                        [dict(name='parameter name', type='parameter data type', bounds=(minimum, maximum))]

        :return : None
        """
        np.random.seed(random_state)
        self.best_category_val = -np.inf
        self.best_param = None
        self.observations = []

        for _ in tqdm(range(n_iter)):
            search_param = {}
            for param in params:
                value = None
                if param['type']=='int':
                    value = np.random.randint(param['bounds']['min'], param['bounds']['max'])
                else:
                    value = np.random.uniform(param['bounds']['min'], param['bounds']['max'])
                search_param[param['name']] = value
            #function evaluation
            this_strat = self.strategy(self.df, **search_param)
            this_strat.run()
            this_strat.get_summary(False)
            val = None
            if category=='sharpe':
                val = this_strat.sharpe
            elif category=='pnl':
                val = this_strat.equity_pnl
            else:
                raise ValueError("Invalid Category.")
            
            constraint_observed = constraint(**search_param)
            feasible = constraint_observed>=constraint_value 
            self.observations.append([val, constraint_observed, feasible])

            if (val > self.best_category_val) and feasible:
                self.best_category_val = val
                self.best_param = search_param
        if verbose:
            print(f"Best {category} score achived : {self.best_category_val}\n")
            print(f"Best parameters for the above score : {self.best_param}")    


class Simulation:
    def __init__(self, acq_func=None, function=None, bounds=None, constraint=None, 
                 init_points=10, n_iter=20, optimizer=None):
        self.acq = acq_func
        self.target_function = function
        self.pbounds = bounds
        self.constraint = constraint
        self.init_points = init_points
        self.n_iter = n_iter
        self.optimizer = optimizer

    def create_job(self, i):
        if self.optimizer is None:
            optimizer = BayesianOptimization(
                                            f=self.target_function,
                                            constraint=self.constraint,
                                            pbounds=self.pbounds,
                                            verbose=0, 
                                            random_state=i+13,
                                        )
            optimizer.maximize(
                        acquisition_function=self.acq,
                        init_points=self.init_points,
                        n_iter=self.n_iter,
                    )
            values = pd.DataFrame(optimizer.res).drop(columns=['params']).values
            return values
        self.optimizer.randomized_search(self.pbounds, self.constraint, 0, 
                                         category='sharpe', n_iter=self.n_iter, 
                                         verbose=False, random_state=i+13)
        return np.array(self.optimizer.observations)
    
    def run(self, N=10, n_cores=-1, verbose=0):
        return Parallel(n_jobs=n_cores, verbose=verbose)(delayed(self.create_job)(i) for i in range(N))


