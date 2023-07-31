import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from operator import itemgetter
import datetime
import seaborn as sns
import statsmodels.api as sm

from quantvale import utils
from quantvale import useful
from quantvale import backtest
from quantvale import access
from qvscripts.index import for_indexdb

PRELOADID_FULL_DAY_BARS = '__PL__A__FULL_DAY_BARS'

class Task(backtest.BaseBacktestTask):
    def execute(self):
        try:
            exe_start_time = utils.now()
            # read parameters from json file
            self.beginDate, self.endDate, self.indexCode, self.QsCodes, self.args, self.taskID \
                    = list(itemgetter('BeginDate', 'EndDate', 'TaskTag', 'QSCode', 'Arguments', 'TASKID')(self.params))
            self.beginDate, self.endDate = map(lambda x: datetime.date(*map(int, x.split('-'))), [self.beginDate, self.endDate])
            self.log.record(f'起始日：{self.beginDate}')
            self.log.record(f'截止日：{self.endDate}')
            self.data_dir = self.gen_data_dir()

            self.progress(10)
            
            # get all qscodes
            if type(self.QsCodes) == list:
                self.allQsCodes = self.QsCodes
            else:
                self.allQsCodes = pd.read_csv(self.QsCodes)['qscode'].tolist()
            
            self.log.record(self.allQsCodes)
            self.progress(20)

            # get data access
            clt = access.get_default_access_client()
            self.trade_cal = clt.trade_cal(self.beginDate, self.endDate)
            targets = ['open', 'close', 'high', 'low']
            # # get data
            # 
            # self.df_cyc_bars = clt.batch_cyc_bar_by_datetime(qscodes = self.allQsCodes, 
            #                                                  fields = targets, 
            #                                                  begin_time = utils.combine_datetime(self.trade_cal[0], utils.zero_hour()), 
            #                                                  end_time = utils.combine_datetime(self.trade_cal[-1], utils.zero_hour()),
            #                                                  cycle = 'day', 
            #                                                  adjfq = useful.AdjFQ.QFQ)
            
            # get data including own database
            self.df_cyc_bars = {}
            for qscode in self.allQsCodes:
                _, _, category, _ = useful.parse_qvcode(qscode)
                if category == useful.QVCodeCategory.UNKNOWN:
                    for_indexdb.ready()
                    df_bars = for_indexdb.fetch_day_bar_by_date(qscode,
                                                                fields=targets,
                                                                begin_date=self.trade_cal[0],
                                                                end_date=self.trade_cal[-1])
                else:
                    df_bars = clt.cyc_bar(qscode=qscode,
                                          fields=targets,
                                          begin_date=self.trade_cal[0],
                                          end_date=self.trade_cal[-1],
                                          cycle='day',
                                          adjfq=useful.AdjFQ.QFQ)
                self.df_cyc_bars[qscode] = df_bars

            # 将数据转换为DataFrame格式
            self.df_cyc_bars = pd.concat(self.df_cyc_bars.values(), keys = self.df_cyc_bars.keys(), names = ['qscode', 'date'])[targets]
            self.log.record(self.df_cyc_bars)
            self.progress(40)

            # make copy for later calculations
            self.df = self.df_cyc_bars.copy()
            self.df.reset_index(inplace=True)
            self.df['date'] = pd.to_datetime(self.df['date'])
            # # calculate returns
            # self.df['return'] = self.df['close'].pct_change(1) + 1
            # self.df['cum_return'] = self.df['return'].cumprod() - 1

            ##########---Extreme_Price_Scan_function---##########
            def extreme_price_scanner(df: pd.DataFrame, window=63, upper_q=95, lower_q=5, change_lim=0.2):
                # Calculate the local top and bottom 5 percentile prices
                df['roll_high'] = df['close'].rolling(window).apply(lambda x: np.percentile(x, upper_q))
                df['roll_low'] = df['close'].rolling(window).apply(lambda x: np.percentile(x, lower_q))
                # df['shift_return'] = np.abs(df['close'].pct_change(window))
                df['max_drawdown'] = df['close'] / df['high'].rolling(window).max() - 1
                df['max_upturn'] = df['close'] / df['low'].rolling(window).min() - 1

                # Create a binary variable for extreme prices
                df['Extreme_High'] = np.where((df['close'] >= df['roll_high']) & (df['max_upturn'] >= change_lim), 1, 0)
                df['Extreme_Low'] = np.where((df['close'] <= df['roll_low']) & (df['max_drawdown'] <= -change_lim), -1, 0)
                df['Extreme'] = df['Extreme_High'] + df['Extreme_Low']
                
                return df
            
            ##########---plot single stock---##########
            def plot_signal(df: pd.DataFrame, qscode, signal, target):
                plot_df = df[df['qscode'] == qscode].set_index('date')
                plot_df.index = pd.to_datetime(plot_df.index)
                fig, ax = plt.subplots(figsize=(20, 12))
                ax.set_title(f"{target} Plot of {qscode} with {signal}")
                ax.set_xlabel('Date')
                ax.set_ylabel(target)
                ax.plot(plot_df.index, plot_df[target], color='black')
                ax.plot(plot_df.index, plot_df[target].where(plot_df[signal] == 1), color='green', alpha=0.3, linewidth=10)
                ax.plot(plot_df.index, plot_df[target].where(plot_df[signal] == -1), color='orange', alpha=0.3, linewidth=10)
                ax.grid()
                plt.savefig(os.path.join(self.data_dir, f"{qscode}/plot_{qscode}.png"))
            
            self.df = self.df.groupby('qscode', group_keys=False).apply(lambda x: extreme_price_scanner(x, 
                                                                                                        self.args['window'], 
                                                                                                        self.args['upper_q'], 
                                                                                                        self.args['lower_q'], 
                                                                                                        self.args['change_lim']))
            self.progress(60)
            
            
            for qscode in self.df['qscode'].unique():
                os.makedirs(os.path.join(self.data_dir, qscode), exist_ok=True)
                plot_signal(self.df, qscode, "Extreme", "close")
                cur_df = self.df[self.df['qscode'] == qscode]
                cur_df.to_csv(os.path.join(self.data_dir, qscode, f"{qscode}_data.csv"))
                
            self.progress(80)

            if self.args['alert_newest']:
                newest_df = self.df[self.df['date'] == pd.to_datetime(self.trade_cal[-1])]
                date_str = pd.to_datetime(self.trade_cal[-1]).strftime("%Y-%m-%d")
                high_alert = newest_df[newest_df['Extreme'] == 1]['qscode']
                low_alert = newest_df[newest_df['Extreme'] == -1]['qscode']
                high_alert.to_csv(os.path.join(self.data_dir, f"high_alert_qscodeList_{date_str}.csv"))
                low_alert.to_csv(os.path.join(self.data_dir, f"low_alert_qscodeList_{date_str}.csv"))

            # self.df.to_csv(os.path.join(self.data_dir, f"data_{self.indexCode}.csv"))
            self.df.rename(columns = {"Extreme": "Signal"}, inplace=True)
            output_df = self.df[['date', 'qscode', 'Signal', 'close']]
            output_df = output_df.pivot(index="date", columns="qscode", values="Signal").sort_index()
            output_df.to_csv(os.path.join(self.data_dir, f"signals_{self.indexCode}.csv"))

            self.progress(100)
            self.chrono.stop('End')
            # 报告
            self.end(dict(
                RunTime=dict(
                    start=utils.strdatetime(exe_start_time),
                    stop=utils.now_str(),
                    total=self.chrono.total(),
                    chrono=self.chrono.recall(),
                ),
            ))
            
        except Exception as e:
            from quantvale import error
            self.log.record(error.get_traceback())
            self.end_with_exception()