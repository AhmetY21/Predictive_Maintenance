import pandas as pd
import numpy as np
np.random.seed=0
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


def crest_factor(x):
    return np.max(np.abs(x))/np.sqrt(np.mean(np.square(x)))

def build_features(time):
    X = pd.DataFrame()
    dayz=time.columns[time.columns.str.contains('Time')]
    for i in range(len(dayz)):
        signal=time[dayz[i]]


        X.loc[i,'Name']=dayz[i].split('_')[0]
        X.loc[i,'Order']=int(dayz[i].split('_')[1])

        #Time Domain

        X.loc[i,'Sum']=signal.sum()
        X.loc[i,'Mean']=signal.mean()
        X.loc[i,'Std']=signal.std()
        X.loc[i,'Max']=signal.max()
        X.loc[i,'Min']=signal.min()

        rms = np.sqrt(np.mean(signal**2))
        X.loc[i,'RMS']=rms


        X.loc[i,'Skew']=signal.skew()
        X.loc[i,'Kurtosis']=signal.kurtosis()
        X.loc[i,'Crest Factor'] = crest_factor(signal)
        X.loc[i,'Impulse Factor'] = (signal.max()) / ((abs(signal)).mean())
        X.loc[i,'Margin Factor'] = (signal.max()) / ((abs(signal)).mean()**2)
        logging.info("Time Domain Features Calculated")
        #Frequency Domain

        f = np.fft.fft(signal)
        f_real = np.real(f)

        X.loc[i,'F_domain_Mean']=f_real.mean()
        X.loc[i,'F_domain_Std']=f_real.std()
        X.loc[i,'F_domain_Max']=f_real.max()
        X.loc[i,'F_domain_Min']=f_real.min()
        X.loc[i,'F_domain_Crest_Factor'] = crest_factor(f_real)
        X.loc[i,'F_domain_Impulse_Factor'] = (f_real.max()) / ((abs(f_real)).mean())
        X.loc[i,'F_domain_Margin_Factor'] = (f_real.max()) / ((abs(f_real)).mean()**2)
        logging.info("Frequency Domain Features Calculated")
    return X


def chunk_it(df,suffix,chunk_size=13000):
    list_df = [df[i:i+chunk_size] for i in range(0,df.shape[0],chunk_size)]
    x=pd.DataFrame()
    for i in range(int(df.shape[0]/chunk_size)):
        x[suffix+'_'+str(i)]=list_df[i].values
        x.add_suffix(suffix)
    logging.info("Signal Chunked into Seconds")
    return x


def m2_converter(Path_name,Label,Machine_Num):
    #Adjust first line according to input type
    time1=pd.read_csv(Path_name,sep=',',skiprows=7,index_col=0).reset_index()
    dayz=time1.columns[time1.columns.str.contains('Time')]
    ch=pd.DataFrame()
    for i in range(len(dayz)):
        signal=time1[dayz[i]]
        temp=chunk_it(signal,dayz[i])
        ch=pd.concat([temp,ch],axis=1)

    tablem=build_features(ch)
    tablem['label']=Label
    logging.info("Signal Ready for Classification")
    return tablem
def lab_break(table,Label,breakdowns,widget=True):
    if widget:
        table.loc[table[table['Name'].isin(breakdowns.value)].index,'label']=Label
    else:
        table.loc[table[table['Name'].isin(breakdowns)].index,'label']=Label

    return table
