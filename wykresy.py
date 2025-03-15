import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

DATA_DIR = "data/"
SP500 = DATA_DIR + "sp500_index.csv"
BTC = DATA_DIR + "BTC-Daily.csv"



def wykres(time,time_offset,fileName,InstrumentName):
    DATA = pd.read_csv(fileName)

    DATA["Date"] = pd.to_datetime(DATA["Date"])
    DATA["Value"] = pd.to_numeric(DATA["Value"]) 

    DATA = DATA.iloc[time_offset:time_offset+time].reset_index(drop=True)

    fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(12, 6))  # Trzy wykresy

    ax1.plot(DATA["Date"], DATA["Value"], label='Value', color='b')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value [$]')
    ax1.set_title(InstrumentName +' Value')
    ax1.legend()
    ax1.grid(True)

    plt.show()
    plt.savefig("wykres " + InstrumentName + " from " + str(yearFrom) + ".pdf",format="pdf")
    plt.close()




        

if __name__ == "__main__":
    wykres(365*3,0,BTC,"BTC")
    wykres(365*3,0,SP500,"S&P 500")
    wykres(365*3,365*3,SP500,"S&P 500")