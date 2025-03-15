import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

SP500 = "sp500_index.csv"
BTC = "BTC-Daily.csv"
DATA_PATH_CSV = 'data/'



def EMA(value_today: float, EMA_yesterday: float, n: int) -> float:
    alpha: float = 2/(n+1)

    EMAtoday: float = (value_today * alpha) + EMA_yesterday * (1 - alpha)
    
    return EMAtoday

def MACD(values: list) -> list:
    macd: list = [0]
    ema12: float = values[0]
    ema26: float = values[0]

    for i in range(1,len(values)):
        ema12 = EMA(values[i],ema12,12)
        ema26 = EMA(values[i],ema26,26)

        macd.append(ema12 - ema26)

    return macd

def SIGNAL(macd: list) -> list:
    signal: list = [macd[0]]

    for i in range(1,len(macd)):
        signal.append(EMA(macd[i],signal[-1],9))
    
    return signal

def checkIfIntersects(figure1: list, figure2: list, i: int) -> bool:
    return (figure1[i] < figure2[i] and figure1[i+1] > figure2[i+1])

def calculateIntersections(xAxis: list, figure1: list, figure2: list, value: list) -> tuple[list,list]:
    buy_intersections = []
    sell_intersections = []


    if len(figure1) != len(figure2):
        raise "Cos sie zesralo, dwie tablice nie maja takich samych dlugosci"
    
    for i in range(0,len(figure1)-1):
        if checkIfIntersects(figure1,figure2,i):
            sell_intersections.append((pd.to_datetime(xAxis[i+1]),value[i+1]))
        elif checkIfIntersects(figure2,figure1,i):
            buy_intersections.append((pd.to_datetime(xAxis[i+1]),value[i+1]))

    return buy_intersections,sell_intersections

def simulateWallet(Dates: list, Values: list, Buys: list, Sells: list):
    LiquidCash = Values[0]*1000
    StockAmount = 0
    StockWorth = 0

    sell_dates = [x[0] for x in Sells]  # Daty przecięć
    buys_dates = [x[0] for x in Buys]  # Daty przecięć
    

    walletSim = []
    walletSim.append((Dates[0],LiquidCash,StockWorth))

    for i in range(1,len(Dates)):
        
        if Dates[i] in buys_dates:
            StockAmount += LiquidCash/Values[i]
            LiquidCash = 0

        elif Dates[i] in sell_dates:
            LiquidCash += StockAmount*Values[i]
            StockAmount = 0

        StockWorth = StockAmount*Values[i]

        walletSim.append((Dates[i],LiquidCash,StockWorth))

    return walletSim

def createPlotMACD(DATA, DATA_Title, macd, signal):
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 12))  # Trzy wykresy

    # Wykres MACD / SIGNAL
    ax.plot(DATA["Date"], macd, label='MACD', color='r')  
    ax.plot(DATA["Date"], signal, label='SIGNAL', color='b')  
    ax.set_ylabel('MACD / SIGNAL')
    ax.set_title(DATA_Title + ' MACD Indicator')
    ax.legend()
    ax.grid(True)

    plt.savefig(DATA_Title + " from "  + ".pdf",format="pdf")

def createPlotBUYSELL(DATA, DATA_Title, buy_intersections, sell_intersections,):
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 12))  # Trzy wykresy

    buy_dates = [x[0] for x in buy_intersections] 
    buy_values = [x[1] for x in buy_intersections] 

    sell_dates = [x[0] for x in sell_intersections]  
    sell_values = [x[1] for x in sell_intersections] 

    # Wykres wartości 
    ax.plot(DATA["Date"], DATA["Value"], label=DATA_Title + ' Value', color='b')
    ax.scatter(buy_dates, buy_values, label='BUY', color='g', zorder=5, marker='s')
    ax.scatter(sell_dates, sell_values, label='SELL', color='r', zorder=5, marker='s')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value [$]')
    ax.set_title(DATA_Title + 'BUY & SELL Points')
    ax.legend()
    ax.grid(True)

def createPlotPortfolioSim(DATA, DATA_Title, walletSim):
    # Wykres stanu konta
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 12))  # Trzy wykresy

    wallet_dates = [x[0] for x in walletSim]  
    walletSim_total = [ (cash + stock_worth) for date, cash, stock_worth in walletSim]

    Values = DATA["Value"].copy()
    Values *= 1000

    ax.plot(wallet_dates, walletSim_total, label='Portfolio value with MACD Trading', color='r')  
    ax.plot(DATA["Date"], Values, label='Portfolio value with buying shares only once', color='b')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value [$]')
    ax.set_title(DATA_Title + ' Portfolio Simulator ')
    formatter = ticker.FuncFormatter(lambda x, _: f"{int(x/1_000_000)} mln")
    ax.yaxis.set_major_formatter(formatter)
    ax.legend()
    ax.grid(True)


def createCombinedPlots(DATA, DATA_Title, macd, signal, buy_intersections, sell_intersections, walletSim):
    buy_dates = [x[0] for x in buy_intersections] 
    buy_values = [x[1] for x in buy_intersections] 

    sell_dates = [x[0] for x in sell_intersections]  
    sell_values = [x[1] for x in sell_intersections] 

    wallet_dates = [x[0] for x in walletSim]  
    walletSim_total = [ (cash + stock_worth) for date, cash, stock_worth in walletSim]

    # Obliczanie procentowego wzrostu
    zarobione_procent = (walletSim_total[-1] / walletSim_total[0] - 1) * 100
    sytuacja_bez_macd_procent = (DATA["Value"].iloc[-1] / DATA["Value"].iloc[0] - 1) * 100

    # Drukowanie wyników w formacie procentowym
    print(f"Zarobione z macd: {zarobione_procent:.2f}% | Gdyby wrzucić bez MACD w 2014: {sytuacja_bez_macd_procent:.2f}%")

    # Trzy wykresy w jednej fig. (2 wiersze, 2 kolumny, trzeci wykres na dole)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 12))  # Trzy wykresy

    # Wykres MACD / SIGNAL
    ax1.plot(DATA["Date"], macd, label='MACD', color='r')  
    ax1.plot(DATA["Date"], signal, label='SIGNAL', color='b')  
    ax1.set_ylabel('MACD / SIGNAL')
    ax1.set_title(DATA_Title + ' Combined Plots')
    ax1.legend()
    ax1.grid(True)

    # Wykres wartości 
    ax2.plot(DATA["Date"], DATA["Value"], label=DATA_Title + ' Value', color='b')
    ax2.scatter(buy_dates, buy_values, label='BUY', color='g', zorder=5, marker='s')
    ax2.scatter(sell_dates, sell_values, label='SELL', color='r', zorder=5, marker='s')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value [$]')
    ax2.legend()
    ax2.grid(True)

    # Wykres stanu konta

    Values = DATA["Value"].copy()
    Values *= 1000

    ax3.plot(wallet_dates, walletSim_total, label='Portfolio value with MACD Trading', color='r')  
    ax3.plot(DATA["Date"], Values, label='Portfolio value with buying shares only once', color='b')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Value [$]')
    formatter = ticker.FuncFormatter(lambda x, _: f"{int(x/1_000_000)} mln")
    ax3.yaxis.set_major_formatter(formatter)
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()  # Lepsze rozmieszczenie wykresów
    

def createAllPlots(DATA, DATA_Title, macd, signal, buy_intersections, sell_intersections, walletSim):
    createPlotMACD(DATA,DATA_Title,macd,signal)
    createPlotBUYSELL(DATA,DATA_Title,buy_intersections,sell_intersections)
    createPlotPortfolioSim(DATA,DATA_Title,walletSim)
    createCombinedPlots(DATA, DATA_Title, macd, signal, buy_intersections, sell_intersections, walletSim)
    
def generate_plot(Data_path, Data_title, time, time_offset):
    DATA = pd.read_csv(Data_path)

    DATA["Date"] = pd.to_datetime(DATA["Date"])
    DATA["Value"] = pd.to_numeric(DATA["Value"]) 

    DATA = DATA.iloc[time_offset:time+time_offset].reset_index(drop=True)

    macd = MACD(DATA["Value"])
    signal = SIGNAL(macd)
    buy_intersections,sell_intersections = calculateIntersections(DATA['Date'],macd,signal,DATA['Value'])
    walletSim = simulateWallet(DATA['Date'],DATA['Value'],buy_intersections,sell_intersections)

    createAllPlots(DATA,Data_title,macd,signal,buy_intersections,sell_intersections,walletSim)

    



def main():
    generate_plot(DATA_PATH_CSV+SP500,"S&P 500",3*365,0)
    generate_plot(DATA_PATH_CSV+SP500,"S&P 500",3*365,3*365)
    generate_plot(DATA_PATH_CSV+BTC,"BTC",3*365,0)
    
    plt.show()






        

if __name__ == "__main__":
    main()