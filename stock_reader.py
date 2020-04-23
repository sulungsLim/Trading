import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import warnings
import seaborn as sns 
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False

def reader():  # read stock information from FinanceData Reader. Insert code and date
    try:
        df = ''
        krx = fdr.StockListing("krx")  #국내시장 변수화
        spx = fdr.StockListing("S&P500") # S&P500 시장  변수화

        code = input("Enter Code or Symbol : ")#코드 입력
        date = input("Enter start date(YYYY-MM--DD) : ") #날짜 입력
        date2 = input("Enter end date(YYYY-MM--DD) : ")

        df = fdr.DataReader(code, date, date2)
        df = df.rename(columns=lambda col: col.lower())  # Change column names from uppercase to lower case
        ix_0_value =  df[df["open"]== 0.].index  # 시가 = 0인 데이터 추출
        df = df.drop(ix_0_value)# 시가 = 0 ---> 거래정지기간  제거

        if code in list(krx["Symbol"]):   #국내종목일 경우 :  종목코드,종목이름, 산업 불러오기
            print('')
            print(krx.loc[krx["Symbol"] == code, ["Symbol", "Name", "Sector"]])
            print('')
            print(df)
            name_section = krx.loc[krx["Symbol"] == code, ["Name"]]
            name = name_section.iloc[0][0]
            plt.figure(figsize=(10, 5), dpi=100)  # 선택된 종목의 종가 그래프
            plt.plot(df.index, df['close'], label=name + '_stockprice')
            plt.xlabel('Date')
            plt.ylabel('Won')
            plt.title('Figure 2:  ' + name + '  stock price')
            plt.legend()
            plt.show()

        elif code in list(spx["Symbol"]):  #S&P종목일 경우 : 종목코드,종목이름, 산업 불러오기
            print('')
            print(spx.loc[spx["Symbol"] == code, ["Symbol", "Name", "Sector"]])
            print('')
            print(df)
            name_section2 = spx.loc[spx["Symbol"] == code, ["Name"]]
            name2 = name_section2.iloc[0][0]
            plt.figure(figsize=(10, 5), dpi=100)  # 선택된 종목의 종가 그래프
            plt.plot(df.index, df['close'], label=name2 + '_stockprice')
            plt.xlabel('Date')
            plt.ylabel('Dollar')
            plt.title('Figure 2:  ' + name2 + '  stock price')
            plt.legend()
            plt.show()
        else:
            print("No Symbol Found")  # 없는 종목일 경우

        print("")
        print('There are {} number of days in the dataset.'.format(df.shape[0]))
        print("")

        return df

    except:
        print("Wrong code")  # if any error occurs -> print "Wrong code"

