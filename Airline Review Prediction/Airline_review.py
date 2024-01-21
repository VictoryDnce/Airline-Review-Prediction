import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import missingno as msno
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings("ignore",category=Warning)
pd.set_option("display.width",500)
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
import matplotlib
matplotlib.use("Qt5Agg")

dff = pd.read_csv("Airline_review/Airline_review.csv")
df = dff.copy()
df.head()
df.shape
df.info()


# Datasetindeki başlıkları düzeltme işlemi
df.columns = [col.replace(" ", "_").upper() for col in df.columns]

# Datasetindeki ilgisiz değişkenleri silme
df.drop("UNNAMED:_0",axis=1,inplace=True)
df.drop("AIRCRAFT",axis=1,inplace=True)
df.drop("ROUTE",axis=1,inplace=True)
df.drop("DATE_FLOWN",axis=1,inplace=True)


# OVERALL_RATING değişkenindeki n`i 10 ile değiştirme
df['OVERALL_RATING'].value_counts()
df['OVERALL_RATING']=df['OVERALL_RATING'].replace('n',10)
df['OVERALL_RATING']=df['OVERALL_RATING'].astype(int)


# Tarihlerdeki --- st,nd,rd,th -- lerin atılması ve formatlama işlemi -----------------------

# -- 'REVIEW_DATE'

df['REVIEW_DATE'] = [col.replace("nd","") if "nd" in col else col for col in df['REVIEW_DATE'].values]
df['REVIEW_DATE'] = [col.replace("th","") if "th" in col else col for col in df['REVIEW_DATE'].values]
df['REVIEW_DATE'] = [col.replace("st","") if "st" in col else col for col in df['REVIEW_DATE'].values]
df['REVIEW_DATE'] = [col.replace("rd","") if "rd" in col else col for col in df['REVIEW_DATE'].values]

df['REVIEW_DATE'] = [col.replace("August","Augu") if "August" in col else col for col in df['REVIEW_DATE'].values]
df['REVIEW_DATE'] = [col.replace("Augu","August") if "Augu" in col else col for col in df['REVIEW_DATE'].values]

df['REVIEW_DATE'] = pd.to_datetime(df['REVIEW_DATE'])

# ------------------------------------- Aykırı deger analizi ----------------------------------------------------
# Puanlandırmadaki dengesizlik giderildi.. 1-5 arası olması gerekirken 0-5 arasıydı

df["SEAT_COMFORT"] = df["SEAT_COMFORT"].apply(lambda x: x + 1.0 if x == 0.0 else x)
df["CABIN_STAFF_SERVICE"] = df["CABIN_STAFF_SERVICE"].apply(lambda x: x + 1.0 if x == 0.0 else x)
df["FOOD_&_BEVERAGES"] = df["FOOD_&_BEVERAGES"].apply(lambda x: x + 1.0 if x == 0.0 else x)
df["INFLIGHT_ENTERTAINMENT"] = df["INFLIGHT_ENTERTAINMENT"].apply(lambda x: x + 1.0 if x == 0.0 else x)
df["WIFI_&_CONNECTIVITY"] = df["WIFI_&_CONNECTIVITY"].apply(lambda x: x + 1.0 if x == 0.0 else x)
df["VALUE_FOR_MONEY"] = df["VALUE_FOR_MONEY"].apply(lambda x: x + 1.0 if x == 0.0 else x)

df["VALUE_FOR_MONEY"].unique()
df["VALUE_FOR_MONEY"].value_counts()

# -----------------------------------------------------------------------------------------

df2 = df.copy()

# Target`ın label encoding işlemi
le = LabelEncoder()
df2["RECOMMENDED"] = le.fit_transform(df2["RECOMMENDED"])
le.inverse_transform([0,1]) # ['no', 'yes'] - [0,1]

# ------------------------------------- Eksik deger analizi ----------------------------------------------------

df2.isnull().sum()

msno.bar(df2,figsize=(9,9),fontsize=10)
msno.matrix(df2,figsize=(9,9),fontsize=10)
msno.heatmap(df2,figsize=(11,11),fontsize=10)
msno.dendrogram(df2,figsize=(9,9),fontsize=10)

sns.heatmap(df.loc[:, df.isnull().any()].isnull().corr(), annot=True, fmt='.2f')
sns.heatmap(df.loc[:, df.notnull().any()].notnull().corr(), annot=True, fmt='.2f')

# df_erase = (df.loc[df["TYPE_OF_TRAVELLER"].isna()]["TYPE_OF_TRAVELLER"]).index
# df.drop(df_erase,inplace=True)


# Yanlıs verilen 10 puanların silinmesi - 666 tane
# Puanlama verilerindeki büyük boslukların birlikte gorulmesine ragmen 10 verilmesi durumu.
# Recommend ve Overall_Rating arasındaki korelasyonda iyileşme gözlemlenmiştir.

veri2 = df2.drop(["AIRLINE_NAME","REVIEW_TITLE","REVIEW_DATE","VERIFIED","TYPE_OF_TRAVELLER","REVIEW","SEAT_TYPE"],axis=1)
veri2 = veri2.loc[veri2["OVERALL_RATING"]==10]

veri2.dropna(how="all").count()
veri2.dropna(thresh=7).index # gecerli olanların indexleri

veri2 = veri2.drop(veri2.dropna(thresh=7).index,axis=0) # gecersiz olanlar
veri2.loc[veri2["OVERALL_RATING"]==10].index # gecersiz olanların indexleri
df2.drop(veri2.loc[veri2["OVERALL_RATING"]==10].index,axis=0,inplace=True) # veri setinden cıkarılması


###########################
# TYPE_OF_TRAVELLER
###########################

df2['TYPE_OF_TRAVELLER'].value_counts()
df2['TYPE_OF_TRAVELLER'].isnull().sum()

# Forward-fill (fill missing values with the previous valid value)
df2['TYPE_OF_TRAVELLER'] = df2['TYPE_OF_TRAVELLER'].fillna(method='ffill')
df2['TYPE_OF_TRAVELLER'].value_counts()

business = 100*df.loc[df["TYPE_OF_TRAVELLER"]=="Business"]["TYPE_OF_TRAVELLER"].count()/len(df["TYPE_OF_TRAVELLER"])
solo = 100*df.loc[df["TYPE_OF_TRAVELLER"]=="Solo Leisure"]["TYPE_OF_TRAVELLER"].count()/len(df["TYPE_OF_TRAVELLER"])
couple = 100*df.loc[df["TYPE_OF_TRAVELLER"]=="Couple Leisure"]["TYPE_OF_TRAVELLER"].count()/len(df["TYPE_OF_TRAVELLER"])
family = 100*df.loc[df["TYPE_OF_TRAVELLER"]=="Family Leisure"]["TYPE_OF_TRAVELLER"].count()/len(df["TYPE_OF_TRAVELLER"])

business2 = 100*df2.loc[df2["TYPE_OF_TRAVELLER"]=="Business"]["TYPE_OF_TRAVELLER"].count()/len(df2["TYPE_OF_TRAVELLER"])
solo2 = 100*df2.loc[df2["TYPE_OF_TRAVELLER"]=="Solo Leisure"]["TYPE_OF_TRAVELLER"].count()/len(df2["TYPE_OF_TRAVELLER"])
couple2 = 100*df2.loc[df2["TYPE_OF_TRAVELLER"]=="Couple Leisure"]["TYPE_OF_TRAVELLER"].count()/len(df2["TYPE_OF_TRAVELLER"])
family2 = 100*df2.loc[df2["TYPE_OF_TRAVELLER"]=="Family Leisure"]["TYPE_OF_TRAVELLER"].count()/len(df2["TYPE_OF_TRAVELLER"])

# atama öncesi ve sonrası yüzdelik değişim (NaN lar dahil)
print(f"bussiness {round(business,2)} - {round(business2,2)}\nsolo {round(solo,2)} - {round(solo2,2)}\ncouple {round(couple,2)} - {round(couple2,2)}\nfamily {round(family,2)} - {round(family2,2)}\n")
"""
solo 30.73 - 36.15
couple 22.72 - 27.1
family 18.78 - 22.23
bussiness 11.64 - 14.53
"""

# kendi içindeki değişim (NaN lar dahil değil)
"""
Solo Leisure      8386-7120 1266 %17
Couple Leisure    6290-5265 1025 %19
Family Leisure    5148-4352 796 %18
Business          3347-2696 651 %24
"""

# NaN ların dagılım oranı -- Toplam NaN sayısı 3738
"""
Solo Leisure      1266 %33.8
Couple Leisure    1025 %27.4
Family Leisure    796 %21.2
Business          651 %17.4
"""

###########################
# SEAT_TYPE
###########################

df2['SEAT_TYPE'].value_counts()
df2['SEAT_TYPE'].isnull().sum()

# Forward-fill (fill missing values with the previous valid value)
df2['SEAT_TYPE'] = df2['SEAT_TYPE'].fillna(method='ffill')

Economy_Class = 100*df.loc[df["SEAT_TYPE"]=="Economy Class"]["SEAT_TYPE"].count()/len(df["SEAT_TYPE"])
Business_Class = 100*df.loc[df["SEAT_TYPE"]=="Business Class"]["SEAT_TYPE"].count()/len(df["SEAT_TYPE"])
Premium_Economy = 100*df.loc[df["SEAT_TYPE"]=="Premium Economy"]["SEAT_TYPE"].count()/len(df["SEAT_TYPE"])
First_Class = 100*df.loc[df["SEAT_TYPE"]=="First Class"]["SEAT_TYPE"].count()/len(df["SEAT_TYPE"])

Economy_Class2 = 100*df2.loc[df2["SEAT_TYPE"]=="Economy Class"]["SEAT_TYPE"].count()/len(df2["SEAT_TYPE"])
Business_Class2 = 100*df2.loc[df2["SEAT_TYPE"]=="Business Class"]["SEAT_TYPE"].count()/len(df2["SEAT_TYPE"])
Premium_Economy2 = 100*df2.loc[df2["SEAT_TYPE"]=="Premium Economy"]["SEAT_TYPE"].count()/len(df2["SEAT_TYPE"])
First_Class2 = 100*df2.loc[df2["SEAT_TYPE"]=="First Class"]["SEAT_TYPE"].count()/len(df2["SEAT_TYPE"])

# atama öncesi ve sonrası yüzdelik değişim (NaN lar dahil)
print(f"Economy_Class {round(Economy_Class,2)} - {round(Economy_Class2,2)}\nBusiness_Class {round(Business_Class,2)} - {round(Business_Class2,2)}\nPremium_Economy {round(Premium_Economy,2)} - {round(Premium_Economy2,2)}\nFirst_Class {round(First_Class,2)} - {round(First_Class2,2)}\n")

"""
Economy_Class 82.62 - 86.77
Business_Class 9.05 - 9.5
Premium_Economy 2.79 - 2.89
First_Class 0.8 - 0.84
"""

# kendi içindeki değişim (NaN lar dahil değil)
"""
Economy Class      20089-19145 944 %4.6
Business Class     2239-2098 141 %6.7
Premium Economy    649-646 3 %0.4
First Class        194-186 8 %4.3
"""

# NaN ların dagılım oranı -- Toplam NaN sayısı 1096
"""
Economy Class      944 944 %86.1
Business Class     141 %12.8
Premium Economy    3 %0.2
First Class        8 %0.7
"""


###########################
# SEAT_COMFORT
###########################
df2.isnull().sum()

df2.corr(numeric_only=True)  # 0.707870 - 0.633572
df2.corr(numeric_only=True,method = "kendall") #  0.628520 - 0.550592
df2.corr(numeric_only=True,method = "spearman") # 0.699010 - 0.604651

df.corr(numeric_only=True)  # 0.707870 - 0.633572
df.corr(numeric_only=True,method = "kendall") #  0.628520 - 0.550592
df.corr(numeric_only=True,method = "spearman") # 0.699010 - 0.604651
le = LabelEncoder()
df["RECOMMENDED"] = le.fit_transform(df["RECOMMENDED"])

df2['SEAT_COMFORT'].value_counts()

# inceleme -----
veri4 = df2.drop(['AIRLINE_NAME', 'OVERALL_RATING', 'REVIEW_TITLE', 'REVIEW_DATE', 'VERIFIED', 'REVIEW', 'TYPE_OF_TRAVELLER', 'SEAT_TYPE'],axis=1)
veri4.loc[veri4["SEAT_COMFORT"].isnull() == False].index # NaN degeri olmayanlar

veri4.drop(veri4.loc[veri4["SEAT_COMFORT"].isnull() == False].index,axis=0,inplace=True) # NaN degeri olanlar 3490

liste = []
veri4.dropna(how="all").count()
veri4.dropna(thresh=3).head()
veri4.tail(25)
veri4.shape


# Uygulama ----------- Most Frequent
mean_imputer = SimpleImputer(strategy='most_frequent')
df2["SEAT_COMFORT"] = mean_imputer.fit_transform(df2[["SEAT_COMFORT"]])

# -----------------------------------------------------------------------------------------------------
df2.reset_index(inplace=True)

###########################
# CABIN_STAFF_SERVICE
###########################
df2["CABIN_STAFF_SERVICE"].value_counts()
df2["CABIN_STAFF_SERVICE"].isnull().sum()

df2['CABIN_STAFF_SERVICE'] = df2['CABIN_STAFF_SERVICE'].fillna(method='ffill')

# -----------------------------------------------------------------------------------------------------

###########################
# FOOD_&_BEVERAGES
###########################
df2["FOOD_&_BEVERAGES"].value_counts()
df2["FOOD_&_BEVERAGES"].isnull().sum()

df2['FOOD_&_BEVERAGES'] = df2['FOOD_&_BEVERAGES'].fillna(method='ffill')

# -----------------------------------------------------------------------------------------------------

###########################
# GROUND_SERVICE
###########################
df2["GROUND_SERVICE"].value_counts()
df2["GROUND_SERVICE"].isnull().sum()

df2['GROUND_SERVICE'] = df2['GROUND_SERVICE'].fillna(method='ffill')

# -----------------------------------------------------------------------------------------------------

###########################
# INFLIGHT_ENTERTAINMENT
###########################
df2["INFLIGHT_ENTERTAINMENT"].value_counts()
df2["INFLIGHT_ENTERTAINMENT"].isnull().sum()

df2['INFLIGHT_ENTERTAINMENT'] = df2['INFLIGHT_ENTERTAINMENT'].fillna(method='ffill')

# -----------------------------------------------------------------------------------------------------
df3=df2.copy()
###########################
# WIFI_&_CONNECTIVITY
###########################
df2["WIFI_&_CONNECTIVITY"].value_counts()
df2["WIFI_&_CONNECTIVITY"].isnull().sum()

df2['WIFI_&_CONNECTIVITY'] = df2['WIFI_&_CONNECTIVITY'].fillna(method='ffill')

# -----------------------------------------------------------------------------------------------------

###########################
# VALUE_FOR_MONEY
###########################
df2["VALUE_FOR_MONEY"].value_counts()
df2["VALUE_FOR_MONEY"].isnull().sum()

df2['VALUE_FOR_MONEY'] = df2['VALUE_FOR_MONEY'].fillna(method='ffill')

# -----------------------------------------------------------------------------------------
df2['INFLIGHT_ENTERTAINMENT'].iloc[0:4,].fillna(round(df2.iloc[0:4,[8,9,10,11,14]].values[0].mean()),inplace=True)
df2['INFLIGHT_ENTERTAINMENT'].iloc[1:2,].fillna(round(df2.iloc[1:2,[8,9,10,11,14]].values[0].mean()),inplace=True)
df2['INFLIGHT_ENTERTAINMENT'].iloc[2:3,].fillna(round(df2.iloc[2:3,[8,9,10,11,14]].values[0].mean()),inplace=True)
df2['INFLIGHT_ENTERTAINMENT'].iloc[3:4,].fillna(round(df2.iloc[3:4,[8,9,10,11,14]].values[0].mean()),inplace=True)
df2['WIFI_&_CONNECTIVITY'].iloc[0:1,].fillna(round(df2.iloc[0:1,[8,9,10,11,14]].values[0].mean()),inplace=True)
df2['WIFI_&_CONNECTIVITY'].iloc[1:2,].fillna(round(df2.iloc[1:2,[8,9,10,11,14]].values[0].mean()),inplace=True)
df2['WIFI_&_CONNECTIVITY'].iloc[2:3,].fillna(round(df2.iloc[2:3,[8,9,10,11,14]].values[0].mean()),inplace=True)
df2['WIFI_&_CONNECTIVITY'].iloc[3:4,].fillna(round(df2.iloc[3:4,[8,9,10,11,14]].values[0].mean()),inplace=True)

# ------------------------------------------ fonksiyonlastırma ----------------------------------

def runner():
    # Datasetindeki başlıkları düzeltme işlemi
    df.columns = [col.replace(" ", "_").upper() for col in df.columns]

    # OVERALL_RATING değişkenindeki n`i 10 ile değiştirme
    df['OVERALL_RATING'] = df['OVERALL_RATING'].replace('n', 10)
    df['OVERALL_RATING'] = df['OVERALL_RATING'].astype(int)

    # Tarihlerin düzeltilmesi
    df['REVIEW_DATE'] = [col.replace("nd", "") if "nd" in col else col for col in df['REVIEW_DATE'].values]
    df['REVIEW_DATE'] = [col.replace("th", "") if "th" in col else col for col in df['REVIEW_DATE'].values]
    df['REVIEW_DATE'] = [col.replace("st", "") if "st" in col else col for col in df['REVIEW_DATE'].values]
    df['REVIEW_DATE'] = [col.replace("rd", "") if "rd" in col else col for col in df['REVIEW_DATE'].values]

    df['REVIEW_DATE'] = [col.replace("August", "Augu") if "August" in col else col for col in df['REVIEW_DATE'].values]
    df['REVIEW_DATE'] = [col.replace("Augu", "August") if "Augu" in col else col for col in df['REVIEW_DATE'].values]

    df['REVIEW_DATE'] = pd.to_datetime(df['REVIEW_DATE'])

    # # Tarihlerden ay ve yıl bilgisinin çekilmesi
    df['MONTH'] = (df['REVIEW_DATE']).dt.month
    df['YEAR'] = (df['REVIEW_DATE']).dt.year

    # Datasetindeki ilgisiz değişkenleri silme
    df.drop(["AIRCRAFT","ROUTE","DATE_FLOWN"], axis=1, inplace=True)

    # OVERALL_RATING değişkenindeki n`i 10 ile değiştirme
    df['OVERALL_RATING'] = df['OVERALL_RATING'].replace('n', 10).astype(int)
    df["SEAT_COMFORT"] = df["SEAT_COMFORT"].apply(lambda x: x + 1.0 if x == 0.0 else x)
    df["CABIN_STAFF_SERVICE"] = df["CABIN_STAFF_SERVICE"].apply(lambda x: x + 1.0 if x == 0.0 else x)
    df["FOOD_&_BEVERAGES"] = df["FOOD_&_BEVERAGES"].apply(lambda x: x + 1.0 if x == 0.0 else x)
    df["INFLIGHT_ENTERTAINMENT"] = df["INFLIGHT_ENTERTAINMENT"].apply(lambda x: x + 1.0 if x == 0.0 else x)
    df["WIFI_&_CONNECTIVITY"] = df["WIFI_&_CONNECTIVITY"].apply(lambda x: x + 1.0 if x == 0.0 else x)
    df["VALUE_FOR_MONEY"] = df["VALUE_FOR_MONEY"].apply(lambda x: x + 1.0 if x == 0.0 else x)

    df2 = df.copy()
    le = LabelEncoder()
    df2["RECOMMENDED"] = le.fit_transform(df2["RECOMMENDED"])

    # Yanlıs verilen 10 puanların silinmesi - 666 tane
    # Puanlama verilerindeki büyük boslukların birlikte gorulmesine ragmen 10 verilmesi durumu.
    # Recommend ve Overall_Rating arasındaki korelasyonda iyileşme gözlemlenmiştir.
    veri2 = df2.drop(["AIRLINE_NAME", "REVIEW_TITLE", "REVIEW_DATE", "VERIFIED", "TYPE_OF_TRAVELLER", "REVIEW", "SEAT_TYPE"], axis=1)
    veri2 = veri2.loc[veri2["OVERALL_RATING"] == 10]
    veri2 = veri2.drop(veri2.dropna(thresh=7).index, axis=0)  # gecersiz olanlar
    df2.drop(veri2.loc[veri2["OVERALL_RATING"] == 10].index, axis=0, inplace=True)  # veri setinden cıkarılması

    # Forward-fill (fill missing values with the previous valid value)
    df2['TYPE_OF_TRAVELLER'] = df2['TYPE_OF_TRAVELLER'].fillna(method='ffill')
    df2['SEAT_TYPE'] = df2['SEAT_TYPE'].fillna(method='ffill')

    mean_imputer = SimpleImputer(strategy='most_frequent')
    df2["SEAT_COMFORT"] = mean_imputer.fit_transform(df2[["SEAT_COMFORT"]])

    df2['VALUE_FOR_MONEY'] = df2['VALUE_FOR_MONEY'].fillna(method='ffill')
    df2['WIFI_&_CONNECTIVITY'] = df2['WIFI_&_CONNECTIVITY'].fillna(method='ffill')
    df2['INFLIGHT_ENTERTAINMENT'] = df2['INFLIGHT_ENTERTAINMENT'].fillna(method='ffill')
    df2['GROUND_SERVICE'] = df2['GROUND_SERVICE'].fillna(method='ffill')
    df2['FOOD_&_BEVERAGES'] = df2['FOOD_&_BEVERAGES'].fillna(method='ffill')
    df2['CABIN_STAFF_SERVICE'] = df2['CABIN_STAFF_SERVICE'].fillna(method='ffill')

    for i in range(4):
        df2['INFLIGHT_ENTERTAINMENT'].iloc[i:i+1, ].fillna(round(df2.iloc[i:i+1, [9, 10, 11, 12, 15]].values[0].mean()),inplace=True)
        df2['WIFI_&_CONNECTIVITY'].iloc[i:i+1,].fillna(round(df2.iloc[i:i+1,[9, 10, 11, 12, 15]].values[0].mean()),inplace=True)
    return df2

df2 = runner()

df2.isnull().sum()
df2.info()
df6.head()

# ------------------------------------------------------ EDA -------------------------------------------------------

# --------------------------------------------------- Creating NPS Data -------------------------------------------
df6=df2.copy()
#creating a new collumn with the NPS profile of each client, to facilitate calculating NPS
def define_profile(score):
    if score <= 6:
        return 'Detractor'
    elif score <= 8:
        return 'Passive'
    else:
        return 'Promoter'

df6['PROFILE'] = df6['OVERALL_RATING'].apply(define_profile)

# ---------------------------------- Percentage of Profiles by Seat Type -------------------------------------------

# calculating the % of each profile by sear type
profile_percentage = df6.groupby(['SEAT_TYPE', 'PROFILE']).size() / df6.groupby('SEAT_TYPE').size() * 100
profile_percentage = profile_percentage.unstack().fillna(0)

# defining colors for profiles
colors = {'Promoter': 'green', 'Passive': 'sandybrown', 'Detractor': 'red'}

# ploting the stacked bar chart
ax = profile_percentage.plot(kind='bar', stacked=True, color=[colors[col] for col in profile_percentage.columns])

plt.xlabel('Seat Type')
plt.ylabel('Percentage')
plt.title('Percentage of Profiles by Seat Type')

# displaying legend with custom labels
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels], labels=labels)

# adding labels with percentage values on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=10, color='white')


plt.ylim(0, 150)
plt.tight_layout()
plt.show()


# ----------------------------------------------- NPS over time -----------------------------------------------

dfeco['NPS'].values
# Economy Class
dfeco= df6[df6['SEAT_TYPE'] == 'Economy Class']
dfeco = dfeco.groupby(['YEAR','PROFILE'])['UNNAMED:_0'].count()
dfeco = dfeco.unstack()
dfeco['TOTAL'] = dfeco.sum(axis=1)
dfeco['NPS'] = (dfeco['Promoter']/dfeco['TOTAL'] - dfeco['Detractor']/dfeco['TOTAL'])*100
dfeco = dfeco.drop(columns = ['Detractor','Passive','Promoter','TOTAL'])
dfeco.reset_index(inplace=True)

sns.lineplot(x='YEAR',y='NPS',color='black',marker='o',markerfacecolor='limegreen', markersize=8,data=dfeco).set_title('NPS for Economy Class over time')


# Business Class
dfbus= df6[df6['SEAT_TYPE'] == 'Business Class']
dfbus = dfbus.groupby(['YEAR','PROFILE'])['UNNAMED:_0'].count()
dfbus = dfbus.unstack()
dfbus['TOTAL'] = dfbus.sum(axis=1)
dfbus['NPS'] = (dfbus['Promoter']/dfbus['TOTAL'] - dfbus['Detractor']/dfbus['TOTAL'])*100
dfbus = dfbus.drop(columns = ['Detractor','Passive','Promoter','TOTAL'])
dfbus.reset_index(inplace=True)

sns.lineplot(x='YEAR',y='NPS',color='black',marker='o',markerfacecolor='limegreen', markersize=8,data=dfbus).set_title('NPS for Business Class over time')

# Premium Economy
dfpre= df6[df6['SEAT_TYPE'] == 'Premium Economy']
dfpre = dfpre.groupby(['YEAR','PROFILE'])['UNNAMED:_0'].count()
dfpre = dfpre.unstack()
dfpre['TOTAL'] = dfpre.sum(axis=1)
dfpre['NPS'] = (dfpre['Promoter']/dfpre['TOTAL'] - dfpre['Detractor']/dfpre['TOTAL'])*100
dfpre = dfpre.drop(columns = ['Detractor','Passive','Promoter','TOTAL'])
dfpre.reset_index(inplace=True)

sns.lineplot(x='YEAR',y='NPS',color='black',marker='o',markerfacecolor='limegreen', markersize=8,data=dfpre).set_title('NPS for Premium Economy over time')

# First Class
dffis= df6[df6['SEAT_TYPE'] == 'First Class']
dffis = dffis.groupby(['YEAR','PROFILE'])['UNNAMED:_0'].count()
dffis = dffis.unstack()
dffis['TOTAL'] = dffis.sum(axis=1)
dffis['NPS'] = (dffis['Promoter']/dffis['TOTAL'] - dffis['Detractor']/dffis['TOTAL'])*100
dffis = dffis.drop(columns = ['Detractor','Passive','Promoter','TOTAL'])
dffis.reset_index(inplace=True)

sns.lineplot(x='YEAR',y='NPS',color='black',marker='o',markerfacecolor='limegreen', markersize=8,data=dffis).set_title('NPS for First Class over time')



fig, axs = plt.subplots(2,2,figsize=(12,8))

sns.lineplot(x='YEAR',y='NPS',color='black',marker='o',markerfacecolor='limegreen', markersize=6,data=dfeco,ax=axs[0][0])
axs[0][0].set_title('NPS for Economy Class over time')
sns.lineplot(x='YEAR',y='NPS',color='#4F2303',marker='o',markerfacecolor='limegreen', markersize=6,data=dfbus,ax=axs[0][1])
axs[0][1].set_title('NPS for Business Class over time')
sns.lineplot(x='YEAR',y='NPS',color='#001241',marker='o',markerfacecolor='limegreen', markersize=6,data=dfpre,ax=axs[1][0])
axs[1][0].set_title('NPS for Premium Economy over time')
sns.lineplot(x='YEAR',y='NPS',color='red',marker='o',markerfacecolor='limegreen', markersize=6,data=dffis,ax=axs[1][1])
axs[1][1].set_title('NPS for First Class over time')
plt.tight_layout()


# ------------------------------------------ Recommended by Seat Type -------------------------------------------

df6.groupby(["RECOMMENDED","SEAT_TYPE"])["RECOMMENDED"].count()
fig, axs = plt.subplots(figsize=(10, 7))
axs = sns.countplot(x='SEAT_TYPE',hue='RECOMMENDED',data=df6,palette=["red","green"]).set_title('RECOMMENDED')

# ------------------------------------------------ Rare Encoding İşlemi ------------------------------------------
pd.DataFrame({'AIRLINE_NAME': df['AIRLINE_NAME'].value_counts(),"Ratio": 100 * df['AIRLINE_NAME'].value_counts()/len(df)})

# --------------------------------------- havayolu şirketlerine göre recommend sayıları --------------------------------------------------
df8=df2.copy()
frequencies = df8['AIRLINE_NAME'].value_counts(ascending=True)
mapping = df8['AIRLINE_NAME'].map(frequencies)
df8['AIRLINE_NAME'].mask(mapping<100,"Other Airlines",inplace=True)

df8['AIRLINE_NAME'].value_counts()
sns.barplot(df7['AIRLINE_NAME'].value_counts(),)


df8.groupby(['AIRLINE_NAME',"RECOMMENDED"])["AIRLINE_NAME"].count()

df6.groupby(["AIRLINE_NAME","RECOMMENDED"]).agg({"AIRLINE_NAME":"count"})

# -----------------------------------------------------------------------------------------

# havayolu şirketlerinin  tavsiyelere gore evet ve hayır olarak ayrılması
liste1=[]
liste2=[]
liste3=[]
for i in df9["AIRLINE_NAME"].unique():

    if i == "Other Airlines":
        continue
        print(i)
    else:
        print(i)
        yes = (df9.loc[(df9["AIRLINE_NAME"] == str(i))]["RECOMMENDED"]).sum()
        no = 100 - (df9.loc[(df9["AIRLINE_NAME"] == str(i))]["RECOMMENDED"].sum())
        liste1.append(yes)
        liste2.append(no)
        liste3.append(i)
df_air = pd.DataFrame({"Yes" :liste1, "No" : liste2}, index = liste3)

df_air.sort_values(by="Yes",ascending=False)

df_air2 = df_air.copy()
df_air2.reset_index(names="AIRLINE_NAME",inplace=True)

sns.barplot(y="Yes",x="AIRLINE_NAME")

df_air2.groupby("AIRLINE_NAME")[["Yes","No"]].count()
# -----------------------------------------------------------------------------------------

# ------------------------------------ Özellik Mühendisliği --------------------------------
df3 = df2.copy()
# Sentiment Intensity Analyzer

analyzer = SentimentIntensityAnalyzer()
df3.head()

def analyze_sentiment_vader(text):
    sentiment_scores = analyzer.polarity_scores(text)['compound']
    if sentiment_scores >= 0.05:
        return "Positive"
    elif sentiment_scores <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# Apply sentiment analysis to your DataFrame
df3['STMNT_REVIEW_TITLE'] = df3['REVIEW_TITLE'].apply(analyze_sentiment_vader)
df3['STMNT_REVIEW'] = df3['REVIEW'].apply(analyze_sentiment_vader)


df3.corr(numeric_only=True)  # 0.707870 - 0.633572
df3.corr(numeric_only=True,method = "kendall") #  0.628520 - 0.550592
df3.corr(numeric_only=True,method = "spearman") # 0.699010 - 0.604651



# ------------------------------------- Encoding ----------------------------------------------

df3 = pd.get_dummies(df3,columns=["STMNT_REVIEW_TITLE","STMNT_REVIEW","TYPE_OF_TRAVELLER","SEAT_TYPE"],drop_first=True,dtype=int)
df3.head()

# ---------------------------------------------- MODELLEME ---------------------------------------------------
df4 = df3.copy()
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, precision_score, recall_score, f1_score,accuracy_score,roc_auc_score

x = df4.drop(["AIRLINE_NAME","REVIEW_TITLE","REVIEW_DATE","VERIFIED","REVIEW","INFLIGHT_ENTERTAINMENT","WIFI_&_CONNECTIVITY","RECOMMENDED","MONTH","YEAR","UNNAMED:_0"],axis=1)
y = df4[["RECOMMENDED"]]

x.head()
y.head()

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.30)

###########################
# Random Forest
###########################
rf_model = RandomForestClassifier()
rf_model.fit(x_train,y_train)

y_pred = rf_model.predict(x_test)
accuracy_score(y_pred,y_test) # 0.96

cm = confusion_matrix(y_pred,y_test)
acc = round(accuracy_score(y_pred,y_test),2)

print(classification_report(y_pred,y_test))

sns.heatmap(cm,annot=True, fmt=".0f",cmap = 'Blues')
plt.xlabel("y_pred")
plt.ylabel("y")
plt.title(f"Accuracy Score {acc}",size=10)

###########################
# XGBoost
###########################
xgboost_model = XGBClassifier()
xgboost_model.fit(x_train,y_train)

y_pred = xgboost_model.predict(x_test)
accuracy_score(y_pred,y_test) # 0.9613431509457916

cm = confusion_matrix(y_pred,y_test)
acc = round(accuracy_score(y_pred,y_test),2)

print(classification_report(y_pred,y_test))
printer()

sns.heatmap(cm,annot=True, fmt=".0f",cmap = 'Blues')
plt.xlabel("y_pred")
plt.ylabel("y")
plt.title(f"Accuracy Score {acc}",size=10)


###########################
# LightGBM
###########################
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(x_train,y_train)

y_pred = lgb_model.predict(x_test)
accuracy_score(y_pred,y_test) # 0.963

cm = confusion_matrix(y_pred,y_test)
acc = round(accuracy_score(y_pred,y_test),2)

print(classification_report(y_pred,y_test))
printer()
roc_auc_score(y_pred, y_test)

sns.heatmap(cm,annot=True, fmt=".0f",cmap = 'Blues')
plt.xlabel("y_pred")
plt.ylabel("y")
plt.title(f"Accuracy Score {acc}",size=10)

# ------------------------------------------ fonksiyonlastırma ----------------------------------
def model_call(mod,plot=False):

    model = mod
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(f"Accuracy:  {(accuracy_score(y_pred,y_test)):.2f}\nPrecision: {(precision_score(y_pred,y_test)):.2f}\nRecall:    {(recall_score(y_pred,y_test)):.2f}\nF1 score:  {(f1_score(y_pred,y_test)):.2f}\nROC:       {(roc_auc_score(y_pred,y_test)):.2f}")
    print(classification_report(y_pred,y_test))
    cm = confusion_matrix(y_pred, y_test)
    if plot:
        sns.heatmap(cm, annot=True, fmt=".0f", cmap='Blues')
        plt.xlabel("y_pred")
        plt.ylabel("y")
        plt.title(f"Accuracy Score {acc}", size=10)

model_call(XGBClassifier(),plot=True)
# ------------------------------------------ H-P-Optimizasyonu ----------------------------------
from sklearn.model_selection import GridSearchCV

# RandomForestClassifier
RandomForestClassifier().get_params()

rf_model = RandomForestClassifier()
rf_params = {"max_depth":[5,8],
             "max_features":[4,8,12,"auto"],
             "min_samples_split":[2,5,8,15,20],
             "n_estimators":[100,200,500]}

rf_best_grid = GridSearchCV(rf_model,rf_params,cv=5,n_jobs=-1,verbose=True).fit(x_train,y_train)
rf_best_grid.best_params_
"""
{'max_depth': 8,
 'max_features': 8,
 'min_samples_split': 5,
 'n_estimators': 500}
"""

# rf_final = RandomForestClassifier().set_params(**rf_best_grid.best_params_).fit(x,y)
rf_final = RandomForestClassifier(max_depth= 8,max_features=8,min_samples_split= 5,n_estimators=500).fit(x_train,y_train)
y_pred = rf_final.predict(x_test)

print(f"Accuracy:  {(accuracy_score(y_pred, y_test)):.2f}\nPrecision: {(precision_score(y_pred, y_test)):.2f}\nRecall:    {(recall_score(y_pred, y_test)):.2f}\nF1 score:  {(f1_score(y_pred, y_test)):.2f}\nROC:       {(roc_auc_score(y_pred, y_test)):.2f}")
accuracy_score(y_pred, y_test)

cm=classification_report(y_pred,y_test)

# -------------------------------------------------------------------------------------------------------
# XGBoost
XGBClassifier().get_params()

xgboost_model = XGBClassifier()
xgboost_params = {"max_depth":[5,6,8],
             "learning_rate":[0.1,0.01],
             "colsample_bytree":[0.7,1],
             "n_estimators":[100,200,300,500]}

xgboost_best_grid = GridSearchCV(xgboost_model,xgboost_params,cv=5,n_jobs=-1,verbose=0).fit(x_train,y_train)
xgboost_best_grid.best_params_
"""
{'colsample_bytree': 0.7,
 'learning_rate': 0.01,
 'max_depth': 8,
 'n_estimators': 200}
"""

# xgboost_final = XGBClassifier().set_params(**xgboost_best_grid.best_params_).fit(x,y)
xgboost_final = XGBClassifier(colsample_bytree= 0.7,learning_rate=0.01,max_depth= 8,n_estimators=200).fit(x_train,y_train)
y_pred = xgboost_final.predict(x_test)

print(f"Accuracy:  {(accuracy_score(y_pred, y_test)):.2f}\nPrecision: {(precision_score(y_pred, y_test)):.2f}\nRecall:    {(recall_score(y_pred, y_test)):.2f}\nF1 score:  {(f1_score(y_pred, y_test)):.2f}\nROC:       {(roc_auc_score(y_pred, y_test)):.2f}")

cm=classification_report(y_pred,y_test)

# -------------------------------------------------------------------------------------------------------
# LightGBM
lgb.LGBMClassifier().get_params()

lgb_model = lgb.LGBMClassifier()
lgbm_params = {"learning_rate":[0.1,0.01],
             "colsample_bytree":[0.7,1],
             "n_estimators":[100,200,300,500]
               }

lgbm_best_grid = GridSearchCV(lgb_model,lgbm_params,cv=5,n_jobs=-1,verbose=0).fit(x_train,y_train)
lgbm_best_grid.best_params_
"""
{'colsample_bytree': 0.7,
 'learning_rate': 0.01,
 'n_estimators': 500}
"""

# lgbm_final = lgb.LGBMClassifier(verbose=-1).set_params(**lgbm_best_grid.best_params_).fit(x,y)
lgbm_final = lgb.LGBMClassifier(colsample_bytree= 0.7,learning_rate=0.01,n_estimators=500).fit(x_train,y_train)
y_pred = lgbm_final.predict(x_test)

print(f"Accuracy:  {(accuracy_score(y_pred, y_test)):.2f}\nPrecision: {(precision_score(y_pred, y_test)):.2f}\nRecall:    {(recall_score(y_pred, y_test)):.2f}\nF1 score:  {(f1_score(y_pred, y_test)):.2f}\nROC:       {(roc_auc_score(y_pred, y_test)):.2f}")

cm=classification_report(y_pred,y_test)
# XGBClassifier() lgb.LGBMClassifier() RandomForestClassifier()



# --------------------------------------- Özellik Önemi----------------------------------------------

def plot_importance(model,features,num=len(x_train),save=False):
    feature_imp = pd.DataFrame({"Value":model.feature_importances_,
                                "Feature":features.columns})
    plt.figure(figsize=(6,6))
    sns.set(font_scale=1)
    sns.barplot(x="Value",y="Feature",data=feature_imp.sort_values(by="Value",ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()

    if save:
        plt.savefig({str(model)}+"importances.png")

# plot_importance(xgboost_final,x_train)

# ------------------------------------------ HPO sonrası fonksiyonlastırma ----------------------------------
df4 = df3.copy()
x = df4.drop(["AIRLINE_NAME","REVIEW_TITLE","REVIEW_DATE","VERIFIED","REVIEW","INFLIGHT_ENTERTAINMENT","WIFI_&_CONNECTIVITY","RECOMMENDED","MONTH","YEAR","UNNAMED:_0"],axis=1)
y = df4[["RECOMMENDED"]]

x.head()
y.head()

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.30)

classifiers = [("RandomForestClassifier",RandomForestClassifier(),RandomForestClassifier(max_depth= 8,max_features=8,min_samples_split= 5,n_estimators=500)),
               ("XGBClassifier",XGBClassifier(),XGBClassifier(colsample_bytree= 0.7,learning_rate=0.01,max_depth= 8,n_estimators=200)),
               ("LGBMClassifier",lgb.LGBMClassifier(verbose=-1),lgb.LGBMClassifier(colsample_bytree= 0.7,learning_rate=0.01,n_estimators=500,verbose=-1))]

# classifiers2 = [("RandomForestClassifier",RandomForestClassifier(max_depth= 8,max_features=8,min_samples_split= 5,n_estimators=500)),
#                 ("XGBClassifier",XGBClassifier(colsample_bytree= 0.7,learning_rate=0.01,max_depth= 8,n_estimators=200)),
#                 ("LGBMClassifier",lgb.LGBMClassifier(colsample_bytree= 0.7,learning_rate=0.01,n_estimators=500,verbose=-1))]


df2.head()
def model_call(x_train, x_test, y_train, y_test,mod_opt,plot=False):

    pl= []
    i = 0
    if mod_opt == False:
        for name,mod1,mod2 in classifiers:

            model = mod1.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            acc,pre,rec,f1,auc = calculate(y_pred)
            pl.append([acc,pre,rec,f1,auc])

            print(f"{name.upper()}\nBefore: Accuracy: {pl[i][0]} Precision: {pl[i][1]} Recall: {pl[i][2]} F1 score: {pl[i][3]} ROC:{pl[i][4]}")
            i = i + 1
    pl2 = []
    if mod_opt:

        for name,mod1,mod2 in classifiers:
            model = mod1.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            acc,pre,rec,f1,auc = calculate(y_pred)
            pl.append([acc,pre,rec,f1,auc])

            final = mod2.fit(x_train,y_train)
            y_pred_final = final.predict(x_test)
            acc,pre,rec,f1,auc = calculate(y_pred_final)
            pl2.append([acc,pre,rec,f1,auc])

            print(f"{name.upper()}\nBefore: Accuracy: {pl[i][0]:.3f} Precision: {pl[i][1]:.3f} Recall: {pl[i][2]:.3f} F1 score: {pl[i][3]:.3f} ROC:{pl[i][4]:.3f}\nAfter:  Accuracy: {pl2[i][0]:.3f} Precision: {pl2[i][1]:.3f} Recall: {pl2[i][2]:.3f} F1 score: {pl2[i][3]:.3f} ROC:{pl2[i][4]:.3f}")

            print(f"{name.upper()}\nBefore: Accuracy: {pl[i][0]:.2f} Precision: {pl[i][1]:.2f} Recall: {pl[i][2]:.2f} F1 score: {pl[i][3]:.2f} ROC:{pl[i][4]:.2f}\nAfter:  Accuracy: {pl2[i][0]:.2f} Precision: {pl2[i][1]:.2f} Recall: {pl2[i][2]:.2f} F1 score: {pl2[i][3]:.2f} ROC:{pl2[i][4]:.2f}")

            i = i + 1

    if plot:
        plot_importance(model, x_train)
        plot_importance(final, x_train)


model_call(x_train, x_test, y_train, y_test,mod_opt=True)
model_call(x_train, x_test, y_train, y_test,mod_opt=False)
# XGBClassifier() lgb.LGBMClassifier() RandomForestClassifier()

def calculate(y):
    acc = accuracy_score(y,y_test)
    pre = precision_score(y,y_test)
    rec = recall_score(y,y_test)
    f1 = f1_score(y,y_test)
    auc = roc_auc_score(y,y_test)
    return acc,pre,rec,f1,auc

print("hello")