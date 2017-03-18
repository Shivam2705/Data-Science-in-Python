# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 11:29:18 2016

@author: 117053
"""
# Assig-3_ ans-1 (Data Cleaning, rename, replace, merge, NA removal, value addition)
def answer_one():
    import pandas as pd
    import numpy as np

    y = pd.ExcelFile('Energy Indicators.xls')
    energy = y.parse(skiprows=17,skip_footer=(38))
    energy = energy[['Unnamed: 1','Petajoules','Gigajoules','%']]
    energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']] =  energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']].replace('...',np.NaN).apply(pd.to_numeric)
    energy['Country'] = energy['Country'].replace({'China, Hong Kong Special Administrative Region':'Hong Kong','United Kingdom of Great Britain and Northern Ireland':'United Kingdom','Republic of Korea':'South Korea','United States of America':'United States','Iran (Islamic Republic of)':'Iran'})
    energy['Country'] = energy['Country'].str.replace(r" \(.*\)","")
    energy['Energy Supply'] = energy['Energy Supply']*1000000
    GDP = pd.read_csv('world_bank.csv',skiprows=4)
    GDP['Country Name'] = GDP['Country Name'].replace('Korea, Rep.','South Korea')
    GDP['Country Name'] = GDP['Country Name'].replace('Iran, Islamic Rep.','Iran')
    GDP['Country Name'] = GDP['Country Name'].replace('Hong Kong SAR, China','Hong Kong')
    GDP = GDP[['Country Name','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
    GDP.columns = ['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']
    ScimEn = pd.read_excel(io='scimagojr-3.xlsx')
    ScimEn_m = ScimEn[:15]
    X = pd.merge(ScimEn_m,energy,how='inner',left_on='Country',right_on='Country')
    W = pd.merge(X,GDP,how='inner',left_on='Country',right_on='Country')
    W = W.set_index('Country')
    
    return W

    
    
    
    
    
    #Ans-2
    def answer_two():
        return 156
        
        
    #Ans-3 (Average across mulitple columns calculation, sorting,rename)
    def answer_three():
    Fifteen_Top_Rank = answer_one()
    avgGDP = Fifteen_Top_Rank[['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']].mean(axis=1).rename('avgGDP').sort_values(ascending=False)
    return avgGDP

    answer_three()
    
    
    
    
    
    # Ans-4 (operation on two columns of df : substraction, to read numeric value of series)
    
    
def answer_four():
    import pandas as pd
    Top15 = answer_one()
    ans = Top15[Top15['Rank'] == 4]['2015'] - Top15[Top15['Rank'] == 4]['2006']
    return pd.to_numeric(ans)[0]

answer_four()



# Ans-5 (average of rows)

def answer_five():
    Top15 = answer_one()
    mean_energy =Top15[['Energy Supply per Capita']].mean()
    return mean_energy
answer_five()




#Ans-6 ( sort by a column, return top 1st with value and identity)
def answer_six():
    Top15 = answer_one()
    country_maximum_Renewable=Top15.sort_values(by='% Renewable',ascending=False)
    return (country_maximum_Renewable.index.tolist()[0],country_maximum_Renewable['% Renewable'].tolist()[0])
answer_six()    






# Ans-7 (find ratio of two columns then sort and then return top one)

def answer_seven():
    Top15 = answer_one()
    Top15['Ratio']=(Top15['Self-citations']/Top15['Citations']).sort_values(ascending=False)
    return (Top15.index.tolist()[0],Top15['Ratio'].tolist()[0])
answer_seven()


#or 
def answer_seven():
    Top15 = answer_one()
    Top15['Citation Ratio'] = Top15['Self-citations']/Top15['Citations']
    ans = Top15[Top15['Citation Ratio'] == max(Top15['Citation Ratio'])]
    return (ans.index.tolist()[0],ans['Citation Ratio'].tolist()[0])

answer_seven()




#Ans-8 (dividing two columns make new col and sort tht)
def answer_eight():
    Top15 = answer_one()
    Top15['Population']=(Top15['Energy Supply']/Top15['Energy Supply per Capita'])
    Top15=Top15.sort_values(by='Population',ascending=False)
    return (Top15.index.tolist()[2])
answer_eight()





#Ans-9 (Correlation between two newly created columns by operations)

def answer_nine():
    Top15 = answer_one()
    Top15['Population']=(Top15['Energy Supply']/Top15['Energy Supply per Capita'])
    Top15['Cit_per_Capita']=Top15['Citable documents']/Top15['Population']
    x=Top15['Cit_per_Capita'].corr(Top15['Energy Supply per Capita'])
    return x
answer_nine()



#Ans-10 (Conditional operator on a column to flag in new column)

def answer_ten():
    Top15 = answer_one()
    Top15['HighRenew'] = [1 if x >= Top15['% Renewable'].median() else 0 for x in Top15['% Renewable']]
    return Top15['HighRenew']
answer_ten()
    
    
    
    
#Ans-11 (resetting of index: from one index to other) (Group by and mean/sum/std dev/)
    
def answer_eleven():
    import pandas as pd
    import numpy as np
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    Top15 = answer_one()
    Top15['Population'] = (Top15['Energy Supply'] / Top15['Energy Supply per Capita']).astype(float)
    Top15 = Top15.reset_index()
    Top15['Continent'] = [ContinentDict[country] for country in Top15['Country']]
    y = Top15.set_index('Continent').groupby(level=0)['Population'].agg({'size': np.size, 'sum': np.sum, 'mean': np.mean,'std': np.std})
    y = y[['size', 'sum', 'mean', 'std']]
    return y
    
    
    
    
    #Ans-12 (Making bins)
    
    def answer_twelve():
    import pandas as pd
    import numpy as np
    Top15 = answer_one()
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    Top15 = Top15.reset_index()
    Top15['Continent'] = [ContinentDict[country] for country in Top15['Country']]
    Top15['bins'] = pd.cut(Top15['% Renewable'],5)
    return Top15.groupby(['Continent','bins']).size()

answer_twelve()




#Ans-13 (adding comma in large number converting into string)

def answer_thirteen():
    import locale
    import pandas as pd
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
    Top15 = answer_one()
    Top15['PopEst'] = (Top15['Energy Supply'] / Top15['Energy Supply per Capita']).astype(float)
    map_str = []
    for num in Top15['PopEst']:
        map_str.append(locale.format('%.2f',num,grouping=True))
    Top15['PopEst_str'] = map_str
    return Top15['PopEst_str']
    
    
    
    
    
    #Optional (BUUBLE CHART)
    
    def plot_optional():
    import matplotlib as plt
    %matplotlib inline
    Top15 = answer_one()
    ax = Top15.plot(x='Rank', y='% Renewable', kind='scatter', 
                    c=['#e41a1c','#377eb8','#e41a1c','#4daf4a','#4daf4a','#377eb8','#4daf4a','#e41a1c',
                       '#4daf4a','#e41a1c','#4daf4a','#4daf4a','#e41a1c','#dede00','#ff7f00'], 
                    xticks=range(1,16), s=6*Top15['2014']/10**10, alpha=.75, figsize=[16,6]);

    for i, txt in enumerate(Top15.index):
        ax.annotate(txt, [Top15['Rank'][i], Top15['% Renewable'][i]], ha='center')

    print("This is an example of a visualization that can be created to help understand the data. \
This is a bubble chart showing % Renewable vs. Rank. The size of the bubble corresponds to the countries' \
2014 GDP, and the color corresponds to the continent.")