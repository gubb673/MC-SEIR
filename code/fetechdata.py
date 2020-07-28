import requests
import pandas as pd
import io
import time
import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
from cov2020model import EpidUnit


class FetchCOVIDData:

    def __init__(self):
        self.rawCOVID_19DataURL = 'https://raw.githubusercontent.com/BlankerL/DXY-COVID-19-Data/master/csv/DXYArea.csv'
        self.rawCOVID_19Data = None

        self.tcCityMigration = None
        self.tcProvinceMigration = None

    def LoadData(self, newest=False):
        if (newest):
            self.LoadNewestCOVIDData()
        else:
            self.LoadCOVIDData()
        self.LoadPOPData()
        self.LoadTencentData()
        self.LoadJHUData()
        self.LoadLaiData()

    def LoadLaiData(self):
        self.LaiData = LaiData();

    def LoadJHUData(self):
        self.JHUData = JHUdata()

    def LoadTencentData(self):
        self.citytc = Tencentdata()
        self.provincetc = Tencentdata(path2='province')

    def LoadPOPData(self, path='pop.csv'):
        # poplation data provided by jixuan cai
        self.population = pd.read_csv('pop.csv')

    def LoadNewestCOVIDData(self):
        try:
            req = requests.request('get', self.rawCOVID_19DataURL)
            rawCOVID_19Data = pd.read_csv(io.StringIO(req.content.decode('utf-8')))
            rawCOVID_19Data.to_csv('rawcoviddata/covid19at' + str(int(time.time())) + '.csv')
            self.rawCOVID_19Data = rawCOVID_19Data
        except Exception as e:
            print(str(e))
        self.CreateRealTimeInfected()
        print('newest coviddata request sucessful')

    def CreateRealTimeInfected(self):
        self.rawCOVID_19Data['province_nowCount'] = self.rawCOVID_19Data['province_confirmedCount'] - \
                                                    self.rawCOVID_19Data[
                                                        'province_deadCount'] - self.rawCOVID_19Data[
                                                        'province_curedCount']
        self.rawCOVID_19Data['city_nowCount'] = self.rawCOVID_19Data['city_confirmedCount'] - self.rawCOVID_19Data[
            'city_deadCount'] - self.rawCOVID_19Data['city_curedCount']

    def LoadCOVIDData(self, filename='rawcoviddata/covid19at1585640085.csv'):
        self.rawCOVID_19Data = pd.read_csv(filename)
        self.CreateRealTimeInfected()

    def GetDayJoinedSeries(self, cityname='Wuhan', provinceid=11, usecity=True):
        if (self.rawCOVID_19Data is None):
            print('pls initialize data with LoadData or LoadNewestData')
            return
        if usecity:
            targetcity = self.rawCOVID_19Data[self.rawCOVID_19Data['cityEnglishName'] == cityname]
            targetcityrecorddate = list(
                map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime('%Y-%m-%d'),
                    targetcity['updateTime']));
            targetcity.insert(0, 'day', targetcityrecorddate)
            return targetcity.groupby('day').mean()
        else:
            targetcity = self.rawCOVID_19Data[self.rawCOVID_19Data['city_zipCode'] // 10000 == provinceid]
            targetcityrecorddate = list(
                map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime('%Y-%m-%d'),
                    targetcity['updateTime']));
            targetcity.insert(0, 'day', targetcityrecorddate)
            return targetcity.groupby(['day', 'province_zipCode']).mean()


class LaiData:
    def __init__(self):

        self.wuhancasedata = pd.read_excel('Cases by onset date_wuhan_province_13Feb.xlsx', sheet_name='wuhan case')
        self.provincecase = pd.read_excel('Cases by onset date_wuhan_province_13Feb.xlsx', sheet_name='province case')
        self.startdate = self.wuhancasedata.iloc[1,0] # 20191211
        self.enddate =  self.wuhancasedata.iloc[-4,0]#20200210
        self.fastseries = self.wuhancasedata.iloc[1:-4,1]




class JHUdata:
    def __init__(self):
        self.datapath = 'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
        self.data = pd.read_csv(self.datapath)
        self.death = pd.read_csv(
            'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
        self.rec = pd.read_csv(
            'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

    def GetSeriesByName(self, country='China', province='Shanghai'):
        return self.data[(self.data['Country/Region'] == country) & (self.data['Province/State'] == province)].iloc[0,
               4:] - self.death[
                         (self.death['Country/Region'] == country) & (self.death['Province/State'] == province)].iloc[0,
                     4:] - self.rec[(self.rec['Country/Region'] == country) & (
                self.rec['Province/State'] == province)].iloc[0, 4:]


class Tencentdata:

    def __init__(self, path1="OneDrive-2020-04-01/", path2="city"):
        # self.contactindex = pd.read_csv(glob.glob(path1 + path2 + "/contact_index_" + path2 + ".csv")[0], header=None)
        # self.contactindex = pd.read_csv('half_person_active.csv')
        # self.contactindex = pd.read_csv('origin.csv') #data.pickle
        self.contactindex = pd.read_csv('raw0.csv') #data raw0 pickle
        self.migration = pd.read_csv(glob.glob(path1 + path2 + '/migration.csv')[0])
        meta = pd.read_csv(path1 + path2 + '/contact_index_meta.txt', header=None)

        self.startdate = datetime.datetime.strptime(str(meta.iloc[1, 0]), "%Y%m%d").date()
        self.citylist = meta.iloc[0, :]
        self.zeroday = self.startdate
        self.migration['date'] = self.migration['date'].apply(
            lambda x: (datetime.datetime.strptime(str(x), "%Y%m%d").date() - self.zeroday).days)
        self.migration['from'] = self.migration['from'].astype('int')
        self.migration['to'].astype('int')
# fed = FetchCOVIDData()
# fed.LoadData()
#
# shd = fed.GetDayJoinedSeriesForOneCity(cityname='Shaoxing')
# plt.plot(shd.city_nowCount)
#
# #total pop
# #start infected
# #start time period
#
# #return r2 estimated
# #
