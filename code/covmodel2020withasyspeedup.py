import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class EpidUnit:
    def __init__(self, Susceptible=1000, Infected=50, Exposed=0, Removed=0, AddedToday=0, Slientinfection=0,
                 rawBeta=2.64 / 6, sigma=1 / 3, gamma=1 / 7, xi=0.0,
                 exposedBeta=(2.64 / 6)*0.12,
                 slientBeta= 2.64 / (6 * 3), slientrate= 0/ 10, slientgamma=1 / 14,
                 randomsize=None,
                 currentTime=0, targetTime=100):

        self.Susceptible = Susceptible;
        self.Infected = Infected;
        self.Silentinfection = Slientinfection;
        self.Exposed = Exposed;
        self.Removed = Removed;
        self.AddedToday = AddedToday;

        self.Totalpop = self.Susceptible + self.Infected + self.Exposed + self.Removed + self.Silentinfection;

        self.rawBeta = rawBeta;

        self.beta = self.rawBeta;
        self.rawexposedBeta = exposedBeta
        self.exposedBeta = exposedBeta# asymptomatic carrier transmission rate
        self.sigma = sigma;
        self.gamma = gamma;  # recovery rate
        self.xi = xi;
        self.randomsize = randomsize

        self.slientrate = slientrate
        self.slientgamma = slientgamma
        self.rawslientbeta = slientBeta
        self.slientbeta = slientBeta
        self.currentTime = currentTime;
        self.targetTime = targetTime;


        self.effectiveContactRateHis = [];
        self.SilentinfectionHis = [];
        self.SusceptibleHis = [];
        self.InfectedHis = [];
        self.ExposedHis = [];
        self.RemovedHis = [];
        self.AddedTodayHis = [];
        self.TotalpopHis = [];
        self.gammahis = [];
        self.fromInfectedHis = [];
        self.fromExposedHis = [];
        self.fromSlientHis = [];
        self.poprecord = []

        self.fromInfected = 0;
        self.fromExposed = 0;
        self.fromSlient = 0;

        self.ExposedDetailedHis = pd.Series([],name = 'converted_date',dtype=int)
        self.ExposedDetailedHisByDay = []

        self.ExportedCasedHis = []

    def updateEpidStatusS2E(self):
        self.fromInfected = np.sum(np.random.poisson(self.beta, np.int(self.Infected)));
        self.fromExposed  = np.sum(np.random.poisson(self.exposedBeta, np.int(self.Exposed)))
        self.fromSlient = np.sum(np.random.poisson(self.slientbeta,np.int(self.Silentinfection)))




        EER = self.equivalenceExposeRatio()
        self.effectiveContactRateHis.append(EER)
        self.Exposed_td = (float(self.fromInfected) + self.fromExposed + self.fromSlient) * EER

        # https: //www.sciencedirect.com/ science / article / pii / S2214109X20300747
        #https://blog.csdn.net/absent1353/article/details/78415118

        l_mean = 5.79100406047167;
        l_sd = 5.692242718086683
        sd = np.log(1+l_sd**2/l_mean**2)
        mean = np.log(l_mean)-0.5*sd
        # np.mean(np.random.lognormal(mean, sd, 1000)), np.quantile(np.random.lognormal(mean, sd, 1000), [0.05,0.5, 0.95])
        ExposedToday = list(np.ceil(np.random.lognormal(mean,sd,int(self.Exposed_td))) + self.currentTime)
        self.ExposedDetailedHis  = self.ExposedDetailedHis.append(pd.Series(ExposedToday))


        self.Susceptible -= self.Exposed_td;
        self.Exposed += self.Exposed_td;

    def equivalenceExposeRatio(self):
        return 1 - (min(1, (self.Exposed + self.Infected + self.Removed) / self.Totalpop))



    def GetCovertedToday(self):
        return (self.ExposedDetailedHis==self.currentTime).sum()
        # num = 0;
        # for i in range(0,27):
        #     if self.ExposedDetailedHis.__len__() <= i:
        #         break
        #     cases = sum(self.ExposedDetailedHis[-(1+i)] == (i+1))
        #     num += cases
        # return num


    def updateEpidStatusE2I(self):
        # self.Convert_td = np.average(np.random.binomial(self.ExposedHis[-1], self.sigma, size=self.randomsize))
        # self.ExposedDetailedHisByDay.append()
        self.Convert_td = self.GetCovertedToday()
        self.ExposedDetailedHisByDay.append(self.Convert_td)
        Apparent = self.Convert_td * (1 - self.slientrate)
        Slient = self.Convert_td * self.slientrate

        self.AddedToday = Apparent
        # self.ExposedDetailedHis = list(filter(lambda x: x >= self.currentTime,self.ExposedDetailedHis))
        self.ExposedDetailedHis = self.ExposedDetailedHis[self.ExposedDetailedHis > self.currentTime]
        self.Exposed = self.ExposedDetailedHis.shape[0]

        # self.Exposed -= self.Convert_td;
        if self.Exposed < 0:
            print('exposed < 0')
        self.Infected += Apparent
        self.Silentinfection += Slient

    def updateEpidStatusI2R(self):
        Removed_td = np.average(np.random.binomial(self.Infected, self.gamma, size=self.randomsize))
        Removed_td_from_slient = np.average(
            np.random.binomial(self.Silentinfection, self.slientgamma, size=self.randomsize))


        self.Removed += Removed_td;
        self.Removed += Removed_td_from_slient
        self.SilentinfectionHis -= Removed_td_from_slient
        self.Infected -= Removed_td;

    def updateEpidStatusI2S(self):
        LosImu = np.average(np.random.binomial(self.Removed, self.xi, size=self.randomsize))

        self.Susceptible += LosImu;
        self.Removed -= LosImu;

    def recordStatus(self):
        self.fromInfectedHis.append(self.fromInfected);
        self.fromExposedHis.append(self.fromExposed);
        self.fromSlientHis.append(self.fromSlient);

        self.AddedTodayHis.append(self.AddedToday);  # new case today
        self.SusceptibleHis.append(self.Susceptible);
        self.TotalpopHis.append(self.Totalpop);
        self.InfectedHis.append(self.Infected);
        self.ExposedHis.append(self.Exposed);
        self.RemovedHis.append(self.Removed);
        self.gammahis.append(self.gamma);

        self.poprecord.append([self.Totalpop,self.Infected,self.Exposed,self.Removed,self.Susceptible])

    def updateEpidStatus(self, moveoutrate=0):

        self.updateEpidStatusS2E();
        self.updateEpidStatusE2I();
        self.updateEpidStatusI2R();
        self.updateEpidStatusI2S()

        if (moveoutrate != 0):
            # here the leave case will revise the updated expose infected etc
            # recorded in next time
            self.createExportcase(moveoutrate)
        self.recordStatus();
        self.currentTime += 1;

    def updatePara(self, socialconnect=1, newgamma=1 / 11, sigma=1 / 3, silentgamma= 1/14):

        self.beta = self.rawBeta * socialconnect;
        self.slientbeta = self.rawslientbeta*socialconnect;
        self.exposedBeta = self.rawexposedBeta*socialconnect;

        self.gamma = newgamma;
        self.sigma = sigma
        self.slientgamma = silentgamma

    def createExportcase(self, movementRate):
        reList = list(map(lambda x: x * movementRate, [self.Susceptible, self.Infected, self.Exposed, self.Removed, self.Silentinfection]))
        adjexposed = 0
        # if (self.AddedTodayHis.__len__() >= 10):
        #     adjr = 1
        #     socialconnection_cari = self.beta / self.rawBeta - adjr if self.beta / self.rawBeta - adjr > 0 else 0;  # adjust if social connection > 1
        #     if socialconnection_cari != 0:
        #         adjexposed = self.Exposed_td * socialconnection_cari / (adjr + socialconnection_cari)

        self.Susceptible -= reList[0];
        self.Infected -= reList[1]
        removedexposed = int(round((reList[2] + adjexposed)))

        #remove some cases
        #all case can't be remove from exposed have been removed
        # ExposeCanBeRemove = [i for i, x in enumerate(self.ExposedDetailedHis) if x > self.currentTime]

        self.ExportedCasedHis.append(reList[1]+removedexposed)

        if(removedexposed < self.Exposed):
            self.Exposed -= removedexposed;
            self.ExposedDetailedHis = self.ExposedDetailedHis.sample( n = self.Exposed)
            # removeindex = (np.random.choice(ExposeCanBeRemove,removedexposed));
            # self.ExposedDetailedHis = list(np.delete(np.array(self.ExposedDetailedHis),removeindex))

        # del l[int(np.random.choice([i for i, x in enumerate(self.ExposedDetailedHis) if x > self.currentTime],removedexposed))]
        self.Removed -= reList[3];
        self.AddedToday -= reList[1];
        self.AddedToday = 0 if self.AddedToday < 0 else self.AddedToday
        self.Silentinfection -= reList[4]
        self.fixTotalpop();

        return reList;

    def updateImportcase(self, Susceptible_Movedin, Infected_Movedin, Removed_Movedin, Exposed_Movedin):
        self.Susceptible += Susceptible_Movedin;
        self.Infected += Infected_Movedin;
        self.Removed += Removed_Movedin;
        self.Exposed += Exposed_Movedin;

        self.fixTotalpop();

    def runEpidSession(self):
        while self.currentTime < self.targetTime:
            self.updateEpidStatus()
            self.currentTime += 1;
        self.plot()

    def plot(self, Acutuall=None):
        # plt.subplot(211)
        plt.plot(self.InfectedHis, '-g', label='Infected')
        plt.plot(self.ExposedHis, '-r', label='Exposed')
        plt.plot(self.AddedTodayHis, '-c', label='TodayAdded')
        # plt.plot(self.SusceptibleHis, '-m', label='Susceptible')
        if (Acutuall is not None):
            plt.plot(list(Acutuall), '-m', label='Acutall')
        plt.legend(loc=0)
        # plt.subplot(212)

        # plt.legend(loc=0)
        plt.show()

    def fixTotalpop(self):
        self.Totalpop = self.Susceptible + self.Infected + self.Exposed + self.Removed;

    def mae(self, InfectedSeries: list):
        return (sum(abs(np.array(InfectedSeries) - np.array(self.AddedTodayHis[1:])) ** 2)) ** 0.5

    def r2(self, InfectedSeries: list, lag=0, first=0, usingexposed=False):
        comparedSeries = self.AddedTodayHis[1:]
        if (usingexposed):
            comparedSeries = self.ExposedHis
        if (comparedSeries.__len__() != len(InfectedSeries)):
            print('unmatched series length')
            return 0
        if (lag == 0):
            correlation_matrix = np.corrcoef(comparedSeries, InfectedSeries)
        else:
            correlation_matrix = np.corrcoef(comparedSeries[lag:], InfectedSeries[:-lag])
        if (first != 0):
            correlation_matrix = np.corrcoef(comparedSeries[:first], InfectedSeries[:first])
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy ** 2
        return r_squared;
