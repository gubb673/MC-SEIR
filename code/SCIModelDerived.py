from covmodel2020withasyspeedup import EpidUnit
from fetechdata import FetchCOVIDData
import numpy as np
import datetime
import pickle
import os
import pandas as pd
from hyperopt import Trials, hp, tpe, fmin
import glob
import ray
import itertools

filepath = 'data.pickle'
if os.path.exists(filepath):
    infile = open(filepath, 'rb')
    fed = pickle.load(infile)
    infile.close()

else:
    fed = FetchCOVIDData()
    fed.LoadData()
    f = open(filepath, 'wb')
    pickle.dump(fed, f)
    f.close()

citytrans = pd.read_csv('contact2/mig_0420.csv')
PopWuhanOutflow = citytrans[citytrans['o_cityid'] == 420100].groupby('date').sum()  # od
PopWuhanInflow = citytrans[citytrans['d_cityid'] == 420100].groupby('date').sum()

epidpara_sigma = 1 / (
    5.791004060471671)  # https://annals.org/AIM/FULLARTICLE/2762808/INCUBATION-PERIOD-CORONAVIRUS-DISEASE-2019-COVID-19-FROM-PUBLICLY-REPORTED

OptimizeTheConfig = True
UsingSocialConnection = True;
CreateOriginalSim = False;
SimRecovery = False
SimResurgence = False

if UsingSocialConnection:

    param = {'gamma1': 0.10438313742445998, 'gamma2': 0.07835566474268932, 'gamma3': 0.16938982582251064,
             'gamma4': 0.3331503715923706, 'gamma5': 0.40442437850161544, 'iniexp': 6.0, 'inik': 10.0,
             'x': 0.04543929449466398,
             'y': 0.49206238687550985}

    op = [10.0, 6.0, 0.04543929449466398, 0.49206238687550985, 0.10438313742445998, 0.07835566474268932,
          0.16938982582251064, 0.3331503715923706, 0.40442437850161544]

    print('Optimized result is used')
else:
    # best base line
    param = {'cd': 0.12677212564855017, 'gammaafter': 0.33004438742040726, 'gammabefore': 0.20004754306993666,
             'iniexp': 33.13734581254948, 'inik': 19.979669811158818, 'x': 0.0659980118887624, 'y': 0.07144049851673497,
             'z': 0.5779503300922744}
    print('Using base line')
    op = [param['inik'], param['iniexp'], param['x'], param['y'], param['z'], param['cd'], param['gammabefore'],
          param['gammaafter']]


def GetGamma(time=0, cutpoint=[15, 20], gamma=[1 / 15, 1 / 3, 1 / 2], defaultgamma=1 / 4):
    index = 0
    while index < len(cutpoint):
        if time < cutpoint[index]:
            return gamma[index]
        else:
            index += 1;

    return gamma[-1];


def BatchEvaluationBaselineModel(param, auxdata, config, testtimes=20):
    r = 0
    k = 0;

    op = [param['inik'], param['iniexp'], param['baseline'], param['socialconnection'],
          param['gamma1'], param['gamma2'], param['gamma3'], param['gamma4'], param['gamma5']]
    for i in range(0, testtimes):
        r += EvaluationBaselineModel(op, auxdata, config)
        k += 1;
    r = r / k
    print('{0:.3f}'.format(r) + " at  input" + str(op))
    return r


def GetAuxDataForOneCity(DataFeature, startdate, enddate, citycode=420100):
    population = sum(DataFeature.population[DataFeature.population['Adcode'] // 100 * 100 == citycode]['resi15'])
    cityname = DataFeature.rawCOVID_19Data.cityEnglishName[DataFeature.rawCOVID_19Data.city_zipCode == citycode].iloc[0]
    socialconnection = DataFeature.citytc.contactindex[DataFeature.citytc.contactindex['cityid'] == citycode]
    baseline = socialconnection.iloc[0, 2]
    socialconnection = socialconnection.sort_values(by=['ds'])
    socialconnection = socialconnection.iloc[startdate:, 2]
    return population, cityname, socialconnection


def GetLaiTestData(DataFeature, citycode=420100):
    startdate = DataFeature.LaiData.startdate  # 20191211
    enddate = DataFeature.LaiData.enddate
    startindex = (startdate.date() - DataFeature.provincetc.zeroday).days
    enddateindex = 91
    newaddedSeries = list(pd.read_excel('/Users/wm1125/PycharmProjects/cov2020_plot/fix2Fit.xlsx')['合计'])[2:]
    population, cityname, socialconnection = GetAuxDataForOneCity(DataFeature, startindex, enddateindex,
                                                                  citycode=citycode)

    return population, cityname, socialconnection, newaddedSeries


def GetInOutFlow(times, InFlow=True):
    if InFlow:
        try:
            result = PopWuhanInflow.iloc[10 + times, -1]
        except:
            result = np.mean(PopWuhanOutflow.iloc[-20:, -1])
    else:
        try:
            result = PopWuhanOutflow.iloc[10 + times, -1]
        except:
            result = np.mean(PopWuhanOutflow.iloc[-20:, -1])  # in = out


    return result


def SimulateOneRecoveryScenario(opparray, auxdata, config, evaluating=True, TargetDay=131, sim=False,
                                adjsocialconnect=None):
    inicase, iniexp, hpy, Socialindex, gamma1, gamma2, gamma3, gamma4, gamma5 = opparray
    pop, name, socialconnection, coviddata = auxdata
    socialtimelag, usingtcsocialconnection, useingexposed, reviewdays, plot, lagr2, usingmae = config
    epid = EpidUnit(Susceptible=pop, Infected=inicase, Exposed=iniexp);
    epid.recordStatus();
    times = 0
    gamma = np.concatenate((
        np.repeat(gamma1, 30),  # 2019-12-11 =》 2020 - 01-09
        np.repeat(gamma2, 13),  # 2020-01-10 =》 2020-01-22
        np.repeat(gamma3, 10),  # 2020-01-23 =》2020-02-01
        np.repeat(gamma4, 15),
        # 2020-02-02=》2020-02-10
    ))
    cpahis = list()
    if sim:
        pass
    else:
        if (TargetDay > socialconnection.shape[0]):
            # print('Using social conntects length: '+ str(socialconnection.shape[0]) + 'replace Targetday = '+str(TargetDay))
            TargetDay = socialconnection.shape[0]

    def CreateTargetSocialContact(v, sccid=0, pop=11000000):
        a = PhysicalDist.iloc[sccid, 1]
        b = PhysicalDist.iloc[sccid, 2]
        return ((v * pop) ** b) * a

    while times < TargetDay:
        timelag = 0;
        targetdate = times - timelag if times > timelag else times
        sc = socialconnection.iloc[targetdate] if targetdate < socialconnection.shape[0] else socialconnection.iloc[-1]
        socialconnectionindex = ((
                                         sc / epid.Totalpop) * Socialindex) ** 0.5 + hpy if usingtcsocialconnection else 1  # best for data.pickle
        if (times < len(gamma)):
            cg = gamma[times]
        else:
            cg = gamma5;
        if adjsocialconnect is not None:
            if times > adjsocialconnect[0]:
                # socialconnectionindex = 1.8807 * adjsocialconnect[1]
                socialconnectionindex = (CreateTargetSocialContact(adjsocialconnect[1], pop=epid.Totalpop,
                                                                   sccid=adjsocialconnect[
                                                                       2]) * Socialindex) ** 0.5 + hpy
                if (len(adjsocialconnect) <= 3):
                    cg = 1 / 3
                else:
                    cg = 1 / adjsocialconnect[3]
                # 1.8807 = average of first 30 days
        cpahis.append(socialconnectionindex)
        epid.updatePara(socialconnect=socialconnectionindex, newgamma=cg, sigma=epidpara_sigma, silentgamma=1 / 14)

        epid.updateImportcase(GetInOutFlow(times), 0, 0, 0)
        epid.updateEpidStatus(moveoutrate=GetInOutFlow(times, InFlow=False) / epid.Totalpop);
        # epid.updateImportcase(0, 0, 0, 0)
        # epid.updateEpidStatus(moveoutrate=0);
        times += 1;

    mae = -1;
    _r2 = -1;
    if evaluating:
        mae = -epid.mae(coviddata[0:socialconnection.shape[0]])
        _r2 = epid.r2(coviddata[0:socialconnection.shape[0]], lag=lagr2, first=reviewdays, usingexposed=useingexposed)

    if (plot):
        print('{0:.2f}'.format(_r2) + "in " + name)
        epid.plot(coviddata)
    return _r2, mae, epid, cpahis





def SimulateContainmentResurgence(opparray, auxdata, config, evaluating=True, TargetDay=131, sim=False,
                                  adjsocialconnect=None, breakpoint=100, pressed=0.5, socialdistancing=1,
                                  contd=10, exitnum=100, targetGamma=7):
    inicase, iniexp, hpy, Socialindex, gamma1, gamma2, gamma3, gamma4, gamma5 = opparray
    pop, name, socialconnection, coviddata = auxdata
    socialtimelag, usingtcsocialconnection, useingexposed, reviewdays, plot, lagr2, usingmae = config
    epid = EpidUnit(Susceptible=pop, Infected=inicase, Exposed=iniexp);
    epid.recordStatus();
    times = 0
    gamma = np.concatenate((
        np.repeat(gamma1, 30),  # 2019-12-11 =》 2020 - 01-09
        np.repeat(gamma2, 13),  # 2020-01-10 =》 2020-01-22
        np.repeat(gamma3, 10),  # 2020-01-23 =》2020-02-01
        np.repeat(gamma4, 15),
        # 2020-02-02=》2020-02-10
    ))
    cpahis = list()
    pressedhis = list()
    if sim:
        pass
    else:
        if (TargetDay > socialconnection.shape[0]):
            # print('Using social conntects length: '+ str(socialconnection.shape[0]) + 'replace Targetday = '+str(TargetDay))
            TargetDay = socialconnection.shape[0]
    socialconnectionindex = 1.88
    pressedmode = False
    pressedtimes = 0;
    while times < TargetDay:
        timelag = 0;
        targetdate = times - timelag if times > timelag else times

        if epid.AddedToday > breakpoint and not pressedmode:
            pressedmode = True
            pressedtimes += 1;

            def CreateTargetSocialContact(v, sccid=0, pop=11000000):
                a = PhysicalDist.iloc[sccid, 1]
                b = PhysicalDist.iloc[sccid, 2]
                return ((v * pop) ** b) * a


            socialconnectionindex = (CreateTargetSocialContact(pressed, pop=epid.Totalpop,
                                                               sccid=socialdistancing) * Socialindex) ** 0.5 + hpy

        if pressedmode and len(epid.AddedTodayHis) > contd:
            if np.mean(epid.AddedTodayHis[-contd:]) <= exitnum:
                pressedmode = False
                socialconnectionindex = 1.88

        pressedhis.append(pressedmode)
        cg = 1 / targetGamma
        cpahis.append(socialconnectionindex)
        epid.updatePara(socialconnect=socialconnectionindex, newgamma=cg, sigma=epidpara_sigma, silentgamma=1 / 14)

        epid.updateImportcase(10 ** 5, 0, 0, 0)
        epid.updateEpidStatus(moveoutrate=10 ** 5 / epid.Totalpop);
        # epid.updateImportcase(0, 0, 0, 0)
        # epid.updateEpidStatus(moveoutrate=0);
        times += 1;

    mae = -1;
    _r2 = -1;
    if evaluating:
        mae = -epid.mae(coviddata[0:socialconnection.shape[0]])
        _r2 = epid.r2(coviddata[0:socialconnection.shape[0]], lag=lagr2, first=reviewdays, usingexposed=useingexposed)
    # print(str(pressedtimes)+"|"+str(sum(pressedhis)))
    if (plot):
        print('{0:.2f}'.format(_r2) + "in " + name)
        epid.plot(coviddata)
    return _r2, mae, epid, cpahis, pressedhis, pressedtimes, sum(pressedhis)





def SimulateBaselineModel(opparray, auxdata, config, evaluating=True, TargetDay=131, sim=False):
    inicase, iniexp, hpy, Socialindex, gamma1, gamma2, gamma3, gamma4, gamma5 = opparray
    pop, name, socialconnection, coviddata = auxdata
    socialtimelag, usingtcsocialconnection, useingexposed, reviewdays, plot, lagr2, usingmae = config
    epid = EpidUnit(Susceptible=pop, Infected=inicase, Exposed=iniexp);
    epid.recordStatus();
    times = 0
    gamma = np.concatenate((
        np.repeat(gamma1, 30),  # 2019-12-11 =》 2020 - 01-09
        np.repeat(gamma2, 13),  # 2020-01-10 =》 2020-01-22
        np.repeat(gamma3, 10),  # 2020-01-23 =》2020-02-01
        np.repeat(gamma4, 15),
        # 2020-02-02=》2020-02-10
    ))
    cpahis = list()
    if sim:
        pass
    else:
        if (TargetDay > socialconnection.shape[0]):
            # print('Using social conntects length: '+ str(socialconnection.shape[0]) + 'replace Targetday = '+str(TargetDay))
            TargetDay = socialconnection.shape[0]

    while times < TargetDay:
        timelag = 0;
        targetdate = times - timelag if times > timelag else times
        sc = socialconnection.iloc[targetdate] if targetdate < socialconnection.shape[0] else socialconnection.iloc[-1]
        socialconnectionindex = ((
                                         sc / epid.Totalpop) * Socialindex) ** 0.5 + hpy if usingtcsocialconnection else 1  # best for data.pickle
        cpahis.append(socialconnectionindex)
        if (times < len(gamma)):
            cg = gamma[times]
        else:
            cg = gamma5;
        epid.updatePara(socialconnect=socialconnectionindex, newgamma=cg, sigma=epidpara_sigma, silentgamma=1 / 14)

        epid.updateImportcase(GetInOutFlow(times), 0, 0, 0)
        epid.updateEpidStatus(moveoutrate=GetInOutFlow(times, InFlow=False) / epid.Totalpop);
        times += 1;

    mae = -1;
    _r2 = -1;
    if evaluating:
        mae = -epid.mae(coviddata[0:socialconnection.shape[0]])
        _r2 = epid.r2(coviddata[0:socialconnection.shape[0]], lag=lagr2, first=reviewdays, usingexposed=useingexposed)

    if (plot):
        print('{0:.2f}'.format(_r2) + "in " + name)
        epid.plot(coviddata)
    return _r2, mae, epid, cpahis


def EvaluationBaselineModel(OptPara, AuxData, config, evaluating=True, TargetEndSimDay=131):
    inicase, iniexp, hpy, gamma1, Socialindex, gamma2, gamma3, gamma4 = OptPara
    pop, name, socialconnection, coviddata = AuxData
    socialtimelag, usingtcsocialconnection, useingexposed, reviewdays, plot, lagr2, usingmae = config
    epid = EpidUnit(Susceptible=pop, Infected=inicase, Exposed=iniexp);
    epid.recordStatus();
    times = 0
    gamma = np.concatenate((
        np.repeat(gamma1, 30),  # 2019-12-11 =》 2020 - 01-09
        np.repeat(gamma2, 13),  # 2020-01-10 =》 2020-01-22
        np.repeat(gamma3, 10),  # 2020-01-23 =》2020-02-01
        np.repeat(gamma4, 9)  # 2020-02-02=》2020-02-10
    ))
    S_cHIS = list()
    while times < TargetEndSimDay:
        timelag = 0;
        targetdate = times - timelag if times > timelag else times

        socialconnectionindex = ((socialconnection.iloc[
                                      targetdate] / epid.Totalpop) * Socialindex) ** 0.5 + hpy if usingtcsocialconnection else 1  # best for data.pickle
        S_cHIS.append(socialconnectionindex)
        cg = gamma[times] if times < gamma.__len__() else gamma[
            -1]  # when plot gamma will exceed the defined range then used the lastest one
        epid.updatePara(socialconnect=socialconnectionindex, newgamma=cg, sigma=epidpara_sigma, silentgamma=1 / 14)
        epid.updateImportcase(GetInOutFlow(times), 0, 0, 0)
        epid.updateEpidStatus(moveoutrate=GetInOutFlow(times, InFlow=False) / epid.Totalpop);
        times += 1;

    MAE = -1;
    R2 = -1;
    if evaluating:
        MAE = -epid.mae(coviddata)
        R2 = epid.r2(coviddata, lag=lagr2, first=reviewdays, usingexposed=useingexposed)

    if (plot):
        print('{0:.2f}'.format(R2) + "in " + name)
        epid.plot(coviddata)
    return R2, MAE, epid, S_cHIS


def GetCombineIndexById(ContactIndex, id):
    result = list(pd.DataFrame(list(map(lambda x: list(ContactIndex[x]['contact']), id))).transpose().sum(axis=1))
    return result

def CreateScenario(citycode=420100, Retype=[2], StartDay=7, RecoverRate=0.1, TargetDate=-1):
    # /Users/wm1125/Documents/Prj/cov2020/contact2/contact0420
    path = 'contact2/contact0420'
    files = glob.glob(path + '/*.csv')  # sch, shop, work, attr, trans
    files = list(map(lambda x: pd.read_csv(x), files))
    files = list(map(lambda x: x[x['cityid'] == citycode].sort_values(by=['ds']), files))

    # 12-1
    fullcontact = np.array(GetCombineIndexById(files, Retype))
    timerange = (datetime.datetime(2020, 4, 8) - datetime.datetime(2019, 12, 1)).days

    fullcontact[timerange + StartDay:] = np.mean(fullcontact[1:30]) * RecoverRate  # 10% Recovery rate
    if (TargetDate > 0):
        fullcontact[timerange + StartDay:TargetDate] = np.mean(fullcontact[1:30]) * RecoverRate  # 10% Recovery rate

    return fullcontact[10:]

def CreateSummary(result, resultname='originalfit', weekshift=None, dayshift=None):
    def Getattributes(attrname):
        rlist = np.array(list(map(lambda x: getattr(x[2], attrname), result)))
        return np.quantile(rlist, (0, 0.25, 0.5, 0.75, 1), 0).T

    namelist = ['AddedTodayHis', 'ExposedHis', 'InfectedHis'];
    r1 = np.concatenate(list(map(lambda x: Getattributes(x), ['AddedTodayHis', 'ExposedHis', 'InfectedHis'])), axis=1)
    r1 = pd.DataFrame(r1)
    r1.columns = [i + j for i in namelist for j in ['_0', '_25', '_50', '_75', '_100']]
    try:
        r1['actual'] = [0] + list(auxdata[3])
    except:
        print('sim data have no actual series')
    r1['rt'] = [float('nan')] + list(np.array(result[1][3]) * 2.64)

    r1.to_csv('result_csv/' + resultname + '_sim_result.csv')

    # calculate peak date
    def GetPEAK(oneresult, attrname):
        index = np.argmax(getattr(oneresult, attrname))
        value = max(getattr(oneresult, attrname))
        return index, value

    def GetDataByAttr(result, attrname, value=10000, startdate=81):
        adl = getattr(result[2], attrname)
        adl = adl[startdate:]
        i = float('nan')
        k = 0
        for i in range(0, len(adl)):
            k += adl[i]
            if k > value:
                break
        return i

    def CreateBoxPlotData(brn):
        if (dayshift is not None) and (weekshift is not None):
            sday = ((datetime.datetime(2020, 4, 8) +
                     datetime.timedelta(days=dayshift + weekshift * 7)) -
                    datetime.datetime(2019, 12, 11)).days
        pd.DataFrame(
            list(map(lambda x: GetDataByAttr(x, attrname='AddedTodayHis', value=brn, startdate=sday), result))).to_csv(
            'result_csv/' + resultname + '_boxplot_' + str(brn) + '.csv'
        )

    CreateBoxPlotData(10000)
    CreateBoxPlotData(1000)
    CreateBoxPlotData(3000)

    Peak = pd.DataFrame(list(map(lambda x: GetPEAK(x[2], 'AddedTodayHis'), result)))
    QuantileFixed = [0.05, 0.25, 0.5, 0.75, 0.95]
    Peak_date = np.quantile(Peak.iloc[:, 0], QuantileFixed)  # [46., 46., 47., 47., 47.]
    Peak_case_perday = np.quantile(Peak.iloc[:, 1],
                                   QuantileFixed)  # 1130.20157288, 1332.2284321 , 1602.59909526
    SumaddedToday = np.quantile(pd.DataFrame(list(map(lambda x: sum(getattr(x[2], 'AddedTodayHis')), result))),
                                QuantileFixed)
    r2_qu = np.quantile(list(map(lambda x: x[0], result)), QuantileFixed)
    mae_qu = np.quantile(list(map(lambda x: -x[1], result)), QuantileFixed)
    ExportedSum = np.quantile(pd.DataFrame(list(map(lambda x: sum(getattr(x[2], 'ExportedCasedHis')), result))),
                              QuantileFixed)

    report5Q = pd.DataFrame([Peak_date, Peak_case_perday, SumaddedToday, r2_qu, mae_qu, ExportedSum]).T
    report5Q.columns = ['PeakData', 'PeakCase', 'SumCases', 'R2', 'MAE', 'ExportedCase']
    report5Q.to_csv('result_csv/' + resultname + '_ModelSummary.csv')
    cpahis = result[1][3]
    periodRt = np.array(list(map(np.mean, [cpahis[0:31], cpahis[31:(31 + 13 + 1)], cpahis[43:53],
                                           cpahis[53:]]))) * 2.64  # rt in different period

    return r1, report5Q, cpahis, periodRt


def SimulationScenarioOrigin():
    config = [0, True, False, 0, False, 0, True]
    result = list(
        map(lambda x: SimulateBaselineModel(op, auxdata, config, TargetDay=81), range(0, 500)))
    r1, report5Q, S_cHIS, periodRt = CreateSummary(result, 'originalResult')

space = {
    'inik': hp.uniformint('inik', 1, 10),
    'iniexp': hp.uniformint('iniexp', 1, 10),
    'baseline': hp.uniform('x', 0.03, 0.07),  # baseline
    'socialconnection': hp.uniform('y', 0.3, 0.7),  # socialconnection
    'gamma1': hp.uniform('gamma1', 1 / 14, 1 / 9),  # gamma1
    'gamma2': hp.uniform('gamma2', 1 / 14, 1 / 6),  # gamma2
    'gamma3': hp.uniform('gamma3', 1 / 6, 1 / 4),  # gamma3
    'gamma4': hp.uniform('gamma4', 1 / 5, 1 / 3),  # gamma4
    'gamma5': hp.uniform('gamma5', 1 / 4, 1 / 2)  # gamma5

}


auxdata = list(GetLaiTestData(fed))


if OptimizeTheConfig:
    config = [0, True, False, 0, False, 0, True]
    best = fmin(fn=lambda x: BatchEvaluationBaselineModel(x, auxdata, config), space=space, algo=tpe.suggest,
                max_evals=10000,
                trials=Trials())
    param = best
    print(best)


if CreateOriginalSim:
    SimulationScenarioOrigin();

ResultName = 'RayTestAfterPeak365_gamma3_usefullrun_move0_'
RecoveryRate = [25, 50, 75, 100]


rawFittedPhysicalDist = pd.read_csv('fitted_curve_withScenario.csv');
PhysicalDist = rawFittedPhysicalDist.groupby(['scenario']).mean()


def CreateTargetSocialContact(v, a=0.0006914826432229847, b=0.5780603094538311, usingPos=False):
    if usingPos:
        a = 0.0004988711819518427
        b = 0.5045597497750869
        pop = 11000000
    return ((v * pop) ** b) * a


ray.init()


@ray.remote
def GetOnePress(v):
    config = [0, True, False, 0, False, 0, True]

    res = list(map(
        lambda x: SimulateContainmentResurgence(op, auxdata, config, evaluating=False, TargetDay=TargetDate,
                                                sim=True, breakpoint=v[0], pressed= v[1] * v[6] / 10000,
                                                socialdistancing=v[5], contd=v[2], exitnum=v[3],
                                                targetGamma=v[4]), range(0, 150)))
    days = list(map(lambda x: x[6], res))
    print(v)
    return list(v) + days


if SimResurgence:
    m = [10, 25, 50, 75]
    cityd = [10, 20, 50, 70, 90, 100]  # original
    cityd = [10, 50, 100]  # test

    k = [0, 1, 2, 3]

    va = itertools.product([50, 100, 500], m, [7], [0], [3], k, cityd)  # original
    va = itertools.product([25, 50, 100], m, [7], [0], [3, 5, 7], k, cityd)  # sensitive analysis days = 4,5,7
    if False:
        # start cases, mobility control,out days, out cri, gamma, measure, density
        va = list(va)
        v = list(va[61])
        v[5] = 1
        config = [0, True, False, 0, True, 0, True]
        tr = evaluateWUHANWITHLAIFixed_dynamicST_v2(op, auxdata, config, evaluating=False, TargetDay=TargetDate,
                                                    sim=True,
                                                    breakpoint=v[0], pressed=v[1] * v[6] / 10000, socialdistancing=v[5],
                                                    contd=v[2], exitnum=v[3], targetGamma=v[4])

    res_press = list()
    config = [0, True, False, 0, False, 0, True]
    futures = [GetOnePress.remote(i) for i in va]
    vaaa = ray.get(futures)
    pd.DataFrame(vaaa).to_csv('pressedResult_bydate_vx_v5_sns.csv')
    print('RecoverySimulation done')


@ray.remote
def EvaluateRecoveryScenarios(RecoveryScenarioConfig):
    TargetDate = 365
    DayShift = -4
    i = (RecoveryScenarioConfig[0] - 119 - DayShift) / 7
    config = [0, True, False, 0, False, 0, True]
    auxdata[2] = pd.DataFrame(
        CreateScenario(Retype=[3], StartDay=RecoveryScenarioConfig[0] - 119,
                       RecoverRate=RecoveryScenarioConfig[1] / 100, TargetDate=TargetDate))
    result = list(map(lambda x: SimulateOneRecoveryScenario(op, auxdata, config,
                                                            evaluating=False, TargetDay=TargetDate, sim=True,
                                                            adjsocialconnect=(
                                                                 RecoveryScenarioConfig[0],
                                                                 RecoveryScenarioConfig[1] / 100,
                                                                 RecoveryScenarioConfig[2], RecoveryScenarioConfig[3])
                                                            ),
                      range(0, 150)))
    testname = ResultName + str(i) + "_Rec" + str(RecoveryScenarioConfig[1]) + "_SD" + str(
        RecoveryScenarioConfig[2]) + '_gamma' + str(RecoveryScenarioConfig[3])
    r1, report5Q, cpahis, periodRt = CreateSummary(result, testname, weekshift=i, dayshift=DayShift)
    print('done ' + testname)
    return r1


RecoveryRate_r = list(map(lambda x: CreateTargetSocialContact(x), RecoveryRate)) + list(
    map(lambda x: CreateTargetSocialContact(x, usingPos=True), RecoveryRate))

ShiftRange = list(range(-6, 4))
DayShift = -4;
TargetDate = 365  # Target end date
RecoveryScenarioList = list()

RecoveryScenarioList = itertools.product([119 + i * 7 + DayShift for i in list(range(-6, 4))],  # lifting timing
                                         [25, 50, 75, 100],  # mobility level
                                         [0, 1, 2, 3],  # physical distancing
                                         [4, 5, 7])  # gamma


if SimRecovery:
    futures = [EvaluateRecoveryScenarios.remote(i) for i in RecoveryScenarioList]
    vaaa = ray.get(futures)
    print('result saved as ' + ResultName)
