import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from numba import jit
from datetime import datetime
import sklearn.metrics as sk
def RegresionLog(x_0,r,n,T):
    x=[x_0/T]
    for i in range(1,n+1):
        x.append((r*x[i-1]*(1-x[i-1])))
    return(np.array(x)*T)
def Exponential(P0,r,n):
    d=np.array(range(n+1))
    return(P0*np.exp(r*d))

@jit(nopython=True)
def probLlegada(Lambda,k):
    return ((np.exp(-Lambda)*(Lambda**k))/np.math.factorial(k))
@jit(nopython=True, nogil=True, parallel=True,fastmath=False)
def Montecarlo(daysrate,n,acceleration):
    #Mediana en Personas/Horas
    RateSickness=(1/daysrate)
    RateHealth=(1/16)
    Days=range(n+1)
    samples=10000
    Samples=range(samples)
    T=48258494
    PersonsMean=np.zeros(n+1)
    Arrivals=np.zeros(n+1)
    for j in Samples:
        Personas=1
        Arrivalers=1
        IncomingSicksperDay=np.zeros(n+1)
        SickinMySystem=np.zeros(n+1)
        for i in Days:
            maximoPersonasLlegan=0
            maximoPersonasSalen=0
            GrowParam=Personas**(1/acceleration)
            maximoPersonasLlegan=np.random.poisson(GrowParam*RateSickness)
            maximoPersonasSalen=np.random.poisson(Personas*RateHealth)
            Arrivalers=Arrivalers+maximoPersonasLlegan
            Personas=Personas+(maximoPersonasLlegan-maximoPersonasSalen);
            if(Personas<0):
                Personas=0
            IncomingSicksperDay[i]=Arrivalers
            SickinMySystem[i]= Personas
        Arrivals=IncomingSicksperDay+Arrivals
        PersonsMean=SickinMySystem+PersonsMean
    return(PersonsMean/samples,Arrivals/samples)
Data=pd.read_csv("Casos1.csv")
Days=[]
set_zero=datetime.strptime(Data["Fecha de diagnóstico"][0], '%d/%m/%Y')
for i in range(len(Data["Fecha de diagnóstico"])):
    Dates=(datetime.strptime(Data["Fecha de diagnóstico"][i], '%d/%m/%Y'))
    Days.append(Dates.day-set_zero.day)
Data["Dias"]=Days
NuevosDia=Data["Dias"].value_counts()
NuevosDia=NuevosDia.sort_index()
dias=range(NuevosDia.index.values[-1]+1)
CumulativeDays=np.ones(len(dias))
for i in dias:
    if (i in NuevosDia.index.values and i>0):
        CumulativeDays[i]=NuevosDia.loc[i]+CumulativeDays[i-1]
    elif(i>0):
        CumulativeDays[i]=CumulativeDays[i-1]
plt.scatter(dias,CumulativeDays)
##Now we make a regression model
Rates=np.linspace(1e-3,3,1000)
a_c= np.linspace(1.3,1.5,10)
besterror=10000000
for rate in Rates:
    for accel in a_c:
    #B=RegresionLog(1,rate,17,48258494)
        B,A=Montecarlo(rate,NuevosDia.index.values[-1],accel)
        MSE=sk.mean_absolute_error(CumulativeDays,A)
        if(MSE<besterror):
            besterror=MSE
            bestaccel=accel
            best=rate
print(best)
z = np.polyfit(dias,CumulativeDays, 3)
p = np.poly1d(z)
B,A=Montecarlo(best,NuevosDia.index.values[-1],bestaccel)
print(A)
plt.plot(dias,p(dias))
plt.plot(dias,A)
#proyeccion de datos
plt.figure()
d=500
da=range(d+1)
LongRange,Arrivals=Montecarlo(best,da[-1],bestaccel)
plt.plot(da,LongRange)
print(np.max(LongRange),Arrivals[19:23])
plt.show()
