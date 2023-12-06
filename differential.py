import numpy as np
import matplotlib.pyplot as plot
import time
from scipy.stats import lognorm
from scipy.optimize import curve_fit
#define the parameters
k1=0.9*0.25
k2=0.025
k3=0.9*0.25
m1=0.9*0.25
m2=0.9*0.25
epi1=0.06
epi2=0.03
beta=0.06
#initial condition
t=np.linspace(0.0,100,10000)
dt=t[1]-t[0]
#differential equation
def dx_dp(a):
    x1=a[0]
    x2=a[1]
    p1=a[2]
    p2=a[3]
    dxdp=np.array([p1/m1, p2/m2,(-k1*x1)+(k2*(x2-x1))-(beta*p1/m1),-k2*(x2-x1)-(k3*x2)-(beta*p2/m2)])
    return dxdp
def run_diff_equat():
    #initial condition
    y0=np.array([0,0,0,0])
    #creating random seed
    rng1=np.random.default_rng()
    rng2=np.random.default_rng()
    #creating empty solution set
    sol=np.zeros((t.size,4))
    sol[0]=y0
    #numerical integration
    for i in range (1,t.size):
        #adding in the stochastic term 
        sol[i]=sol[i-1]+dx_dp(sol[i-1])*dt+np.sqrt(np.array([0,0,2*epi1,2*epi2]))*np.array([0,0,rng1.normal(),rng2.normal()])*np.sqrt(dt)
    return sol
def area_function(a,b):
    sol=run_diff_equat()
    A=np.zeros_like(t)
    for i in range (1,t.size):
        A[i]=A[i-1]+0.5*(sol[i-1,a]*sol[i,b]-sol[i,a]*sol[i-1,b])
    return A
def avg_area(num_sim):
    array=np.zeros_like(t)
    for i in range(num_sim):
        array+=area_function(0,3)
        print(i)
    average=array/num_sim
    return average
def average_momentumcrossed(num_sim):
    array=np.zeros_like(t)
    for i in range(num_sim):
        sol=run_diff_equat()
        array+=(sol[:,2]*sol[:,3])
    return array/num_sim
def avg_areaxp(num_sim):
    array=np.zeros_like(t)
    for i in range(num_sim):
        array+=area_function(0,3)
        #print(i)
    average=array/num_sim
    return average
def save_runs(num_sim):
    for i in range(num_sim):
        sol=run_diff_equat()
        np.savetxt("numericalsim{}runtime=10000beta=0.06dt={}.csv".format(i,dt), sol, delimiter=",")

def sample_hist(time_index,num_sim):
    bins=np.linspace(-1,1,40)
    array=np.zeros(num_sim)
    array2=np.zeros(num_sim)
    array3=np.zeros(num_sim)
    array4=np.zeros(num_sim)
    for i in range(num_sim):
        sol=run_diff_equat()
        momentum=(sol[:,2]*sol[:,3])
        array[i]=momentum[time_index[0]]
        array2[i]=momentum[time_index[1]]
        array3[i]=momentum[time_index[2]]
        array4[i]=momentum[time_index[3]]
    figure,axis=plot.subplots(2,2)
    axis[0,0].hist(array,bins)
    axis[0, 0].set_title("100 time index")
    axis[0,1].hist(array2,bins)
    axis[0, 1].set_title("1000 time index")
    axis[1,0].hist(array3,bins)
    axis[1, 0].set_title("3000 time index")
    axis[1,1].hist(array4,bins)
    axis[1, 1].set_title("5000 time index")
    plot.show()
def sample_logcountssingle(time_index,num_sim):
    array=np.zeros((num_sim,len(time_index)))
    for i in range(num_sim):
        sol=run_diff_equat()
        momentum=np.power(sol[:,2],3)
        for j in range(len(time_index)):
                  array[i][j]=momentum[time_index[j]]
    for j in range(len(time_index)):
                  hist, bin_edges=np.histogram(array[:,j])
                  centers=np.divide(bin_edges[1:]+bin_edges[:-1],2)
                  logcounts=np.log(hist)
                  plot.scatter(centers,logcounts)
                  plot.show()
def sample_logcountshc(time_index,num_sim):
    array=np.zeros((num_sim,len(time_index)))
    for i in range(num_sim):
        sol=run_diff_equat()
        momentum=(sol[:,2]*sol[:,3])
        for j in range(len(time_index)):
                  array[i][j]=momentum[time_index[j]]
    for j in range(len(time_index)):
                  hist, bin_edges=np.histogram(array[:,j])
                  centers=np.divide(bin_edges[1:]+bin_edges[:-1],2)
                  logcounts=np.log(hist)
                  plot.scatter(centers,logcounts)
                  plot.show()
            
def sample_logcounts(time_index,num_sim):
    array=np.zeros(num_sim)
    array2=np.zeros(num_sim)
    array3=np.zeros(num_sim)
    array4=np.zeros(num_sim)
    for i in range(num_sim):
        sol=run_diff_equat()
        momentum=(sol[:,2]*sol[:,3])
        array[i]=momentum[time_index[0]]
        array2[i]=momentum[time_index[1]]
        array3[i]=momentum[time_index[2]]
        array4[i]=momentum[time_index[3]]
    hist,bin_edges=np.histogram(array)
    hist2, bin_edges2=np.histogram(array2)
    hist3, bin_edges3=np.histogram(array3)
    hist4, bin_edges4=np.histogram(array4)
    centers=np.divide(bin_edges[1:]+bin_edges[:-1],2)
    centers2=np.divide(bin_edges2[1:]+bin_edges2[:-1],2)
    centers3=np.divide(bin_edges3[1:]+bin_edges3[:-1],2)
    centers4=np.divide(bin_edges4[1:]+bin_edges4[:-1],2)
    logcounts=np.log(hist)
    logcounts2=np.log(hist2)
    logcounts3=np.log(hist3)
    logcounts4=np.log(hist4)
    
    figure,axis=plot.subplots(2,2)
    figure.tight_layout(pad=5.0)
    figure.suptitle("Log Counts of p1p2 at various time indices")
    axis[0,0].scatter(centers,logcounts)
    axis[0, 0].set_title("100 time index")
  
    
    axis[0,1].scatter(centers2,logcounts2)
    axis[0, 1].set_title("1000 time index")
    
    
    axis[1,0].scatter(centers3,logcounts3)
    axis[1, 0].set_title("3000 time index")


    axis[1,1].scatter(centers4,logcounts4)
    axis[1, 1].set_title("5000 time index")
    for ax in axis.flat:
        ax.set(xlabel='p1p2 value', ylabel='Log of Counts')
    plot.show()
    
    
def sample_log(time_index,num_sim):
    array=np.zeros(num_sim)
    array2=np.zeros(num_sim)
    array3=np.zeros(num_sim)
    array4=np.zeros(num_sim)
    for i in range(num_sim):
        sol=run_diff_equat()
        momentum=(sol[:,2]*sol[:,3])
        array[i]=momentum[time_index[0]]
        array2[i]=momentum[time_index[1]]
        array3[i]=momentum[time_index[2]]
        array4[i]=momentum[time_index[3]]
    log_data=np.exp(array)
    log_data2=np.exp(array2)
    log_data3=np.exp(array3)
    log_data4=np.exp(array4)
    x = np.linspace(0, log_data.max(), 100)
    x2 = np.linspace(0, log_data2.max(), 100)
    x3= np.linspace(0, log_data3.max(), 100)
    x4= np.linspace(0, log_data4.max(), 100)
    
    pdf = lognorm.pdf(x, 1, scale=np.exp(0))
    pdf2 = lognorm.pdf(x2, 1, scale=np.exp(0))
    pdf3 = lognorm.pdf(x3, 1, scale=np.exp(0))
    pdf4 = lognorm.pdf(x4, 1, scale=np.exp(0))
    figure,axis=plot.subplots(2,2)
    figure.tight_layout(pad=5.0)
    axis[0,0].plot(x, pdf, 'r-', lw=2)
    axis[0, 0].set_title("100 time index")
  
    
    axis[0,1].plot(x2, pdf, 'r-', lw=2)
    axis[0, 1].set_title("1000 time index")
    
    
    axis[1,0].plot(x3, pdf, 'r-', lw=2)
    axis[1, 0].set_title("3000 time index")


    axis[1,1].plot(x4, pdf, 'r-', lw=2)
    axis[1, 1].set_title("5000 time index")
    for ax in axis.flat:
        ax.set(xlabel='Value', ylabel='Probability Density')

    plot.show()
def variance(num_sim,time_index):
    array=np.zeros((num_sim,len(time_index)))
    array1=np.zeros(len(time_index))
    for i in range(num_sim):
        sol=run_diff_equat()
        momentum=(sol[:,2]*sol[:,3])
        for j in range(len(time_index)):
                  array[i][j]=momentum[time_index[j]]
    for j in range(len(time_index)):
        array1[j]=np.var((array[:,j]))
    """    
    plot.scatter(time_index, array1)
    plot.xlabel("Time Indices")
    plot.ylabel("Value of Variance")
    plot.title("Variance of p1p2 distribution over time")
    plot.show()
    """
    return array1

    
starttime=time.time()
#sample_logcountshc([100,200,300,400,500,600,700,800,900,1000,1500,2000,2500,3000,3500,4000,4500,5000,9000],1000)
x=np.linspace(0,5000,251)
y=variance(1000,np.linspace(0,5000,251).astype(int))
def func(x,a,b):
    return b*(1-np.exp(-a*(x**2)))
popt, pcov=curve_fit(func,x,y)
plot.scatter(x,y)
plot.plot(np.linspace(0,5000,1000),func(np.linspace(0,5000,1000),popt[0],popt[1]))
plot.show()
print(popt)

endtime=time.time()
print(endtime-starttime)

