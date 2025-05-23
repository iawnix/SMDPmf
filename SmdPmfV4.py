#!/home/iaw/soft/conda/2024.06.1/envs/pytorch3.9/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import os
import glob
import logging
import matplotlib.ticker as ticker
import math

# 这里修改使用正则匹配符合要求的
class ReadDat():
    def __init__(self, fp, log) -> None:
        self.fp = fp
        self.log = log
        self.Dmatrix = None
        self.Fmatrix = None
        self.D = self._Read("D")
        self.F = self._Read("F")
        self._check()
        self._DF()
  
    def _Read(self, sign):
        f = os.path.join(self.fp,"{}*.dat".format(sign))
        f_list = glob.glob(f)
        out = sorted(f_list, key=lambda x:eval((os.path.split(x)[1][1:].split("."))[0]))
        return out
    
    def _check(self):
        if len(self.D) != len(self.F):
            self.log.warning("Please Check D fles and F file")
            #print("Please Check D fles and F file")

    def _DF(self):
        data = {}
        for i,f in enumerate(self.D):
            dat = pd.read_csv(f,sep = ",",header=None)
            data[str(i)] = dat.iloc[:,1].to_list()
        self.Dmatrix = pd.DataFrame(data)
        data = {}
        for i,f in enumerate(self.F):
            dat = pd.read_csv(f,sep = ",",header=None)
            data[str(i)] = dat.iloc[:,1].to_list()
        self.Fmatrix = pd.DataFrame(data)

# 此函数用于生成时间序列
def GTimeSeries(Force,fout,dt):
    t = {"fs":[],"ps":[],"ns":[]}
    for i in range(Force.shape[0]):
        t["fs"].append(i*fout*dt)
        t["ps"].append(i*fout*dt/1000)
        t["ns"].append(i*fout*dt/1000000)
    return t


# 设置字体格式
from matplotlib.font_manager import FontProperties
font_path="/home/iaw/MYscrip/SMD/font/arial.ttf"
font_prop =  FontProperties(fname=font_path)
plt.switch_backend("agg")
plt.rcParams["axes.labelweight"] ="bold"
plt.rcParams["font.family"]=font_prop.get_name()
plt.rcParams["font.weight"]="bold"
plt.rcParams["font.size"]=12
plt.tight_layout()

def initdraw():
    # 创建你画布
    fig,ax = plt.subplots(figsize=(7,5),dpi=300)
    # 设置边框
    ax.tick_params(
         which='both'
        ,bottom=True
        ,left=True
        ,direction='out'
        ,width=2 
        ,length=6
    )
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    return fig, ax

# 绘图模块
def DPlot(x,y,xlabel,ylabel,title):
    
    color_s5 = ["#2878B5","#9AC9D8","#F8AC8C","#C82423","#FF8884"
                ,"#2878B5","#9AC9D8","#F8AC8C","#C82423","#FF8884"]
    lstyle = ["-","-","-","-","-","--","--","--","--","--"]
    #mtype = ["x","x","x","x","x","s","s","s","s","s"]
    fig,ax = initdraw()
    # 新增代表曲线

    if type(y) == tuple:
        # 绘制PMF
        if len(y) == 2:
            y_ = y[0]
            y_std = y[1]
            ax.plot(x,y_,alpha=0.9,linewidth=1.2,linestyle="-",color = "black")
            #y_std_center = [i/2 for i in y_std]
            ZZY = [y_[i] - y_std[i] for i in range(len(y_))]
            ax.fill_between(
                x,
                [y_[i] + y_std[i] for i in range(len(y_))],
                [y_[i] - y_std[i] for i in range(len(y_))],
                facecolor="#9AC9D8",
                alpha=0.8)
            #ax.errorbar(x[::100],y_[::100],yerr=y_std[::100],color="darkgreen",linewidth=2,linestyle="-",elinewidth=1,ecolor="blue")
            ax.set_ylim([min(ZZY)-10,max(y_)*1.5])
        # 绘制带有代表曲线，平均曲线标准差的曲线图
        else:
            y_ave = y[0]
            y_std = y[1]
            y_rep = y[2]
            ax.plot(x,y_ave,alpha=0.9,linewidth=1.2,linestyle="-",color = "black")
            ax.plot(x,y_rep,alpha=0.9,linewidth=1.2,linestyle="-",color = "green")
            ZZY = [y_ave[i] - y_std[i] for i in range(len(y_ave))]
            ax.fill_between(
                x,
                [y_ave[i] + y_std[i] for i in range(len(y_ave))],
                [y_ave[i] - y_std[i] for i in range(len(y_ave))],
                facecolor="#9AC9D8",
                alpha=0.3)
            ax.set_ylim([min(ZZY)-10,max(y_rep)*1.5])

            # 20240910
            ax.yaxis.set_major_locator(ticker.MultipleLocator(((max(y_rep)*1.5-min(ZZY)+10)/4)))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(((max(y_rep)*1.5-min(ZZY)+10)/8)))

    else:
        for i in range(y.shape[1]):
            ax.plot(x,y.iloc[:,i],alpha=0.9,markersize= 0.5,color = color_s5[i],linewidth=2,linestyle=lstyle[i])  
        ymin = min(y.min().to_list())
        ymax = max(y.max().to_list())
        ax.set_ylim([ymin-2,ymax*1.5])

        # 20240910
        ax.yaxis.set_major_locator(ticker.MultipleLocator(((ymax*1.5-ymin+2)/4)))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(((ymax*1.5-ymin+2)/8)))
    

    # 20240910
    ax.xaxis.set_major_locator(ticker.MultipleLocator(((max(x)+2-min(x)+2)/4)))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(((max(x)+2-min(x)+2)/8)))

    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([min(x)-2,max(x)+2])
    plt.tight_layout()
    plt.savefig("{}.png".format(title))
    plt.savefig("{}.tiff".format(title),dpi=300)

def DST(S,t,v,xlb,ylb,dist):

    color_s5 = ["#2878B5","#9AC9D8","#F8AC8C","#C82423","#FF8884"
                ,"#2878B5","#9AC9D8","#F8AC8C","#C82423","#FF8884"]
    lstyle = ["-","-","-","-","-","--","--","--","--","--"]

    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["font.family"] = "Arial"
    #plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = 15
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    plt.figure(figsize=(7,5),dpi=300)
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["top"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    
    
    for i in range(S.shape[1]):
        ax.plot(t
                , S.iloc[:,i], label="Curve[{}]: {:.2f}".format(i,dist[i])
                , linewidth=0.2
                , color = color_s5[i]
                , linestyle = lstyle[i])
        #ax.scatter(t,S.iloc[:,i],s = 100,alpha=0.1,color= "Blue")

    tmin = min(t)
    tmax = max(t)
    ax.plot([tmin,tmax],[0,v*(tmax-tmin)],label="Dummy Atom",color = "black", linewidth=1.5)
    ax.set_xlabel(xlb)
    ax.set_ylabel(ylb)
    ax.set_xlim([tmin-2,tmax+2])
        
    plt.tight_layout()
    plt.savefig("{}.png".format("S-T"))
    plt.savefig("{}.tiff".format("S-T"),dpi=300)


# 修正，计算dist，将理想S(t)曲线视为拟合曲线
class STdist():
    def __init__(self,x,Y,func):
        self.F = func           # 用于计算理想S(t)
        self.x = np.array(x)    # t
        self.y = self.F(self.x) # 理想的S
        self.dist = []  
        self.dist_ = 0
        for i in range(Y.shape[1]):
            self.dist.append(self._cal(Y.iloc[:,i].to_numpy()))
        self.dist_ = sum(self.dist) / len(self.dist)

    # y1是真实值
    def _cal(self, y1):
        #print(self.y.mean(),y1.mean())
        return np.sum(y1 - self.y)/len(y1)

    def pDist(self):
        var = '\n'
        for i,t in enumerate(self.dist):
            if (i != 0 and i % 4 == 0) or i == len(self.dist) - 1:    
                var += "Curve[{}]: {:.2f}\n".format(i,t)
            else:
                var += "Curve[{}]: {:.2f}\t".format(i,t)
        var += "\nThe average of Dist[S-Smd -> S-Dummy] is {:.2f}.".format(self.dist_)
        return var
        

# 计算行平均，以及标准差
class RepSMD():
    def __init__(self, matrix):
        self.dat = matrix
        self.ave = matrix.mean(axis=1).to_numpy()
        self.std = matrix.std(axis=1).to_numpy()
        self.max_ = None
        self._maxAve()

    def _rep(self):
        dic = {}
        for i in range(self.dat.shape[1]):
            var = self.dat.iloc[:,i].to_numpy()
            dic[str(i)] = np.sum(var - self.ave)/len(var)
     
        return sorted(dic.items(),  key=lambda d: d[0], reverse=False)

    # 计算平均的最大拉力
    def _maxAve(self):
        max_index = np.argmax(self.ave)
        self.max_ = (self.ave[max_index], self.std[max_index], max_index)

    def pRep(self):
        dist = self._rep()
        var = "\n"
        for t in dist:
            i = eval(t[0])
            d = t[1]
            if (i != 0 and i % 4 == 0) or i == len(dist) - 1:
                var += "Curve[{}]: {:.2f}\n".format(i,d)
            else:
                var += "Curve[{}]: {:.2f}\t".format(i,d)
        return var
# 用于对矩阵进行W
class Wmatrix():
    def __init__ (self, Force, v):
        self.Force = Force
        self.v = v
        self.Work = self._w()

    def _calW(self, F):
        w = []
        w_= 0
        for i in F:
            w_ = w_ + i*self.v*1
            w.append(w_)
        return w
    
    def _w(self):
        w = {}
        for i in range(self.Force.shape[1]):
            w_ = self._calW(self.Force.iloc[:,i].to_list())
            w[str(i)] = w_
        return pd.DataFrame(w)

# 用于对w进行Jarzynski转换
def w_Jarzynski(W):
    K_B = 1.38*10**(-23)               # J/K
    NA = 6.02*10**23                   # mol-1
    T = 300                            # K
    beta = 1/(K_B * NA * 0.001 * T)    # kcal/mol
    pmf_Jarzynski = []
    std = []                           # 标准差
    for i in range(W.shape[0]):
        sum1 = 0
        sum2 = 0
        N = W.shape[1]
        std_ = []
        for j in range(N):
            sum1 += W.iloc[i,j]
            sum2 += W.iloc[i,j]**2
            std_.append(W.iloc[i,j])
        pmf = sum1/N -beta*(N/(N-1))*(sum2/N - (sum1/N)**2)/2
        std.append(np.array(std_).std())
        pmf_Jarzynski.append(pmf)
    return (pmf_Jarzynski,std)


def Parm():
    parser = argparse.ArgumentParser(description=
                                     "Fast Pulling Method Based on NAMD SMD for Calculating PMF\n"
                                     "The PMF calculation adopts the Jarzynski equation and undergoes second-order truncation\n"
                                     "Author: ZJH [HENU]"
                                    )
    parser.add_argument("-outF",type=int, nargs=1, help="SMDOutFreq")
    parser.add_argument("-k",type=str, nargs=1, help="SMDk [kcal/mol/A^2]")
    parser.add_argument("-v",type=str, nargs=1, help="SMDVel [A/timestep]")
    parser.add_argument("-dt",type=int, nargs=1, help="timestep [fs/timestep]")
    return parser.parse_args()

# 用于离散化矩阵
class discretize():
    def __init__(self, data, log) -> None:
        self.Matrix = data
        self.log = log
        self.dx = self._dx()
        self.S = self._dS()
        self._check()

    def _dx(self):
        new = {}
        for i in range(self.Matrix.shape[1]):
            # 起始归零
            var = [0]
            for j in range(1,self.Matrix.shape[0]):
                var.append(self.Matrix.iloc[j,i] - self.Matrix.iloc[j-1,i])
            new[self.Matrix.columns[i]] = var
        return pd.DataFrame(new)
    def _dS(self):
        new = {}
        for i in range(self.dx.shape[1]):
            # 注意是否需要var归零
            var = []
            sum = 0
            for j in range(self.dx.shape[0]):
                sum += self.dx.iloc[j,i]
                var.append(sum)
            new[self.dx.columns[i]] = var
        return pd.DataFrame(new)

    def _check(self):
        if self.dx.shape != self.S.shape:
            self.log.warning("Pleas check S matrix and dx matrix")

def outData(x,Y,fout):
    with open(fout,"w+") as F:
        if type(Y) == list:
            for i,x_ in enumerate(x):
                F.writelines(str(x_)+","+str(Y[i])+"\n")
        else:
            for i,x_ in enumerate(x):
                F.writelines(str(x_) + "," + ",".join([str(a) for a in  list(Y.iloc[i,:])])+"\n")
    return 0

def main():

    global loger 
    logging.basicConfig(filename="./SmdPmf.log"
                                 ,filemode="w+"
                                 ,format="%(asctime)s %(levelname)s:%(message)s"
                                 ,datefmt="%d-%M-%Y %H:%M:%S"
                                 ,level=logging.DEBUG)
    loger = logging.getLogger("logger")
    parm = Parm()
    loger.info("\nWelcome to use Smdpmf\nThis Scrip is used to process the Distance and files obtained from NAMD Smd Simulation")
    loger.info('Start to read dat from current directory.')
    data = ReadDat("./",loger)
    Force = data.Fmatrix
    Dist = data.Dmatrix
    t = GTimeSeries(Force,parm.outF[0],parm.dt[0])
    loger.info("\nForce: {} x {}\t Dist: {} x {}\nTime: {:.3f} -> {:.3f} (ns)\nFinish".format(
        Force.shape[0],Force.shape[1]
        ,Dist.shape[0],Dist.shape[1]
        ,t["ns"][0], t["ns"][-1])
        )
    
    loger.info("Start to cal S-T, the figure will be saved as S-T.jpg")
    data_discretize = discretize(Dist, loger)
    dx = data_discretize.dx
    S = data_discretize.S
    STd = STdist(t["ps"], S, lambda t: (eval(parm.v[0])*1000/(parm.dt[0])) * t )  
    DST(S,t["ps"],eval(parm.v[0])*1000/(parm.dt[0]),"Time (ps)","Distance (A)",STd.dist) 
    STd_out_str = STd.pDist()
    loger.info("\nThe R2 of S[Dummy Atom] and S[Smd Atom]:\n{}".format(STd_out_str))
    loger.info("S[real] will be saved in Sreal-T.csv")
    outData(t["ps"],S,"S-T.csv")

    loger.info("Start to draw F-T, the figure will be saved as pN-Time.jpg and the Force[pN] will be saved as F-T.csv")
    Rep = RepSMD(Force)
    Rep_out_str = Rep.pRep()
    Rep_max = Rep.max_
    loger.info(Rep_out_str)
    loger.info("Max Average : {} ± {}, index : {}".format(Rep_max[0], Rep_max[1], Rep_max[2]))
    DPlot(t["ps"],(Rep.ave,Rep.std),"Time (ps)","Pulling Force (pN)","Pulling_pN-Time")

    Force_ = Force.map(lambda x: x/69.479)
    loger.info("Start to draw F-T, the figure will be saved as Pulling_kcal_mol_A-Time.jpg and the Force[kcal/mol/A] will be saved as F-T.csv")
    Rep_ = RepSMD(Force_)
    Rep_out_str_ = Rep_.pRep()
    Rep_max_ = Rep_.max_
    loger.info(Rep_out_str_)
    loger.info("Max Average : {} ± {}, index : {}".format(Rep_max_[0], Rep_max_[1], Rep_max_[2]))
    
    DPlot(t["ps"],(Rep_.ave,Rep_.std),"Time (ps)","Pulling Force (kcal/mol/Å)","Pulling_kcal_mol_A-Time")
    outData(t["ps"],Force_,"F-T.csv") 

    loger.info("Start to draw W-T, the figure will be saved as PullingWork-Time.jpg and The Work[kcal/mol] will be saved as W-T.csv")
    w_data = Wmatrix(Force_,eval(parm.v[0]))
    w = w_data.Work
    DPlot(t["ps"],w,"Time (ps)","Pulling Work (kcal/mol)","PullingWork-Time")
    outData(t["ps"],w,"W-T.csv")

    loger.info("Start to draw pmf-d, the figure will be saved as PMF-Distance.jpg") 
    Jarzynski = w_Jarzynski(w)
    pmf,w_std = Jarzynski
    # pmf-distance
    loger.info("Max pmf is {:.2f}, and min is {:.2f}".format(max(pmf),min(pmf))) 
    DPlot([(eval(parm.v[0])/parm.dt[0])*i for i in t["fs"]],Jarzynski,"Distance (Å)","PMF (kcal/mol)","PMF-Distance") 
    pmf_ = pd.DataFrame({"pmf":pmf,"w_std":w_std})
    outData([(eval(parm.v[0])/parm.dt[0])*i for i in  t["fs"]],pmf_,"pmf-d.csv") 
    loger.info("The pmf[kcal/mol] will be saved as pmf-d.csv")
    loger.info("Finish")
    return 0

if __name__ == "__main__":
    main()
