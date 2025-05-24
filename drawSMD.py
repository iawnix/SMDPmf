#!/home/hang/.conda/envs/py37/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
import os
import math


# 设置字体格式
plt.rcParams["axes.labelweight"] ="bold"
plt.rcParams["font.family"]="Times New Roman"
plt.rcParams["font.weight"]="bold"
plt.rcParams["font.size"]=10



def GetDAT(path):
    out = []
    for file in os.listdir(path):
                if ".dat" in file:
                    out.append(file.split(".")[0])
    return out
        


def draw(var : dict , dt,cnum):
    plt.figure(figsize=(7,5),dpi=300)
    key = list(var.keys())
    ax = plt.gca()

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)


    # 计算平均曲线
    x_ = var[key[0]]["fra"]
    t_ = [dt*i for i in x_]

    x_len = len(x_)
    y_average = []
    for i in range(x_len):
        var_sum = 0
        for j in range(len(key)):
            var_sum += var[key[j]]["F"][i]
        y_average.append(var_sum/len(key))
    
    # 计算欧式距离
    D = {}
    for i in range(len(key)):
        var_sum = 0
        for j in range(x_len):
            var_sum += math.pow((var[key[i]]["F"][j]) -(y_average[j]), 2 )
        D[key[i]] = math.pow(var_sum, 0.5)
    nearD = sorted(D.items(),key=lambda s:s[1])[0][0]
    
    # 绘制10条曲线
    for i in range(len(key)):
        x = var[key[i]]["fra"]
        t = [dt*i for i in x]

        if key[i] == nearD:
            plt.plot(t,var[key[i]]["F"],"#091A7A", linestyle='-',marker = 'o',markersize = 0.01, linewidth = 3,label="Rep",alpha=1)
        else:
            # 除rep之外的SMD曲线
            plt.plot(t,var[key[i]]["F"],color[cnum], linestyle='-',marker = 'o',markersize = 0.01, linewidth = 10,alpha=0.3)


    print("Distance to Average output")
    for i in D.keys():
         print("{}\t{:.2f}".format(i,D[i]))       
    print()

    select = var[nearD]
    x_i,y_i = select.idxmax(axis = 0)
    print("The frame of {}'s Max Pulling Force = {}, and index = {}".format(nearD,select["fra"][y_i], y_i+1))
    print()

    # 绘制平均曲线
    plt.plot(t_,y_average,"#B70040", linestyle='-',marker = 'o',markersize = 0.01, linewidth = 3, label = "Average",alpha=1)

    
    # 绘制平均曲线的峰值线
    plt.plot(t_,[max(y_average) for  i in t_],color = "black",alpha = 0.6,linewidth= 2,linestyle='--')
    plt.text(0,max(y_average)+30,s="{:.2f}".format(max(y_average)),ha = "left")
    #ax.set_xlabel("Times (fs)")
    ax.set_xlabel("Times (ps)")
    ax.set_ylabel("Pulling Force (pN)")
    plt.legend(markerscale= 200)
    plt.savefig("./SMD-10.jpg")
    plt.savefig("./SMD-10.tiff")

    # 返回平均曲线
    return [t_, y_average]

def main():

    path = "./"
    dt = eval(sys.argv[1])
    cnum = eval(sys.argv[2])
    global color 
    color = ["#38184C","#164C45","#CC8D1A","#10454F","#16232E","#024059","#802922","#4A4633","#BDE038","#FF81D0"]

    #color = ["#38184C","#164C45","#10454F","#16232E","#024059","#CC8D1A","#802922","#4A4633","#BDE038","#FF81D0"]

    dat = GetDAT(path)
    var_dict = {}
    max_list = []
    for i in dat:
        var_dict["SMD{}".format(i)] = pd.read_csv("{}.dat".format(i),header=None,names=["fra","F"],sep=",")
        tmp = var_dict["SMD{}".format(i)]["F"].max()
        max_list.append(tmp)
        print("SMD{} : {}".format(i,tmp))
    print("\n")
    min_ = min(max_list)
    max_ = max(max_list)
    print("max_Fmax : {}".format(max_))
    print("min_Fmax : {}".format(min_))
    

    Avg = draw(var_dict, dt, cnum)
    #print(Avg)
    pd.DataFrame({"fra":Avg[0],"F":Avg[1]}).to_csv("./Avg-SMD.csv",index=0)

if __name__ == "__main__":
    main()


