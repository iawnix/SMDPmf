#!/home/hang/.conda/envs/py37/bin/python
import subprocess
import sys
import math
from decimal import Decimal

##############################################################
##Author:Zhai Jihang                                     #####
##Date:2022-01-08                                        #####
##Institution:HENU                                       #####
##Description: python GetSmdxyz.py x y z yourfile        #####
##Other:Linux;python3.8                                  #####
##############################################################
class parmdata():
    def __init__(self):
        self.X = Decimal(sys.argv[1])
        self.Y = Decimal(sys.argv[2])
        self.Z = Decimal(sys.argv[3])
        self.file = sys.argv[4]

def Cal(line_list,parmdata):
    dout = []
    fout = []
    for line in line_list:
        var = line.split()
        #['SMD  10 26.6816 31.1275 7.85791 -1.0346 -1.24614 -0.701124']
        #   0    1    2       3      4       5        6       7
        if len(var) == 8:
            #tmp1 = [var[1],str(math.sqrt(pow(Decimal(var[2]),2)+pow(Decimal(var[3]),2)+pow(Decimal(var[4]),2)))]
            tmp1 = [var[1],str(parmdata.X*Decimal(var[2])+parmdata.Y*Decimal(var[3])+parmdata.Z*Decimal(var[4]))]           # 20230215
            tmp = [var[1],str(parmdata.X*Decimal(var[5])+parmdata.Y*Decimal(var[6])+parmdata.Z*Decimal(var[7]))]
            fout.append(",".join(tmp))
            dout.append(",".join(tmp1))
        else:
            print("Error: SmdDataBase is not 8 \n")

    return (dout,fout)

def GetSMD(parmdata):
    
    cmd1 = lambda file:"cat {}".format(file)
    cmd2 = 'grep "^\<SMD\>"'
    p1 = subprocess.Popen(cmd1(parmdata.file), bufsize=-1, shell=True, encoding = "utf-8", stdout=subprocess.PIPE)
    p2 = subprocess.Popen(cmd2,bufsize=-1,shell=True,encoding = "utf-8",stdin=p1.stdout,stdout=subprocess.PIPE)
    ret = p2.communicate(input=None)
    
    if not p2.returncode:
        line_list = ret[0].splitlines()
        print("num of line is {} \n".format(len(line_list)))  
        
        return Cal(line_list,parmdata) 
            
def writelist(result,parmdata,SIGN):
    savefile = SIGN + parmdata.file.split(".")[0] + ".dat"
    with open(savefile,mode="w") as S:
        for line in result:
            S.writelines(line+"\n")

def main():
    myargv = parmdata()
    result = GetSMD(myargv)
    writelist(result[0],myargv,"D")
    writelist(result[1],myargv,"F")
    
    return 0

if __name__ == "__main__":
    main()
