#!/home/iaw/soft/conda/2024.06.1/envs/pytorch3.9/bin/python
##################################################################################
##python GetPBC.py pdb.pdb
##Zhai Jihang               HENU
##Linux python3.7
##################################################################################




from decimal import Decimal
import subprocess
import os
import sys

def Pipe(cmd1 : str, cmd2 : str) -> str:
    ret = subprocess.Popen(cmd1,bufsize=-1,shell=True,encoding="utf-8",stdout = subprocess.PIPE,stderr=subprocess.PIPE)
    out = ret.communicate(input=None)
    out1,error1 = out[0],out[1]
    code1 = ret.returncode
    if error1 != "":
        if not code1:								# 解决UBUNTU上浮点数的Note导致程序跳出
            print("Sucessful: {},But: {}".format(cmd1,error1))
            return out1
        else:
            print("Error: {}".format(error1))
            sys.exit(1)
    else:
        ret_ = subprocess.Popen(cmd2,bufsize=-1,shell=True,encoding="utf-8",stdout = subprocess.PIPE,stdin = subprocess.PIPE,stderr=subprocess.PIPE)
        out_ = ret_.communicate(input=out1)
        out2,error2 = out_[0],out_[1]
        code2 = ret_.returncode
        if error1 != "":
            if not code2:								# 解决UBUNTU上浮点数的Note导致程序跳出
                print("Sucessful: {},But: {}".format(cmd2,error2))
                return out2
            else:
                print("Error: {}".format(error2))
                sys.exit(1)
        else:
            print("Sucessful: {}".format(cmd2))
            return out1
        
def runCMD(cmd:str) -> str:
    ret1 = subprocess.Popen(cmd,bufsize=-1,shell=True,encoding="utf-8",stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    ret1_ = ret1.communicate(input=None)
    out1,error1 = ret1_[0],ret1_[1]
    code = ret1.returncode
    if error1 != "":
        if not code:								# 解决UBUNTU上浮点数的Note导致程序跳出
            print("Sucessful: {},But: {}".format(cmd,error1))
            return out1
        else:
            print("Error: {}".format(error1))
            sys.exit(1)
    else:
        print("Sucessful: {}".format(cmd))
        return out1

def cmdout2list(string : str) -> list:
    var = []
    for i in string.split("\n"):
        if i != "":
            var.append(i)
    return var

def FILE(pdb : str,parm : list,out : str):
    var = ["mol new {%s} type {pdb} first 0 last -1 step 1 waitfor 1 \n"%(pdb)
            ,"animate style Loop \n"
            ,"set allatoms [atomselect top all] \n"
            ,"$allatoms set beta 0 \n"
            ,"set fixedatom [atomselect top \"resid %s and name %s \"] \n"%(parm[0],parm[1])
            ,"$fixedatom set beta 1 \n"
            ,"$allatoms set  occupancy 0 \n"
            ,"set smdatom [atomselect top \"resid %s and name %s \"] \n"%(parm[2],parm[3])
            ,"$smdatom set occupancy 1 \n"
            ,"$allatoms writepdb %s.ref \n"%(out)
            ,"set smdpos [lindex [$smdatom get {x y z}] 0] \n"
            ,"set fixedpos [lindex [$fixedatom get {x y z}] 0] \n"
            ,"puts \"zjhzjhzjhzjh\" \n"
            ,"vecnorm [vecsub $smdpos $fixedpos] \n"
            ,"quit \n"
    ]
    return var


def main():
    Pdb = sys.argv[1]                        # pdb
    in_ = sys.argv[2].split(",")
    out_ = sys.argv[3]
    var_s = "var"

    with open("var",mode='w+') as I:
        file = FILE(Pdb,in_,out_)
        I.writelines(file)

    out = cmdout2list(runCMD("vmd -e {}".format(var_s)))
    os.system("rm var")
    for i  in range(len(out)):
        if "zjhzjhzjhzjh"in out[i]:
            for j in out[i+1].split(" ") :
                print(j,end="\t")
            print("\n")
            break
            #print(out[out.index(i)+1])
    #print(out)
if __name__ == "__main__":
    main()
