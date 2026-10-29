import numpy as np, pandas as pd
TYPES=[(10_000.0,60.0,150.0),(20_000.0,90.0,250.0),(40_000.0,150.0,450.0)]; MEAN_LEN=6e4
def scales(nh,vph,rho):
    hm=np.array([TYPES[h%3][0] for h in range(nh)]); pi=np.array([TYPES[h%3][1] for h in range(nh)])
    total=hm.sum(); rate=rho*total/MEAN_LEN; m=1/(0.55*rate); e=pi.sum()*m; mean_exec=MEAN_LEN/(total/(nh*vph)); return m,e,mean_exec
def add_components(R,nh,vph,rho):
    m,e,x=scales(nh,vph,rho); R=R.copy()
    R["c_mk"]=R.makespan/m; R["c_en"]=(R.idle_j+R.dyn_j)/e; R["c_late"]=R.late.astype(float); R["c_resp"]=R.resp/(5*x)
    R["kwh"]=(R.idle_j+R.dyn_j)/3.6e6; R["sla"]=100*R.late/R.N
    R["J"]=-(R.c_mk+R.c_en+R.c_late+R.c_resp); R["J1000"]=R.J/R.N*1000; return R
