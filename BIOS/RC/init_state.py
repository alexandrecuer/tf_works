# building volume in m3
vb=300
# air bulk density in kg/m3
rho_air=1.22
# air heat capacity in J/(kg.K)
c_air=1004
# heated floor area in m2
floor=80
# inertia in J/(K.m2)
inertia = 260000
# q4pasurf in m3/(h.m2)
q4pasurf=1.2
# atbat in m2 - off-floor loss area
atbat=217

# cs in J/K
cs=inertia*floor*100
# res in J/K
cres=c_air*rho_air*vb
# leakage resistance in K/W
rf=3600/(rho_air*c_air*q4pasurf*atbat)
# internal wall resistance in K/W
ri=1/(3*3*(atbat+floor))
# external wall resistance in K/W
r0=1/(17*(atbat+floor))
