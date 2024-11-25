import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from galpy.orbit import Orbit
import astropy.units as u
from astropy.coordinates import Galactocentric, ICRS, SkyCoord
import astropy.coordinates as coord
from astropy.coordinates import ICRS,Galactic
from scipy.stats import binned_statistic_2d
import pandas as pd
from astropy.coordinates import CartesianDifferential
from astropy.table import Table
import time
from galpy.potential import MWPotential2014
from galpy.potential.mwpotentials import McMillan17, Irrgang13I, DehnenBinney98I
from galpy.util.bovy_conversion import get_physical, physical_compatible
import statsmodels.api as sm
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.odr import ODR, Model, Data, RealData
import seaborn as sns
from galpy.potential import MWPotential2014
from scipy.stats import gaussian_kde
import scipy.ndimage as ndimage
import gc

file = Table.read("/home/chenao/for_draft/dr8_age_rclog_rgb_kinematic_final_printer.fits")
age_data = file.to_pandas()
age_data = age_data.set_index('obsid')
file = Table.read("/home/chenao/for_draft/dr8_final_vac_flag_parameter.fits")
other = file.to_pandas()
other = other.set_index('obsid')

use_list = ['FEH_APOGEE',
 'MH_APOGEE',
 'CH_APOGEE',
 'NH_APOGEE',
 'CFE_APOGEE',
 'NFE_APOGEE',
 'AFE_APOGEE',
 'LOGG_APOGEE',
 'snrg',
 'rv',
 'err_rv',
 'err_feh_apogee',
 'err_mh_apogee',
 'err_afe_apogee',
 'err_cfe_apogee',
 'err_nfe_apogee',
 'err_logg_apogee',
 'mh_afe',
 'err_mh_afe',
 'GAIAID',
 'uqflag',
 'dist_phot',
 'err_dist_phot',
 'x',
 'err_x',
 'y',
 'err_y',
 'z',
 'err_z',
 'r',
 'err_r',
 'phi',
 'err_phi',
 'vr',
 'vphi',
 'vz',
 'e_vr',
 'e_vphi',
 'e_vz']
other = other[(other['flag_feh_apogee']==0)&(other['flag_afe_apogee']==0)]
index = age_data.index.intersection(other.index)
age_data.loc[index,use_list] = other.loc[index,use_list]
age_data['R'] = (age_data['x']**2+age_data['y']**2)**(1/2)
age_data.loc[age_data['status']>0,'status']=1

data = age_data[(age_data['snr']>20)&(((age_data['status']==0)&(age_data['err_age_rgb']<0.3)&(age_data['mass_rgb']>0.7))|((age_data['status']==1)&(age_data['err_age_rc']<0.3)&(age_data['mass_rc']>0.7)))&(np.abs(age_data['z'])<10000)&((age_data['R']>5000)&(age_data['R']<20000))&np.all(np.abs(age_data[['err_r','e_vr','e_vphi','err_z','e_vz']].values/age_data[['r','vr','vphi','z','vz']]).values<0.7,axis=1)]

pot_ = Irrgang13I
ro = 8.27*1000
vo = 236
i=2

normalized = data[['r','vr','vphi','z','vz','err_r','e_vr','e_vphi','err_z','e_vz']].values/np.array([ro,vo,vo,ro,vo,ro,vo,vo,ro,vo])
N = len(normalized)
print(N)
X = np.random.randn(100*N, 5)
copies = (normalized*np.ones((100,N,10))).reshape((100*N,10),order="F")
re_xxvvs = X*copies[:,5:] + copies[:,:5]
o = Orbit(re_xxvvs,**get_physical(pot_))
samp_rgs=o.rguiding(pot=pot_).reshape((N,100))
rg = np.mean(samp_rgs,axis=1)
erg = np.std(samp_rgs,axis=1)
np.save(f'/home/chenao/for_draft/other_pot/rg{i}.npy',rg)
np.save(f'/home/chenao/for_draft/other_pot/erg{i}.npy',erg)
for j in range(8):
    o=Orbit(normalized[40000*j:40000*(j+1),:5],**get_physical(pot_))
    ts=np.linspace(0.,150.,10001)
    o.integrate(ts,pot_)
    zmax = o.zmax()
    ecc = o.e()
    np.save(f'/home/chenao/for_draft/other_pot/zmax_40k_{i}_{j}.npy',zmax)
    np.save(f'/home/chenao/for_draft/other_pot/ecc_40k_{i}_{j}.npy',ecc)
    del o
    gc.collect()

o=Orbit(normalized[320000:,:5],**get_physical(pot_))
ts=np.linspace(0.,150.,10001)
o.integrate(ts,pot_)
zmax = o.zmax()
ecc = o.e()
np.save(f'/home/chenao/for_draft/other_pot/zmax_40k_{i}_8.npy',zmax)
np.save(f'/home/chenao/for_draft/other_pot/ecc_40k_{i}_8.npy',ecc)
del o
gc.collect()

ecc = []
zmax = []
for j in range(9):
    ecc.append(np.load(f'/home/chenao/for_draft/other_pot/ecc_40k_{i}_{j}.npy'))
    zmax.append(np.load(f'/home/chenao/for_draft/other_pot/zmax_40k_{i}_{j}.npy'))
ecc = np.hstack(ecc)
zmax = np.hstack(zmax)
print(len(ecc))
rg = np.load(f'/home/chenao/for_draft/other_pot/rg{i}.npy')
erg = np.load(f"/home/chenao/for_draft/other_pot/erg{i}.npy")
data[['zmax','ecc','rg','erg']] = np.vstack((zmax,ecc,rg,erg)).T
data = data[(data['FEH_APOGEE']>=-1.0)&(data['vphi']>=0)]
data.loc[data['status']==0,'age'] = data.loc[data['status']==0,'age_rgb'].values
data.loc[data['status']==1,'age'] = data.loc[data['status']==1,'age_rc'].values

def linear_func (p,x):
    return p[0]*x + p[1]
agebin = np.linspace(1,12.25,10,endpoint=True)
age_binedge = (agebin[:-1] + agebin[1:])/2
def chem_select(meta,a=-.3,b=.12,k=-.14):
    if meta<a:
        return b
    else: return k*(meta-a) + b
chem_select = np.vectorize(chem_select)
def gradient_cal(data,rg_name, feh_name, age_name,erg_name=None, efeh_name=None):
    if erg_name:
        rg, feh, age, erg, efeh = data[[rg_name,feh_name,age_name,erg_name,efeh_name]].values.T
        agebin = np.linspace(1,12.25,10,endpoint=True)
        meta_gradients = []
        b_=[]
        error_meta_gra = []
        for i in range(len(agebin)-1):
            w = np.bitwise_and((age>agebin[i]),(age<agebin[i+1]))
            x = rg[w]
            y = feh[w]
            xe = erg[w]
            ye = efeh[w]
            model = Model(linear_func)
            data_ = RealData(x, y, sx=xe, sy=ye)
            odr = ODR(data_, model, beta0=[0., -5.])
            odr.set_job(fit_type=2)
            out = odr.run()
            slope = out.beta[0]
            intercept = out.beta[1]
            eslope = out.sd_beta[0]
            einter = out.sd_beta[1]
            meta_gradients.append(slope)
            b_.append(intercept)
            error_meta_gra.append(eslope)
            feh = feh[w == False]
            rg = rg[w == False]
            age = age[w == False]
            erg = erg[w == False]
            efeh = efeh[w == False]
    return (meta_gradients,b_, error_meta_gra)
def thin_and_thick_grad(data,i,data_name='W23', a=-0.3, b=0.12, k=-0.14,rg_name='rg',feh_name='FEH_APOGEE',age_name='age',erg_name='erg',afe_name='AFE_APOGEE',zmax_name='zmax',efeh_name='err_feh_apogee',ecc_name=False):
    if ecc_name:
        thin = data[(data[afe_name]<chem_select(data[feh_name],a,b,k))&(data[zmax_name]<0.25)&(data[ecc_name]<=(1-0.8**2)**(1/2))]
        thick = data[(data[afe_name]>chem_select(data[feh_name],a,b,k))&(data[zmax_name]>2)&(data[ecc_name]<(1-0.2**2)**(1/2))&(data[ecc_name]>(1-0.8**2)**(1/2))]
    else:
        thin = data[(data[afe_name]<chem_select(data[feh_name],a,b,k))&(data[zmax_name]<0.25)]
        thick = data[(data[afe_name]>chem_select(data[feh_name],a,b,k))&(data[zmax_name]>2)]
    thin_meta_grad,thin_b, thin_err = gradient_cal(thin,rg_name, feh_name, age_name,erg_name, efeh_name)
    thick_meta_grad,thick_b, thick_err = gradient_cal(thick,rg_name, feh_name, age_name,erg_name, efeh_name)
    np.save(f"/home/chenao/for_draft/other_pot/thin_meta_grad{i}.npy",thin_meta_grad)
    np.save(f"/home/chenao/for_draft/other_pot/thick_meta_grad{i}.npy",thick_meta_grad)
    np.save(f"/home/chenao/for_draft/other_pot/thin_err{i}.npy",thin_err)
    np.save(f"/home/chenao/for_draft/other_pot/thick_err{i}.npy",thick_err)
    return 0
thin_and_thick_grad(data,i)