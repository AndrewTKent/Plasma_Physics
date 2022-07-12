import matplotlib.pyplot as plt
import numpy as np

from read_hdf5_v2 import readhdf5
import matplotlib.colors as colors
import imageio
from pathlib import Path
from scipy import signal
from scipy.optimize import curve_fit
from pygifsicle import optimize

#%%

### Whistler Data for Different Frequencies  ###

whistler = {}
whistler['2MHz'] = readhdf5('./data/day_1_lab_3_f_2MHz.hdf5')
whistler['10MHz'] = readhdf5('./data/day_1_lab_3_f_10MHz.hdf5')
whistler['20MHz'] = readhdf5('./data/day_1_lab_3_f_20MHz.hdf5')
whistler['30MHz'] = readhdf5('./data/day_1_lab_3_f_30MHz.hdf5')
whistler['40MHz'] = readhdf5('./data/day_2_lab_3_f_40MHz.hdf5')
whistler['50MHz'] = readhdf5('./data/day_1_lab_3_f_50MHz.hdf5')
whistler['60MHz'] = readhdf5('./data/day_2_lab_3_f_60MHz.hdf5')
whistler['70MHz'] = readhdf5('./data/day_1_lab_3_f_70MHz.hdf5')
whistler['100MHz'] = readhdf5('./data/day_1_lab_3_f_2MHz.hdf5')
whistler['pulse'] = readhdf5('./data/day_2_lab_3_pulse.hdf5')

whistler['f_HDL_10MHz'] = readhdf5('./data/day_3 HDL 10MHz.hdf5')
whistler['f_HDL_20MHz'] = readhdf5('./data/day_3 HDL 20MHz.hdf5')
whistler['f_HDL_30MHz'] = readhdf5('./data/day_3 HDL 30MHz.hdf5')
whistler['f_HDL_40MHz'] = readhdf5('./data/day_3 HDL 40MHz.hdf5')
whistler['f_HDL_50MHz'] = readhdf5('./data/day_3 HDL 50MHz.hdf5')
whistler['f_HDL_60MHz'] = readhdf5('./data/day_3 HDL 60MHz.hdf5')
whistler['f_HDL_70MHz'] = readhdf5('./data/day_3 HDL 70MHz.hdf5')
whistler['f_HDL_100MHz'] = readhdf5('./data/day_3 HDL 100MHz.hdf5')
whistler['f_HDP_70MHz'] = readhdf5('./data/day_3 HDP 70MHz.hdf5')

whistler['iv_00'] = readhdf5('./data/day_3_IV A 00.hdf5')
whistler['iv_01'] = readhdf5('./data/day_3_IV A 01.hdf5')
whistler['iv_02'] = readhdf5('./data/day_3_IV A 02.hdf5')
whistler['iv_03'] = readhdf5('./data/day_3_IV A 03.hdf5')
whistler['iv_04'] = readhdf5('./data/day_3_IV A 04.hdf5')
whistler['iv_05'] = readhdf5('./data/day_3_IV A 05.hdf5')
whistler['iv_06'] = readhdf5('./data/day_3_IV A 06.hdf5')
whistler['iv_07'] = readhdf5('./data/day_3_IV A 07.hdf5')
whistler['iv_08'] = readhdf5('./data/day_3_IV A 08.hdf5')
whistler['iv_09'] = readhdf5('./data/day_3_IV A 09.hdf5')
whistler['iv_10'] = readhdf5('./data/day_3_IV A 10.hdf5')
whistler['iv_11'] = readhdf5('./data/day_3_IV A 11.hdf5')
whistler['iv_12'] = readhdf5('./data/day_3_IV A 12.hdf5')
whistler['iv_13'] = readhdf5('./data/day_3_IV A 13.hdf5')
whistler['iv_14'] = readhdf5('./data/day_3_IV A 14.hdf5')
whistler['iv_15'] = readhdf5('./data/day_3_IV A 15.hdf5')
whistler['iv_16'] = readhdf5('./data/day_3_IV A 16.hdf5')

whistler['lang_00'] = readhdf5('./data/day_3_Lang 00.hdf5')
whistler['lang_01'] = readhdf5('./data/day_3_Lang 01.hdf5')
whistler['lang_02'] = readhdf5('./data/day_3_Lang 02.hdf5')
whistler['lang_03'] = readhdf5('./data/day_3_Lang 03.hdf5')
whistler['lang_04'] = readhdf5('./data/day_3_Lang 04.hdf5')
whistler['lang_05'] = readhdf5('./data/day_3_Lang 05.hdf5')
whistler['lang_06'] = readhdf5('./data/day_3_Lang 06.hdf5')
whistler['lang_07'] = readhdf5('./data/day_3_Lang 07.hdf5')
whistler['lang_08'] = readhdf5('./data/day_3_Lang 08.hdf5')
whistler['lang_09'] = readhdf5('./data/day_3_Lang 09.hdf5')
whistler['lang_10'] = readhdf5('./data/day_3_Lang 10.hdf5')
whistler['lang_11'] = readhdf5('./data/day_3_Lang 11.hdf5')
whistler['lang_12'] = readhdf5('./data/day_3_Lang 12.hdf5')
whistler['lang_13'] = readhdf5('./data/day_3_Lang 13.hdf5')
whistler['lang_14'] = readhdf5('./data/day_3_Lang 14.hdf5')
whistler['lang_15'] = readhdf5('./data/day_3_Lang 15.hdf5')
whistler['lang_16'] = readhdf5('./data/day_3_Lang 16.hdf5')

#%%

### Generally Useful Vectors ###

# Creation of Label Vectors 
ch_label = ['ch1', 'ch2', 'ch3', 'ch4', 'time','pos']
b_label = [r'$\dot{B}_z$', r'$\dot{B}_y$', r'$\dot{B}_x$']
data_label_f = ['2MHz', '10MHz', '20MHz', '30MHz', '40MHz', '50MHz', '60MHz', '70MHz', '100MHz', 'pulse']
data_label_HD = ['f_HDL_10MHz','f_HDL_20MHz', 'f_HDL_30MHz', 'f_HDL_40MHz', 'f_HDL_50MHz', 'f_HDL_60MHz', 
                 'f_HDL_70MHz', 'f_HDL_100MHz', 'f_HDP_70MHz']
data_label_iv = ['iv_00', 'iv_01', 'iv_02', 'iv_03', 'iv_04', 'iv_05', 'iv_06', 'iv_07', 'iv_08', 'iv_09', 'iv_10', 
                 'iv_11', 'iv_12', 'iv_13', 'iv_14', 'iv_15', 'iv_16']
data_label_lang = ['lang_00', 'lang_01', 'lang_02', 'lang_03', 'lang_04', 'lang_05', 'lang_06', 'lang_07', 'lang_08', 
                   'lang_09', 'lang_10', 'lang_11', 'lang_12', 'lang_13', 'lang_14', 'lang_15', 'lang_16']

data_label_total = data_label_f + data_label_HD + data_label_iv + data_label_lang

# Creation of the Size Vectors for Total
elem_num_total = np.zeros(np.size(data_label_total))
for i in range(0, np.size(data_label_total)):
    elem_num_total[i] = np.size(whistler[data_label_total[i]][ch_label[3]][0,:])
    
# Creation of the Size Vectors for _f
elem_num_f = np.zeros(np.size(data_label_f))
for i in range(0, np.size(data_label_f)):
    elem_num_f[i] = np.size(whistler[data_label_f[i]][ch_label[3]][0,:])

# Creation of the Size Vectors for _HD
elem_num_HD = np.zeros(np.size(data_label_HD))
for i in range(0, np.size(data_label_HD)):
    elem_num_HD[i] = np.size(whistler[data_label_HD[i]][ch_label[3]][0,:])
    
# Creation of the Size Vectors for _iv
elem_num_iv = np.zeros(np.size(data_label_iv))
for i in range(0, np.size(data_label_iv)):
    elem_num_iv[i] = np.size(whistler[data_label_iv[i]][ch_label[3]][0,:])
    
# Creation of the Size Vectors for _lang
elem_num_lang = np.zeros(np.size(data_label_lang))
for i in range(0, np.size(data_label_lang)):
    elem_num_lang[i] = np.size(whistler[data_label_lang[i]][ch_label[3]][0,:])

#%%

### Construction of Theoretical Dispersion Relation ###

#Physical Paremeters (in MKS)
m_e =9.10938356*10**(-31)
c = 299792458
q = 1.60217662*10**(-19)
e_0 = 8.85418782*10**(-12)
n_e = 5*10**(16)
B = 0.0052

# Combined Parameters
W_e = q*B/m_e
W_p = np.sqrt(n_e*q**2/(e_0*m_e))
d_e = c/W_p

# k Parallel Vector 
k_par = np.linspace(0,75,num=1000)

# Dispersion Relation
w_k = W_e * k_par**2 * d_e**2 / (1 + k_par**2 * d_e**2 )

# Data for W vs. K from Experiement (in MKS)
w_k_data = [10*10**6*2*np.pi, 20*10**6*2*np.pi, 30*10**6*2*np.pi, 40*10**6*2*np.pi, 50*10**6*2*np.pi, 60*10**6*2*np.pi, 70*10**6*2*np.pi, 100*10**6*2*np.pi] 
k_par_data = [2*np.pi/.1427, 2*np.pi/.1671, 2*np.pi/.1904, 2*np.pi/.2148, 2*np.pi/.2029, 2*np.pi/.1691, 2*np.pi/.1461, 2*np.pi/.1053]

#%% 

### Dispersion Relation Graph ###

# Initialize Plot
plt.figure(figsize = (10, 8))

# General Plot
plt.plot(k_par, w_k, label = r'$\omega(k) = \Omega_e \frac{k_\parallel^2 \delta_e^2}{1 + k_\parallel^2 \delta_e^2}$')
plt.scatter(k_par_data, w_k_data, label = 'Data')

# Label Plot
plt.title('Dispersion Relation: Data and Theory', fontsize = 22)
plt.xlabel(r'$k_\parallel$ [1/m]', fontsize = 18)
plt.ylabel(r'$\omega$ [rad/s]', fontsize = 18)
plt.legend(loc = 'best', fontsize = 18)


#%%

### Tricontour Graph, Saving PNG's ###

# Intrerpolation Parameters
z_max = 60
z_min = 10
theta_max = 6
theta_min = -6
levels_num = 25
linewidth_num = 0.1
color_map_max = 1
shot = 100
n_col = 1
n_row = 4
data_run = 9
elem_min = 850
elem_max = 1000
single_shot = 1050

# Create the Position Vectors
num_pos = np.size(whistler[data_label_f[data_run]][ch_label[5]][:])
z_pos = np.zeros(num_pos)
theta_pos = np.zeros(num_pos)

for l in range(0, num_pos):        
    z_pos[l] = whistler[data_label_f[data_run]][ch_label[5]][l][1]
    theta_pos[l] = whistler[data_label_f[data_run]][ch_label[5]][l][2]

for k in range(single_shot, single_shot + 1):#(0, int(elem_num_HD[data_run])):        
    print(k)
    
    fig, axs = plt.subplots(nrows = n_row, ncols = n_col, figsize = (14, 9))
    divnorm = colors.DivergingNorm(vmin=-1*color_map_max, vcenter=0, vmax=color_map_max)
    
    axs[0].plot(whistler[data_label_f[data_run]][ch_label[4]], whistler[data_label_f[data_run]][ch_label[0]][shot,:])   
    axs[0].scatter(whistler[data_label_f[data_run]][ch_label[4]][k], whistler[data_label_f[data_run]][ch_label[0]][shot,k], s = 100)
    axs[0].set(ylabel = 'RF Wave Amplitude') 
    axs[0].set_title('RF Waveform: f = ' + data_label_f[data_run][-5:], fontsize = 18)
    
    for i in range(1, n_row):
            
            data_vec = whistler[data_label_f[data_run]][ch_label[i]][:,k]
        
            axs[i].tricontour(z_pos, theta_pos, data_vec, levels = levels_num, linewidths = linewidth_num, colors='k')
            cntr2 = axs[i].tricontourf(z_pos, theta_pos, data_vec, levels = levels_num, cmap="RdBu_r",norm=divnorm)

            
            axs[i].scatter(z_pos,theta_pos, c = data_vec, s = 20, edgecolors = 'k', alpha = 0.5, cmap = 'RdBu_r', norm = divnorm)
            axs[i].set(xlim=(z_min, z_max), ylim=(theta_min, theta_max), ylabel = 'Angle [Degrees]')
            axs[i].set_title(b_label[i - 1], fontsize = 18)
            
            if i == 3:
                axs[i].set(xlim=(z_min, z_max), ylim=(theta_min, theta_max), ylabel = 'Angle [Degrees]', xlabel = 'Distance [cm]')
                
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=divnorm)
    plt.subplots_adjust(top=0.95, bottom=0.065, left=0.055, right=0.825, hspace=0.45, wspace=0.085)
    cbar_ax = fig.add_axes([0.85, 0.065, 0.02, 0.65])
    plt.colorbar(sm, cax=cbar_ax)
    fig.savefig('source_images/'+str(k) + '.png', dpi=600)
    plt.close()          
    
#%%

### Creation of the GIF ###

# Data Initializers
image_path = Path('source_images')
images = []
image_list = []

# Creation of GIF
for i in range(elem_min, elem_max):#(0, int(elem_num_f[data_run])): 
    images = images + list(image_path.glob(str(i)+'.png'))

for file_name in images:
    image_list.append(imageio.imread(file_name))

imageio.mimwrite('pulse.gif', image_list, fps = 40)

#%%

### Optimize the GIF ###

gif_path = 'f_70MHz_HDP.gif'

# create a new one
optimize(gif_path, 'f_70MHz_HDP_optimized.gif')

# overwrite the original one
optimize(gif_path)

#%%

### Looking at the High Density Linear Maps After Smoothing, Then Fitting ###

# Plot Parameters
data_run = 6
shot = 675

# Create the Position Vectors
num_pos = np.size(whistler[data_label_HD[data_run]][ch_label[5]][:])
z_pos = np.zeros(num_pos)

for l in range(0, num_pos):        
    z_pos[l] = whistler[data_label_HD[data_run]][ch_label[5]][l][1]

# Wave Function Definition
def damped_fit(z, A, g, a, w, d):
    return  A*np.exp(-g*(z-a))*np.sin(w*(z-a)) + d

# Smoothing Parameters
window = 31
order = 4

# Creating Analyzed Data Vectors
smoothed_data = signal.savgol_filter(whistler[data_label_HD[data_run]][ch_label[3]][:,shot], window, order)
wave = whistler[data_label_HD[data_run]][ch_label[3]][:,shot]
params, params_covariance = curve_fit(damped_fit, z_pos, smoothed_data)
fitted_z_pos = np.linspace(z_pos[0], z_pos[num_pos - 1], 1000)
fitted_wave = params[0]*np.exp(-1*params[1]*(fitted_z_pos - params[2]))*np.sin(params[3]*(fitted_z_pos - params[2])) + params[4]

# Initialize Plot
fig = plt.figure(figsize = (14, 8))

# General Plot
plt.plot(z_pos, wave, label = 'Whister Data')
plt.plot(z_pos, smoothed_data, label = 'Smoothed Data')
plt.plot(fitted_z_pos, fitted_wave, label = 'Fitted Data')

# Label Plot
plt.title('Whistler Wave Amplitude vs. Z Position, ' + 'f = ' + str(data_label_HD[data_run][-6:]) + ' 1/s ' + r', $\lambda = $ ' + str(2*np.pi/params[3])[:5] + r', $|\Gamma| =$ ' + str(np.abs(params[1]))[:5] + ' [cm]', fontsize = 22)
plt.xlabel('Z Position [cm]', fontsize = 18)
plt.ylabel('Wave Amplitude', fontsize = 18)
plt.legend(loc = 'best', fontsize = 18)

fig.show()


#%% 

### Density Calculation ###

# Parameters
R = 20/3
data_run = 0
left_offset = 3500
right_offset = 4125
voltage_left = whistler[data_label_iv[data_run]][ch_label[1]][0,left_offset]
voltage_right = whistler[data_label_iv[data_run]][ch_label[1]][0,int(elem_num_iv[data_run]) - right_offset]

# Exp Function Definition
def exp_funct(x, m, b, a):
    return  np.exp(m * x + b) + a

# Smoothing Parameters
window = 31
order = 4

# Data Vectors
voltage_data = whistler[data_label_iv[data_run]][ch_label[1]][0,:]
current_data = whistler[data_label_iv[data_run]][ch_label[2]][0,:]/R

# Cut Up Vectors 
cut_current = np.zeros(int(elem_num_iv[data_run]) - right_offset - left_offset)
cut_voltage = np.zeros(int(elem_num_iv[data_run]) - right_offset - left_offset)

for i in range(left_offset, int(elem_num_iv[data_run]) - right_offset):
    cut_current[i - left_offset] = current_data[i]
    cut_voltage[i - left_offset] = voltage_data[i]

# Smoothing Parameters
window = 501
order = 5

# Fitting Vectors
smoothed_current = signal.savgol_filter(cut_current, window, order)
smoothed_voltage = signal.savgol_filter(cut_voltage, window, order)
params, params_covariance = curve_fit(exp_funct, smoothed_voltage, smoothed_current)
fitted_voltage = np.linspace(voltage_left, voltage_right, 1000)
fitted_current = np.exp(params[0] * fitted_voltage + params[1]) + params[2]

# Finding the Plasma Temperature
slope = params[0]
T_ev = 1/slope

# Finding the Plasma Density
def plasma_density(I_esat, area_probe, kT, mass_electron, charge_electron):
    return I_esat / ( area_probe * charge_electron  *np.sqrt( kT / (2 * np.pi * mass_electron)) )

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

# Problem Parameters (in MKS)
area_probe = 6.33*10**(-5)
mass_electron = 9.11*10**(-31)
charge_electron = 1.602*10**(-19)
ev_joule_conversion = 6.242*10**18

# Density Calculation 
I_esat = .187
kT = T_ev/ev_joule_conversion     
density = format_e(plasma_density(I_esat, area_probe, kT, mass_electron, charge_electron))


# Initialize Plot
fig = plt.figure(figsize = (14, 8))

# General Plot
plt.scatter(voltage_data, current_data, label = 'Raw IV Data', s = .05)
plt.scatter(cut_voltage, cut_current, label = 'Cut Raw IV Data', s = .6)
plt.plot(smoothed_voltage, smoothed_current, label = 'Smoothed Data')
plt.plot(fitted_voltage, fitted_current, label = 'Fitted Data')

# Label Plot
plt.title('Z = ' + str(data_label_iv[data_run][-2:]) + r' cm,  $K_B T_e = $' + str(T_ev)[:5] + ' eV' + r',  $n_e$ = ' + str(density) + r' $ \frac{1}{m^3}$ ', fontsize = 22)
plt.xlabel('Voltage [V]', fontsize = 18)
plt.ylabel('Current [A]', fontsize = 18)
plt.legend(loc = 'best', fontsize = 18, markerscale=9)

fig.show()

#%%

### I_{sat} Calculation ### 

# Parameters
R = 20/3
data_run = 7
shot = 565

# Create the Position Vector
num_pos = np.size(whistler[data_label_lang[data_run]][ch_label[5]][:])
z_pos = np.zeros(num_pos)

for l in range(0, num_pos):        
    z_pos[l] = whistler[data_label_lang[data_run]][ch_label[5]][l][1]

# Data Vector Creation
i_sat = whistler[data_label_lang[data_run]][ch_label[2]][0,:]/R
time = whistler[data_label_lang[data_run]][ch_label[4]][0,:]

# Initialize Plot
fig = plt.figure(figsize = (14, 8))

# General Plot
plt.plot(z_pos, wave, label = 'Whister Data')
plt.plot(z_pos, smoothed_data, label = 'Smoothed Data')
plt.plot(fitted_z_pos, fitted_wave, label = 'Fitted Data')

# Label Plot
plt.title(r'$I_{sat}$ vs. Time', fontsize = 22)
plt.xlabel('Z Position [cm]', fontsize = 18)
plt.ylabel('Wave Amplitude', fontsize = 18)
plt.legend(loc = 'best', fontsize = 18)

fig.show()

#%%

### Construction of Theoretical Dispersion Relation ###

#Physical Paremeters (in MKS)
m_e =9.10938356*10**(-31)
c = 299792458
q = 1.60217662*10**(-19)
e_0 = 8.85418782*10**(-12)
ev_joule_conversion = 6.242*10**18
n_e = 5*10**(16)
B = 0.0052
n_g = 10**(17)
sigma = 10**(-9)
T_ev = 7.7*10**(16)
KT = T_ev/ev_joule_conversion  
lambda_mfp = 1/(n_g * sigma)
v_th = np.sqrt(2*kT / m_e)

# Combined Parameters
W_e = q*B/m_e
W_p = np.sqrt(n_e*q**2/(e_0*m_e))
d_e = c/W_p
nu = v_th/lambda_mfp

# k Parallel Vector 
omega = np.linspace(5*10**6*2*np.pi,100*10**6*2*np.pi,num=100)

# Dispersion Relation
gamma_fit = np.sqrt(    (  nu*omega  /d_e**2) * 1/( (W_e - omega)**2  + nu**2) )

# Data for W vs. K from Experiement (in MKS)
w_k_data = [10*10**6*2*np.pi, 20*10**6*2*np.pi, 30*10**6*2*np.pi, 40*10**6*2*np.pi, 50*10**6*2*np.pi, 60*10**6*2*np.pi, 70*10**6*2*np.pi, 100*10**6*2*np.pi] 
#gamma_data = [1/0.0036, 1/0.097, 1/0.032, 1/0.054, 1/0.032, 1/0.023, 1/0.029, 1/0.054]
gamma_data = [0.036, 0.0097, 0.032, 0.054, 0.032, 0.023, 0.029, 0.054]

### Spatial Damping Relation Graph ###

# Initialize Plot
plt.figure(figsize = (12, 8))

# General Plot
plt.scatter(w_k_data, gamma_data, label = 'Data')
plt.plot(omega, gamma_fit, label = 'Theory')

# Label Plot
plt.title(r'Spatial Damping $(1/\Gamma)$ vs RF Angular Frequency $(2 \pi /f)$, ', fontsize = 22)
plt.ylabel(r'Spatial Damping $[1/cm]$', fontsize = 18)
plt.xlabel(r'RF Angular Frequency $[rad/s]$', fontsize = 18)
plt.legend(loc = 'best', fontsize = 18)




#%%

print(lambda_mfp)




















