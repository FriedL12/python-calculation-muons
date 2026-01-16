import numpy as np
from scipy.integrate import quad
from scipy.integrate import simpson
import matplotlib.pyplot as plt

############
# Parameters
# programm parameters
binsize = 5 # in degrees for plotting

# building parameters
thickness = 50
length = 800
height = 290
setup_height = 100
roof = 30

dEdx = 0.004 # GeV cm^2/g for muons 2 to 3 GeV (2 MeV/cm for muon with constant 2 GeV, and 4 MeV/cm for distribution (Geant4)
rho = 2.3 # g/cm^3 for concrete

# muon constants
# parameters at 0° sea level (durham UK)
I0 = 72.5 # m^-2 s^-1 sr^-1
n = 3.06
E0 = 3.87 # GeV
epsilon = 854
R = 6360 # km
d = 10 # km
Ec = 0 # GeV (cut-off value of the data)
N = (n-1)*(E0 + Ec)**(n-1)


#############



# function that returns a tuple of 2 numoy arrays with a list of theta angles in degrees (0) and radian (1)
def thetas(start=2.5, end=90.5, step=5):
    theta_deg = np.arange(start, end, step)
    theta_rad = np.deg2rad(theta_deg)
    return theta_deg, theta_rad


# functions for the arrangement of concrete floors for the three different locations

def ceil_b(thickness=40, height=290, setup_height=100, roof=roof):
    #define y positions of the ceilings
    y11 = height * 1 - setup_height
    y12 = height * 1 + thickness * 1 - setup_height
    y21 = height * 2 + thickness * 1 - setup_height
    y22 = height * 2 + thickness * 2 - setup_height
    y31 = height * 3 + thickness * 2 - setup_height
    y32 = height * 3 + thickness * 3 - setup_height
    y41 = height * 4 + thickness * 3 - setup_height
    y42 = height * 4 + thickness * 4 - setup_height
    y51 = height * 5 + thickness * 4 - setup_height
    y52 = height * 5 + thickness * 5 - setup_height + roof
    
    ceilings = np.array([[y11, y12], [y21, y22], [y31, y32], [y41, y42], [y51,y52]])
    return ceilings


def ceil_1(thickness=40, height=290, setup_height=100):
    #define y positions of the ceilings
    y11 = height * 1 - setup_height
    y12 = height * 1 + thickness * 1 - setup_height
    y21 = height * 2 + thickness * 1 - setup_height
    y22 = height * 2 + thickness * 2 - setup_height
    y31 = height * 3 + thickness * 2 - setup_height
    y32 = height * 3 + thickness * 3 - setup_height + roof
    
    ceilings = np.array([[y11, y12], [y21, y22], [y31, y32]])
    return ceilings


def ceil_3(thickness=40, height=290, setup_height=100):
    #define y positions of the ceilings
    y11 = height * 1 - setup_height
    y12 = height * 1 + thickness * 1 - setup_height + roof
    
    ceilings = np.array([[y11, y12]])
    return ceilings


# Pathlength calculation:
def get_slab_path_length(theta_rad, y_bottom, y_top, max_x):

    # Avoid division by zero for perfectly horizontal rays
    if np.isclose(theta_rad, np.pi/2):
        return 0.0   

    slab_thickness = y_top - y_bottom
    raw_path = slab_thickness / np.cos(theta_rad)
    
    # 2. Check Building Constraints (Finite Length)
    # Calculate x-position where the ray enters and exits the slab layer
    tan_theta = np.tan(theta_rad)
    x_enter = y_bottom * tan_theta
    x_exit  = y_top * tan_theta
    
    # Case A: Ray starts beyond the building length (misses entirely)
    if x_enter >= max_x:
        return 0.0
    
    # Case B: Ray exits through the side wall (clips the corner)
    if x_exit > max_x:
        # Calculate distance from entry point (x_enter, y_bottom) to corner (max_x, y_corner)
        # We need to find y at x=max_x -> y = x / tan(theta)
        y_at_wall = max_x / tan_theta
        
        # Distance formula between (x_enter, y_bottom) and (max_x, y_at_wall)
        dx = max_x - x_enter
        dy = y_at_wall - y_bottom
        return np.sqrt(dx**2 + dy**2)
        
    # Case C: Standard transmission (enters bottom, exits top)
    return raw_path


def total_path(theta_rad, ceilings, length=1000):
    # Sum the path lengths through all defined ceilings
    total = 0.0
    for y_bot, y_top in ceilings:
        total += get_slab_path_length(theta_rad, y_bot, y_top, length)
    return total


def E_loss(theta_rad, ceilings, length, dEdx=0.004, rho=2.3):
    return dEdx * rho * total_path(theta_rad, ceilings)


def D_theta(theta_rad, R=R, d=d):
    R_d = R / d
    cos_theta = np.cos(theta_rad)
    D_val = np.sqrt(R_d**2 * cos_theta**2 + 2 * R_d + 1) - (R_d * cos_theta)
    return D_val


def cos2(theta_rad):
    return np.cos(theta_rad)**2


def I_E_theta(E, I0, N, E0, n, theta_rad, dterm=2):
    if dterm == 1:
        D_term = D_theta(theta_rad, R, d)**(-(n - 1))
    elif dterm == 2:
        D_term = cos2(theta_rad)
    else:
        print("Error: dterm not valid")

    E_term = (E0 + E)**(-n)
    e_corr = (1+ E/epsilon)**(-1)
    return I0 * N * E_term * dterm * e_corr # * np.sin(theta_rad)


def integrate_I_analy(theta_rad, Emin, Emax):
    # I0 * N * cos^2(theta) * Energy_Integral_Term
    term_energy = (1/(-n + 1)) * ((E0 + Emax)**(-n + 1) - (E0 + Emin)**(-n + 1))
    return I0 * N * cos2(theta_rad) * term_energy


def mean_cos2(theta_min, binsize=5):
    result, error = quad(cos2, theta_min, theta_min + np.deg2rad(binsize))
    return 1/(theta_min+np.deg2rad(binsize)-theta_min)*result


def mean_theta(theta_min, binsize=5):
    return np.arccos(np.sqrt(mean_cos2(theta_min, binsize)))


def I_in(theta, ceilings, length):
    # If theta is extremely close to pi/2 (horizon), usually 0 flux or cutoff
    if theta > 1.56: # ~89 degrees
        return 0
        
    energy_loss = E_loss(theta, ceilings, length)
    val = integrate_I_analy(theta, energy_loss, 1000)
    return val


def I_in_int(theta, ceilings, length):
    return I_in(theta, ceilings, length) * np.sin(theta) * 2 * np.pi


def I_out(theta, ceilings, length):
    if theta > 1.56: # ~89 degrees
        return 0

    energy_loss = E_loss(theta, ceilings, length)
    val = integrate_I_analy(theta, 0, 1000)
    return val


def I_out_int(theta, ceilings, length):
    return I_out(theta, ceilings, length) * np.sin(theta) * 2 * np.pi


def muon_flux(thickness, height, setup_height=100, ceilingf=ceil_1, angle=np.pi/2):
    theta_samples = np.linspace(0, angle, 2000)
    c = ceilingf(thickness, height, setup_height)
    y_samples = np.array([I_in_int(t, c, length) for t in theta_samples])

    result = simpson(y_samples, x=theta_samples)
    return result