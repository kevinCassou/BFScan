import scipy.constants
from numpy import exp, sqrt, arctan, vectorize, real, pi, arange, poly1d, vstack, array
from math import log
from scipy.optimize import curve_fit

#######################################################################
################### Laser Wakefield with ionization ###################
#######################################################################

# longitudinale profiles function 
 
## plasma profile


def plasmaProfile(ne1, L1, ne2, Lx, laser_fwhm, print_flag= False, plot_flag = False):
    """
    return the longitudinal plasma profile of a chair-like target 
    ne1 = first plateau electron density    [cm^-3]
    L1 = length of the first plateau        [m]
    ne2 = density of the second plateau     [cm^-3]
    return numpy array (x,ne)  in m and cm^-3
    """
    # conversion mm2m
    mm2m = 1e-3
    # polygonal characteristics point 
    x0 = 0.0
    xupramp1 = Lx+1.2*laser_fwhm # starting point
    lupramp1 = 1.e-3  # first up ramp  
    xupramp2 = xupramp1+lupramp1
    lupramp2 = 0.7e-3  #second upramp length of the input diameter l2,d2
    xupramp3 = xupramp2+lupramp2
    lupramp3 = 0.3e-3   #third upramp
    # plateau region #  
    xplateau1 = xupramp3 + lupramp3
    lplateau1 = 0.85*L1                 #region 1 plateau length 
    # downramp#
    xbegindownramp1 = xplateau1+lplateau1
    ldownramp1 = 0.35e-3
    # plateau region 2 with correction due to non null flow 1->2
    xplateau2 = xbegindownramp1 + ldownramp1
    lplateau2 = 0.8*ldownramp1
    xbegindownramp2 = xplateau2 + lplateau2
    ldownramp2 = ldownramp1/2.
    xbegindownramp3 = xbegindownramp2 + ldownramp2
    ldownramp3 = 0.7e-3
    xbegindownramp4 = xbegindownramp3 + ldownramp3
    ldownramp4 = 1e-3
    xend = xbegindownramp4 + ldownramp4
    ne_up1 = 0.5*ne1
    ne_up2 = 0.75*ne1
    r = ne2/ne1
    l1 = [ 0.52644434 -0.60966651 -0.1897902   0.58150618]
    x1 = np.poly1d(l1)
    l2 = [ 0.84934688 -0.82402032 -0.30158281  0.75062812]
    x2 = np.poly1d(l2)
    l3 = [ 0.26187251 -0.21280877  0.15913015  0.62090305]
    x3 = np.poly1d(l3)
    x4 = 0.72
    x5 = 1.49
    
    k1 = [-5.91588353e+24,  5.45382530e+24,  5.11741745e+24,  3.87320393e+23]
    y1 = np.poly1d(k1)
    k2 = [-2.67412997e+24,  3.65119495e+24,  3.70868391e+24,  1.15712555e+24]
    y2 = np.poly1d(k2)
    k3 = [-2.06670632e+24,  3.28005947e+24,  3.39737877e+24,  1.25963404e+24]
    y3 = np.poly1d(k3)
    k4 = [-2.02897508e+24,  3.40346262e+24,  3.39462911e+24,  1.11544168e+24]
    y4 = np.poly1d(k4)
    k5 = [-3.97425183e+24,  4.52137775e+24,  4.14970701e+24,  4.51583537e+23]
    y5 = np.poly1d(k5)
    
    xr = np.array([x0,xupramp1,xupramp2,xupramp3,xplateau1,xbegindownramp1,
                   xbegindownramp1 + x1(r)*mm2m,
                   xbegindownramp1 + x2(r)*mm2m,
                   xbegindownramp1 + x3(r)*mm2m,
                   xbegindownramp1 + x4*mm2m,
                   xbegindownramp1 + x5*mm2m,
                   xend])
    
    ner = np.array([0,0,ne_up1,ne_up2,ne1,ne1,
                    y1(r),
                    y2(r),
                    y3(r),
                    y4(r),
                    y5(r),
                    0])

    if plot_flag == True:
        fig, ax = plt.subplots()
        ax.plot(xr,ner)
        ax.set_xlabel('x[m]')
        ax.set_ylabel('ne (x)')
    
    if print_flag == True:
        print("###########################################################\n",
        xr,
        ner,
        "\n ###########################################################")

    return np.vstack((xr,ner))

## dopant profile with leak correction 

def dopantProfile(C_N2,ne1,ne2,xr,ner):
    """ return the longitudinale profile of dopant taking into account the
    a rough correction for the leak depending on the ratio of ne2/ne1
    return a numpy array (x,nN2) [m,cm^-3]
    """
    # correction of the density compared to null flow between region 1 and 
    # region 2
    r = ne2/ne1
    correction1 = [ 0.73285714, -1.07571428,  0.41714286]
    correction2 = [-0.03891875,  0.26215569, -0.07071524]

    pcorrection1 = np.poly1d(correction1)
    pcorrection2 = np.poly1d(correction2)
    # correction factor for zone 1 density
    xN2 = xr 
    nN2 = ner*C_N2

    nN2[6:] = ner[6:]*C_N2* pcorrection1(r)

    # diffusion correction 
    nN2[-2] = ner[-2]]*C_N2* pcorrection1(r)*pcorrection2(r)

    if plot_flag == True:
        fig, ax = plt.subplots()
        ax.plot(xN2,nN2)
        ax.set_xlabel('x[m]')
        ax.set_ylabel('ne (x)')
    
    if print_flag == True:
        print("###########################################################\n",
        xN2,
        nN2,
        "\n ###########################################################")

    return np.vstack((xN2,nN2))

## vacuum focus offset as a factor of Zr 

# laser wavelength (Ti-Sa)
lambda_0  = 0.8e-6                # [m] IMPORTANT: this value is used for conversions
omega0 = 2*pi*scipy.constants.c/lambda_0
onel = lambda_0/(2*pi)

################### Main simulation parameters

# Mesh resolution and integration timestep
dx       = 1.0                  # lambda0/(2pi) units, longitudinal resolution
dtrans   = 2.                   # lambda0/(2pi) units, transverse resolution
dt       = 0.83*dx             # 1/omega0 units

# Number of points in the simulation window
nx       = 448                 # longitudinal grid points
ntrans   = 384                 # transverse  grid points

# Size of the simulation window (simulation window = moving window)
Lx       = nx * dx              # lambda0/(2pi), longitudinal size
Ltrans   = ntrans*dtrans        # lambda0/(2pi), transverse size (half plane since we are in cylindrical geometry)

# Number of patches for parallelization (see parallelization doc) check patch / node / cpu
npatch_x = 32                   # longitudinal direction
npatch_r = 32                   # transverse direction

# Laser fwhm duration in field (i.e. sqrt(2) times longer than the fwhm duration in intensity)
laser_fwhm = 40e-15 * omega0 *2**0.5   # 1/omega0 units

# Compute time when moving window starts to move. Initially it is at rest while the laser enters  
laser_right_border_distance   = 1.5*laser_fwhm                                                # lambda0/(2pi)/c units, initial space between laser peak and right simulation border  
time_laser_peak_enters_window = 3.*laser_fwhm                                                 # lambda0/(2pi)/c units, time when the laser peak enters the left border of the simulation window   
time_start_moving_window      =  Lx-laser_right_border_distance+time_laser_peak_enters_window # lambda0/(2pi)/c units

Main(
    geometry = "AMcylindrical", # please see the documentation of the azimuthal modes decomposition technique

    interpolation_order = 2,
    number_of_AM        = 1, # number of azimuthal modes included (for LWFA the minimum is 2: mode 0 for wakefield, mode 1 for the laser, see documentation)
    timestep            = dt,
    simulation_time     = 21000, 

    cell_length         = [ dx,  dtrans],
    grid_length         = [ Lx,  Ltrans],

    number_of_patches   = [npatch_x, npatch_r],
    
    clrw = nx/npatch_x,

    EM_boundary_conditions = [
       ["silver-muller","silver-muller"],
       ["buneman","buneman"],
    ],

    solve_poisson                  = False,
    print_every                    = 100,
    reference_angular_frequency_SI = omega0, # necessary to compute the ionization rate

    random_seed = smilei_mpi_rank # for the random number extraction in the ADK ionization model
)


######################### Enabling the moving window following the region around the laser
MovingWindow(
    time_start = 0, #time_start_moving_window,
    velocity_x = 1. # propagation speed of the moving window along the positive x direction, in c units 
)

######################### Load balancing, essential for performances of LWFA simulations, where normally these parameters are ok
#LoadBalancing(
#    initial_balance = False,
#    every = 20,
#    cell_load = 5.,
#    frozen_particle_load = 0.1
#)

######################### Define the particles Species

# Masses are normalized by the electron mass
me_over_me = 1.0                                                              # normalized electron mass
mp_over_me = scipy.constants.proton_mass / scipy.constants.electron_mass      # normalized proton mass
mn_over_me = scipy.constants.neutron_mass / scipy.constants.electron_mass     # normalized neutron mass

# External_config


#config_external = {'l_1': 0.3, 'n_e_2': 1.5e+18, 'n_e_1': 3e+18, 'c_N2': 0.3, 'x_foc': 0.1, 'Config': 0.0}

# Definition of the density profiles (density is normalized by the critical density referred  to lambda_0)
# The plasma is assumed already ionized by the prepulse (hydrogen completely ionized, nitrogen up to the first two ionization levels)
ncrit = scipy.constants.epsilon_0*scipy.constants.electron_mass*omega0**2/scipy.constants.e**2; # Critical density in [m^-3] (NOT [cm^-3])

dopant_level = config_external['c_N2'] #  dopant in % of the 

# longitudinal definition of the plasma profile

pp = plasmaProfile(config_external['n_e_1'],config_external['l_1'],config_external['n_e_2'],Lx,laser_fwhm)

h_level     = 1.-dopant_level
xh_points = pp[0,:]/onel
xh_values = pp[1,:]/ncrit

# longitudinal definition of the dopant profile 

dp = dopantProfile(config_external['c_N2'],config_external['n_e_1'],config_external['n_e_2'],pp[0,:],pp[1,:])
xd_points = dp[0,:]/onel
xd_values = dp[1,:]/ncrit

longitudinal_density_profile_h =        polygonal(xpoints = xh_points, xvalues = xh_values)
longitudinal_density_profile_dopant =   polygonal(xpoints = xd_points, xvalues =xd_values)

# Radius of the plasma, i.e. half its transverse width
R_plasma = 785. # lambda0/(2pi) units

# Density profile of the hydrogen
def my_density_profile_h(x,r):
    radial_profile = 1.
    if (r>R_plasma):
	    radial_profile = 0.
    
    return longitudinal_density_profile_h(x,r)*radial_profile

# Density profile of the dopant (nitrogen)
def my_density_profile_dopant(x,r):
    radial_profile = 1.
    if (r>R_plasma):
        radial_profile = 0.	
    
    return longitudinal_density_profile_dopant(x,r)*radial_profile


# Hydrogen electrons 
# (no need to define the hydrogen ions - why? read section Why Ions Are Not Present in tutorial https://smileipic.github.io/tutorials/advanced_wakefield_electron_beam.html)
Species( 
    name = "electron",
    position_initialization = "regular",
    momentum_initialization = "cold",
    ionization_model = "none",
    particles_per_cell = 1,
    regular_number = [1,1,1], # [x,r,theta=1] 
    c_part_max = 1.0,
    mass = me_over_me,
    charge = -1.0,
    charge_density = my_density_profile_h,
    mean_velocity = [0., 0., 0.],
    time_frozen = 0.0,
    pusher = "ponderomotive_boris",
    ponderomotive_dynamics = "True",
    boundary_conditions = [
       ["remove", "remove"],
       ["reflective", "remove"],
    ], 
 
)

# Electrons that are  obtained from the ionization of the nitrogen (last five ionization levels)
Species( 
    name = "electronfromion",
    position_initialization = "regular",
    momentum_initialization = "cold",
    ionization_model = "none",
    particles_per_cell = 0,
    c_part_max = 1.0,
    mass = me_over_me,
    charge = -1.0,
    charge_density = 0.,  # Here absolute value of the charge is 1 so charge_density = nb_density
    mean_velocity = [0., 0., 0.],
    time_frozen = 0.0,
    pusher = "ponderomotive_boris",
    ponderomotive_dynamics = "True",
    boundary_conditions = [
       ["remove", "remove"],
       ["reflective", "remove"],
    ], 
  
)

# Nitrogen N5+ ions (i.e. already ionized up to the first two ionization levels over seven)
Species( 
    name = "nitrogen5plus",
    position_initialization = "regular",
    momentum_initialization = "cold",
    particles_per_cell = 1, 
    regular_number = [1,1,1],
    atomic_number = 7,
    ionization_model = "tunnel_envelope_averaged",
    #ionization_model = "none",
    ionization_electrons = "electronfromion",
    maximum_charge_state = 7,
    c_part_max = 1.0,
    mass = 7.*mp_over_me + 7.*mn_over_me + 2.*me_over_me,
    charge = 5.0,
    charge_density = my_density_profile_dopant,  
    mean_velocity = [0., 0., 0.],
    time_frozen = 20000.0,
    pusher = "ponderomotive_boris",
    ponderomotive_dynamics = "True",
    boundary_conditions = [
       ["remove", "remove"],
       ["reflective", "remove"],
    ],
)

# The nitrogen electrons obtained from the ionization of the first two ionization level
Species( 
    name = "neutralizingelectron",
    position_initialization = "nitrogen5plus",
    #position_initialization = "regular",
    momentum_initialization = "cold",
    particles_per_cell = 1, #for tests,
    ionization_model = "none",
    c_part_max = 1.0,
    mass = me_over_me,
    charge = -1,
    charge_density = my_density_profile_dopant,  
    mean_velocity = [0., 0., 0.],
    time_frozen = 0.0,
    pusher = "ponderomotive_boris",
    ponderomotive_dynamics = "True",
    boundary_conditions = [
       ["remove", "remove"],
       ["reflective", "remove"],
    ],

)


############################# Laser pulse definition #############################

# The laser pulse is injected from the left boundary of the window
# the peak of the laser enters the window at time = center (lambda0/(2pi)/c units)


w_0                  = 18.0e-6                 # m 
Z_r                  = pi*w_0**2/lambda_0     # m 
# position of the focal spot in vacuum in lambda0/(2pi) unit
# pp[0,5] is the xbegindownramp position 

xfocus               = pp[0,5] + config_external['x_foc']*Z_r/onel    # lambda0/(2pi) unit


LaserEnvelopeGaussianAM( 
    a0              = 1.15,                          # peak normalized field
    focus           = [xfocus, 0.],
    waist           = w_0*omega0/scipy.constants.c,                        # lambda0/(2pi) units
    time_envelope   = tgaussian(center=Lx-1.5*laser_fwhm, fwhm=laser_fwhm),
  
)


############################# Checkpoint for  the simulation #############################

Checkpoints(
    dump_step = 5000,
    dump_minutes = 0.0,
    exit_after_dump = False,
)

############################ Diagnostics #############################

list_fields = ['Ex','Ey','Rho','Jx','Env_A_abs','Env_Chi','Env_E_abs',"Rho_electron","Rho_electronfromion","Rho_nitrogen5plus"]

field_lists_forprobes=["Ex","Ey","Rho","Rho_electron","Rho_electronfromion","Rho_nitrogen5plus","Env_E_abs"]

# 1D probe near the propagation axis
DiagProbe(	
	every = 300,
	
	origin = [0., 2*dtrans, 0.],
	
	corners = [
              [Main.grid_length[0], 2*dtrans, 0.]
                  ],

	number = [nx],
	fields = field_lists_forprobes,
)

# 2D probe on the xy plane	
DiagProbe(	
	every = 300, #1000,
	
	origin = [0., -Main.grid_length[1], 0.],
	
	corners =  [	
           [Main.grid_length[0], -Main.grid_length[1], 0.],	
	   [0., Main.grid_length[1], 0.],
                   ],
	
 	number = [nx,2*ntrans],	
        fields = field_lists_forprobes,
)

#DiagFields(
#    every = 300,
#    fields = ["El","Er","Rho_electronfromion","Rho"],
#)

#DiagScalar(every = 10, vars=['Env_A_absMax','Env_E_absMax','Zavg_nitrogen5plus'])

# Diag to track the particles

# filter on the particles to be tracked (now only particles with px>50MeV/c are tracked)
def my_filter(particles):
    return ((particles.px>16.))


DiagTrackParticles(
    species = "electronfromion", # species to be tracked
    every = 300,
#    flush_every = 100,
    filter = my_filter,
    attributes = ["x","y", "z", "px", "py", "pz","weight"]
)
