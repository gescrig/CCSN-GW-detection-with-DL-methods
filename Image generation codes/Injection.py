import numpy as np
import bilby
from scipy.interpolate import interp1d
from gwpy.timeseries import TimeSeries
import glob


##################################################################################################################
def richers_time_domain_template(time, dist, theta, t0, Rich_time,Rich_strain):
        """
        Rescale Richers waveform to a given time, distance and polarization angle.
        Returns both polarizations (plus and cross)
        """
        # Rescale data to dist, theta and t0:
        X = (Rich_time) + t0
        Y = (Rich_strain / (dist*3.085677581*10**21) ) * np.sin(theta)**2 # dist from kpc to cm

        # Making sure injection is done within our time lapse and amplitude is the correct one
        #    print ("X    : ",min(X), max(X))
        #    print ("time : ", min(time), max(time))
        #    print ("Y    : ",min(Y), max(Y))
        #    plt.figure()
        #    plt.plot(X, Y)
        #    plt.show()

        # Interpolate so that we can get Y data for any time, making sure that if time array contains elements out of the
        # interval for which we have calculated our master waveform, those Y elements are zero, instead of an extrapolation
        # of original data:
        f_Y_interp = interp1d(X, Y, kind='cubic', bounds_error = False, fill_value=0)
        plus = f_Y_interp(time)

        # We do not consider cross polarization:
        cross = np.zeros(len(time))

        # Testing again:
        #    print ("hplus     ", min(plus), max(plus))
        #    print ("hcross    ", min(cross), max(cross))
        #    print ("N        =", len(time))
        #    print ("fs       =",1./(time[1]-time[0]))
        #    plt.figure()
        #    plt.plot(time, plus)
        #    plt.show()

        return {'plus': plus, 'cross': cross}
##################################################################################################################

def make_bilby_injection(injection_parameters,interferometers):
    """
    injection_parameters: dictionary with the injection parameters
    time_domain_source_model: function used to generate the injection, tipically returns plus and cross polarizations
    interferometers: list of interferometers Livingston (L), Handford (H), Virgo (V)
    return:
        bilby ifos variable with the injection
    """

    duration = injection_parameters['duration']
    sampling_frequency = injection_parameters['fs']
    noise_type = injection_parameters['noise_type']
    start_time = injection_parameters['gps0']
    np.random.seed(88170235)
    
    waveform_injection = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        time_domain_source_model=richers_time_domain_template,
        start_time=0.0) # Offset


    # Inject the signal into three interferometers:
    ifos = bilby.gw.detector.InterferometerList(interferometers)
    if noise_type == 'Gaussian':
    
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=start_time) # Offset
    else: 
        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=start_time) # Offset
        
    ifos.inject_signal(waveform_generator=waveform_injection,
                       parameters=injection_parameters)
    
    return ifos

##################################################################################################################

    
def get_segments_gps(file_path):
    files = glob.glob(file_path +'*.hdf5')
    gpss = []
    for f in files:
        name,ext = f.split('.')
        det,_,gps,dur = (name.split('-'))
        gpss.append(gps)

    gpss = list(set(gpss))
    return gpss
        
                      
def get_filename(file_path,gps_0):    
    segments = get_segments_gps(file_path)
    for seg in segments:
        if gps_0 >= float(seg) and gps_0 <= float(seg) + 4096:
            f = glob.glob(file_path+'*'+seg+'*.hdf5' )
    f.sort()
    return f
                      
def real_noise_injection(bg,h,filename):
    h = h.taper()
    dur = h.dt.value * len(h.times)
    gps0 = h.times[0].value
    bg = bg.crop(gps0,gps0+dur)
    #h.t0 = bg.times[0].value
    sig = bg.inject(h)
    
    return sig,bg,h
          
#def inyection(injection_parameters,interferometers,signal):
#    noise_type = injection_parameters['noise_type']
#    injection_parameters['Rich_time'] = signal.time
#    injection_parameters['Rich_strain'] = signal.h
#    ifos = make_bilby_injection(injection_parameters,interferometers)
#    if noise_type == 'Gaussian':

def ifo_to_TS (ifo):
    hp = ifo.time_domain_strain
    tp = ifo.time_array
    dt = tp[1]-tp[0]
    h = TimeSeries(hp,dt = dt)
    h.t0 = tp[0]
    return h
                      
                      
                      