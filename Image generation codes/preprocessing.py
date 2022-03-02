import gwpy
import numpy as np




def compute_asd(signal):
    asd = signal.spectrogram2(fftlength=4, overlap=2, window='hanning') ** (1/2.)
    asd = asd.percentile(50)
       
    return asd

def load_asd_from_file(filename):
    asd = np.loadtxt(filename)
    f = asd[:,0]
    asd = asd[:,1]
    asd= gwpy.frequencyseries.FrequencySeries(asd,frequencies=f)

    return asd

def asd_calculation(signal,gps_ini,gps_end,file_path,det):
    import os
    
    dur = gps_end - gps_ini
    filename = det + '_ASD_'+str(gps_ini)+'_'+str(dur)+'.dat'
    #Check if file exists
    if os.path.exists(file_path+filename):
        print('File exists - loading')
        asd = load_asd_from_file(file_path+filename)
    else:
        print('Computing ASD for %s'%det)
        asd = compute_asd(signal)
        np.savetxt(file_path+filename,np.c_[asd.frequencies.value,asd.value], fmt='%.18e', delimiter=' ', newline='\n')
    return asd

def Q_to_image(sig_qt,m=-1):
    sig_qt_norm=sig_qt-sig_qt.min()
    if m == -1:
        sig_qt_norm=np.uint8(sig_qt_norm*255/sig_qt_norm.max()) 
    else: 
        sig_qt_norm=np.uint8(sig_qt_norm*255/m) 
    return sig_qt_norm


    