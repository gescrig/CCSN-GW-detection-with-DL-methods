import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import bilby
import os
import h5py
import logging


def get_omega0(gw_name):
    """
    gw_name (string): name of the Richer's simulation. See get_name
    
    return:
        gw_omega
        # The 'w' is added because some EOS have also a number in its name so that we might get two outputs for the same gw
        # if that number is in the list.
        
    """
    range_of_omega0 = ['w0.0', 'w0.5', 'w1.0', 'w1.5', 'w2.0', 'w2.5', 'w3.0', 'w3.5', 'w4.0', 'w4.5', 'w5.0', 'w5.5',
                           'w6.0', 'w6.5', 'w7.0', 'w7.5', 'w8.0', 'w8.5', 'w9.0', 'w9.5', 'w10.0', 'w10.5', 'w11.0',
                           'w11.5', 'w12.0', 'w12.5', 'w13.0', 'w13.5', 'w14.0', 'w14.5', 'w15.0', 'w15.5']
    
    for number in range_of_omega0:
        if number in gw_name:
            gw_omega0 = (float(number[1:])) # We remove the 'w'.
                   
    return gw_omega0

def get_index_of_waveform(waveform, reduced_data, gw_name):

        """
        Returns the index of omega_0 (rad/s)
        omega_0 has an additional problem. For some reason it has been saved as an int. Therefore, regardless of
        being (e.g.) 9 or 9.5, we well read 9 if we use gw_omega0 = waveform.attrs.get('omega_0(rad|s)'). Instead:
        """
        gw_A = waveform.attrs.get('A(km)')
        gw_EOS = waveform.attrs.get('EOS')
       
        gw_omega0 = get_omega0(gw_name)


        #print('Working with waveform {0}.\nIts atributes are {1} = {2}, {3} = {4}, {5} = {6}. \n'
        #      .format(gw, list(waveform.attrs)[0], gw_A, list(waveform.attrs)[1], gw_EOS,
        #              list(waveform.attrs)[2], gw_omega0))

        # Look for our current waveform's attributes in 'reduced_data':
        A = reduced_data.get('A(km)')
        EOS = reduced_data.get('EOS')
        omega0 = reduced_data.get('omega_0(rad|s)')

        for element_EOS in EOS:
            if element_EOS == gw_EOS:
                index_EOS = list(EOS).index(element_EOS)
                break
        #print('EOS \'{0}\' start at index {1}. \n'.format(gw_EOS, index_EOS))
        Aless = A[index_EOS:]

        for element_A in Aless:
            if element_A == gw_A:
                index_A = list(Aless).index(element_A) + index_EOS
                break
        #print('A(km) = \'{0}\' start at index {1} for our EOS. \n'.format(gw_A, index_A))
        omega0less = omega0[index_A:]

        for element_omega0 in omega0less:
            if element_omega0 == gw_omega0:
                index_omega0 = list(omega0less).index(element_omega0) + index_A
                break

        #print('Data of waveform {0} are stored in \'reduced_data\' with index {1}. \n'.format(gw, index_omega0))

        return index_omega0
    
def read_Richers_file(gw_name,filename):
        """
        Opens Richars file, read the paremeters and return the strain
        return:
            t_minus_tb: time vector (s) the t=0 corresponds to bounce
            strain_dist: strain normalized to distance (cm). If you want it to a given distance you must multiply by it (in cm)
            Deltah: parameter from the simulation
            fpeak: parameter from the simulation
        """
        with h5py.File(filename, 'r') as data:

            # Parent datasets:
            waveforms = data.get('waveforms')
            reduced_data = data.get('reduced_data')
            
            # Sub-datasets for fpeak and DeltaH:
            frequencies = reduced_data.get('fpeak(Hz)')
            D_amp_1 = reduced_data.get('D*bounce_amplitude_1(cm)')
            D_amp_2 = reduced_data.get('D*bounce_amplitude_2(cm)')

            # Get a certain waveform (name specified as an argument) and its index in reduced_data:
            waveform = waveforms.get(gw_name)
            gw_index = get_index_of_waveform(waveform, reduced_data, gw_name)

            # Grab (t-tb) and strain*dist data from waveform dataset & fpeak and Deltah from reduced_data dataset:
            t_minus_tb = np.array(waveform.get('t-tb(s)'))
            strain_dist = np.array(waveform.get('strain*dist(cm)'))
            fpeak = frequencies[gw_index]
            Deltah = D_amp_1[gw_index] - D_amp_2[gw_index]

            # Print info of selected waveform:
            print('\nWaveform {0} (index {1}): \nfpeak = {2},\nD_amp1 = {3},\nD_amp2 = {4},\nDeltah = {5}.\n\n'
                  .format(gw_name, gw_index, fpeak, D_amp_1[gw_index], D_amp_2[gw_index], Deltah))

            # Normalize values:
            #NormalXValues_Rich = t_minus_tb * fpeak
            #NormalYValues_Rich = strain_dist / Deltah

        return t_minus_tb, strain_dist, Deltah, fpeak
    
def get_Richers_waveforms_name(filename):
    """
    Returns the list of simulation names.
    """
    with h5py.File(filename, 'r') as data:
        waveforms = data.get('waveforms')
        waveforms_name = list(waveforms)
    return waveforms_name

