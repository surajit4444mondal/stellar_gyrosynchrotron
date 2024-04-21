import numpy as np
import os, platform
import astropy.units as u
from astropy import constants as const
import ctypes
from numpy.ctypeslib import ndpointer
from scipy import interpolate

def initGET_MW(libname):
    """
    Python wrapper for fast gyrosynchrotron codes.
    Identical to GScodes.py in https://github.com/kuznetsov-radio/gyrosynchrotron
    This is for the single thread version
    @param libname: path for locating compiled shared library
    @return: An executable for calling the GS codes in single thread
    """
    _intp = ndpointer(dtype=ctypes.c_int32, flags='F')
    _doublep = ndpointer(dtype=ctypes.c_double, flags='F')

    libc_mw = ctypes.CDLL(libname)
    mwfunc = libc_mw.pyGET_MW
    mwfunc.argtypes = [_intp, _doublep, _doublep, _doublep, _doublep, _doublep, _doublep]
    mwfunc.restype = ctypes.c_int

    return mwfunc

class spectrum_generator(): 
    '''
    This class computes the total GS spectrum provided
    a LOS and a star. Star is an object generated using
    the star_generator class
    
    :param los: Line of sight. Right now the code will always use Z-axis as the LOS.
    :param freq_fghz: Frequencies in GHz for which the spectrum has to be calculated.
    :type freq_fghz: numpy ndarray
    :param binary_path: The path where the binary file used to compute a single GS
                        spectrum is located. SHOULD be a compiled version of the fast
                        GS code developed in Fleishmann & Kuznetsov 2010
    :type binary_path: str
    :param log_freq_step: The GS code uses this and the first entry of the freq_ghz to 
                            compute at which frequencies it should compute the flux.
                            This step is used in the logarithmic space.
    :type log_freq_step: float
    :param num_freq_calculated: The GS code will compute the fluxes at different frequencies
                                than what is provided through freq_ghz. The actually frequencies
                                at which this will be computed depends on the start of freq_ghz,
                                log_freq_step and the number of frequencies provided through
                                this parameter. We interpolate at the freq_ghz using that 
                                computed spectrum.
    :type num_freq_calculated: int 
    '''
    def __init__(self,star,los,freq_ghz,binary_path,log_freq_step=0.1, num_freq_calculated=100):
        self.star=star
        self.los=[0,0,1]
        self.freq_ghz=freq_ghz
        self.num_freq_calculated=num_freq_calculated
        self.log_freq_step=log_freq_step
        if os.path.isfile(binary_path):
            self.binary_path=binary_path
        else:
            raise IOError("Binary file does not exist")
        
    
    def single_spectrum_calculator(self,xindex,yindex):
        '''
        This function computes the spectrum along a single axis. The area
        needed for computing the spectrum will be a projection of the area
        computed if the LOS is pointed directly to the centre of the disc,
        and hence will not change the shape of the spectrum. For now, let
        us ignore the projection. The LOS is mostly aligned along Z-axis
        :param xindex,yindex: X,Y index of the X,Y matrix which will be used
                                to extract the parameters
        '''
        
        libname=self.binary_path
        GET_MW = initGET_MW(libname)  # load the library
        
        star=self.star
        grid=star.grid
        X,Y,Z=grid
        x=X[:,0,0]
        sep=np.diff(x)

        mean_sep=np.mean(np.log10(sep)) ### will use the mean separation to compute
                                        ### src_area. 
        src_area_cm2=mean_sep**2*star.stellar_radius_cm**2
        
        NSteps=np.shape(X)[0]-1
        
        Nf=self.num_freq_calculated
        Lparms = np.zeros(11, dtype='int32')  # array of dimensions etc.
        Lparms[0] = NSteps
        Lparms[1] = self.num_freq_calculated
        
        

        Rparms = np.zeros(5, dtype='double')  # array of global floating-point parameters
        Rparms[0] = src_area_cm2  # Area, cm^2
        Rparms[1] = self.freq_ghz[0]*1e9  # starting frequency to calculate spectrum, Hz
        Rparms[2] = self.log_freq_step  # logarithmic step in frequency
        Rparms[3] = 0  # f^C
        Rparms[4] = 0  # f^WH
        
        Parms = np.zeros((24, NSteps), dtype='double', order='F')  # 2D array of input parameters - for multiple voxels
        

        
        for i in range(NSteps):
            Parms[0, i] = sep[i]*star.stellar_radius_cm   # depth for one voxel in cm
            Parms[1,i]=star.params['T'][xindex,yindex,i]
            Parms[2,i]=10**star.params['log_nth'][xindex,yindex,i]
            Parms[3,i]=star.params['Bx100'][xindex,yindex,i]*100
            Parms[6,i]=3  ### powerlaw distribution
            Parms[7,i]=10**star.params['log_nnth'][xindex,yindex,i]
            Parms[9,i]=star.params['emin_keV'][xindex,yindex,i]*1e-3
            Parms[10,i]=star.params['emax_MeV'][xindex,yindex,i]
            Parms[12,i]=star.params['delta'][xindex,yindex,i]
        #print (Parms[3,:])

        RL = np.zeros((7, Nf), dtype='double', order='F')  # input/output array
        dummy = np.array(0, dtype='double')

        # calculating the emission for array distribution (array -> on)
        res = GET_MW(Lparms, Rparms, Parms, dummy, dummy, dummy, RL)
        
        
        f = RL[0]
        I_L = RL[5]
        I_R = RL[6]
        #print (I_L+I_R)
        all_zeros = not RL.any()
        if not all_zeros:
            flux_model = I_L + I_R
            flux_model = np.nan_to_num(flux_model) + 1e-11
            logf = np.log10(f)
            logflux_model = np.log10(flux_model)
            logfreqghz = np.log10(self.freq_ghz)
            interpfunc = interpolate.interp1d(logf, logflux_model, kind='linear')
            logmflux = interpfunc(logfreqghz)
            mflux = 10. ** logmflux
        
        else:
            print("Calculation error! Assign an unrealistically huge number")
            mflux = np.ones_like(self.freq_ghz) * 1e-11
        
        return mflux
        
    def calculate_spectrum(self):
        '''
        This function computes the total spectrum. The area
        needed for computing the spectrum will be a projection of the area
        computed if the LOS is pointed directly to the centre of the disc,
        and hence will not change the shape of the spectrum. For now, let
        us ignore the projection. The LOS is mostly aligned along Z-axis
        '''
        grid=self.star.grid
        X,Y,Z=grid
        
        shape=np.shape(X)
        num_cell=shape[0]
        
        spectrum=np.zeros(np.size(self.freq_ghz))
        
        for i in range(num_cell):
            print (i)
            for j in range(num_cell):
                spectrum+=self.single_spectrum_calculator(i,j)
                
            
        return spectrum
        
            
