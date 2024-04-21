import numpy as np



class star_generator():
    '''
    This class will prepare the star will all the necessary 
    parameters. The star will be prepared in cartesian coordinates.
    gridding in log space between min_rad and max_rad. Within +-
    min_rad, we will have a constant grid spacing, where the grid
    spacing will be equal to the minimum grid spacing of the logarithmically
    spaced grid. The line of sight is assumed to be along the Z-axis 
    by default
    :param min_rad: Minimum radius till which the grid
                    will be prepared.
    :type min_rad: float
    :param max_rad: Maximum radius for the grid
    :type max_rad: float
    :param num_cell: Number of cells along each axis between min_rad and max_rad
    :type num_cell: int
    :type los: line of sight in cartesian coordinates. Default: Z-axis
                specified as [0,0,1]. LOS will be normalised by default.
    :type los: numpy ndarray/list
    :param footpoint_B: footpoint magnetic field which is used to calculate the
                        magnetic field.
    :type footpint_B: float
    :param inclination: Angle betwenn los and rotation axis
    :type inclication: float
    :param obliquity: Angles between rotation axis and dipole axis
    :type obliquity: float
    :param stellar_radius_sun: stellar units in units of solar radius
    :type stellar_radius_sun: float
    '''
    def __init__(self, min_rad=1.01,max_rad=10,num_cell=10, footpoint_B=2000,\
                        inclination=0, obliquity=0, stellar_radius_sun=1.1, rot_phase=0,\
                        los=[0,0,1],orientation=90):
        if min_rad<=1:
            raise RuntimeError("Radius should be assigned to each voxel")
        self.min_rad=min_rad
        self.max_rad=max_rad
        self.num_cell=num_cell
        self.los=los/np.linalg.norm(np.array(los))  ### normalising the vector
        self.footpoint_B=footpoint_B
        self.stellar_radius_cm=stellar_radius_sun*696700*1e5
        self.inclination=inclination
        self.orientation=orientation
        self.rot_phase=rot_phase
    
    def generate_grid(self):
        '''
        gridding in log space between min_rad and max_rad. Within +-
        min_rad, we will have a constant grid spacing, where the grid
        spacing will be equal to the minimum grid spacing of the logarithmically
        spaced grid.
        So if min_rad=1.1 and max_rad=10, the grid will go from -10 to 10. The 
        space between (-10,-1.1) and (1.1, 10) will have logarithmic spacing.
        Within (-1.1,1.1), the spacing is uniform. We use meshgrid to generate 
        the final grid, which can be accessed by using self.grid.
        '''
        large_grid=np.logspace(np.log10(self.min_rad),np.log10(self.max_rad),self.num_cell)
        min_grid_sep=large_grid[1]-large_grid[0]
        small_grid=np.arange(-self.min_rad+min_grid_sep,self.min_rad,min_grid_sep)
        small_grid_cells=np.size(small_grid)
        tot_cells=2*self.num_cell+small_grid_cells
        
        x=np.zeros(tot_cells)
        x[0:self.num_cell]=-large_grid[::-1]
        x[self.num_cell:self.num_cell+small_grid_cells]=small_grid
        x[self.num_cell+small_grid_cells:]=large_grid

        X,Y,Z=np.meshgrid(x,x,x,indexing='ij')
        self.grid=(X,Y,Z)
        self.radius=np.sqrt(X**2+Y**2+Z**2)

    
    def generate_parameter_grid(self,params=None,param_funcs=None):
        '''
        This function generates the grid of the parameters relevant for
        computing the GS spectrum. All the parameters needed by the user
        and their values can be passed through the dictionary params.
        If not provided, or if any of the essential parameters are missing,
        they will be added by default. If any parameter is provided, we will
        assume that its values are also appropriately provided by the user, 
        unless it is an empty list. However this exception only holds tue for
        the essential parameters. For any other parameters, whatever is provided
        in the params dictionary is assumed to be right. This also means that the
        essential parameters should have a fixed format. The format is that the
        shape of the parameters should be same as the radius variable of this 
        class. param_funcs is another dictionary which can be used to pass custom
        functions for each of the default parameters. The key of each function should
        be same as the name of the corresponding essential parameter. The essential
        parameters are: Bx100: B in units of 100Gauss
        log_nnth: Log10 of the density of nonthermal electrons/cc
        log_nth: Log10 of the density of thermal electrons /cc
        theta: Angle between the local magnetic field and los (degree)
        T: temperature
        emin_keV: Minimum energy of the nonthermal electrons in keV. Powerlaw distribution 
                    is assumed
        emax_MeV: Maximum energy of the nonthermal electrons in MeV. Powerlaw distribution
                    is assumed
        delta: Powerlaw distribution of the nonthermal electrons
        '''
        if params is None:
            params={}
        
        self.params=params
        essential_keys=['Bx100','log_nnth', 'log_nth','theta','T','emin_keV',\
                        'emax_MeV','delta']
        keys=params.keys()
        for ek in essential_keys:
            if ek not in keys:
                self.params[ek]=None
        
        keys=self.params.keys()
        func_keys=[]
        if param_funcs is not None:
            func_keys=param_funcs.keys()
        
        if len(func_keys)==0:
            func_keys=[]
        
        for key in keys:
            if self.params[key] is not None and len(self.params[key])!=0:
                continue    
            if key=='Bx100':
                if key in func_keys:
                    self.params[key]=param_funcs[key](self.grid,self.footpoint_B)
                else:
                    self.params[key]=self.get_magnetic_field(self.grid,self.footpoint_B)

                    
            if key=='theta':
                if key in func_keys:
                    self.params[key]=param_funcs[key](self.grid,self.los,self.rot_phase)
                else:
                    self.params[key]=self.get_LOS_angle(self.grid,self.los,self.rot_phase)
                    
            if key=='log_nnth':
                if key in func_keys:
                    self.params[key]=param_funcs[key](self.grid)
                else:
                    self.params[key]=self.get_log_nnth(self.grid)
                    
            if key=='log_nth':
                if key in func_keys:
                    self.params[key]=param_funcs[key](self.grid)
                else:
                    self.params[key]=self.get_log_nth(self.grid)

            if key=='T':
                if key in func_keys:
                    self.params[key]=param_funcs[key](self.grid)
                else:
                    self.params[key]=self.get_temperature(self.grid)
                
            if key=='emin_keV':
                if key in func_keys:
                    self.params[key]=param_funcs[key](self.grid)
                else:
                    self.params[key]=self.get_emin(self.grid)

            if key=='emax_MeV':
                if key in func_keys:
                    self.params[key]=param_funcs[key](self.grid)
                else:
                    self.params[key]=self.get_emax(self.grid)

            if key=='delta':   
                if key in func_keys:
                    self.params[key]=param_funcs[key](self.grid)
                else:
                    self.params[key]=self.get_powerlaw_index(self.grid)

            
            
    def get_magnetic_field(self,grid,B0=2000):
        '''
        Assumes a dipolar field model. given by B0/(r-1)**3
        B0: footpoint magnetic field in Gauss
        Gives magnetic field in units of 100G
        :param grid: X,Y,Z which was returned by meshgrid
                    at radius smaller than 1, we assume that 
                    the B is equal to footpoint
        :type grid: tuple of ndarays
        :param B0: footpoint magnetic field. default: 2000
        :type B0: float
        '''
        X,Y,Z=grid
        mag=np.zeros_like(X)
        pos=np.where(self.radius>1)
        mag[pos]=B0/(self.radius[pos])**3
        mag/=100
        shape=X.shape
        return mag
    
    def get_LOS_angle(self,grid, los,rot_phase):
        '''
        Compute the LOS angle in units of theta between the LOS and the local magnetic field
        Assumes a dipolar field geometry. Use the radius of each cell and compute this.
        :param grid: X,Y,Z which was returned by meshgrid
                    
        :type grid: tuple of ndarays
        :param los: LOS vector in cartesian coordinates
        :type los: ndarray/list
        '''
        X,Y,Z=grid
        
        return np.zeros_like(X)
    
    def get_log_nnth(self,grid):
        '''
        Returns the log10 of density of nonthermal electrons in the voxel. 
        :param grid: X,Y,Z which was returned by meshgrid
                    
        :type grid: tuple of ndarays
        :param los: LOS vector in cartesian coordinates
        :type los: ndarray/list
        '''
        X,Y,Z=grid
        return np.ones_like(X)*8
    
    def get_temperature(self,grid):
        '''
        Returns the temperature. Assume 10000K
        :param grid: X,Y,Z which was returned by meshgrid
                    
        :type grid: tuple of ndarays
        '''
        X,Y,Z=grid
        return 1e4*np.ones_like(X)
    
    def get_emin(self,grid):
        '''
        Returns the minimum electron energy for the nonthermal electrons, assuming a powerlaw
        distribution. Units is keV
        :param grid: X,Y,Z which was returned by meshgrid
        :type grid: tuple of ndarays
        '''
        X,Y,Z=grid
        return 1*np.ones_like(X)
    
    def get_emax(self,grid):
        '''
        Returns the maximum electron energy for the nonthermal electrons. Assume a powerlaw 
        distribution. Unit is MeV
        :param grid: X,Y,Z which was returned by meshgrid        
        :type grid: tuple of ndarays
        
        '''
        X,Y,Z=grid
        return 10*np.ones_like(X)
    
    def get_powerlaw_index(self,grid):
        '''
        Returns the powerlaw index of the underlying electron distribution. 
        :param grid: X,Y,Z which was returned by meshgrid        
        :type grid: tuple of ndarays
        
        '''
        X,Y,Z=grid
        return 4*np.ones_like(X)
    
    def get_log_nth(self,grid):
        '''
        Returns the log10 of density of thermal electrons in the voxel. 
        :param grid: X,Y,Z which was returned by meshgrid
                    
        :type grid: tuple of ndarays
        :param los: LOS vector in cartesian coordinates
        :type los: ndarray/list
        '''
        X,Y,Z=grid
        return 10*np.ones_like(X)/self.radius**2
        
        
    
    
