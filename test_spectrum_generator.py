import generate_star as gen_star
import spectrum_generator as gen_spec
import numpy as np
import matplotlib.pyplot as plt

star=gen_star.star_generator(footpoint_B=2000)
star.generate_grid()
star.generate_parameter_grid()

los=[0,0,1]
freq_ghz=np.logspace(0,3,50)
binary_path='/home/barnali/suncasa_pygsfit/pygsfit/binaries/MWTransferArr.so'

spec=gen_spec.spectrum_generator(star=star,los=los,freq_ghz=freq_ghz, binary_path=binary_path,log_freq_step=0.4)

int_spec=spec.calculate_spectrum()

plt.loglog(freq_ghz,int_spec)
plt.show()
