import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import cmath

wavelength = 600e-9
thickness = 500e-9        # Thickness of LC layer
V_app = 0   # V_applied
V_thresh = 3 # semi arbitrary

input_jones = np.array([1, 0])

wavelengths = np.linspace(300, 800, 500)
voltages = np.linspace(0, 5, 40)
angles_in = np.linspace(0, 0.5 * np.pi, 100)
ant_absorb = []
x_ints = []
y_ints = []
x_phs = []
output_phs = []
out_ints = []

for a in angles_in:


    def get_rotation_matrix(theta):
        rotation_matrix = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        return rotation_matrix

    def get_neff(theta_lc, thick2):
        n_ord = 1.5
        n_ext = 1.8
        n_eff = (n_ord * n_ext) / (((n_ext**2 * np.sin(theta_lc)**2) + (n_ord**2 * np.cos(theta_lc)**2))**0.5)
        brf = (1 / wavelength) * 2 * np.pi * thick2 * (n_eff - n_ord)
        return n_eff, brf

    def get_gen_ret(brf):
        gen_retarder = np.array([[np.exp(-brf * 1j/2), 0], [0, np.exp(brf * 1j/2)]])
        return gen_retarder

    # Force in E field ~ F=qE
    def get_e_force(V, thick_E):
        E_field =  V / thick_E
        e_force = E_field
        return e_force

    # Alignment force scales as r^2 from edges

    def get_align_force(a_const, y, thickness):
        a_force = a_const * min(y, thickness-y) ** (-1/2)
        return a_force


    def get_lc_rotation(e_force, a_force, on_v, thickness):
        resultant = max(0, e_force - a_force)
        threshold = on_v / thickness
        if resultant > threshold:
            lc_rot = np.pi / 2
        elif resultant < 0:
            lc_rot = 0
        else:
            lc_rot = (np.pi / 2) * resultant / threshold
        return lc_rot


    segments = 100
    section = thickness/segments
    current = 0
    current += section
    e_force = get_e_force(V_app, thickness)
    threshold = 3
    matrix = np.array([[1,0], [0,1]])
    angles = []
    ns = []
    phs_gain = 0
    rot_minus = get_rotation_matrix(a * -1)
    rot_plus = get_rotation_matrix(a)

    for i in range(segments - 1):
        a_force = get_align_force(100, current, thickness)
        lc_rot = get_lc_rotation(e_force, a_force, V_thresh, thickness)
        angles.append(lc_rot)
        n_eff, brf = get_neff(lc_rot, section)
        ns.append(n_eff)
        ret = get_gen_ret(brf)
        matrix = np.matmul(ret, matrix)
        phs_gain += (2 * np.pi / wavelength) * section * n_eff
        current += section

    output = np.matmul(matrix, input_jones)
    output = np.matmul(rot_minus, output)
    output = np.matmul(output, rot_plus)

    output_x_int = (output[0].real ** 2 + output[0].imag ** 2) ** 0.5
    output_y_int = (output[1].real ** 2 + output[1].imag ** 2) ** 0.5
    #output_x_phs = np.arctan(output[0].imag / output[0].real)
    #output_x_phs = cmath.phase(output[0])
    output_x_phs = phs_gain % (2 * np.pi)

    #print(output_x_int, output_x_phs)

    output_phs.append(output_x_phs)

    #plt.plot(angles)
    #plt.show()

    # Metasurface Antenna
    fsw = 500e-9 # free space wavelength
    rfrw = 1/1.5 * fsw
    abs_tail = 100e-9

    gaussian = signal.gaussian(201, std=7)
    sig_x = np.linspace(-5, 5, 201)
    sigmoid = 1/(1+np.exp(-sig_x))

    n_bot = get_neff(lc_rot, section)[0]
    n_avg = sum(ns[-20:])/len(ns[-20:])
    bot_wav = 1/n_bot * wavelength
    bot_wav_avg = 1/n_avg * wavelength
    y_wav = 1/1.5 * wavelength

    def calc_antenna_absorb(wav_1):
        diff = abs(wav_1 - rfrw)
        absorbed = -1
        if diff > abs_tail:
            absorbed = 0
        else:
            d1 = int(diff * 1e9 // 1)
            absorbed = gaussian[100 - d1]
        return absorbed

    def calc_anteanna_phase(wav_1):
        diff = (wav_1 - rfrw)
        phase = -1
        if abs(diff) > abs_tail:
            phase = 0
        else:
            d1 = int(diff * 1e9 // 1)
            phase = sigmoid[100 - d1] * np.pi
        return phase

    #print(calc_antenna_absorb(bot_wav_avg))
    #print(calc_anteanna_phase(bot_wav_avg))

    output_x_int *= (1 - calc_antenna_absorb(bot_wav_avg))
    output_y_int *= (1 - calc_antenna_absorb(y_wav))
    output_x_phs += (calc_anteanna_phase(bot_wav_avg) % (2 * np.pi))
    output_int = (output_x_int ** 2 + output_y_int ** 2) ** 0.5

    ant_absorb.append(calc_antenna_absorb(bot_wav_avg))
    x_ints.append(output_x_int)
    y_ints.append(output_y_int)
    out_ints.append(output_int)
    #x_phs.append(output_x_phs % (2 * np.pi))
    x_phs.append(output_x_phs)

'''xnew = np.linspace(voltages.min(), voltages.max(), 100)
spl = make_interp_spline(voltages, ant_absorb, k=3)
smoothed = spl(xnew)

plt.plot(xnew, smoothed)
plt.show()'''

plt.plot(angles_in, ant_absorb)
plt.title('Nanodisc SLM Absorption vs Voltage')
plt.xlabel('Voltage Applied / V')
plt.ylabel('Absorption')
plt.show()

plt.plot(angles_in, x_ints)
plt.plot(angles_in, y_ints)
plt.plot(angles_in, out_ints)
plt.show()

plt.plot(angles_in, x_phs)
plt.title('Nanodisc SLM Output Phase (along primary axis) vs Voltage')
plt.xlabel('Voltage Applied / V')
plt.ylabel('Phase Difference')
plt.show()

plt.plot(angles_in, output_phs)
plt.title('LC Only Phase Change (along primary axis) vs Voltage')
plt.xlabel('Voltage Applied / V')
plt.ylabel('Phase Change / rad')
plt.show()

plt.plot(angles_in, x_phs)
plt.plot(angles_in, output_phs)
plt.title('Nanodisc SLM Output Phase (along primary axis) vs Voltage')
plt.xlabel('Voltage Applied / V')
plt.ylabel('Phase Difference / rad')
plt.legend(['With Nanodisc', 'LC Only'])
plt.show()
