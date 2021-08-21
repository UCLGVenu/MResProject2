import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def get_jones_matrix(theta):
    if theta == 90:
        jones = np.array([1,0])
    else:
        y = np.tan(theta * np.pi/180)
        coeff = (1 + y**2)**0.5
        jones = 1/coeff * np.array([y, 1])
    return jones

wavelength = 500e-9
thickness = 5e-6        # Thickness of LC layer
V_app = 3   # V_applied
V_thresh = 3 # semi arbitrary
input_angles = np.linspace(0, np.pi / 2, 100)
#input_pol_angle = 89
ant_abs = []
x_ints = []
x_phs = []

for ia in input_angles:
    input_pol_angle = ia
    input_jones = get_jones_matrix(input_pol_angle)

    def get_rotation_matrix(theta):
        rotation_matrix = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        return rotation_matrix

    def get_neff(theta_lc):
        n_ord = 1.5
        n_ext = 1.8
        n_eff = (n_ord * n_ext) / (((n_ord**2 * np.sin(theta_lc)**2) + (n_ext**2 * np.cos(theta_lc)**2))**0.5)
        brf = (1 / wavelength) * 2 * np.pi * thickness * (n_eff - n_ord)
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
        threshold = on_v/thickness
        if resultant > threshold:
            lc_rot = np.pi / 2
        else:
            lc_rot = (np.pi / 2) * resultant/threshold
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

    for i in range(segments - 1):
        a_force = get_align_force(100, current, thickness)
        lc_rot = get_lc_rotation(e_force, a_force, V_thresh, thickness)
        angles.append(lc_rot)
        n_eff, brf = get_neff(lc_rot)
        ns.append(n_eff)
        ret = get_gen_ret(brf)
        matrix = np.matmul(ret, matrix)
        current += section

    output = np.matmul(matrix, input_jones)
    #print(output)
    output_x_int = (output[0].real ** 2 + output[0].imag ** 2) ** 0.5
    output_x_phs = np.arctan(output[0].imag / output[0].real)
    output_y_int = (output[1].real ** 2 + output[1].imag ** 2) ** 0.5
    output_y_phs = np.arctan(output[1].imag / output[1].real)
    output_magnitude = np.linalg.norm(output)
    #print(output_x_phs, output_y_phs, output_magnitude)

    #plt.plot(angles)
    #plt.show()

    # Metasurface Antenna
    fsw = 500e-9 # free space wavelength
    rfrw = 1/1.5 * fsw
    abs_tail = 100e-9

    gaussian = signal.gaussian(201, std=7)
    sig_x = np.linspace(-5, 5, 201)
    sigmoid = 1/(1+np.exp(-sig_x))

    n_bot = get_neff(lc_rot)[0]
    n_avg = sum(ns)/len(ns)
    bot_wav = 1/n_bot * fsw
    bot_wav_avg = 1/n_avg * fsw

    def calc_antenna_absorb(wav_1, rot):
        diff = abs(wav_1 - rfrw)
        absorbed = -1
        if diff > abs_tail:
            absorbed = 0
        else:
            d1 = int(diff * 1e9 // 1)
            absorbed = gaussian[100 - d1] * (np.cos(rot)**2)
        return absorbed

    def calc_anteanna_phase(wav_1):
        diff = (wav_1 - rfrw)
        phase = -1
        if abs(diff) > abs_tail:
            phase = 0
        else:
            d1 = int(diff * 1e9 // 1)
            phase = sigmoid[100 - d1] * 2 * np.pi
        return phase

    #print(calc_antenna_absorb(bot_wav_avg))
    #print(calc_anteanna_phase(bot_wav_avg))

    output_x_int *= (1 - calc_antenna_absorb(bot_wav_avg, input_pol_angle))
    output_x_phs += calc_anteanna_phase(bot_wav_avg)

    #print(output_x_int, output_x_phs)

    ant_abs.append(calc_antenna_absorb(bot_wav_avg, input_pol_angle))
    x_ints.append(output_x_int)
    x_phs.append(output_x_phs)

plt.plot(ant_abs)
plt.show()

plt.plot(x_ints)
plt.show()

