import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from collections import defaultdict

# ─── Physical constants ───────────────────────────────────────────────────────
HBAR_C    = 197.3269804
EP2       = 1.43996
M_NUCLEON = 931.494

def nuclear_radius(A):
    return 1.28*A**(1/3) - 0.76 + 0.8*A**(-1/3)

def reduced_mass(A1, A_alpha=4):
    return M_NUCLEON * A_alpha * A1 / (A1 + A_alpha)

def bessel_zero(l):
    zeros = {0: np.pi, 1: 4.493409, 2: 5.763459,
             3: 6.987932, 4: 8.182561, 5: 9.355812}
    return zeros.get(l, np.pi * (l + 1))

# ─── Hulthén ──────────────────────────────────────────────────────────────────
def V0_V1(a, Z1, l, mu):
    V0 = 2 * a * EP2 * Z1
    V1 = (a**2 * HBAR_C**2 * (l + 0.5)**2) / (2 * mu)
    return V0, V1

def stable_inv_expm1(a, r):
    z = a * r
    if abs(z) < 1e-8:
        return (1.0/z) - 0.5 + (z/12.0) - (z**3/720.0) + (z**5/30240.0)
    return 1.0 / np.expm1(z)

def barrier_exit_radius_hulthen(a, Z1, l, mu, Qa):
    V0, V1 = V0_V1(a, Z1, l, mu)
    disc  = np.sqrt(V0**2 + 4*V1*Qa)
    return (1.0/a) * np.log(2*V1/(disc - V0) + 1.0)

def I1(r, a, V0, V1, Qa):
    x = stable_inv_expm1(a, r)
    val = (V1*x + V0)*x - Qa
    term1 = -np.sqrt(max(val, 0))
    Qa_over_x = Qa / x if x > 1e-300 else 0.0
    arg_asin = np.clip((V0 - 2*Qa_over_x) / np.sqrt(4*Qa*V1 + V0**2 + 1e-300), -1.0, 1.0)
    term2 = np.sqrt(Qa) * np.arcsin(arg_asin)
    inner = 2*np.sqrt(max(V1,0))*np.sqrt(max(val,0)) + V0 + 2*V1*x
    term3 = -(V0/(2*np.sqrt(max(V1,1e-25)))) * np.log(max(inner, 1e-300))
    return term1 + term2 + term3

def I2(r, a, V0, V1, Qa):
    U0 = V0 - 2*V1
    U1 = Qa + V0 - V1
    x = stable_inv_expm1(a, r)
    y = x + 1.0
    disc = (V1*y + U0)*y - U1
    term1 = np.sqrt(max(disc, 0))
    denom_atan = 2*np.sqrt(max(U1,1e-25)) * np.sqrt(max(disc,1e-300))
    arg_atan = (y*U0 - 2*U1) / denom_atan if denom_atan != 0 else 0.0
    term2 = -np.sqrt(max(U1,0)) * np.arctan(arg_atan)
    inner = 2*np.sqrt(max(V1,0))*np.sqrt(max(disc,0)) + U0 + 2*V1*y
    term3 = (U0/(2*np.sqrt(max(V1,1e-25)))) * np.log(max(abs(inner),1e-300))
    return term1 + term2 + term3

def barrier_integral_hulthen(Rt, Rout, a, V0, V1, Qa):
    F = lambda rv: (1.0/a)*(I1(rv,a,V0,V1,Qa)+I2(rv,a,V0,V1,Qa))
    return F(Rout) - F(Rt)

def half_life_hulthen(A_parent, Z_parent, Qa, a, l=0, h=0):
    A1=A_parent-4; Rt=nuclear_radius(A1)+nuclear_radius(4)
    mu=reduced_mass(A1); xl=bessel_zero(l); Z1=Z_parent-2
    nu=(xl*HBAR_C*3e23)/(2.0*mu*Rt**2)
    V0,V1=V0_V1(a,Z1,l,mu)
    Rout=barrier_exit_radius_hulthen(a,Z1,l,mu,Qa)
    I_val=barrier_integral_hulthen(Rt,Rout,a,V0,V1,Qa)
    ln_P=-(2.0/HBAR_C)*np.sqrt(2.0*mu)*I_val
    return np.log10(np.log(2))+h-np.log10(nu)-(ln_P/np.log(10))


h_unfav = {"eo": 0, "oe": 0, "oo": 0}

# ─── Nuclear data ─────────────────────────────────────────────────────────────
all_nuclei = {

84: {
    "nuclei": [
        (186,84,8.501,"ee",0,0),(188,84,8.082,"ee",0,0),(189,84,7.701,"eo",2,h_unfav["eo"]),
        (190,84,7.693,"ee",0,0),(191,84,7.501,"eo",5,h_unfav["eo"]),(192,84,7.320,"ee",0,0),
        (193,84,7.094,"eo",0,0),(194,84,6.987,"ee",0,0),(195,84,6.749,"eo",0,0),
        (196,84,6.658,"ee",0,0),(197,84,6.412,"eo",0,0),(198,84,6.309,"ee",0,0),
        (199,84,6.074,"eo",0,0),(200,84,5.981,"ee",0,0),(201,84,5.798,"eo",0,0),
        (202,84,5.701,"ee",0,0),(203,84,5.496,"eo",2,h_unfav["eo"]),(204,84,5.484,"ee",0,0),
        (205,84,5.325,"eo",0,0),(206,84,5.327,"ee",0,0),(207,84,5.215,"eo",0,0),
        (208,84,5.215,"ee",0,0),(209,84,4.979,"eo",2,h_unfav["eo"]),(210,84,5.407,"ee",0,0),
        (211,84,7.594,"eo",5,h_unfav["eo"]),(212,84,8.954,"ee",0,0),(213,84,8.536,"eo",0,0),
        (214,84,7.833,"ee",0,0),(215,84,7.526,"eo",0,0),(216,84,6.906,"ee",0,0),
        (217,84,6.662,"eo",0,0),(218,84,6.114,"ee",0,0),(219,84,5.910,"eo",0,0),
    ],
    "exp": {
        186:-4.698970004336, 188:-3.5686,          189:-1.3590,
        190:-2.6108,         191:-0.8268,           192:-1.4921,
        193:-0.6090,         194:-0.3752,           195: 0.6928,
        196: 0.7767,         197: 2.0889,           198: 2.2677,
        199: 3.6434,         200: 3.7857,           201: 4.9097,
        202: 5.3782,         203: 8.3014,           204: 6.2745,
        205: 7.2070,         206: 7.1445,           207: 7.9975,
        208: 7.9609,         209:10.4351,           210: 7.0776,
        211:-0.2831,         212:-6.5302,           213:-5.4311,
        214:-3.7865,         215:-2.7493,           216:-0.8416,
        217: 0.1956,         218: 2.2690,           219: 3.3420,
    },
},

85: {
    "nuclei": [
        (191,85,7.820,"oe",0,0),(196,85,7.198,"oo",0,0),(197,85,7.104,"oe",0,0),
        (198,85,6.889,"oo",0,0),(199,85,6.777,"oe",0,0),(200,85,6.596,"oo",0,0),
        (201,85,6.473,"oe",0,0),(202,85,6.353,"oo",0,0),(203,85,6.210,"oe",0,0),
        (204,85,6.070,"oo",0,0),(205,85,6.019,"oe",0,0),(207,85,5.872,"oe",0,0),
        (208,85,5.751,"oo",0,0),(209,85,5.757,"oe",0,0),(210,85,5.631,"oo",2,h_unfav["oo"]),
        (211,85,5.982,"oe",0,0),(212,85,7.817,"oo",5,h_unfav["oo"]),(213,85,9.254,"oe",0,0),
        (214,85,8.987,"oo",0,0),(215,85,8.178,"oe",0,0),(216,85,7.950,"oo",0,0),
        (217,85,7.201,"oe",0,0),(219,85,6.324,"oe",0,0),
    ],
    "exp": {
        191:-2.7696, 196:-0.4127, 197:-0.4018, 198: 0.6955, 199: 0.8762,
        200: 1.9175, 201: 2.0678, 202: 2.6805, 203: 3.1560, 204: 4.1509,
        205: 4.2079, 207: 4.8793, 208: 6.0281, 209: 5.6670, 210: 7.7405,
        211: 4.7931, 212:-0.5031, 213:-6.9066, 214:-6.2534, 215:-4.0000,
        216:-3.5229, 217:-1.4949, 219: 1.7768,
    },
},

86: {
    "nuclei": [
        (194,86,7.862,"ee",0,0),(195,86,7.690,"eo",0,0),(196,86,7.617,"ee",0,0),
        (197,86,7.411,"eo",0,0),(199,86,7.140,"eo",0,0),(200,86,7.043,"ee",0,0),
        (202,86,6.774,"ee",0,0),(203,86,6.630,"eo",0,0),(204,86,6.547,"ee",0,0),
        (205,86,6.387,"eo",2,h_unfav["eo"]),(206,86,6.384,"ee",0,0),(207,86,6.251,"eo",0,0),
        (208,86,6.261,"ee",0,0),(209,86,6.155,"eo",0,0),(210,86,6.159,"ee",0,0),
        (211,86,5.965,"eo",2,h_unfav["eo"]),(212,86,6.385,"ee",0,0),
        (213,86,8.244,"eo",5,h_unfav["eo"]),(214,86,9.208,"ee",0,0),
        (215,86,8.839,"eo",0,0),(216,86,8.197,"ee",0,0),(217,86,7.887,"eo",0,0),
        (218,86,7.263,"ee",0,0),(219,86,6.946,"eo",2,h_unfav["eo"]),
        (220,86,6.405,"ee",0,0),(221,86,6.148,"eo",2,h_unfav["eo"]),
        (222,86,5.590,"ee",0,0),
    ],
    "exp": {
        194:-3.1079, 195:-2.2218, 196:-2.3565, 197:-1.2596, 199:-0.2291,
        200: 0.0453, 202: 1.0667, 203: 1.8239, 204: 2.0124, 205: 4.6990,
        206: 2.7391, 207: 3.4220, 208: 3.3722, 209: 4.0071, 210: 3.9577,
        211: 5.7521, 212: 3.1565, 213:-1.7022, 214:-6.5867, 215:-5.6383,
        216:-4.3468, 217:-3.2291, 218:-1.4717, 219: 0.6979, 220: 1.7451,
        221: 3.9786, 222: 5.5188,
    },
},

87: {
    "nuclei": [
        (200,87,7.621,"oo",0,0),(201,87,7.516,"oe",0,0),(202,87,7.389,"oo",0,0),
        (203,87,7.260,"oe",0,0),(204,87,7.170,"oo",0,0),(205,87,7.054,"oe",0,0),
        (206,87,6.923,"oo",0,0),(207,87,6.900,"oe",0,0),(208,87,6.772,"oo",0,0),
        (209,87,6.778,"oe",0,0),(210,87,6.671,"oo",2,h_unfav["oo"]),
        (211,87,6.663,"oe",0,0),(212,87,6.528,"oo",2,h_unfav["oo"]),
        (213,87,6.904,"oe",0,0),(214,87,8.589,"oo",5,h_unfav["oo"]),
        (215,87,9.540,"oe",0,0),(216,87,9.174,"oo",0,0),(217,87,8.469,"oe",0,0),
        (218,87,8.013,"oo",0,0),(219,87,7.448,"oe",0,0),
        (220,87,6.800,"oo",1,h_unfav["oo"]),(221,87,6.457,"oe",2,h_unfav["oe"]),
        (223,87,5.561,"oe",4,h_unfav["oe"]),
    ],
    "exp": {
        200:-1.3098, 201:-1.1612, 202:-0.4295, 203:-0.2604, 204: 0.3464,
        205: 0.5966, 206: 1.2596, 207: 1.1923, 208: 1.8222, 209: 1.7539,
        210: 2.6509, 211: 2.3664, 212: 4.1060, 213: 1.5415, 214:-2.1379,
        215:-7.0655, 216:-6.1549, 217:-4.6576, 218:-2.9586, 219:-1.6990,
        220: 1.6197, 221: 2.5386, 223: 7.5185,
    },
},

88: {
    "nuclei": [
        (202,88,7.880,"ee",0,0),
        (203,88,7.730,"eo",0,0),
        (204,88,7.637,"ee",0,0),
        (205,88,7.486,"eo",0,0),
        (206,88,7.415,"ee",0,0),
        (208,88,7.273,"ee",0,0),
        (209,88,7.143,"eo",0,0),
        (210,88,7.151,"ee",0,0),
        (211,88,7.043,"eo",0,0),
        (212,88,7.032,"ee",0,0),
        (213,88,6.862,"eo",2,h_unfav["eo"]),
        (214,88,7.273,"ee",0,0),
        (215,88,8.864,"eo",5,h_unfav["eo"]),
        (216,88,9.526,"ee",0,0),
        (217,88,9.161,"eo",0,0),
        (218,88,8.540,"ee",0,0),
        (219,88,8.138,"eo",2,h_unfav["eo"]),
        (220,88,7.592,"ee",0,0),
        (221,88,6.880,"eo",2,h_unfav["eo"]),
        (222,88,6.678,"ee",0,0),
        (223,88,5.978,"eo",2,h_unfav["eo"]),
        (224,88,5.788,"ee",0,0),
        (226,88,4.870,"ee",0,0),
    ],
    "exp": {
        202: -2.4202, 203:-1.6198, 204:-1.2441, 205:-0.6778, 206:-0.6234, 208: 0.1335,
        209: 0.6812, 210: 0.6085, 211: 1.1452, 212: 1.1844, 213: 2.6578,
        214: 0.3909, 215:-2.7620, 216:-6.7645, 217:-5.7959, 218:-4.5865,
        219:-1.4844, 220:-1.7447, 221: 1.9420, 222: 1.5400, 223: 7.9948,
        224: 5.4999, 226:10.7037,
    },
},

89: {
    "nuclei": [
        (205,89,8.090,"oe",0,0),(207,89,7.840,"oe",0,0),(209,89,7.730,"oe",0,0),
        (211,89,7.620,"oe",0,0),(213,89,7.501,"oe",0,0),(215,89,7.746,"oe",0,0),
        (216,89,9.235,"oo",5,h_unfav["oo"]),(217,89,9.832,"oe",0,0),
        (218,89,9.380,"oo",0,0),(219,89,8.830,"oe",0,0),
        (220,89,8.348,"oo",2,h_unfav["oo"]),(221,89,7.780,"oe",4,h_unfav["oe"]),
        (222,89,7.137,"oo",0,0),(223,89,6.783,"oe",2,h_unfav["oe"]),
        (224,89,6.327,"oo",1,h_unfav["oo"]),(225,89,5.935,"oe",2,h_unfav["oe"]),
        (227,89,5.042,"oe",0,0),
    ],
    "exp": {
        205:-1.6990, 207:-1.5686, 209:-1.0605, 211:-0.6778, 213:-0.1319,
        215:-0.7696, 216:-3.0332, 217:-7.1612, 218:-5.9872, 219:-4.9281,
        220:-0.1871, 221:-1.1413, 222: 0.7210, 223: 2.1045, 224: 5.7216,
        225: 6.2314, 227:10.6968,
    },
},

90: {
    "nuclei": [
        (208,90,8.200,"ee",0,0), (212,90,7.958,"ee",0,0),(213,90,7.837,"eo",0,0),(214,90,7.827,"ee",0,0),
        (215,90,7.665,"eo",2,h_unfav["eo"]),(216,90,8.072,"ee",0,0),
        (217,90,9.435,"eo",5,h_unfav["eo"]),(218,90,9.849,"ee",0,0),
        (219,90,9.510,"eo",0,0),(220,90,8.973,"ee",0,0),
        (221,90,8.626,"eo",2,h_unfav["eo"]),(222,90,8.127,"ee",0,0),
        (223,90,7.567,"eo",2,h_unfav["eo"]),(224,90,7.298,"ee",0,0),
        (225,90,6.920,"eo",2,h_unfav["eo"]),(226,90,6.452,"ee",0,0),
        (227,90,6.146,"eo",2,h_unfav["eo"]),(228,90,5.520,"ee",0,0),
        (229,90,5.168,"eo",2,h_unfav["eo"]),(230,90,4.770,"ee",0,0),
        (232,90,4.081,"ee",0,0),
    ],
    "exp": {
        208: -2.5086, 212:-1.4989, 213:-0.8416, 214:-1.0000, 215: 0.4771, 216:-1.5850,
        217:-3.5850, 218:-6.9136, 219:-5.9788, 220:-4.9991, 221:-2.3261,
        222:-2.7069, 223: 1.7782, 224: 0.0212, 225: 3.8102, 226: 3.2656,
        227: 6.8241, 228: 7.7804, 229:14.6655, 230:12.3761, 232:17.6449,
    },
},

91: {
    "nuclei": [
        (213,91,8.384,"oe",0,0),
        (215,91,8.240,"oe",0,0),(217,91,8.489,"oe",0,0),
        (219,91,10.08,"oe",0,0),
        (221,91,9.250,"oe",0,0),(227,91,6.580,"oe",0,0),
        (229,91,5.835,"oe",1,h_unfav["oe"]), (231,91,5.150,"oe",0,0),
    ],
    "exp": {
        213:-2.2757, 215:-1.8539, 217:-2.4202,
        219:-7.2757, 221:-5.2291,227: 3.7309, 228: 8.5453, 
        229:10.0334, 231: 12.014150769062,
    },
},

92: {
    "nuclei": [
        (216,92,8.531,"ee",0,0),(218,92,8.775,"ee",0,0),
        (219,92,9.940,"eo",5,h_unfav["eo"]),(221,92,9.890,"eo",0,0),
        (222,92,9.480,"ee",0,0),(223,92,9.158,"eo",0,0),(224,92,8.633,"ee",0,0),
        (226,92,7.701,"ee",0,0),(227,92,7.211,"eo",2,h_unfav["eo"]),
        (228,92,6.800,"ee",0,0),(229,92,6.475,"eo",0,0),(230,92,5.993,"ee",0,0),
        (231,92,5.576,"eo",3,h_unfav["eo"]),(232,92,5.414,"ee",0,0),
        (233,92,4.909,"eo",0,0),(234,92,4.858,"ee",0,0),
        (235,92,4.678,"eo",1,h_unfav["eo"]),(236,92,4.573,"ee",0,0),
        (238,92,4.270,"ee",0,0),
    ],
    "exp": {
        216:-2.3468, 218:-3.2924, 219:-4.2218, 221:-6.1805, 222:-5.3279,
        223:-4.2076, 224:-3.3873, 226:-0.5719, 227: 2.5185, 228: 2.7591,
        229: 4.2405, 230: 6.2546, 231:12.1126, 232: 9.3370, 233:12.7007,
        234:12.8889, 235:17.6678, 236:14.8684, 238:17.1489,
    },
},

93: {
    "nuclei": [
        (231,93,6.370,"oe",0,0),
        (235,93,5.1938,"oe",1 ,h_unfav["oe"]),
        (237,93,4.9573,"oe",1 ,h_unfav["oe"]),
    ],
    "exp": {
        231: 5.1655, 235: 13.9432, 237: 15.4516,
    },
},

94: {
    "nuclei": [
        (228,94,7.940,"ee",0,0),
        (230,94,7.180,"ee",0,0),
        (231,94,6.839,"eo",0,0),
        (232,94,6.716,"ee",0,0),
        (234,94,6.310,"ee",0,0),
        (236,94,5.867,"ee",0,0),
        (237,94,5.748,"eo",1,h_unfav["eo"]),
        (238,94,5.593,"ee",0,0),
        (240,94,5.256,"ee",0,0),
        (241,94,5.140,"eo",2,h_unfav["eo"]),
        (242,94,4.985,"ee",0,0),
        (244,94,4.666,"ee",0,0),
        
    ],
    "exp": {
        228: 0.0414, 230: 2.0086, 231: 3.7126, 232: 4.2979, 234: 5.7226,
        236: 7.9549, 237: 12.1467, 238: 9.4418, 240: 11.3158, 241: 15.7208,
        242: 13.0705, 244: 15.4089,
    },

},

95: {
    "nuclei": [
        (235,95,6.576,"oe",0,0),
        (237,95,6.200,"oe",1 ,h_unfav["oe"]),
        (239,95,5.922,"oe",1 ,h_unfav["oe"]),
    ],
    "exp": {
        235: 5.1889, 237: 7.2471, 239: 10.1133,
    },
},

96: {
    "nuclei": [
        (234,96,7.365,"ee",0,0),
        (238,96,6.620,"ee",0,0),
        (240,96,6.398,"ee",0,0),
        (241,96,6.185,"eo",3,h_unfav["eo"]),
        (242,96,6.215,"ee",0,0),
        (243,96,6.169,"eo",2,h_unfav["eo"]),
        (244,96,5.902,"ee",0,0),
        (245,96,5.622,"eo",2,h_unfav["eo"]),
        (246,96,5.475,"ee",0,0),
        (247,96,5.354,"eo",1,h_unfav["eo"]),
        (248,96,5.162,"ee",0,0),
        
    ],
    "exp": {
        234: 3.0881, 238: 6.3521, 240: 6.3679, 241: 11.2762,
        242: 7.1483, 243: 9.7106, 244: 8.7567, 245: 13.6608,
        246: 11.1715, 247: 15.5520, 248: 13.07859, 
    },

},

97: {
    "nuclei": [
        (243,97,6.874,"oe",2 ,h_unfav["oe"]),
        (244,97,6.779,"oo",2 ,h_unfav["oo"]),
        (245,97,6.454,"oe",2 ,h_unfav["oe"]),
        (247,97,5.890,"oe",2 ,h_unfav["oe"]),
        (249,97,5.521,"oe",2 ,h_unfav["oe"]),
    ],
    "exp": {
        243: 7.8478, 244: 8.4788, 245: 9.3616, 247: 11.8983, 249: 13.4970,
    },
},

98: {
    "nuclei": [
        (240,98,7.711,"ee",0,0),
        (242,98,7.517,"ee",0,0),
        (244,98,7.329,"ee",0,0),
        (245,98,7.2585,"eo",0,0),
        (246,98,6.862,"ee",0,0),
        (248,98,6.361,"ee",0,0),
        (249,98,6.2933,"eo",1,h_unfav["eo"]),
        (250,98,6.128,"ee",0,0),
        (251,98,6.177,"eo",5,h_unfav["eo"]),
        (252,98,6.217,"ee",0,0),
        (254,98,5.926,"ee",0,0),
    ],
    "exp": {
        240: 1.6053, 242: 2.5355, 244: 3.0681, 245: 3.8349,
        246: 5.10897, 248: 7.4596, 249: 11.6567, 250: 8.6157,
        251: 12.0371, 252: 7.9352, 254: 9.3083,
    },
},

99: {
    "nuclei": [
        (245,99,7.909,"oe",3 ,h_unfav["oe"]),
        (251,99,6.598,"oe",0 ,0),
        (253,99,6.739,"oe",0 ,0),
    ],
    "exp": {
        245: 3.5185, 251: 7.4620, 253: 6.2476,
    },
},

100: {
    "nuclei": [
        (246,100,8.379,"ee",0,0),
        (248,100,7.995,"ee",0,0),
        (250,100,7.557,"ee",0,0),
        (251,100,7.425,"eo",1,h_unfav["eo"]),
        (252,100,7.153,"ee",0,0),
        (253,100,7.198,"eo",5,h_unfav["eo"]),
        (254,100,7.307,"ee",0,0),
        (255,100,7.240,"eo",4,h_unfav["eo"]),
        (256,100,7.027,"ee",0,0),
        (257,100,6.863,"eo",2,h_unfav["eo"]),
    ],
    "exp": {
        246: 0.1875, 248: 1.5378, 250: 3.2695,
        251: 7.8492, 252: 4.9609, 253: 8.2095, 254: 4.0669,
        255: 8.01375, 256: 5.0658, 257: 9.1752,
    },
},

101: {
    "nuclei": [
        (253,101,7.573,"oe",1 ,h_unfav["oe"]),
        (255,101,7.905,"oe",2 ,h_unfav["oe"]),
        (256,101,7.737,"oo",3 ,h_unfav["oo"]),
        (257,101,7.557,"oe",1 ,h_unfav["oe"]),
    ],
    "exp": {
        253: 4.7426, 255: 6.3644, 256: 6.2872, 257: 7.9999,
    },
},

102: {
    "nuclei": [
        (251,102,8.752,"eo",0,0),
        (252,102,8.549,"ee",0,0),
        (253,102,8.415,"eo",2,h_unfav["eo"]),
        (254,102,8.226,"ee",0,0),
        (255,102,8.428,"eo",5,h_unfav["eo"]),
        (256,102,8.582,"ee",0,0),
    ],
    "exp": {
        251: 0.0086, 252: 0.7007, 253: 3.6509, 254: 1.7092,
        255: 2.8474, 256: 0.4638,
    },
},

104: {
    "nuclei": [
        (256,104,8.926,"ee",0,0),
        (258,104,9.196,"ee",0,0),
    ],
    "exp": {
        256: 0.3385, 258: -0.6985,
    },
},

106: {
    "nuclei": [
        (260,106,9.901,"ee",0,0),
    ],
    "exp": {
        260: -2.0043, 
    },
},

108: {
    "nuclei": [
        (264,108,10.591,"ee",0,0),
        (266,108,10.346,"ee",0,0),
        (268,108,9.760,"ee",0,0),
        (270,108,9.070,"ee",0,0),
    ],
    "exp": {
        264: -3.1037, 266:-2.4080, 268: -0.3979, 270: 1.1024,
    },
},

110: {
    "nuclei": [
        (270,110,11.12,"ee",0,0),
    ],
    "exp": {
        270: -3.7212, 
    },
},

114: {
    "nuclei": [
        (286,114,10.36,"ee",0,0),
        (288,114,10.17,"ee",0,0),
    ],
    "exp": {
        286: -0.6953, 288: -0.3979,
    },
},

116: {
    "nuclei": [
        (290,116,11,"ee",0,0),
        (292,116,10.791,"ee",0,0),
    ],
    "exp": {
        290: -2.0809, 292: -1.8927,
    },
},


}

chain_names = {84:"Po",85:"At",86:"Rn",87:"Fr",88:"Ra",89:"Ac",90:"Th",91:"Pa",92:"U",
               93:"Np",94:"Pu",95:"Am",96:"Cm",97:"Bk",98:"Cf",99:"Es",100:"Fm",
               101:"Md",102:"No",104:"Rf",106:"Sg",108:"Hs",110:"Ds",114:"Fl",116:"Lv"}

# ─── Build groups as the article ─────────────────
groups_fav   = defaultdict(list)  
groups_unfav = defaultdict(list)   

for Z, chain in all_nuclei.items():
    exp_dict = chain["exp"]
    for A, Z_, Qa, pairing, l, h_tuple in chain["nuclei"]:
        if A not in exp_dict:
            continue
        logT_exp = exp_dict[A]
        if l == 0:
            groups_fav[pairing].append((A, Z_, Qa, pairing, l, logT_exp))
        else:
            groups_unfav[pairing].append((A, Z_, Qa, pairing, l, logT_exp))

# ─── Sigma functions ──────────────────────────────────────────────────────────
def sigma_hulthen_fav(log_a, members):
    a = 10**log_a
    sq = []
    for A, Z_, Qa, pairing, l, logT_exp in members:
        try:
            logT_th = half_life_hulthen(A, Z_, Qa, a, l=0, h=0)
            if np.isfinite(logT_th):
                sq.append((logT_th - logT_exp)**2)
        except:
            pass
    return np.sqrt(np.mean(sq)) if sq else 1e10

def sigma_hulthen_unfav(params, members):
    log_a, h = params
    a = 10**log_a
    sq = []
    for A, Z_, Qa, pairing, l, logT_exp in members:
        try:
            logT_th = half_life_hulthen(A, Z_, Qa, a, l=l, h=h)
            if np.isfinite(logT_th):
                sq.append((logT_th - logT_exp)**2)
        except:
            pass
    return np.sqrt(np.mean(sq)) if sq else 1e10

# ─── Fit a for favored (h=0 fixed) ───────────────────────────────────────────
print("="*65)
print("FAVORED DECAYS — fit a only (h=0 fixed)")
print("="*65)

best_a_fav     = {}
best_sigma_fav = {}

for pairing in ["ee", "eo", "oe", "oo"]:
    members = groups_fav.get(pairing, [])
    if not members:
        print(f"  {pairing}: no nuclei")
        continue

    best_s      = 1e10
    best_log_a  = None

    # Fine grid search across full range including very small a
    for log_a_init in np.linspace(-12, -2, 500):
        s = sigma_hulthen_fav(log_a_init, members)
        if s < best_s:
            best_s     = s
            best_log_a = log_a_init

    # Refine
    res = minimize_scalar(
            lambda la, m=members: sigma_hulthen_fav(la, m),
            bounds=(best_log_a - 1.0, best_log_a + 1.0),
            method='bounded',
            options={'xatol': 1e-12})

    a_best = 10**res.x
    s_best = sigma_hulthen_fav(res.x, members)
    best_a_fav[pairing]     = a_best
    best_sigma_fav[pairing] = s_best
    print(f"  {pairing}: n={len(members):3d}  "
          f"a={a_best:.15e} fm⁻¹   σ={s_best:.4f}")
    

# ─── Fit a AND h simultaneously for unfavored ────────────────────────────────
print("\n" + "="*65)
print("UNFAVORED DECAYS — fit a and h simultaneously")
print("="*65)

best_a_unfav     = {}
best_h_unfav     = {}
best_sigma_unfav = {}

for pairing in ["eo", "oe", "oo"]:
    members = groups_unfav.get(pairing, [])
    if not members:
        print(f"  {pairing}: no nuclei")
        continue

    best_s   = 1e10
    best_params = None

    # Coarse grid search first, then refine
    # log_a grid covers the article's range
    log_a_grid = np.linspace(-10, -2, 17)
    h_grid     = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    for log_a_init in log_a_grid:
        for h_init in h_grid:
            s = sigma_hulthen_unfav([log_a_init, h_init], members)
            if s < best_s:
                best_s = s
                best_params = [log_a_init, h_init]

    # Refine the best point found in the grid
    try:
        res = minimize(
            sigma_hulthen_unfav,
            x0=best_params,
            args=(members,),
            method='Nelder-Mead',
            options={'xatol': 1e-10, 'fatol': 1e-10,
                     'maxiter': 50000, 'adaptive': True}
        )
        log_a_best, h_best = res.x
    except:
        log_a_best, h_best = best_params

    a_best = 10**log_a_best
    s_best = sigma_hulthen_unfav([log_a_best, h_best], members)
    best_a_unfav[pairing]     = a_best
    best_h_unfav[pairing]     = h_best
    best_sigma_unfav[pairing] = s_best
    print(f"  {pairing}: n={len(members):3d}  "
          f"a={a_best:.15e} fm⁻¹   h={h_best:.6f}   σ={s_best:.4f}")


# ─── Plot — one figure per chain ─────────────────────────────────────────────
for Z, chain in all_nuclei.items():
    nuclei_list = chain["nuclei"]
    exp_dict    = chain["exp"]
    name        = chain_names[Z]

    A_exp_s    = sorted(exp_dict.keys())
    logT_exp_s = [exp_dict[a] for a in A_exp_s]

    A_fav_v,   logT_fav_v   = [], []
    A_unfav_v, logT_unfav_v = [], []

    for A, Z_, Qa, pairing, l, h in nuclei_list:
        if A not in exp_dict:
            continue

        if l == 0:
            a_use = best_a_fav.get(pairing)
            if a_use:
                logT_th = half_life_hulthen(A, Z_, Qa, a_use, l=0, h=0)
                if np.isfinite(logT_th):
                    A_fav_v.append(A)
                    logT_fav_v.append(logT_th)
        else:
            a_use = best_a_unfav.get(pairing)
            h_use = best_h_unfav.get(pairing, 0.0)
            if a_use:
                logT_th = half_life_hulthen(A, Z_, Qa, a_use, l=l, h=h_use)
                if np.isfinite(logT_th):
                    A_unfav_v.append(A)
                    logT_unfav_v.append(logT_th)

    N126_A = 126 + Z
    A_vals = [A for A, *_ in nuclei_list]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(A_exp_s, logT_exp_s, 'k-', lw=1.8, label="Exp", zorder=5)
    if A_fav_v:
        ax.scatter(A_fav_v, logT_fav_v, marker='o', s=45,
                   color='red', zorder=4, label="Th. fav ($l=0$)")
    if A_unfav_v:
        ax.scatter(A_unfav_v, logT_unfav_v, marker='o', s=45,
                   color='blue', zorder=4, label="Th. unfav ($l>0$)")
    if min(A_vals) <= N126_A <= max(A_vals):
        ax.axvline(x=N126_A, color='gray', ls='--',
                   alpha=0.5, label="N=126")

    ax.set_xlabel("A", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(T_{1/2})$ [s]", fontsize=12)
    ax.set_title(f"{name} (Z={Z}) — α decay half-lives", fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()