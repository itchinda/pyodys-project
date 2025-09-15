import numpy as np
import json
import os
from fractions import Fraction


def _get_erk1():
    """Returns the Butcher tableau data for the 1st order Euler explicit scheme."""
    return {
        "A": [[0]],
        "B": [1],
        "C": [0],
        "order": 1
    }

def _get_erk2_midpoint():
    """Returns the Butcher tableau data for the 2nd explicit mid-point scheme."""
    return {
        "A": [[0.0, 0.0],
              [0.5, 0.0]],
        "B": [0.0, 1.0],
        "C": [0.0, 0.5],
        "order": 2
    }

def _get_erk4():
    """Returns the data for the classical RK4 scheme."""
    return {
        "A": [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ],
        "B": [1/6, 1/3, 1/3, 1/6],
        "C": [0.0, 0.5, 0.5, 1.0],
        "order": 4
    }


def _get_sdirk1():
    """Returns the Butcher tableau data for the 1st order Euler implicit scheme."""
    return {
        "A": [[1]],
        "B": [1],
        "C": [1],
        "order": 1
    }

def _get_sdirk2_midpoint():
    """Returns the butcher tableau data for the 2nd order 1 stage implicit mid-point scheme."""
    return {
        "A": [[0.5]],
        "B": [1.0],
        "C": [0.5],
        "order": 2
    }

def _get_sdirk43_crouzeix():
    """Returns the butcher tableau data for the 4th order 3 stages Crouzeix scheme."""
    gamma = 1.068579021300
    delta = 6*(2*gamma - 1)**2
    return {
        "A": [
            [gamma, 0.0, 0.0],
            [0.5-gamma, gamma, 0.0],
            [2*gamma, 1-4*gamma, gamma]
        ],
        "B": [1/delta, 1-2/delta, 1/delta],
        "C": [gamma, 0.5, 1-gamma],
        "order": 4
    }

def _get_cooper_verner():
    """Calculates the Cooper-Verner coefficients and returns them."""
    # Define coefficients
    sqrt21 = np.sqrt(21)
    
    a21 = 1/2
    a31 = 1/4; a32 = 1/4
    a41 = 1./7; a42 = (-7+3*sqrt21)/98; a43 = (21-5*sqrt21)/49
    a51 = (11-sqrt21)/84; a52=0; a53 = (18-4*sqrt21)/63; a54 = (21+sqrt21)/252
    a61 = (5-sqrt21)/48;  a62=0; a63 = (9-sqrt21)/36; a64 = (-231-14*sqrt21)/360; a65 = (63+7*sqrt21)/80
    a71 = (10+sqrt21)/42; a72=0; a73 = (-432-92*sqrt21)/315; a74=(633+145*sqrt21)/90; a75=(-504-115*sqrt21)/70; a76 = (63+13*sqrt21)/35
    a81 = 1./14; a82=0; a83=0; a84=0; a85 = (14+3*sqrt21)/126; a86=(13+3*sqrt21)/63; a87 = 1./9
    a91 = 1./32; a92=0; a93=0; a94=0; a95 = (91+21*sqrt21)/576; a96=11./72; a97=(-385+75*sqrt21)/1152; a98 = (63-13*sqrt21)/128
    a101= 1./14; a102=0;a103=0;a104=0;a105=1./9; a106=(-733+147*sqrt21)/2205; a107 = (515-111*sqrt21)/504; a108 = (-51+11*sqrt21)/56; a109 = (132-28*sqrt21)/245
    a111= 0; a112=0; a113=0; a114=0; a115=(-42-7*sqrt21)/18; a116 = (-18-28*sqrt21)/45; a117=(-273+53*sqrt21)/72; a118=(301-53*sqrt21)/72; a119=(28+28*sqrt21)/45; a1110=(49+7*sqrt21)/18
    b1 = 1/20; b2=0; b3=0; b4=0; b5=0; b6=0; b7=0; b8=49/180; b9=16/45; b10=49/180; b11=1/20
    c1=0; c2 = 1./2; c3=1./2; c4 = (7-sqrt21)/14; c5 = (7-sqrt21)/14; c6=1/2; c7=(7+sqrt21)/14; c8=(7+sqrt21)/14; c9=1./2; c10=(7-sqrt21)/14; c11=1
    A = [
        [   0,      0,    0,    0,    0,    0,    0,    0,    0,     0,   0 ],
        [ a21,      0,    0,    0,    0,    0,    0,    0,    0,     0,   0 ],
        [ a31,    a32,    0,    0,    0,    0,    0,    0,    0,     0,   0 ],
        [ a41,    a42,  a43,    0,    0,    0,    0,    0,    0,     0,   0 ],
        [ a51,    a52,  a53,  a54,    0,    0,    0,    0,    0,     0,   0 ],
        [ a61,    a62,  a63,  a64,  a65,    0,    0,    0,    0,     0,   0 ],
        [ a71,    a72,  a73,  a74,  a75,  a76,    0,    0,    0,     0,   0 ],
        [ a81,    a82,  a83,  a84,  a85,  a86,  a87,    0,    0,     0,   0 ],
        [ a91,    a92,  a93,  a94,  a95,  a96,  a97,  a98,    0,     0,   0 ],
        [ a101,  a102, a103, a104, a105, a106, a107, a108, a109,     0,   0 ],
        [ a111,  a112, a113, a114, a115, a116, a117, a118, a119, a1110,   0 ]
    ]
    B = [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11]
    C = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11]

    return {
        "A": A,
        "B": B,
        "C": C,
        "order": 8
    }

def _get_euler_heun():
    """
    Embedded ERK: Euler-Heun method (Order 1/2, 2 stages)
    """
    A = [
            [0.0, 0.0],
            [1.0, 0.0]
        ]
    b = [0.5, 0.5]       # higher order solution (order 2)
    bh = [1.0, 0.0]      # lower order embedded solution (order 1)
    B = [b, bh]
    C = [0.0, 1.0]
    return {"A": A, "B": B, "C": C, "order": 2, "embedded_order": 1}

def _get_bogacki_shampine():
    """
    Embedded ERK: Bogacki–Shampine method (Order 3/4, 4 stages)
    """
    A = [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 3/4, 0.0, 0.0],
            [2/9, 1/3, 4/9, 0.0]  # Last row not used in A for this tableau
        ]
    b = [7/24, 1/4, 1/3, 1/8]     # order 4 solution
    bh = [2/9, 1/3, 4/9, 0.0]   # order 3 embedded solution
    B = [b, bh]
    C = [0.0, 0.5, 3/4, 1.0]
    return {"A": A, "B": B, "C": C, "order": 4, "embedded_order": 3}


def _get_fehlberg45():
    """
    Embedded ERK: Fehlberg 4(5) method (Order 4/5, 6 stages)
    """
    A = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1/4, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3/32,9/32, 0.0, 0.0, 0.0, 0.0],
            [1932/2197, -7200/2197, 7296/2197, 0.0, 0.0, 0.0],
            [439/216, -8, 3680/513, -845/4104, 0.0, 0.0],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0.0]
        ]
    b = [25/216, 0.0, 1408/2565, 2197/4104, -2/10, 0.0]       # order 4
    bh = [16/135, 0.0, 6656/12825, 28561/56430, -9/50, 2/55]  # order 5 embedded
    B = [b, bh]
    C = [0.0, 1/4, 3/8, 12/13, 1.0, 1/2]
    return {"A": A, "B": B, "C": C, "order": 5, "embedded_order": 4}

def _get_dopri5():
    """
    Embedded explicit Runge-Kutta: Dormand-Prince (Order 4/5, 7 stages)
    """
    A = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1/5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3/40, 9/40, 0.0, 0.0, 0.0, 0.0, 0.0],
        [44/45, -56/15, 32/9, 0.0, 0.0, 0.0, 0.0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0.0, 0.0, 0.0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0.0, 0.0],
        [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0],
    ]
    
    b = [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0]
    bh = [5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    B = [b, bh]

    C = [0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]

    return {"A": A, "B": B, "C": C, "order": 5, "embedded_order": 4}

def _get_sdirk21_crouzeix_raviart():
    """
    Embedded SDIRK: Crouzeix-Raviart method (Order 1/2, 2 stages)
    A-stable.
    """
    
    gamma = 1 - 1.0/ np.sqrt(2)
    
    A = [
            [gamma, 0.0],
            [1.0 - gamma, gamma]
        ]
    
    b = [1-gamma, gamma] # Coefficients for the higher-order solution (Order 2)
    b_embedded = [1.0, 0.0]  # Coefficients for the lower-order embedded solution (Order 1)
    B = [b, b_embedded]
    C = [gamma, 1.0]
    
    return {"A": A, "B": B, "C": C, "order": 2, "embedded_order": 1}

def _get_sdirk_norsett_thomson_23():
    """
    Embedded SDIRK: Norsett & Thomson (Order 2/3, 3 stages)
    """
    A = [
        [5/6, 0.0, 0.0],
        [-61/108, 5/6, 0.0],
        [-23/183, -33/61, 5/6]
    ]
    b = [25/61, 36/61, 0.0] # Order 3 solution for step
    bh = [26/61, 324/671, 1/11] # Order 2 solution for prediction
    B = [b, bh]
    C = [5/6, 29/108, 1/6]
    return {"A": A, "B": B, "C": C, "order": 3, "embedded_order": 2}

def _get_sdirk_norsett_thomson_34():
    """Returns the data for the SDIRK scheme."""
    alpha = 5.0/6
    A = [
        [alpha, 0, 0, 0],
        [-15/26, alpha, 0, 0],
        [215/54, -130/27, alpha, 0],
        [4007/6075, -31031/24300, -133/2700, alpha]
    ]
    B = [
        [32/75, 169/300, 1/100, 0],
        [61/150, 2197/2100, 19/100, -9/14]
    ]
    C = [alpha, 10/39, 0, 1/6]
    return {
        "A": A,
        "B": B,
        "C": C,
        "order": 4,
        "embedded_order": 3
    }

def _get_sdirk_hairer_norsett_wanner_45():
    """
    Embedded SDIRK: Hairer, Norsett & Wanner (Order 4/5, 5 stages)
    """
    A = [
        [1/4, 0.0, 0.0, 0.0, 0.0],
        [1/2,    1/4, 0.0, 0.0, 0.0],
        [17/50, -1/25,    1/4, 0.0, 0.0],
        [371/1360, -137/2720, 15/544, 1/4, 0.0],
        [25/24, -49/48, 125/16, -85/12,    1/4]
    ]
    b = [59/48, -17/96, 225/32, -85/12, 0.0] # Order 5 solution for step
    bh = [25/24, -49/48, 125/16, -85/12, 1/4] # Order 4 solution for prediction
    B = [b, bh]
    C = [1/4, 3/4, 11/20, 1/2, 1.0]
    return {"A": A, "B": B, "C": C, "order": 5, "embedded_order": 4}

def _get_esdirk6():

    A = [
            [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)],
            [Fraction(5, 16), Fraction(5, 16), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)],
            [Fraction(-6647797099592, 102714892273533), Fraction(-6647797099592, 102714892273533), Fraction(5, 16), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)],
            [Fraction(-87265218833, 1399160431079), Fraction(-87265218833, 1399160431079), Fraction(3230569391728, 5191843160709), Fraction(5, 16), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)],
            [Fraction(-3742173976023, 7880396319491), Fraction(-4537732256035, 9784784042546), Fraction(32234033847818, 24636233068093), Fraction(1995418204833, 9606020544314), Fraction(5, 16), Fraction(0, 1), Fraction(0, 1)],
            [Fraction(-460973220726, 7579441323155), Fraction(-113988582459, 8174956167569), Fraction(-679076942985, 7531712581924), Fraction(1946214040135, 12392905069014), Fraction(-2507263458377, 16215886710685), Fraction(5, 16), Fraction(0, 1)],
            [Fraction(2429030329867, 4957732179206), Fraction(5124723475981, 12913403568538), Fraction(3612624980699, 11761071195830), Fraction(714493169479, 5549220584147), Fraction(-4586610949246, 13858427945825), Fraction(-4626134504839, 7500671962341), Fraction(5, 16)]
        ]

    b = [Fraction(541976983222, 5570117184863), Fraction(424517620289, 10281234581904), Fraction(3004784109584, 2968823999583), Fraction(-1080268266981, 2111416452515), Fraction(3198291424887, 7137915940442), Fraction(-6709580973937, 9894986011196), Fraction(4328230890552, 7324362344791)]
    bhat = [Fraction(23807813993, 6613359907661), Fraction(122567156372, 6231407414731), Fraction(5289947382915, 9624205771537), Fraction(-132784415823, 2592433009541), Fraction(2055455363695, 9863229933602), Fraction(-686952476184, 6416474135057), Fraction(2766631516579, 7339217152243)]

    # Convert to floats for numerical use
    A = [[float(x) for x in row] for row in A]
    b_float = [float(x) for x in b]
    bhat_float = [float(x) for x in bhat]
    B = [b_float, bhat_float]
    c1 = 0; c2 = 5./8; c3 = 5.*(2-np.sqrt(2))/16; c4 = 81./100; c5 = 89./100; c6 = 3./20; c7 = 11./16
    C = [c1,c2,c3,c4,c5,c6,c7]
    return {"A": A, "B": B, "C": C, "order": 6, "embedded_order": 4}

def available_butcher_table_data_to_json():
    """
    Collects data from all scheme functions and generates a single JSON file.
    """
    all_schemes = {
        "erk1": _get_erk1(),
        "erk2_midpoint": _get_erk2_midpoint(),
        "erk4": _get_erk4(),
        "sdirk1": _get_sdirk1(),
        "sdirk2_midpoint": _get_sdirk2_midpoint(),
        "sdirk43_crouzeix": _get_sdirk43_crouzeix(),
        "cooper_verner": _get_cooper_verner(),
        "euler_heun": _get_euler_heun(),
        "bogacki_shampine": _get_bogacki_shampine(),
        "fehlberg45": _get_fehlberg45(),
        "dopri5": _get_dopri5(),
        "sdirk21_crouzeix_raviart": _get_sdirk21_crouzeix_raviart(),
        "sdirk_norsett_thomson_23": _get_sdirk_norsett_thomson_23(),
        "sdirk_norsett_thomson_34": _get_sdirk_norsett_thomson_34(),
        "sdirk_hairer_norsett_wanner_45": _get_sdirk_hairer_norsett_wanner_45(),
        "esdirk6": _get_esdirk6()
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "tableaux_de_butcher_disponibles.json")

    with open(file_path, 'w') as f:
        json.dump(all_schemes, f, indent=4)
    
if __name__ == '__main__':
    available_butcher_table_data_to_json()