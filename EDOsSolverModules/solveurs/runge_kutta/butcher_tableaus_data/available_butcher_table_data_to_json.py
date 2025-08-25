import numpy as np
import json
import os

def _get_euler_explicite():
    """Returns the data for the Explicit Euler scheme."""
    return {
        "A": [[0]],
        "B": [1],
        "C": [0],
        "ordre": 1
    }

def _get_euler_implicite():
    """Returns the data for the Implicit Euler scheme."""
    return {
        "A": [[1]],
        "B": [1],
        "C": [1],
        "ordre": 1
    }

def _get_rk4():
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
        "ordre": 4
    }

def _get_cooper_verner():
    """Calculates the Cooper-Verner coefficients and returns them."""
    # Define coefficients
    sqrt21 = np.sqrt(21)
    
    a21 = 1/2
    a31 = 1/4; a32 = 1/4
    a41 = 1/7; a42 = (-7-3*sqrt21)/98; a43 = (21+5*sqrt21)/49
    a51 = (11+sqrt21)/84; a52=0; a53 = (18+4*sqrt21)/63; a54 = (21-sqrt21)/252
    a61 = (5+sqrt21)/48;  a62=0; a63 = (9+sqrt21)/36; a64 = (-231+14*sqrt21)/360; a65 = (63-7*sqrt21)/80
    a71 = (10-sqrt21)/42; a72=0; a73 = (-432+92*sqrt21)/315; a74=(633-145*sqrt21)/90; a75=(-504+115*sqrt21)/70; a76 = (63-13*sqrt21)/35
    a81 = 1/14; a82=0; a83=0; a84=0; a85 = (14-3*sqrt21)/126; a86=(13-3*sqrt21)/63; a87 = 1/9
    a91 = 1/32; a92=0; a93=0; a94=0; a95 = (91-21*sqrt21)/576; a96=11/72; a97=(-385-75*sqrt21)/1152; a98 = (63+13*sqrt21)/128
    a101= 1/14; a102=0;a103=0;a104=0;a105=1/9; a106=(-733-147*sqrt21)/2205; a107 = (515+111*sqrt21)/504; a108 = (-51-11*sqrt21)/56; a109 = (132+28*sqrt21)/245
    a111= 0; a112=0; a113=0; a114=0; a115=(-42+7*sqrt21)/18; a116 = (-18+28*sqrt21)/45; a117=(-273-53*sqrt21)/72; a118=(301+53*sqrt21)/72; a119=(28-28*sqrt21)/45; a1110=(49-7*sqrt21)/1
    b1 = 1/20; b2=0; b3=0; b4=0; b5=0; b6=0; b7=0; b8=49/180; b9=16/45; b10=49/180; b11=1/20
    c1=0; c2 = 1/2; c3=1/2; c4 = (7+sqrt21)/14; c5 = (7+sqrt21)/14; c6=1/2; c7=(7-sqrt21)/14; c8=(7-sqrt21)/14; c9=1/2; c10=(7+sqrt21)/14; c11=1
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
        "ordre": 8
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
    return {"A": A, "B": B, "C": C, "ordre": 2}

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
    return {"A": A, "B": B, "C": C, "ordre": 4}


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
    return {"A": A, "B": B, "C": C, "ordre": 5}

def _get_sdirk21_crouzeix_raviart():
    """
    Embedded SDIRK: Crouzeix-Raviart method (Order 1/2, 2 stages)
    A-stable.
    """
    
    gamma = (3 + np.sqrt(3)) / 6.0
    
    A = [
            [gamma, 0.0],
            [1.0 - gamma, gamma]
        ]
    
    b = [0.5, 0.5] # Coefficients for the higher-order solution (Order 2)
    b_embedded = [1.0, 0.0]  # Coefficients for the lower-order embedded solution (Order 1)
    B = [b, b_embedded]
    C = [gamma, 1.0]
    
    return {"A": A, "B": B, "C": C, "ordre": 2}

def _get_sdirk32():
    """
    Embedded SDIRK: Standard SDIRK32 method (Order 2/3, 3 stages)
    A-stable.
    """
    gamma = (3 + np.sqrt(3)) / 6.0
    
    A = [
            [gamma, 0.0, 0.0],
            [1.0 - gamma, gamma, 0.0],
            [0.0, 1.0 - gamma, gamma]
        ]
    
    b = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0] # Coefficients for the higher-order solution (Order 3)
    b_embedded = [1.0 / 2.0, 1.0 / 2.0, 0.0]  # Coefficients for the lower-order embedded solution (Order 2)
    B = [b, b_embedded]
    C = [gamma, 1.0, 1.0]
    
    return {"A": A, "B": B, "C": C, "ordre": 3}

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
    return {"A": A, "B": B, "C": C, "ordre": 3}

def _get_sdirk_norsett_thomson_34():
    """Returns the data for the SDIRK scheme."""
    alpha = 5/6
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
        "ordre": 4
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
    return {"A": A, "B": B, "C": C, "ordre": 5}

def available_butcher_table_data_to_json():
    """
    Collects data from all scheme functions and generates a single JSON file.
    """
    all_schemes = {
        "euler_explicite": _get_euler_explicite(),
        "euler_implicite": _get_euler_implicite(),
        "rk4": _get_rk4(),
        "cooper_verner": _get_cooper_verner(),
        "euler_heun": _get_euler_heun(),
        "bogacki_shampine": _get_bogacki_shampine(),
        "fehlberg45": _get_fehlberg45(),
        "sdirk21_crouzeix_raviart": _get_sdirk21_crouzeix_raviart(),
        "sdirk32": _get_sdirk32(),
        "sdirk_norsett_thomson_23": _get_sdirk_norsett_thomson_23(),
        "sdirk_norsett_thomson_34": _get_sdirk_norsett_thomson_34(),
        "sdirk_hairer_norsett_wanner_45": _get_sdirk_hairer_norsett_wanner_45()
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "tableaux_de_butcher_disponibles.json")

    with open(file_path, 'w') as f:
        json.dump(all_schemes, f, indent=4)
    
    #print(f"Successfully generated {file_path} with {len(all_schemes)} schemes.")

if __name__ == '__main__':
    available_butcher_table_data_to_json()