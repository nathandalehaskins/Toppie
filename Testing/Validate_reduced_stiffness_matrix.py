"""
Author: Nathan Dale Haskins
Date:   23/10/23


This program is was used to validate the calculation of the reduced stiffness matrix [Qbar] in global coordinates.
Example 2.6 from "Mechanics of Composite Materials" by Autar K. Kaw was used to validate this calculation.

"""


import numpy as np

# Orthotropic Material Properties:
E1    = 181*10**9        
E2    = 10.30*10**9       
v12   = 0.28  
v21   = (v12/E1)*E2
G12   = 7.17*10**9       
theta = 60

# reduced stiffness matrix [Q] in local coordinates:
Q11    = E1/(1-v12*v21)
Q12    = (v12*E2)/(1-v12*v21)
Q22    = E2/(1-v12*v21)
Q66    = G12

Q = np.array([[Q11, Q12, 0], 
              [Q12, Q22, 0], 
              [0  , 0, Q66]])   # correct

print("Q = \n", Q)
print("Q is correct! \n")

# compliance matrix [S]:
S11 = 1/E1
S12 = -v12/E1
S22 = 1/E2
S66 = 1/G12

S = np.array([[S11, S12, 0],
              [S12, S22, 0],
              [0,   0, S66]])   # correct

print("S = \n", S)  
print("S is correct! \n")                              
Q1 = np.linalg.inv(S)            # correct
print("Q_alternative = \n", Q1)
tolerance = 1e-6  
print("Does Q_alternative match Q? ", np.allclose(Q1, Q, atol=tolerance))
if np.allclose(Q1, Q, atol=tolerance) == True:
    print("Q_alternative is correct! \n")
else:
    print("Q_alternative is NOT correct! \n")

# Reuter Matrix [R]:    
R = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 2]])

# Transformation Matrix [T]:
c = np.cos(np.radians(theta))
s = np.sin(np.radians(theta))

T = np.array([[c**2, s**2,    2*c*s],
              [s**2, c**2,   -2*c*s],
              [-s*c, s*c, c**2-s**2]])
# reduced complaince matrix [Sbar] in global coordinates:
Sbar_11 = S11*c**4 + (2*S12 + S66)*s**2*c**2 + S22*s**4                     # correct            
Sbar_12 = S12*(s**4 + c**4) + (S11 + S22 - S66)*s**2*c**2                   # correct
Sbar_22 = S11*s**4 + (2*S12 + S66)*s**2*c**2 + S22*c**4                     # correct
Sbar_16 = (2*S11 - 2*S12 - S66)*s*c**3 - (2*S22 - 2*S12 - S66)*s**3*c       # correct
Sbar_26 = (2*S11 - 2*S12 - S66)*s**3*c - (2*S22 - 2*S12 - S66)*s*c**3       # correct
Sbar_66 = 2*(2*S11 + 2*S22 - 4*S12 - S66)*s**2*c**2 + S66*(s**4 + c**4)     # correct

Sbar = np.array([[Sbar_11, Sbar_12, Sbar_16],
                 [Sbar_12, Sbar_22, Sbar_26],
                 [Sbar_16, Sbar_26, Sbar_66]])

print("Sbar = \n", Sbar)
print("Sbar is correct! \n")
Qbar = np.linalg.inv(Sbar)            
print("Qbar = \n", Qbar)
print("Qbar is correct! \n")

Qbar_matrix = np.linalg.inv(T) @ Q @ R @ T @ np.linalg.inv(R)
print("Qbar_matrix = \n", Qbar_matrix)
print("Does Qbar_matrix match Qbar? ", np.allclose(Qbar_matrix, Qbar, atol=tolerance))
if np.allclose(Qbar_matrix, Qbar, atol=tolerance) == True:
    print("Qbar_matrix is correct! \n")
else:
    print("Qbar_matrix is NOT correct! \n")


# Reduced Stiffness Matrix [Qbar] in global coordinates:
Qbar_11 = Q11*c**4 + Q22*s**4 + 2*(Q12 + 2*Q66)*s**2*c**2                   # correct
Qbar_12 = (Q11 + Q22 - 4*Q66)*s**2*c**2 + Q12*(c**4 + s**2)                 # correct
Qbar_22 = Q11*s**4 + Q22*c**4 + 2*(Q12 + 2*Q66)*s**2*c**2                   # correct
Qbar_16 = (Q11 - Q12 - 2*Q66)*c**3*s - (Q22 - Q12 - 2*Q66)*s**3*c           # correct
Qbar_26 = (Q11 - Q12 - 2*Q66)*c*s**3 - (Q22 - Q12 - 2*Q66)*c**3*s           # correct
Qbar_66 = (Q11 + Q22 - 2*Q12 - 2*Q66)*s**2*c**2 + Q66*(s**4 + c**4)         # correct

Qbar1 = np.array([[Qbar_11, Qbar_12, Qbar_16],
                 [Qbar_12, Qbar_22, Qbar_26],
                 [Qbar_16, Qbar_26, Qbar_66]])

print("Qbar alternative = \n", Qbar1)
print("Does Q_bar_alternative match Qbar? ", np.allclose(Qbar1, Qbar, atol=tolerance))

