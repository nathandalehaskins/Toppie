"""
Author: Nathan Dale Haskins
Date:   23/10/23


This program is used to generate the edofMat matrix, plotted to the terminal.
The goal is to showcase the unique numbering for each DOF in a FEA problem.
This complements Section 4.4.1 Element Degrees of Freedom in the main report.

"""

import numpy as np

X = 4           # number of elements in the x-direction
Y = 3           # number of elements in the y-direction

n_el = X * Y    # number of elements

elx = np.repeat(np.arange(X), Y).reshape((n_el, 1))    # column vector where x indices are repeated y times 
ely = np.tile(np.arange(Y), X).reshape((n_el, 1))      # column vector where y indices are repeated x times 


n1 = (Y+1)*elx + ely                   # bottom-left node of the element
n2 = (Y+1)*(elx + 1) + ely             # bottom-right node of the element
n3 = (Y+1)*(elx + 1) + (ely + 1)       # top-right node of the element
n4 = (Y+1)*elx + (ely + 1)             # top-left node of the element

edofMat = np.hstack([2*n1+0, 2*n1+1, 
                     2*n2+0, 2*n2+1, 
                     2*n3+0, 2*n3+1, 
                     2*n4+0, 2*n4+1])

edofMat = np.hstack([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1])

print("edofMat: ")
print(edofMat)
print("")
print("Notice that some entries have the same number, indicating a level of coupling between the DOFs.")