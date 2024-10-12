"""
Author: Nathan Dale Haskins
Date:   23/10/23

Here Toppie is modified to showcase the changing langrange multiplier (lmid) and 
the volume constraint (volume_constraint) during the optimisation process. This
section complements 4.6.3 OC Method in the main report.

"""


import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import math
from scipy.ndimage import convolve
from scipy.sparse.linalg import spsolve, splu, cg, gmres, bicgstab, lgmres, qmr
from scipy.sparse import csc_matrix, diags
from cvxopt import matrix, cholmod, spmatrix


class Material:

    def __init__(self, material_type=None, E=None, v=None, E1=None, E2=None, v12=None, G12=None, theta=None):
        
        self.E = E
        self.v = v
        self.E1 = E1
        self.E2 = E2
        self.v12 = v12
        self.G12 = G12
        self.theta = theta
        self.material_type = material_type

    def compute_material_matrix(self): 
        """
        Compute the material matrix based on the given type
        """
        if self.material_type == "I":
            return self.Q_iso()
        elif self.material_type == "O":
            return self.Q_ortho()
        else:
            raise ValueError("Invalid material type. Choose 'I' for Isotropic and 'O' for Orthotropic.")
        
    def Q_iso(self):
        """
        Computes Reduced stiffness matrix for an isotropic element in plane-stress

        Parameters:
        E: Young's Modulus
        v: Poisson's Ratio

        """
        return (self.E/ (1 - self.v**2) * np.array([[1, self.v, 0], [self.v, 1, 0], [0, 0, (1 - self.v) / 2]]))

    def Q_ortho(self):
        """
        Computes Reduced stiffness matrix for an Orthotropic element in plane-stress

        Parameters:

        E1: Young's Modulus in fibre-direction
        E2: Young's Modulus in transverse direction
        v12: Poisson's Ratio
        G12: Shear Modulus
        theta: fibre angle (degrees) relative to the global x-axis

        T: Transformation matrix
        Q: Reduced stiffness matrix in the material coordinate system
        R: Reuter matrix which accounts for the difference between the engineering and tensorial shear strain

        Returns:
        - np.ndarray: Reduced stiffness matrix for an Orthotropic element in plane-stress 
        
        """
        v21 = self.v12 * self.E2 / self.E1
        theta_radians = -np.radians(self.theta)
        c = np.cos(theta_radians)
        s = np.sin(theta_radians)
        x = 1 - self.v12 * v21
        T = np.array([[c**2, s**2, 2*s*c],
                      [s**2, c**2, -2*s*c],
                      [-s*c, s*c, c**2 - s**2]])
        Q = np.array([[self.E1/x, v21*self.E1/x, 0],
                      [self.v12*self.E2/x, self.E2/x, 0],
                      [0, 0, self.G12]])
        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 2]])

        return np.linalg.inv(T) @ Q @ R @ T @ np.linalg.inv(R)

class Problem:
    
    # F = [[F1x], [F1y], [F2x], [F2y], [F3x], [F3y], ...] 
    # node 1: F1x, F1y = F[0], F[1]
    # node 2: F2x, F2y = F[2], F[3]
    # node 3: F3x, F3y = F[4], F[5]
    # ...
    # if you want to apply a force at a specific node, you would use the formula:
    # index in F = 2 * (node number) + direction
    # where:
    # node number -  the number assigned to a particular node in your grid (starting from 0) also known as the nodal index
    # direction   -  0 for x-direction and 1 for y-direction.

    DOF_MULTIPLIER = 2

    def __init__(self, X, Y, force):

        self.n_dof = 2 * (X + 1) * (Y + 1)                              # number of degrees of freedom
        self.X = X                                                      # number of elements in the x-direction
        self.Y = Y                                                      # number of elements in the y-direction 
        self.n_el  = X * Y                                              # number of elements in the grid
        self.F = np.zeros((self.n_dof, 1))                              # Initialise the force vector as a column vector of zeros (considering only y direction)   
        self.alldofs = np.arange(self.n_dof)                            # Numpy array containing indices representing all degrees of freedom in the system.
        self.elx, self.ely, self.edofMat = self.initialize_elements()
        self.n_dof = self.DOF_MULTIPLIER * (X + 1) * (Y + 1)            # number of degrees of freedom
        self.force = force      

    def half_MBB(self):

        self.F[1]           = -self.force
        left_edge_dofs      = np.arange(0, 2*(self.Y + 1), 2)
        bottom_right_node   = np.array([self.n_dof - 1])
        fixeddofs           = np.union1d(left_edge_dofs, bottom_right_node)
        
        return self.F, fixeddofs, self.compute_freedofs(fixeddofs)


    def full_MBB(self):
        
        node_of_interest          = 0 + (self.X//2)*(self.Y + 1) 
        direction                 = 1                                               # 0 for x-direction and 1 for y-direction
        self.F[2*node_of_interest + direction] = -self.force
        bottom_left_node          = np.array([2*self.Y, 2*self.Y+1])
        bottom_right_node         = np.array([self.n_dof - 1])
        fixeddofs                 = np.union1d(bottom_left_node, bottom_right_node)
        
        return self.F, fixeddofs, self.compute_freedofs(fixeddofs)
    
    def cantilever(self):

        node_of_interest          = self.Y//2 + self.X * (self.Y + 1)
        direction                 = 1
        self.F[2*node_of_interest + direction] = -self.force
        left_edge_dofs_x          = np.arange(0, 2*(self.Y + 1), 2)              # x-direction degrees of freedom on the left edge
        left_edge_dofs_y          = np.arange(1, 2*(self.Y + 1), 2)              # y-direction degrees of freedom on the left edge
        fixeddofs                 = np.union1d(left_edge_dofs_x, left_edge_dofs_y)

        return self.F, fixeddofs, self.compute_freedofs(fixeddofs)
    
    def cantileverV2(self):

        node_of_interest          = 0 + self.X * (self.Y + 1)
        direction                 = 1
        self.F[2*node_of_interest + direction] = -self.force
        left_edge_dofs_x          = np.arange(0, 2*(self.Y + 1), 2)              # x-direction degrees of freedom on the left edge
        left_edge_dofs_y          = np.arange(1, 2*(self.Y + 1), 2)              # y-direction degrees of freedom on the left edge
        fixeddofs                 = np.union1d(left_edge_dofs_x, left_edge_dofs_y)

        return self.F, fixeddofs, self.compute_freedofs(fixeddofs)
    
    def compute_freedofs(self, fixeddofs):
        
        """
        computes the difference between alldofs and fixeddofs to get the free degrees of freedom
        """
        return np.setdiff1d(self.alldofs, fixeddofs)

    def initialize_elements(self):
       
        """
        Computes the element degrees of freedom matrix (edofMat) and the x and y indices of each element in the grid.
        
        Returns:
        tuple: Contains arrays representing the x-indices, y-indices of each element in the grid and the element degrees of freedom matrix.
        """

        elx = np.repeat(np.arange(self.X), self.Y).reshape((self.n_el, 1))    # column vector where x indices are repeated y times 
        ely = np.tile(np.arange(self.Y), self.X).reshape((self.n_el, 1))      # column vector where y indices are repeated x times 
        
        # np.repeat and np.tile are used to construct these arrays (elx and ely) 
        # such that each pair (elx[i], ely[i]) represents the coordinates of the 
        # i-th element in the grid. Similar functionality to the meshgrid function. 

        # calculate the node numbers of the four corner nodes of each element:

        n1 = (self.Y+1)*elx + ely                   # bottom-left node of the element
        n2 = (self.Y+1)*(elx + 1) + ely             # bottom-right node of the element
        n3 = (self.Y+1)*(elx + 1) + (ely + 1)       # top-right node of the element
        n4 = (self.Y+1)*elx + (ely + 1)             # top-left node of the element

        # Define the DOF of each node (n):

        # ------> 2*n    = degrees of freedom in x direction
        # ------> 2*n+1  = degrees of freedom in y direction

        # The element degrees of freedom matrix (edofMat) contains the 
        # degrees of freedom associated with each node of each element.
        
        # The size of edofMat is:
        # number of elements * number of nodes per element * number of degrees of freedom per node.
        
        edofMat = np.hstack([2*n1+0, 2*n1+1, 
                             2*n2+0, 2*n2+1, 
                             2*n3+0, 2*n3+1, 
                             2*n4+0, 2*n4+1])

        Problem.edofMat = np.hstack([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1])

        return elx, ely, edofMat
    
class Element:

    GAUSS_POINTS_2D = np.array([(-0.5773502691896257, -0.5773502691896257),
                                (0.5773502691896257,  -0.5773502691896257),
                                (0.5773502691896257,   0.5773502691896257),
                                (-0.5773502691896257,  0.5773502691896257)])
    
    GAUSS_WEIGHTS_2D    = np.ones(4)
    NATURAL_COORD_NODES = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])

    def __init__(self, material_matrix, thickness=1.0):
        
        self.Q = material_matrix
        self.thickness = thickness
        self.KE = self.elemental_stiffness_matrix()
    
    def _shape_function_derivatives(self, zeta, eta):
       
        """
        Private function:

        Computes the derivatives of the shape functions with respect to the natural coordinates

        Parameters:
        zeta:       natural coordinate in the zeta direction
        eta:        natural coordinate in the eta direction

        Returns:
        
        Derivatives of the shape functions with respect to the natural coordinates
        """

        dN1_dzeta = -0.25 * (1 - eta)
        dN2_dzeta =  0.25 * (1 - eta)
        dN3_dzeta =  0.25 * (1 + eta)
        dN4_dzeta = -0.25 * (1 + eta)

        dN1_deta  = -0.25 * (1 - zeta)
        dN2_deta  = -0.25 * (1 + zeta)
        dN3_deta  =  0.25 * (1 + zeta)
        dN4_deta  =  0.25 * (1 - zeta)

        return (dN1_dzeta, dN2_dzeta, dN3_dzeta, dN4_dzeta), (dN1_deta, dN2_deta, dN3_deta, dN4_deta)

    def _compute_jacobian(self, dNdzeta, dNdeta, x_coords, y_coords):
        
        """
        Private function:

        Computes the Jacobian matrix and its determinant

        Parameters:
        dNdzeta:    derivatives of the shape functions with respect to the natural coordinate zeta
        dNdeta:     derivatives of the shape functions with respect to the natural coordinate eta
        x_coords:   x-coordinates of the nodes
        y_coords:   y-coordinates of the nodes

        returns:
        J:          Jacobian matrix
        detJ:       determinant of the Jacobian matrix

        """
        J11  = sum(np.array(dNdzeta) * x_coords)
        J12  = sum(np.array(dNdzeta) * y_coords)
        J21  = sum(np.array(dNdeta)  * x_coords)
        J22  = sum(np.array(dNdeta)  * y_coords)
        J    = np.array([[J11, J12], [J21, J22]])
        detJ = sc.linalg.det(J)

        return J, detJ

    def _compute_B_matrix(self, dNdzeta, dNdeta, J):
        
        """
        Private function:
        
        Computes the strain-displacement (B) matrix

        Parameters:
        dNdzeta:    derivatives of the shape functions with respect to the natural coordinate zeta
        dNdeta:     derivatives of the shape functions with respect to the natural coordinate eta
        J:          Jacobian matrix

        returns:
        B: strain-displacement matrix

        """
        invJ = sc.linalg.inv(J)
        dNdx = np.dot(invJ[0], [dNdzeta, dNdeta])
        dNdy = np.dot(invJ[1], [dNdzeta, dNdeta])
        B    = np.zeros((3,8))

        B[0, 0::2] = dNdx
        B[1, 1::2] = dNdy
        B[2, 0::2] = dNdy
        B[2, 1::2] = dNdx

        return B

    def elemental_stiffness_matrix(self):
        
        """
        Computes the elemental stiffness matrix using Gaussing Quadrature with helper functions 
        _shape_function_derivatives, _compute_jacobian and _compute_B_matrix.

        Returns:

        ke: elemental stiffness matrix
        """

        x_coords = self.NATURAL_COORD_NODES[:, 0]           # fetch x-coordinates of the nodes from the NATURAL_COORD_NODES array
        y_coords = self.NATURAL_COORD_NODES[:, 1]           # fetch y-coordinates of the nodes from the NATURAL_COORD_NODES array
        
        ke = np.zeros((8, 8))                               # initialise the elemental stiffness matrix as a 8x8 matrix of zeros
        
        for i in range(4):
            
            zeta, eta    = self.GAUSS_POINTS_2D[i]      
            dNdz, dNdeta = self._shape_function_derivatives(zeta, eta)
            J, detJ      = self._compute_jacobian(dNdz, dNdeta, x_coords, y_coords)
            B            = self._compute_B_matrix(dNdz, dNdeta, J)
            ke          += self.GAUSS_WEIGHTS_2D[i] * detJ * self.thickness * (B.T @ self.Q @ B)    
        
        return ke

class Optimisation:
    
    def __init__(self, X, Y, move):
        self.X = X    
        self.T = Y    
        self.move = move

    def initialisation(self, X, Y, f: float, C):
       
        """
        Initialisation of the optimisation problem

        Parameters:
        X: number of elements in the x-direction
        Y: number of elements in the y-direction
        f: volume fraction
        C: reduced stiffness matrix in global domain
        
        Returns:
        tuple: Containing density distribution, loop counter, error, penalisation factor, 
        compliance derivative and elemental stiffness matrix.
        """

        x     = np.ones((Y, X)) * f                                 
        loop  = 0                                                   
        error = 1.0                                                 
        c     = 0.0                                                 
        dc    = np.zeros((Y, X))                               
        inst_element = Element(material_matrix=C, thickness=1.0)
        ke    = inst_element.elemental_stiffness_matrix()

        return x, loop, error, c, dc, ke
       
    @staticmethod
    def comp(x, u, ke, P: int, edof):

        """
        Computes the compliance and its derivative (sensitivity).

        Parameters:
        x (numpy.ndarray):      Density distribution.
        u (numpy.ndarray):      Displacement vector.
        ke (numpy.ndarray):     Elemental stiffness matrix.
        P (int):                Penalisation factor.
        edof :                  Element degrees of freedom.
        
        Returns:
        tuple: Containing compliance and its derivative.
        """

        x  = x.T.flatten()                                         # density array transposed and flattened to 1-d array
        u  = u[edof]                                               # extract the displacement variables corresponding to each element DOF                                
        
        # the following line computes the dot product of the elemental stiffness matrix ke and 
        # the element displacement array ue. 
        # ue array after reshaping has shape (X*Y, 8, 1), representing X*Y elements, each with an 8x1 column vector.
        # It reshapes the element displacement array ue into 
        # a 3D array of shape (X*Y, 8, 1) where X*Y is the number of elements in the grid and 8 
        # is the number of nodes per element (2 DOFs per node). It then multiplies this array by 
        # the elemental stiffness matrix ke.
        
        f  = np.dot(ke, u.reshape((X*Y, 8, 1)))                   # compute the element forces for each element  
        # unsqueeze removes the single-dimensional entry from the shape of the array, which was needed
        # for the previous calculation. Ue and fe are now the shape (X*Y, 8).
        C_e  = np.sum(u.squeeze() * f.squeeze().T, axis=1)         # compute the element compliance for each element
        dc   = -P * (x ** (P-1)) * C_e
        
        C    = np.sum(C_e * x.T**P)                                 # total compliance of the structure                                                     
        dc   = dc.reshape((X, Y)).T                                             

        return C, dc

    def filt(self, x, rmin, dc):
        
        """
        Filters sensitivies to mitigate checkerboarding effect. [TopOpt] 

        Parameters:
        x (numpy.ndarray):  density distribution
        rmin (float):       minimum radius
        dc (numpy.ndarray): compliance derivative (sensitivity)

        Returns:
        numpy.ndarray: filtered sensitivities
        """

        rminf   = math.floor(rmin)

        size    = rminf*2+1
        kernel  = np.array([[max(0, rmin - np.sqrt((rminf-i)**2 + (rminf-j)**2)) for j in range(size)] for i in range(size)])
        kernel /= kernel.sum()                              # kernel is normalised such that its elements sum to 1
        
        xdc     = dc * x                                    # sensitivies (dc) are weighed by the densisties, x (element-wise multiplication)
        xdcn    = convolve(xdc, kernel, mode='reflect')     # convolve the sensitivities with the kernel
        dcn     = xdcn / x                                  # normalised convutioned sensitivities [care for near zero densities]

        return dcn
    
    @staticmethod
    def OC(X, Y, x, f, dc):
        
        """
        Optimality criteria

        Parameters:
        X:      number of elements in the x-direction
        Y:      number of elements in the y-direction
        x:      density distribution
        f:      volume fraction
        dc:     compliance derivative (sensitivity)

        Returns:
        xnew: new density distribution
        """

        l1    = 0
        l2    = 100000
        eta   = 0.5
        x_min = 0.001 # ensures that minimum density is not 0, which would causse numerical instabilities.

        while l2 - l1 > 1e-4:
            
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(x_min, 
                              np.maximum(x - move, 
                                         np.minimum(1, 
                                                    np.minimum(x + move, 
                                                               x * (-dc / ((f/1)*lmid))**(eta)))))
            # check if the volume constraint is satisfied
            if np.sum(xnew) - f * X * Y > 0:
                l1 = lmid
            else:
                l2 = lmid

        return xnew, lmid

class FEAnalysis:
    
    ELEMENT_SIZE        = 64
    NODES_PER_ELEMENT   = 8
    DOF_MULTIPLIER      = 2

    def __init__(self, X, Y, E0, elx, ely, edofMat):
        
        self.X = X
        self.Y = Y
        self.n_el = X * Y
        self.n_dof = self.DOF_MULTIPLIER * (X + 1) * (Y + 1)
        self.elx, self.ely, self.edofMat = elx, ely, edofMat
        self.E0 = E0
    
    @staticmethod
    def csc_to_cvxopt(K_free):
        """
        Convert CSC matrix to CVXOPT sparse matrix
        """
        data = K_free.data.tolist()
        row  = K_free.indices.tolist()
        col  = np.repeat(np.arange(K_free.shape[1]), np.diff(K_free.indptr)).tolist()

        return spmatrix(data, row, col, size=K_free.shape)
    
    def compute_displacements(self, x, P, ke, F, fixeddofs, freedofs, E0, solver_type):
        
        """
        Computes the nodal displacesments. 

        parameters:
        x:              density distribution
        P:              penalisation factor
        ke:             elemental stiffness matrix
        F:              force vector
        fixeddofs:      fixed degrees of freedom
        freedofs:       free degrees of freedom
        E0:             Young's modulus
        solver_type:    solver type

        returns:

        U:              nodal displacements
        
        """
        K = self._assemble_stiffness_matrix(x, P, ke, E0)  
        K_free = K[np.ix_(freedofs, freedofs)]             
        F_free = F[freedofs]                               
        U = np.zeros((self.n_dof, ))                       

        if solver_type == "spsolve":
            U_free = spsolve(K_free, F_free)
        elif solver_type == "cg":
            M_inv = diags(1 / K_free.diagonal())    
            U_free, _ = cg(K_free, F_free, M=M_inv)
        elif solver_type == "gmres":
            U_free, _ = gmres(K_free, F_free)
        elif solver_type == "bicgstab":
            U_free, _ = bicgstab(K_free, F_free)
        elif solver_type == "lgmres":
            U_free, _ = lgmres(K_free, F_free)
        elif solver_type == "qmr":
            U_free, _ = qmr(K_free, F_free)
        elif solver_type == "SuperLU":
            lu = splu(K_free.tocsc())
            U_free = lu.solve(F_free)
            U_free = U_free.flatten()  
        elif solver_type == "cvxopt":
            K_free_cvx = self.csc_to_cvxopt(K_free)
            F_free_cvx = matrix(F[freedofs])
            cholmod.linsolve(K_free_cvx, F_free_cvx)
            U_free = np.array(F_free_cvx).flatten()
        else:
            raise ValueError("Invalid solver type")

        U[freedofs]     = U_free                               
        U[fixeddofs]    = 0                                   

        return U.reshape(-1, 1)

    def _assemble_stiffness_matrix(self, x, P, ke, E0):
        
        """
        Assembles the global stiffness matrix.

        Parameters:
        x:      density distribution
        P:      penalisation factor
        ke:     elemental stiffness matrix
        E0:     Young's modulus

        returns:
        K:      global stiffness matrix
    
        """
        
        Emin = 10e-6

        ke_flat          = ke.flatten()  
        xe               = (Emin + x[self.ely, self.elx] ** P * (E0 - Emin))        

        edofMat_repeated = np.repeat(self.edofMat[:, :, np.newaxis], self.NODES_PER_ELEMENT, axis=2)
        edofMat_tiled    = np.tile(self.edofMat[:, np.newaxis, :], (1, self.NODES_PER_ELEMENT, 1))

        assembled_rows   = edofMat_repeated.reshape(self.n_el, -1).flatten()
        assembled_cols   = edofMat_tiled.reshape(self.n_el, -1).flatten()
        assembled_data   = (xe[:, np.newaxis] * ke_flat[np.newaxis, :]).reshape(self.n_el, -1).flatten()

        K = csc_matrix((assembled_data, (assembled_rows, assembled_cols)), shape=(self.n_dof, self.n_dof))
        
        return K
    
class Visualization:
    
    def __init__(self):
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 5))

        # Initialize empty lists for data storage
        self.lmid_values = []
        self.volume_constraints = []
        self.iterations = []

        # Bisection Method Visualization
        self.bisection_plot, = self.axs[0].plot([], [], label='lmid Convergence')
        self.axs[0].set_title('Bisection Method Visualization')
        self.axs[0].set_xlabel('Iterations')
        self.axs[0].set_ylabel('lmid Value')

        # Constraint Satisfaction Plot
        self.constraint_plot, = self.axs[1].plot([], [], label='Volume Constraint')
        self.axs[1].set_title('Constraint Satisfaction Plot')
        self.axs[1].set_xlabel('Iterations')
        self.axs[1].set_ylabel('Volume')

    def display_results(self, loop, x, X, Y, error, dc, lmid, volume_constraint):
        # Append new data
        self.lmid_values.append(lmid)
        self.volume_constraints.append(volume_constraint)
        self.iterations.append(loop)

        # Update Bisection Method Visualization
        self.bisection_plot.set_xdata(self.iterations)
        self.bisection_plot.set_ydata(self.lmid_values)
        self.axs[0].relim()
        self.axs[0].autoscale_view()

        # Update Constraint Satisfaction Plot
        self.constraint_plot.set_xdata(self.iterations)
        self.constraint_plot.set_ydata(self.volume_constraints)
        self.axs[1].relim()
        self.axs[1].autoscale_view()

        # Refresh the plot
        plt.pause(0.0001)



# Initialize the sensitivity map plot

def main(X, Y, f, P, rmin, material_type, E, v, E1, E2, v12, G12, theta, problem_type, move, solver_type, force, max_iterations):
    
    if material_type == "I":
        E0  = E
    elif material_type == "O":
        E0  = E2

    # Instantiate classes
    inst_problem       = Problem(X, Y, force)
    elx, ely, edofMat  = inst_problem.initialize_elements()
    inst_material      = Material(material_type, E, v, E1, E2, v12, G12, theta)
    inst_optimisation  = Optimisation(X, Y, move)
    inst_FEAnalysis    = FEAnalysis(X, Y, E0, elx, ely, edofMat)
    inst_visualization = Visualization()
    
    # Problem initialisation
    if problem_type == "h":
        F, fixeddofs, freedofs = inst_problem.half_MBB()
    elif problem_type == "f":
        F, fixeddofs, freedofs = inst_problem.full_MBB()
    elif problem_type == "c":
        F, fixeddofs, freedofs = inst_problem.cantilever()
    elif problem_type == "c2":
        F, fixeddofs, freedofs = inst_problem.cantileverV2()
    else:
        raise ValueError("Unknown problem type: " + problem_type)
    
    
    Q = inst_material.compute_material_matrix()
    x, loop, error, C, dc, ke, = inst_optimisation.initialisation(X, Y, f, Q)

    x_values = []
    current_iteration = 0

    while error > 0.01 and current_iteration < max_iterations:

        current_iteration += 1   
        loop += 1
        xold  = x                                   
        
        U       = inst_FEAnalysis.compute_displacements(x, P, ke, F, fixeddofs, freedofs, E0, solver_type)
        C, dc   = inst_optimisation.comp(x, U, ke, P, edofMat)  
        dc      = inst_optimisation.filt(x, rmin, dc)
        x, lmid = inst_optimisation.OC(X, Y, x, f, dc)
        error   = np.max(np.abs(x - xold))   

        print(f"Iteration: {current_iteration}, lmid: {lmid}")
        volume = np.sum(x)
        volume_constraint = volume / (X * Y) 
        # print(f"Iteration: {current_iteration}, Constraint Satisfaction: {volume_constraint}")
        inst_visualization.display_results(loop, x, X, Y, error, dc, lmid, volume_constraint)

    plt.show()


if __name__ == '__main__':
    
    material_type = "I"
    problem_type  = "h"  

    # Isotropic Material Properties:
    E  = 1          # GPa   
    v  = 0.36            

    # Orthotropic Material Properties (Glass/epoxy) 

    E1    = 36      # GPa     
    E2    = 8.27    # GPa    
    v12   = 0.26                      
    G12   = 4.14    # GPa    
    theta = -30     # degrees

    X    = 60       # number of elements in the x-direction
    Y    = 20       # number of elements in the y-direction
    f    = 0.5      # volume fraction
    P    = 3        # penalisation factor
    rmin = 2.5      # filter radius 
    move = 0.2      # move limit


    force = 50      # N

    max_iterations = 200 
    
    # cg
    # spsolve
    # gmres --- way too slow!
    # bicgstab
    # cvxopt
    # lgmres, 
    # qmr
    # SuperLU
    # UMFPACK

    solver_type = "cvxopt"

    main(X, Y, f, P, rmin, material_type, E, v, E1, E2, v12, G12, theta, problem_type, move, solver_type, force, max_iterations)

