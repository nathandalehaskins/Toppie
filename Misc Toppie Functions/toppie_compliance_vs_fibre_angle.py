"""
Author: Nathan Dale Haskins
Date:   23/10/23

This is a modified version of Toppie.py that plots a compliance vs fibre angle graph for a given problem type.
It complements the results Section 5.4 Orthotropic Implementation - Compliance Trend in the final report.

A user can modify the function " run_main_multiple_times ". Specifically the user can specify the range 
of angles that they wish to test the compliance for. For examples, to test the compliance for fibre angles
-90 to 90  in incremenets of 10, the user can modify the line in the function " run_main_multiple_times ":

theta_values = np.arange(-90, 90, 10)

"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import numba
import math
import time
from scipy.ndimage import convolve
from scipy.sparse.linalg import spsolve, splu, cg, gmres, bicgstab, lgmres, qmr, minres
from scipy.sparse import csc_matrix, diags
from cvxopt import matrix, cholmod, spmatrix
from matplotlib.animation import FuncAnimation, FFMpegWriter


plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

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
            return self.q_iso()
        elif self.material_type == "O":
            return self.q_ortho()
        else:
            raise ValueError("Invalid material type. Choose 'I' for Isotropic and 'O' for Orthotropic.")
        
    def q_iso(self):
        """
        Computes Reduced stiffness matrix for an isotropic element in plane-stress

        Parameters:
        E: Young's Modulus
        v: Poisson's Ratio

        """
        return (self.E/ (1 - self.v**2) * np.array([[1, self.v, 0], [self.v, 1, 0], [0, 0, (1 - self.v) / 2]]))

    def q_ortho(self):
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
    # force vector is a column vector where each 2 entries represent the force
    # in the x and y directions, respectively, at a node. So if we want to apply 
    # a force of -1 to the first node in the x direction, we would do F[0] = -1. 
    # If we want to apply a force of -1 to the first node in the y direction,
    # we would do F[1] = -1. Then the next node would be F[2] and F[3] for the
    # x and y directions, respectively, and so on. F looks like:

    # [[F1x], [F1y], [F2x], [F2y], [F3x], [F3y], ...] = F
    # node 1: F1x, F1y = F[0], F[1]
    # node 2: F2x, F2y = F[2], F[3]
    # node 3: F3x, F3y = F[4], F[5]
    # ...
    # if you want to apply a force at a specific node, you would use the formula:
    # index in F=2 * (node number) + direction
    # where:

    # node number -  the number assigned to a particular node in your grid (starting from 0).
    # direction   -  0 for x-direction and 1 for y-direction.

    # Messerschmitt-Bölkow-Blohm beam 

    DOF_MULTIPLIER = 2

    def __init__(self, X, Y, force):

        self.n_dof = 2 * (X + 1) * (Y + 1)                              # number of degrees of freedom
        self.X = X                                                      # number of elements in the x-direction
        self.Y = Y                                                      # number of elements in the y-direction 
        self.n_el  = X * Y       
        self.F = np.zeros((self.n_dof, 1))                              # Initialise the force vector as a column vector of zeros (considering only y direction)   
        self.alldofs = np.arange(self.n_dof)                            # Numpy array containing indices representing all degrees of freedom in the system.
        self.elx, self.ely, self.edofMat = self.initialize_elements()
        self.n_dof = self.DOF_MULTIPLIER * (X + 1) * (Y + 1)
        self.force = force

    def half_MBB(self):

        self.F[1] = -self.force
        left_edge_dofs = np.arange(0, 2*(self.Y + 1), 2)
        bottom_right_node = np.array([self.n_dof - 1])
        fixeddofs = np.union1d(left_edge_dofs, bottom_right_node)
        
        return self.F, fixeddofs, self.compute_freedofs(fixeddofs)


        # should self.Y rather be 0?
        # node_of_interest = 0 + (self.Y + 1)                  # node just to the right of the top left node
        # direction = 1                                        # 0 for x-direction and 1 for y-direction
        # self.F[2*node_of_interest + direction] = -self.force                            
        # left_edge_dofs    = np.arange(0, 2*(self.Y + 1), 2)  # degrees of freedom on the left edge of the domain
        # bottom_right_node = np.array([self.n_dof - 1])
        # fixeddofs = np.union1d(left_edge_dofs, bottom_right_node)
        
        return self.F, fixeddofs, self.compute_freedofs(fixeddofs)

    def full_MBB(self):
        # should self.Y rather be 0? 
        node_of_interest          = self.Y + (self.X//2)*(self.Y + 1) 
        direction                 = 1                                               # 0 for x-direction and 1 for y-direction
        self.F[2*node_of_interest + direction] = -self.force
        bottom_left_node          = np.array([2*self.Y, 2*self.Y+1])
        bottom_right_node         = np.array([self.n_dof - 1])
        fixeddofs                 = np.union1d(bottom_left_node, bottom_right_node)
        
        return self.F, fixeddofs, self.compute_freedofs(fixeddofs)
    
    def cantilever(self):

        node_of_interest = self.Y//2 + self.X * (self.Y + 1)
        direction = 1
        self.F[2*node_of_interest + direction] = -self.force
        left_edge_dofs_x = np.arange(0, 2*(self.Y + 1), 2)   # x-direction degrees of freedom on the left edge
        left_edge_dofs_y = np.arange(1, 2*(self.Y + 1), 2)   # y-direction degrees of freedom on the left edge
        fixeddofs = np.union1d(left_edge_dofs_x, left_edge_dofs_y)

        return self.F, fixeddofs, self.compute_freedofs(fixeddofs)
    
    def cantileverV2(self):

        node_of_interest = 0 + self.X * (self.Y + 1)
        direction = 1
        self.F[2*node_of_interest + direction] = -self.force
        left_edge_dofs_x = np.arange(0, 2*(self.Y + 1), 2)   # x-direction degrees of freedom on the left edge
        left_edge_dofs_y = np.arange(1, 2*(self.Y + 1), 2)   # y-direction degrees of freedom on the left edge
        fixeddofs = np.union1d(left_edge_dofs_x, left_edge_dofs_y)

        return self.F, fixeddofs, self.compute_freedofs(fixeddofs)
    
    def compute_freedofs(self, fixeddofs):
        """
        computes the difference between alldofs and fixeddofs to get the free degrees of freedom
        """
        return np.setdiff1d(self.alldofs, fixeddofs)

    def initialize_elements(self):
        """
        
        Returns:
        tuple: Contains arrays representing the x-indices, y-indices of each element in the grid and the element degrees of freedom matrix.
        """
        elx = np.repeat(np.arange(self.X), self.Y).reshape((self.n_el, 1))    # column vector where x indices are repeated y times 
        ely = np.tile(np.arange(self.Y), self.X).reshape((self.n_el, 1))      # column vector where y indices are repeated x times 
        
        # np.repeat and np.tile are used to construct these arrays (elx and ely) 
        # such that each pair (elx[i], ely[i]) represents the coordinates of the 
        # i-th element in the grid.

        # calculate the node numbers of the four corner nodes of each element:

        n1 = (self.Y+1)*elx + ely                   # bottom-left node of the element
        n2 = (self.Y+1)*(elx + 1) + ely             # bottom-right node of the element
        n3 = (self.Y+1)*(elx + 1) + (ely + 1)       # top-right node of the element
        n4 = (self.Y+1)*elx + (ely + 1)             # top-left node of the element

        # Define the DOF of each node:
        # For each node index n, 2*n and 2*n+1 represent the degrees of freedom 
        # in the x and y directions, respectively;

        # The element degrees of freedom matrix (edofMat) contains the 
        # degrres of freedom associated with each node of each element.
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
        Computes the derivatives of the shape functions with respect to the natural coordinates

        Parameters:
        zeta: natural coordinate in the zeta direction
        eta: natural coordinate in the eta direction

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
        Computes the Jacobian matrix and its determinant

        Parameters:
        dNdzeta: derivatives of the shape functions with respect to the natural coordinate zeta
        dNdeta: derivatives of the shape functions with respect to the natural coordinate eta
        x_coords: x-coordinates of the nodes
        y_coords: y-coordinates of the nodes

        returns:
        J: Jacobian matrix
        detJ: determinant of the Jacobian matrix

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
        Computes the strain-displacement (B) matrix

        Parameters:
        dNdzeta: derivatives of the shape functions with respect to the natural coordinate zeta
        dNdeta: derivatives of the shape functions with respect to the natural coordinate eta
        J: Jacobian matrix

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
        Computes the elemental stiffness matrix

        Returns:
        ke: elemental stiffness matrix
        """
        x_coords = self.NATURAL_COORD_NODES[:, 0]
        y_coords = self.NATURAL_COORD_NODES[:, 1]
        
        ke = np.zeros((8, 8))
        
        for i in range(4):
            zeta, eta = self.GAUSS_POINTS_2D[i]      
            dNdz, dNdeta = self._shape_function_derivatives(zeta, eta)
            J, detJ = self._compute_jacobian(dNdz, dNdeta, x_coords, y_coords)
            B = self._compute_B_matrix(dNdz, dNdeta, J)
            ke += self.GAUSS_WEIGHTS_2D[i] * detJ * self.thickness * (B.T @ self.Q @ B)
        
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
        C: material matrix
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
        x (numpy.ndarray): Density distribution.
        u (numpy.ndarray): Displacement vector.
        ke (numpy.ndarray): Elemental stiffness matrix.
        P (int): Penalisation factor.
        edof : Element degrees of freedom.
        
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


    # inspired from TopOpt
    def filt(self, x, rmin, dc):
        """
        Filtering method - inspired from TopOpt [SOURCE]

        Parameters:
        x (numpy.ndarray): density distribution
        rmin (float): minimum radius
        dc (numpy.ndarray): compliance derivative (sensitivity)

        Returns:
        numpy.ndarray: filtered sensitivity
        """
        rminf = math.floor(rmin)

        size    = rminf*2+1
        kernel  = np.array([[max(0, rmin - np.sqrt((rminf-i)**2 + (rminf-j)**2)) for j in range(size)] for i in range(size)])
        kernel /= kernel.sum()              # kernel is normalised such that its elements sum to 1
        
        xdc     = dc * x                    # sensitivies (dc) are weighed by the densisties, x (element-wise multiplication)
        xdcn    = convolve(xdc, kernel, mode='reflect')
        dcn     = xdcn / x                  # normalised convutioned sensitivities [care for near zero densities]

        return dcn
    
    # investigating plotting of lagrange multiplier history
    @staticmethod
    #@numba.jit(nopython=True, cache=True)
    def OC(X, Y, x, f, dc):
        """
        Optimality criteria

        Parameters:
        X: number of elements in the x-direction
        Y: number of elements in the y-direction
        x: density distribution
        f: volume fraction
        dc: compliance derivative (sensitivity)

        Returns:
        xnew: new density distribution
        """
        l1   = 0
        l2   = 100000
        eta  = 0.5

        while l2 - l1 > 1e-4:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0.001, 
                              np.maximum(x - move, 
                                         np.minimum(1, 
                                                    np.minimum(x + move, 
                                                               x * (-dc / ((f/1)*lmid))**(eta)))))
            if np.sum(xnew) - f * X * Y > 0:
                l1 = lmid
            else:
                l2 = lmid
        return xnew


class FEAnalysis:
    
    ELEMENT_SIZE = 64
    NODES_PER_ELEMENT = 8
    DOF_MULTIPLIER = 2

    def __init__(self, X, Y, E0, elx, ely, edofMat):
        self.X = X
        self.Y = Y
        self.n_el = X * Y
        self.n_dof = self.DOF_MULTIPLIER * (X + 1) * (Y + 1)
        self.elx, self.ely, self.edofMat = elx, ely, edofMat
        self.E0 = E0
    
    @staticmethod
    def csc_to_cvxopt(K_free):

        data = K_free.data.tolist()
        row  = K_free.indices.tolist()
        col  = np.repeat(np.arange(K_free.shape[1]), np.diff(K_free.indptr)).tolist()

        return spmatrix(data, row, col, size=K_free.shape)
    
    def compute_displacements(self, x, P, ke, F, fixeddofs, freedofs, E0, solver_type):
        
        K = self._assemble_stiffness_matrix(x, P, ke, E0)  
        K_free = K[np.ix_(freedofs, freedofs)]             
        F_free = F[freedofs]                               
        U = np.zeros((self.n_dof, ))                       

        # Create a Jacobi preconditioner (Diagonal of K_free)
        M_inv = diags(1 / K_free.diagonal())


        if solver_type == "spsolve":
            U_free = spsolve(K_free, F_free)
        elif solver_type == "cg":
            U_free, _ = cg(K_free, F_free, M=M_inv)
        elif solver_type == "gmres":
            U_free, _ = gmres(K_free, F_free)
        elif solver_type == "bicgstab":
            U_free, _ = bicgstab(K_free, F_free)
        elif solver_type == "lgmres":
            U_free, _ = lgmres(K_free, F_free)
        elif solver_type == "qmr":
            U_free, _ = qmr(K_free, F_free)
        elif solver_type == "minres":
            U_free, _ = minres(K_free, F_free)
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

        U[freedofs] = U_free                               
        U[fixeddofs] = 0                                   

        return U.reshape(-1, 1)

    def _assemble_stiffness_matrix(self, x, P, ke, E0):
        Emin = 10e-6

        ke_flat = ke.flatten()  
        xe = (Emin + x[self.ely, self.elx] ** P * (E0 - Emin))

        edofMat_repeated = np.repeat(self.edofMat[:, :, np.newaxis], self.NODES_PER_ELEMENT, axis=2)
        edofMat_tiled = np.tile(self.edofMat[:, np.newaxis, :], (1, self.NODES_PER_ELEMENT, 1))

        assembled_rows = edofMat_repeated.reshape(self.n_el, -1).flatten()
        assembled_cols = edofMat_tiled.reshape(self.n_el, -1).flatten()
        assembled_data = (xe[:, np.newaxis] * ke_flat[np.newaxis, :]).reshape(self.n_el, -1).flatten()

        K = csc_matrix((assembled_data, (assembled_rows, assembled_cols)), shape=(self.n_dof, self.n_dof))
        
        return K


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


    current_iteration = 0


    while error > 0.01 and current_iteration < max_iterations:


        current_iteration += 1  
        loop += 1
        xold  = x                                   
        
        U     = inst_FEAnalysis.compute_displacements(x, P, ke, F, fixeddofs, freedofs, E0, solver_type)
        C, dc = inst_optimisation.comp(x, U, ke, P, edofMat)  
        dc    = inst_optimisation.filt(x, rmin, dc)
        x     = inst_optimisation.OC(X, Y, x, f, dc)
        error = np.max(np.abs(x - xold))   
 

    plt.show()

    return C

def run_main_multiple_times(X, Y, f, P, rmin, material_type, E, v, E1, E2, v12, G12, theta, problem_type, move, solver_type, force, max_iterations):
    theta_values = np.arange(-90, 90, 10)  # From -90 to 90 degrees with 5-degree increments
    compliance_values = {}
    
    for theta in theta_values:
        print(f"Running for theta = {theta} degrees")
        
        C = main(X, Y, f, P, rmin, material_type, E, v, E1, E2, v12, G12, theta, problem_type, move, solver_type, force, max_iterations)
        
        compliance_values[theta] = C
        
    return compliance_values


if __name__ == '__main__':

    # Isotropic Material Properties:
    E  = 1        
    v  = 0.36            

    # Orthotropic Material Properties:
    E1    = 36      
    E2    = 8.27     
    v12   = 0.26                      
    G12   = 4.14       
    theta = 0 
    
    X    = 60
    Y    = 20
    f    = 0.5
    P    = 3
    rmin = 2.5 
    move = 0.2

    material_type = "O"
    problem_type  = "f"  

    force = 50

    max_iterations = 1000
    
    solver_type = "cvxopt"

    compliance_values = run_main_multiple_times(X, Y, f, P, rmin, material_type, E, v, E1, E2, v12, G12, theta, problem_type, move, solver_type, force, max_iterations)
    
    thetas = list(compliance_values.keys())
    compliances = list(compliance_values.values())
    
    plt.plot(thetas, compliances)
    plt.xlabel('Fiber Angle (degrees)')
    plt.ylabel('Compliance (C)')
    plt.title('Compliance vs Fiber Angle')
    plt.grid(True)
    plt.show()
