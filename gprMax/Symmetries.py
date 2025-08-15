import numpy as np
from gprMax.grid import FDTDGrid

class Symmetry():

    def __init__(self, G: FDTDGrid, direction, phase):
        
        # We have to be careful: a symmetry for E is an antisymmetry for H !
        self.phase = 1

        self.direction = direction

        # We can only have one symmetry per direction: already checked in input_cmds_multiuse !
        if self.direction == 'x':
            G.new_dimensions[0] = G.nx//2
        elif self.direction == 'y':
            G.new_dimensions[1] = G.ny//2
            self.new_ny = np.arange(0, G.ny//2, 1)
        else:
            G.new_dimensions[2] = G.nz//2
            self.new_nz = np.arange(0, G.nz//2, 1)
        
    def position_rxs(self, G: FDTDGrid):
        # To easily compute the fields on rxs outside the computation zone, we save the parameters
        for rx in G.rxs:
            if self.direction == 'x':
                x_coord = rx.xcoord
                if x_coord > G.new_dimensions[0]:
                    rx.outside_planes.append(('x', self.phase))
            elif self.direction == 'y':
                y_coord = rx.ycoord
                if y_coord > G.new_dimensions[1]:
                    rx.outside_planes.append(('y', self.phase))
            elif self.direction == 'z':
                z_coord = rx.zcoord
                if z_coord > G.new_dimensions[2]:
                    rx.outside_planes.append(('z', self.phase))

    def calculation_fluxes(self, G:FDTDGrid):
        # Same as rxs, but we have to be careful that flux surfaces may not respect the symmetry
        for flux in G.fluxes:
            x_begin = flux.bottom_left_corner[0]
            y_begin = flux.bottom_left_corner[1]
            z_begin = flux.bottom_left_corner[2]

            nx = flux.cells_number[0]
            ny = flux.cells_number[1]
            nz = flux.cells_number[2]

            # Check what part of the flux is inside the computing space
             
