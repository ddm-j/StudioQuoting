from scipy.spatial import ConvexHull
import numpy as np
from rtree import index
import os, sys
import pyclipper
import math
import logging
from stl import mesh
from contextlib import contextmanager
import warnings
warnings.filterwarnings("ignore", message= ".*mesh is not closed.*")

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class STLUtils(object):

    def __init__(self):
        # Studio System V1 Materials
        self.materials = {
            '316L': {
                'scale': 1.16,
                'cost': 0.6,
                'density':5.01 # g/cc
            },
            '17-4PH': {
                'scale': 1.19,
                'cost': 0.6, # $/cc
                'density': 5.01 # g/cc
            },
            '4140': {
                'scale': 1.19,
                'cost': 0.6,
                'density': 4.92  # g/cc
            },
            'H13': {
                'scale': 1.15,
                'cost': 0.6,
                'density': 5.01  # g/cc
            }
        }

        # Matrix of profile densities (reverse engineered from Fabricate interface)
        self.densities = {
            'model': 0.88665,
            'support': 0.728,
            'raft': 2.455
        }

        # Equipment Sizes
        self.equipment = {
            "printer":{
                "size":[289,189,195],
                "padding":14,
                "level_size":0,
                "n_levels":1,
                "cost":21.66
            },
            "debinder":{
                "size":[292.1,187.96,220.98],
                "padding": 5,
                "level_size":73.66,
                "n_levels":3,
                "cost": 58.90
            },
            "furnace":{
                "size":[321.564,219.202,127],
                "padding": 10,
                "level_size":25.4,
                "n_levels":5,
                "cost": 153.81
            }
        }

        self.pco = pyclipper.PyclipperOffset()

    def get_proj_area(self, obj, i_facet):
        facet = obj.vectors[i_facet]
        v0 = facet[2] - facet[0]
        v1 = facet[1] - facet[0]
        return np.abs(v0[1] * v1[0] - v0[0] * v1[1])

    def is_XY_pt_in_XY_projection(self, P, obj, i_facet, **kw):
        facet = obj.vectors[i_facet]
        A = facet[0]
        B = facet[1]
        C = facet[2]
        v0 = (C - A)[0:2]
        v1 = (B - A)[0:2]
        v2 = (P - A)[0:2]

        dot00 = kw['dot00'][i_facet]
        dot11 = kw['dot11'][i_facet]
        dot01 = kw['dot01'][i_facet]
        dot02 = v0.dot(v2)
        dot12 = v1.dot(v2)
        invDenom = kw['invDenom'][i_facet]

        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        if u > -1.0E-6 and v > -1.0E6 and u + v <= 1.000001:
            height = A[2] + u * (C[2] - A[2]) + v * (B[2] - A[2])

            return True, height
        else:
            return False, 0.0

    def get_facet_XY_bounding_box(self, obj, i_facet):
        return (obj.vectors[i_facet, :, 0].min(), obj.vectors[i_facet, :, 1].min(),
                obj.vectors[i_facet, :, 0].max(), obj.vectors[i_facet, :, 1].max())

    def get_support_height(self, **kw):

        support_bottom = kw['minZ']
        r_tree = kw['rtree']
        pt = kw['point']

        # print('top: {:.2f}'.format(pt[2]))
        for possible_support_facet_i in r_tree.intersection(tuple(pt[0:2])):
            is_in, height = self.is_XY_pt_in_XY_projection(pt, kw['obj'], possible_support_facet_i, dot00=kw['dot00'],
                                                           dot11=kw['dot11'], dot01=kw['dot01'],
                                                           invDenom=kw['invDenom'])
            if is_in:
                pass
                # print('possible support facet {} has a support height {:.8f}'.format(possible_support_facet_i, height))
            if is_in and height > support_bottom and height < pt[2] - 1.0E-4:
                support_bottom = height
                # print('adjusting support bottom to {:.8f}'.format(support_bottom))

        return pt[2] - support_bottom

    # determine the volume of support material necessary for a given object:
    def get_amount_support(self, obj):
        # critical angle; facets more overhung than this angle will require support;
        # an angle of 0 means _all_ faces with an outward normal that points at all downward
        # would require support:
        CRITICAL_ANGLE_DEG = 40.0
        CRITICAL_ANGLE_RAD = CRITICAL_ANGLE_DEG * np.pi / 180.0

        # create an R-tree of the XY-plane projections of the upward-facing facets;
        # these facets represent potential support facets:
        minZ = obj.vectors[:, :, 2].min()
        N_facets = obj.normals.shape[0]

        As = obj.vectors[:, 0, 0:2]
        Bs = obj.vectors[:, 1, 0:2]
        Cs = obj.vectors[:, 2, 0:2]
        v0s = Cs - As
        v1s = Bs - As
        dot00s = (v0s * v0s).sum(axis=1)
        dot11s = (v1s * v1s).sum(axis=1)
        dot01s = (v0s * v1s).sum(axis=1)
        denoms = dot00s * dot11s - dot01s * dot01s

        a = obj.vectors[:, 0]
        b = obj.vectors[:, 1]
        c = obj.vectors[:, 2]
        normals = np.cross(b - a, c - a)
        normals = normals / (np.sqrt((obj.normals ** 2).sum(axis=1)) + 1.0E-9)[:, np.newaxis]

        up_facets = np.argwhere(np.logical_and((normals[:, 2] > 0.0), (np.abs(denoms) > 1.0E-8)))
        denoms = np.where(np.abs(denoms) > 1.0E-8, denoms, 1.0)
        invDenoms = 1.0 / denoms

        up_index = index.Index()
        for i_facet in up_facets:
            bbox = self.get_facet_XY_bounding_box(obj, i_facet)
            up_index.insert(i_facet, bbox)

        crit_z = -np.sin(CRITICAL_ANGLE_RAD)
        need_support_facets = np.argwhere(normals[:, 2] < crit_z).flatten()
        # print('facets needing support:', need_support_facets.shape[0])
        # for i_facet in need_support_facets:
        #    print('facet {}: {}, normal: {}'.format(i_facet, obj.vectors[i_facet], normals[i_facet]))

        # for each facet needing support, determine the distance down to support:
        support_vol = 0.0

        supports = []
        for i_facet in need_support_facets:
            pts = obj.vectors[i_facet]
            # get the support height at each corner of the facet:
            support_heights = []
            for ipt, pt in enumerate(pts):
                # print('getting support height for facet {}, pt {} = {}'.format(i_facet, ipt, pt))
                height = self.get_support_height(obj=obj, point=pt, rtree=up_index, minZ=minZ, dot00=dot00s,
                                                 dot11=dot11s, dot01=dot01s, invDenom=invDenoms)
                support_heights.append(height)

            if sum(support_heights) > 1.0E-4:
                supports.append([i_facet, support_heights])
            addl_support_vol = self.get_proj_area(obj, i_facet) * sum(support_heights) / 3.0
            # print('facet:', pts, 'support vol:', addl_support_vol)
            # the volume of support needed to suspend this facet is the XY-plane-projected area
            # of the facet times the average needed support height:
            support_vol += addl_support_vol

        return support_vol, supports

    def get_rotated_obj(self, obj, v):
        v_norm = v / np.sqrt(v.dot(v))

        # get the cross-product of the z-axis and v:
        z = np.array([0, 0, 1.0])
        if abs(v[2]) == 1.0:
            # v = z or v = -z
            c_norm = np.array([1.0, 0, 0])
        else:
            c = np.cross(z, v_norm)
            c_norm = c / np.sqrt(c.dot(c))
        rot_ang = np.arccos(v_norm.dot(z))
        new_mesh = mesh.Mesh(obj.data.copy())
        new_mesh.rotate(c_norm, rot_ang)

        return new_mesh

    def PolyArea2D(self, pts):
        lines = np.hstack([pts, np.roll(pts, -1, axis=0)])
        area = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in lines))
        return area

    def supportVolume(self, obj):

        # Rotation vectors for minimizing support volume
        vecs = {
            "x": [1.0, 0, 0],
            "-x": [-1.0, 0, 0],
            "y": [0, 1.0, 0],
            "-y": [0, -1.0, 0],
            "z": [0, 0, 1.0],
            "-z": [0, 0, -1.0],
        }

        # Calculate Support Volumes for Different Orientations
        sup_vols = {i: self.get_amount_support(self.get_rotated_obj(obj, np.array(vecs[i])))[0] for i in vecs.keys()}

        # Get optimium orientation
        minimum = min(sup_vols, key=sup_vols.get)
        axis = minimum.split('-')[-1]

        supVol = sup_vols[minimum]

        # Get Raft Area
        raftA, bV = self.raftArea(obj, axis)

        return supVol, raftA, bV

    def meshVolume(self, obj):

        logging.getLogger("stl").setLevel(logging.ERROR)
        volume, cog, inertia = obj.get_mass_properties()
        logging.getLogger("stl").setLevel(logging.INFO)

        return volume

    def raftArea(self, obj, axis):

        axes = ['x', 'y', 'z']
        axes_ind = [0, 1, 2]

        axes_ind.pop(axes.index(axis))

        # Calculate bounding box
        x = max(obj.x.flatten()) - min(obj.x.flatten())
        y = max(obj.y.flatten()) - min(obj.y.flatten())
        z = max(obj.z.flatten()) - min(obj.z.flatten())

        # Re-orient bounding box so that maximum bound is in X direction
        dx = max([x, y, z][i] for i in axes_ind)
        dy = min([x, y, z][i] for i in axes_ind)
        dz = [x, y, z][axes.index(axis)]

        # Reduce point cloud to 2D projection on best support axis
        points = np.array(list(zip(obj.x.flatten(), obj.y.flatten(), obj.z.flatten())))
        points = points[:, axes_ind]

        # Generate Convex Hull around 2D projection (raft outline)
        hull = ConvexHull(points)

        hullVertices = tuple(zip(points[hull.vertices, 0], points[hull.vertices, 1]))

        # Offset the convex hull with PyClipper & Round Corners
        self.pco.AddPath(hullVertices, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        solution = self.pco.Execute(3)
        self.pco.Clear()

        hullVertices = list(zip(*hullVertices))
        offset = list(zip(*solution[0]))

        # Get Area of Offset Outline
        area = self.PolyArea2D(np.array(list(zip(offset[0], offset[1]))))

        # plt.plot(hullVertices[0],hullVertices[1],'r')
        # plt.plot(offset[0], offset[1])
        # plt.show()

        return area, [dx, dy, dz]

    def calculateVolume(self, path, material):

        # Load Object & Suppress ANNOYING FUCKING MESSAGES from mesh class
        with suppress_stdout():
            obj = mesh.Mesh.from_file(path)

        # Rescale according to material properties
        obj.points = obj.points * (self.materials[material]['scale'])

        # Calculate Mesh Volume
        vol = self.meshVolume(obj)

        # Calculate Support Volume, Raft size, and Bounding Box (will calculate optimum orientation)
        supVol, raftA, bV = self.supportVolume(obj)

        # Calculate total volume of material used in printing:
        totalVol = self.densities['model'] * vol + self.densities['support'] * supVol + self.densities['raft'] * raftA

        return totalVol, bV

    def pack2d(self,plate,part,padding):

        nx = round(plate[0]/(part[0]+padding))
        ny = round(plate[1]/(part[1]+padding))

        n = nx*ny

        return n

    def calculateQuantities(self, boundingVolume, vol, material):

        # Calculate Maximum n_parts in sinter (3kg mass limit)
        max_n_furnace = math.floor(3000/(vol/1000)*self.materials[material]['density'])

        # Calculate the maximum number of parts per print, debind, and sinter (2d Plate)
        quantities = {}
        for appliance, stats in self.equipment.items():

            # 2D Pack
            n = self.pack2d(stats['size'],boundingVolume,stats['padding'])

            # Check how many levels we can use
            if stats['level_size'] > boundingVolume[2] or appliance=="printer":
                # Part is smaller than level height, we can use all of the available levels
                n = n*stats['n_levels']
            else:
                # Part is larger than level height, calculate how many levels to use
                n_used = math.floor(boundingVolume[2]/stats['level_size'])
                n_stack = math.floor(stats['n_levels']/n_used)
                n = n*n_stack

            # Check furnace weight limit
            if appliance=="furnace":
                if n > max_n_furnace:
                    print('Furnace weight limit exceed with maximum pack of {0} units. Reducing to 3kg with {1} units.'.format(n,max_n_furnace))
                    n = max_n_furnace


            quantities.update({appliance:n})

        return quantities

    def calculateCycles(self,qtys):

        n_prints = math.ceil(qtys['furnace']/qtys['printer'])
        n_debinds = math.ceil(qtys['furnace']/qtys['debinder'])
        n_sinter = 1

        cycles = {'printer':n_prints,
                'debinder':n_debinds,
                'furnace':n_sinter}

        costs = {'printer':n_prints*self.equipment['printer']['cost'],
                'debinder':n_debinds*self.equipment['debinder']['cost'],
                'furnace':n_sinter*self.equipment['furnace']['cost']}

        return cycles, costs



if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Define model to calculate volume ej: python measure_volume.py torus.stl")
    else:
        mySTLUtils = STLUtils()
        if (len(sys.argv) > 2 and sys.argv[2] == "inch"):
            mySTLUtils.calculateVolume(sys.argv[1], "inch")
        else:
            mySTLUtils.calculateVolume(sys.argv[1], "cm")
