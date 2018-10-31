import numpy as np

from menpo.io import *
from menpo3d.io import *
from menpo.shape import ColouredTriMesh

def export_coloured_mesh(cmesh, fp):
    points = cmesh.points
    colours = cmesh.colours
    trilist = cmesh.trilist
    with open(fp, 'w') as f:
        for p, c in zip(points, colours):
            f.write('v {} {}\n'.format(' '.join(map(str, p)), ' '.join(map(str, c))))
        f.write('\n')
        for t in trilist + 1:
            f.write('f {}\n'.format(' '.join(map(str, t))))


def import_coloured_mesh(fp):
    points = []
    colours = []
    trilist = []
    
    with open(fp, 'r') as f:
        for l in f.readlines():
            if l.startswith('v '):
                pts = list(map(float, l.strip().split(' ')[1:7]))
                points.append(pts[:3])
                colours.append(pts[3:])

            elif l.startswith('f '):
                faces = [int(f.split('/')[0]) - 1 for f in l.strip().split(' ')[1:4]]
                trilist.append(faces)


    return ColouredTriMesh(
        np.array(points),
        trilist=np.array(trilist),
        colours=np.array(colours)
    )