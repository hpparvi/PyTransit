#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2021  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

# This script can generate spheres, rounded cubes, and capsules.
# For more information, see https://prideout.net/blog/octasphere/
# Copyright (c) 2019 Philip Rideout
# Distributed under the MIT License, see bottom of file.

from math import sin, cos, acos, pi
from numpy import empty, array, vstack, cross, dot
from pyrr import quaternion


def octasphere(ndivisions: int):
    """Creates a unit sphere using octagon subdivision.

    Creates a unit sphere using octagon subdivision. Modified slightly from the original code
    by Philip Rideout (https://prideout.net/blog/octasphere).
    """

    n = 2**ndivisions + 1
    num_verts = n * (n + 1) // 2
    verts = empty((num_verts, 3))
    j = 0
    for i in range(n):
        theta = pi * 0.5 * i / (n - 1)
        point_a = [0, sin(theta), cos(theta)]
        point_b = [cos(theta), sin(theta), 0]
        num_segments = n - 1 - i
        j = compute_geodesic(verts, j, point_a, point_b, num_segments)
    assert len(verts) == num_verts

    num_faces = (n - 2) * (n - 1) + n - 1
    faces = empty((num_faces, 3), dtype='int')
    f, j0 = 0, 0
    for col_index in range(n-1):
        col_height = n - 1 - col_index
        j1 = j0 + 1
        j2 = j0 + col_height + 1
        j3 = j0 + col_height + 2
        for row in range(col_height - 1):
            faces[f + 0] = [j0 + row, j1 + row, j2 + row]
            faces[f + 1] = [j2 + row, j1 + row, j3 + row]
            f = f + 2
        row = col_height - 1
        faces[f] = [j0 + row, j1 + row, j2 + row]
        f = f + 1
        j0 = j2

    euler_angles = array([
        [0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0],
        [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 0, 3],
    ]) * pi * 0.5
    quats = (quaternion.create_from_eulers(e) for e in euler_angles)

    offset, combined_verts, combined_faces = 0, [], []
    for quat in quats:
        rotated_verts = [quaternion.apply_to_vector(quat, v) for v in verts]
        rotated_faces = faces + offset
        combined_verts.append(rotated_verts)
        combined_faces.append(rotated_faces)
        offset = offset + len(verts)

    return vstack(combined_verts), vstack(combined_faces)


def compute_geodesic(dst, index, point_a, point_b, num_segments):
    """Given two points on a unit sphere, returns a sequence of surface
    points that lie between them along a geodesic curve."""

    angle_between_endpoints = acos(dot(point_a, point_b))
    rotation_axis = cross(point_a, point_b)
    dst[index] = point_a
    index = index + 1
    if num_segments == 0:
        return index
    dtheta = angle_between_endpoints / num_segments
    for point_index in range(1, num_segments):
        theta = point_index * dtheta
        q = quaternion.create_from_axis_rotation(rotation_axis, theta)
        dst[index] = quaternion.apply_to_vector(q, point_a)
        index = index + 1
    dst[index] = point_b
    return index + 1
