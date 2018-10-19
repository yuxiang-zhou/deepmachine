# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import time
import numpy as np
import tensorflow as tf

import camera_utils
import mesh_renderer

import menpo3d.io as m3io
import menpo.io as mio

from menpo.image import Image


def main():

    test_data_directory = (
        './test_data_face/render')

    # load obj
    face_mesh = m3io.import_mesh(
        './test_data_face/mesh.obj')

    texture_index = (face_mesh.tcoords.points[:, ::-1] *
                     face_mesh.texture.shape).astype(np.int32)

    vertex_color = face_mesh.texture.pixels[
        :,
        1 - texture_index[:, 0], texture_index[:, 1]].T

    tf.reset_default_graph()
    # Set up a basic cube centered at the origin, with vertex normals pointing
    # outwards along the line from the origin to the cube vertices:
    face_vertices = tf.constant(
        face_mesh.points,
        dtype=tf.float32)
    face_normals = tf.nn.l2_normalize(face_vertices, dim=1)
    face_triangles = tf.constant(
        face_mesh.trilist,
        dtype=tf.int32)

    # testRendersSimpleCube:
    """Renders a simple cube to test the full forward pass.

    Verifies the functionality of both the custom kernel and the python wrapper.
    """

    n_randering = 16

    model_transforms = camera_utils.euler_matrices(
        tf.random_uniform([n_randering, 3]) * np.pi / 2 - np.pi / 4.
    )[:, :3, :3]

    vertices_world_space = tf.matmul(
        tf.stack([face_vertices for _ in range(n_randering)]),
        model_transforms,
        transpose_b=True)

    normals_world_space = tf.matmul(
        tf.stack([face_normals for _ in range(n_randering)]),
        model_transforms,
        transpose_b=True)

    # camera position:
    eye = tf.constant(n_randering * [[0.0, 0.0, 6.0]], dtype=tf.float32)
    center = tf.constant(n_randering * [[0.0, 0.0, 0.0]], dtype=tf.float32)
    world_up = tf.constant(n_randering * [[0.0, 1.0, 0.0]], dtype=tf.float32)
    ambient_colors = tf.constant(
        n_randering * [[0.2, 0.2, 0.2]], dtype=tf.float32)
    image_width = 256
    image_height = 256
    light_positions = tf.constant(
        n_randering * [[[6.0, 6.0, 6.0], [-6.0, -6.0, 6.0]]])
    light_intensities = tf.ones([n_randering, 1, 3], dtype=tf.float32)
    vertex_diffuse_colors = tf.constant(
        np.stack([vertex_color for _ in range(n_randering)]), dtype=tf.float32)

    rendered = mesh_renderer.mesh_renderer(
        vertices_world_space,
        triangles=face_triangles,
        normals=normals_world_space,
        diffuse_colors=vertex_diffuse_colors,
        camera_position=eye,
        camera_lookat=center,
        camera_up=world_up,
        light_positions=light_positions,
        light_intensities=light_intensities,
        image_width=image_width,
        image_height=image_height,
        ambient_color=ambient_colors
    )

    image_id = 0
    with tf.Session() as sess:
        fps_list = []
        while(image_id < 100):
            start_time = time.time()
            images = sess.run(rendered, feed_dict={})
            for image in images:
                target_image_name = 'Gray_face_%i.png' % image_id
                image_id += 1
                baseline_image_path = os.path.join(test_data_directory,
                                                   target_image_name)

                mio.export_image(Image.init_from_channels_at_back(
                    image[..., :3].clip(0, 1)), baseline_image_path, overwrite=True)

            end_time = time.time()
            fps = n_randering / (end_time - start_time)
            fps_list.append(fps)
            if len(fps_list) > 5:
                fps_list.pop(0)
            print(np.mean(fps_list))


if __name__ == '__main__':
    main()
