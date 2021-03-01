#!/usr/bin/env python3

import pyrender
import trimesh
from trimesh import transformations as T
import numpy as np
import imageio
import json
from tqdm import tqdm

#random = np.random.RandomState(0)
random = np.random

COLORS = {
    "red": [1.0, 0.0, 0.0],
    "yellow": [1.0, 1.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "cyan": [0.0, 1.0, 1.0],
    "blue": [0.0, 0.0, 1.0],
    "magenta": [1.0, 0.0, 1.0],
}

def make_pose(x, y, z):
    return T.translation_matrix([x, y, z])

def make_sphere():
    return trimesh.creation.uv_sphere(radius=0.06)

def make_cube():
    return trimesh.creation.box(extents=(0.13, 0.13, 0.13))

def make_cylinder():
    return trimesh.creation.cylinder(radius=0.06, height=0.13)

def make_cone():
    return trimesh.creation.cone(radius=0.06, height=0.13)

SHAPES = {
    "sphere": make_sphere,
    "cube": make_cube,
    "cylinder": make_cylinder,
    "cone": make_cone,
}

def sample_object():
    color = random.choice(list(COLORS.keys()))
    shape = random.choice(list(SHAPES.keys()))

    obj = SHAPES[shape]()
    obj.visual.vertex_colors = COLORS[color]
    mesh = pyrender.Mesh.from_trimesh(obj)

    return f"{color} {shape}", mesh

def sample_coords(cx=0, cy=0, cz=0):
    return (
        cx + np.random.random() * 0.3 - 0.15,
        cy + np.random.random() * 0.3 - 0.15,
        cz
    )

def sample_scene():
    scene = pyrender.Scene()
    obj1_desc, obj1 = sample_object()
    #p1x, p1y, p1z = sample_coords()
    p1x, p1y, p1z = 0, -.15, 0
    scene.add(obj1, pose=make_pose(p1x, p1y, p1z))
    desc = obj1_desc

    if random.random() < 0.5:
    #if True:
        obj2_desc, obj2 = sample_object()
        p2x, p2y, p2z = 0, .15, 0
        #scene.add(obj2, pose=make_pose(*sample_coords(-p1x, -p1y, p1z)))
        scene.add(obj2, pose=make_pose(p2x, p2y, p2z))
        desc += " , " + obj2_desc

    camera = pyrender.PerspectiveCamera(yfov=np.pi/3, aspectRatio=1.0)
    s = np.sqrt(2)/2
    camera_pose = np.array([
       [0.0, -s,   s,   0.3],
       [1.0,  0.0, 0.0, 0.0],
       [0.0,  s,   s,   0.35],
       [0.0,  0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)

    light1 = pyrender.SpotLight(
            color=np.ones(3),
            intensity=0.3,
            innerConeAngle=np.pi/16.0,
            outerConeAngle=np.pi/6.0
    )
    scene.add(light1, pose=camera_pose)

    light2_pose = make_pose(0.1, -0.4, 0.2)
    light2 = pyrender.PointLight(
            color=np.ones(3),
            intensity=0.2,
    )
    scene.add(light2, pose=light2_pose)

    light3_pose = make_pose(0.0, -0.3, 0.3)
    light3 = pyrender.PointLight(
            color=np.ones(3),
            intensity=0.2,
    )
    scene.add(light3, pose=light3_pose)


    renderer = pyrender.OffscreenRenderer(64, 64)
    color, _ = renderer.render(scene)

    return desc, color

def main():
    data = []
    splits = {"train":[],"test":[]}
    for i in tqdm(range(2000)):
        obj_desc, img = sample_scene()
        img_path = f"images/{i}.png"
        imageio.imsave(img_path, img)
        data.append({
            "image": img_path,
            "description": obj_desc
        })
        if "red cube" in obj_desc:
            splits["test"].append(i)
        else:
            splits["train"].append(i)

    with open("data.json", "w") as writer:
        json.dump(data, writer)

    with open("splits.json", "w") as writer:
        json.dump(splits, writer)


if __name__ == "__main__":
    main()
