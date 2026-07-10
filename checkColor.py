import trimesh
import numpy as np

path = "data/OakInk-v2/object_preview/align_ds/O02@0020@00001/scan.ply"
mesh = trimesh.load(path, process=False, force="mesh")

colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32)
if colors.max() <= 1.0:
    colors *= 255.0

red = (
    (colors[:, 0] > 150)
    & (colors[:, 1] < 100)
    & (colors[:, 2] < 100)
)
black = (
    (colors[:, 0] < 80)
    & (colors[:, 1] < 80)
    & (colors[:, 2] < 80)
)

print("path:", path)
print("vertices:", len(mesh.vertices))
print("unique colors:", np.unique(colors, axis=0))
print("red_valid:", bool(red.any()))
print("black_valid:", bool(black.any()))
print("red_count:", int(red.sum()))
print("black_count:", int(black.sum()))
