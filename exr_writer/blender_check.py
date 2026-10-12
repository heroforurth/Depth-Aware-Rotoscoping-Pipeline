"""Check that our EXR opens in Blender with a usable depth channel (design decision 6).

Runs INSIDE Blender, no GUI:
    blender.exe -b --factory-startup --python exr_writer/blender_check.py -- <in.exr> <out.exr> <width> <height>

It loads <in.exr> in Blender's compositor, connects the image node's Depth
output to the Composite node and renders, which writes <out.exr>: Blender's
own reading of our depth. `python -m exr_writer.verify_blender` runs this and
compares <out.exr> with the Z channel we wrote.
"""
import json
import sys

import bpy

src, dst, W, H = sys.argv[sys.argv.index("--") + 1:][:4]
scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = int(W), int(H)
scene.render.resolution_percentage = 100
if scene.camera is None:                       # rendering needs a camera, even for compositing only
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam

scene.use_nodes = True
tree = scene.node_tree
tree.nodes.clear()
node_img = tree.nodes.new("CompositorNodeImage")
node_img.image = bpy.data.images.load(src)
outputs = [s.name for s in node_img.outputs]
depth = "Depth" if "Depth" in outputs else ("Z" if "Z" in outputs else None)
result = {"image_node_outputs": outputs, "depth_socket": depth}

if depth:
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(node_img.outputs[depth], comp.inputs["Image"])
    scene.view_settings.view_transform = "Raw"           # no colour management on the depth values
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = dst
    bpy.ops.render.render(write_still=True)
    result["written"] = dst
print("BLENDER_CHECK " + json.dumps(result))
