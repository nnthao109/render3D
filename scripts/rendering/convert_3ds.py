import bpy
import sys
import argparse
# Replace 'input_file.blend' with the path to your .blend file
parser = argparse.ArgumentParser()
parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the object file",
    )
parser.add_argument(
    "--output_path",
        type=str,
        required=True,
        help="Path to the object file",
)

# Replace 'output_file.obj' with the path where you want to save the .obj file
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)
# output_file = 'output_file.obj'
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Import .3DS file
bpy.ops.import_scene.autodesk_3ds(filepath=args.object_path)

# Export to .OBJ format
bpy.ops.export_scene.obj(filepath=args.output_path, use_selection=True)
# Load the .blend file
# bpy.ops.wm.open_mainfile(filepath=args.object_path)

# # Select all objects in the scene
# bpy.ops.object.select_all(action='SELECT')

# # Export selected objects as .obj file
# bpy.ops.export_scene.obj(filepath=args.output_path, use_selection=True)

# Print a message indicating successful export
print("Exported to", args.output_path)
# blender-3.2.2-linux-x64/blender --background  --python convert_3ds.py -- --object_path './3dsfile.3DS' --output_path './3dsfile.obj'