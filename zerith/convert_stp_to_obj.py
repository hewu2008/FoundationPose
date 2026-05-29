# encoding:utf8
import argparse
import cadquery as cq
import trimesh


def convert_stp_to_obj(stp_path, obj_path, tolerance=0.1):
    """
    Convert STP/STEP file to OBJ format
    :param stp_path: Path to input .stp or .step file
    :param obj_path: Path to output .obj file
    :param tolerance: Mesh tolerance (smaller value = finer mesh, smoother curves)
    """
    print(f"Reading STP file: {stp_path} ...")
    # 1. Import STP model
    shape = cq.importer.importStep(stp_path)
    
    # 2. Discretize high-precision CAD surfaces (tessellation)
    # tolerance controls triangle mesh roughness, typically 0.1mm or smaller
    vertices, triangles = shape.toMesh(tolerance=tolerance)
    
    # 3. Build polygon mesh using trimesh and export as OBJ
    print("Generating mesh data and exporting to OBJ...")
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh.export(obj_path, file_type='obj')
    
    print(f"Conversion successful! File saved to: {obj_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert STP/STEP CAD files to OBJ mesh format')
    parser.add_argument('--input', type=str, required=True, help='Input STP/STEP file path')
    parser.add_argument('--output', type=str, required=True, help='Output OBJ file path')
    parser.add_argument('--tolerance', type=float, default=0.1, 
                        help='Mesh tolerance (smaller = finer mesh, default: 0.1)')
    args = parser.parse_args()
    
    convert_stp_to_obj(args.input, args.output, args.tolerance)


if __name__ == '__main__':
    main()
