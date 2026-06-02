# encoding:utf8
import argparse
from cadquery.occ_impl.importers import importStep 
import trimesh


def convert_stp_to_obj(stp_path, obj_path, tolerance=0.1, unit='meter'):
    """
    Convert STP/STEP file to OBJ format
    :param stp_path: Path to input .stp or .step file
    :param obj_path: Path to output .obj file
    :param tolerance: Mesh tolerance (smaller value = finer mesh, smoother curves)
    :param unit: Output unit, 'meter' or 'millimeter' (default: 'meter')
    """
    print(f"Reading STP file: {stp_path} ...")
    wp = importStep(stp_path)
    shape = wp.val()
    if shape is None:
        raise ValueError("无法从该 STP 文件中提取到有效的几何实体。")
    print("Generating mesh data and exporting to OBJ...")
    cq_vertices, triangles = shape.tessellate(tolerance=tolerance, angularTolerance=0.1)
    
    # STP files typically use millimeter units
    # Convert to meters if specified (divide by 1000)
    scale_factor = 1.0
    if unit.lower() == 'meter':
        scale_factor = 0.001
        print("Converting units from millimeter to meter...")
    
    vertices = [(v.x * scale_factor, v.y * scale_factor, v.z * scale_factor) for v in cq_vertices]
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh.merge_vertices() 
    mesh.export(obj_path, file_type='obj')
    print(f"Conversion successful! File saved to: {obj_path} (unit: {unit})")

def main():
    parser = argparse.ArgumentParser(description='Convert STP/STEP CAD files to OBJ mesh format')
    parser.add_argument('--input', type=str, required=True, help='Input STP/STEP file path')
    parser.add_argument('--output', type=str, required=True, help='Output OBJ file path')
    parser.add_argument('--tolerance', type=float, default=0.1, 
                        help='Mesh tolerance (smaller = finer mesh, default: 0.1)')
    parser.add_argument('--unit', type=str, default='meter', choices=['meter', 'millimeter'],
                        help='Output unit (default: meter)')
    args = parser.parse_args()
    
    convert_stp_to_obj(args.input, args.output, args.tolerance, args.unit)


if __name__ == '__main__':
    main()
