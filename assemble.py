#!/usr/bin/env python3
"""
Convert ShapeR DPRecon GLB outputs to Replica-style OBJ names.

Example:
  python assemble.py /path/to/output_dir \
      --instance_id_json ~/Downloads/replica-scan1/instance_id.json

Input GLB name:
  dprecon__sofa_4.glb

instance_id.json rank:
  sofa_4 has the 10th-smallest instance id.

Output OBJ name:
  obj_10.obj

Combined GLB:
  assembled.glb
"""

import argparse
import json
import re
from pathlib import Path


DPRECON_GLB_RE = re.compile(r"^dprecon__(?P<object_name>.+)\.glb$")


def expand_path(path):
    return Path(path).expanduser().resolve()


def parse_object_name(glb_path):
    match = DPRECON_GLB_RE.match(glb_path.name)
    if match is None:
        raise ValueError(f"Expected GLB name like dprecon__<object_name>.glb: {glb_path}")

    return match.group("object_name")


def load_instance_ids(instance_id_json):
    with instance_id_json.open("r", encoding="utf-8") as f:
        instance_ids = json.load(f)

    if not isinstance(instance_ids, dict):
        raise ValueError(f"Expected JSON object in {instance_id_json}")

    loaded = {}
    for name, instance_id in instance_ids.items():
        try:
            loaded[str(name)] = int(instance_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Instance id for {name!r} must be an integer, got {instance_id!r}"
            ) from exc

    return loaded


def build_instance_rank_map(instance_ids):
    sorted_items = sorted(instance_ids.items(), key=lambda item: item[1])
    sorted_ids = [instance_id for _, instance_id in sorted_items]
    if len(sorted_ids) != len(set(sorted_ids)):
        raise ValueError("instance_id.json contains duplicate instance IDs; rank is ambiguous")

    return {
        object_name: rank
        for rank, (object_name, _) in enumerate(sorted_items, start=1)
    }


def require_trimesh():
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "assemble.py requires trimesh to convert GLB files. "
            "Install it in the active environment with: pip install trimesh"
        ) from exc

    return trimesh


def load_mesh(glb_path, trimesh_module):
    mesh = trimesh_module.load(str(glb_path), force="mesh", process=False)
    if mesh is None or mesh.is_empty:
        raise ValueError(f"Loaded empty mesh from {glb_path}")
    return mesh


def resolve_output_path(path, base_dir):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def convert_glbs(
    output_dir,
    instance_ids,
    obj_output_dir,
    combined_glb_path,
    overwrite=False,
    skip_missing=False,
):
    glb_paths = sorted(output_dir.glob("dprecon__*.glb"))
    if not glb_paths:
        raise FileNotFoundError(f"No dprecon__*.glb files found in {output_dir}")

    obj_output_dir.mkdir(parents=True, exist_ok=True)
    combined_glb_path.parent.mkdir(parents=True, exist_ok=True)
    if combined_glb_path.exists() and not overwrite:
        raise FileExistsError(
            f"{combined_glb_path} already exists; pass --overwrite to replace it"
        )

    trimesh_module = require_trimesh()
    scene = trimesh_module.Scene()
    rank_by_object_name = build_instance_rank_map(instance_ids)

    converted = []
    missing_names = []
    for glb_path in glb_paths:
        object_name = parse_object_name(glb_path)
        obj_id = instance_ids.get(object_name)
        rank = rank_by_object_name.get(object_name)
        if obj_id is None or rank is None:
            if skip_missing:
                missing_names.append(object_name)
                continue
            raise KeyError(
                f"{object_name!r} from {glb_path.name} is missing in instance_id.json"
            )

        obj_path = obj_output_dir / f"obj_{rank}.obj"
        if obj_path.exists() and not overwrite:
            raise FileExistsError(f"{obj_path} already exists; pass --overwrite to replace it")

        mesh = load_mesh(glb_path, trimesh_module)
        mesh.export(str(obj_path))
        scene.add_geometry(mesh, geom_name=f"obj_{rank}", node_name=f"obj_{rank}")
        converted.append(
            {
                "glb_path": glb_path,
                "obj_path": obj_path,
                "object_name": object_name,
                "obj_id": obj_id,
                "rank": rank,
            }
        )

    if len(scene.geometry) == 0:
        raise ValueError("No meshes were converted; combined GLB would be empty")

    scene.export(str(combined_glb_path))
    return converted, missing_names


def main():
    parser = argparse.ArgumentParser(
        description="Convert dprecon__<name>.glb files to obj_<rank>.obj and assemble them into a GLB."
    )
    parser.add_argument(
        "output_dir",
        help="Directory containing dprecon__*.glb files from ShapeR inference.",
    )
    parser.add_argument(
        "--instance_id_json",
        "--instance-id-json",
        required=True,
        dest="instance_id_json",
        help="Path to DPRecon instance_id.json mapping object names to instance IDs.",
    )
    parser.add_argument(
        "--obj_output_dir",
        "--obj-output-dir",
        default=None,
        dest="obj_output_dir",
        help="Directory for exported OBJ files. Defaults to output_dir.",
    )
    parser.add_argument(
        "--combined_glb",
        "--combined-glb",
        default="assembled.glb",
        dest="combined_glb",
        help="Output path/name for the combined GLB. Relative paths are resolved under obj_output_dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing OBJ files.",
    )
    parser.add_argument(
        "--skip_missing",
        "--skip-missing",
        action="store_true",
        dest="skip_missing",
        help="Skip GLBs whose object names are not present in instance_id.json.",
    )
    args = parser.parse_args()

    output_dir = expand_path(args.output_dir)
    instance_id_json = expand_path(args.instance_id_json)
    obj_output_dir = expand_path(args.obj_output_dir) if args.obj_output_dir else output_dir
    combined_glb_path = resolve_output_path(args.combined_glb, obj_output_dir)

    if not output_dir.is_dir():
        raise NotADirectoryError(f"output_dir does not exist or is not a directory: {output_dir}")
    if not instance_id_json.is_file():
        raise FileNotFoundError(f"instance_id_json does not exist: {instance_id_json}")

    instance_ids = load_instance_ids(instance_id_json)
    converted, missing_names = convert_glbs(
        output_dir=output_dir,
        instance_ids=instance_ids,
        obj_output_dir=obj_output_dir,
        combined_glb_path=combined_glb_path,
        overwrite=args.overwrite,
        skip_missing=args.skip_missing,
    )

    for item in converted:
        print(
            f"{item['glb_path'].name} "
            f"(obj_id={item['obj_id']}, rank={item['rank']}) -> {item['obj_path'].name}"
        )
    if missing_names:
        print("Skipped missing instance_id entries: " + ", ".join(missing_names))
    print(f"Converted {len(converted)} GLB file(s) into {obj_output_dir}")
    print(f"Assembled GLB: {combined_glb_path}")


if __name__ == "__main__":
    main()
