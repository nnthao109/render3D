import glob
import json
import multiprocessing
import os
import platform
import random
import subprocess
import tempfile
import time
import zipfile
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import fire
# from app.fire import fire
import fsspec
import GPUtil
import pandas as pd
from loguru import logger

import objaverse.xl as oxl
from objaverse.utils import get_uid_from_str
import random
import string

def generate_random_string(length):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))
                   
def log_processed_object(csv_filename: str, *args) -> None:
    """Log when an object is done being used.

    Args:
        csv_filename (str): Name of the CSV file to save the logs to.
        *args: Arguments to save to the CSV file.

    Returns:
        None
    """
    args = ",".join([str(arg) for arg in args])
    # log that this object was rendered successfully
    # saving locally to avoid excessive writes to the cloud
    dirname = os.path.expanduser(f"./.objaverse2/logs/")
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, csv_filename), "a", encoding="utf-8") as f:
        f.write(f"{time.time()},{args}\n")

def zipdir(path: str, ziph: zipfile.ZipFile) -> None:
    """Zip up a directory with an arcname structure.

    Args:
        path (str): Path to the directory to zip.
        ziph (zipfile.ZipFile): ZipFile handler object to write to.

    Returns:
        None
    """
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            # this ensures the structure inside the zip starts at folder/
            arcname = os.path.join(os.path.basename(root), file)
            ziph.write(os.path.join(root, file), arcname=arcname)

def render_3d(
    local_path: str,
    # file_identifier: str,
    # sha256: str,
    # metadata: Dict[str, Any],
    num_renders: int,
    render_dir: str ,
    only_northern_hemisphere: bool,
    gpu_devices: Union[int, List[int]],
    render_timeout: int,
    successful_log_file: Optional[str] = "handle-found-object-successful.csv",
    # failed_log_file: Optional[str] = "handle-found-object-failed.csv",
) -> bool:
    
    # args = 
    if local_path.endswith(".blend"):
        output_path = os.path.join(os.path.splitext(local_path)[0] + ".obj")

        print("Output path for obj file:", output_path)

        command1 = f"blender-3.2.2-linux-x64/blender --background  --python convert_obj.py -- --object_path '{local_path}' --output_path '{output_path}'"
        subprocess.run(
            ["bash", "-c", command1],
            timeout=render_timeout,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        local_path = output_path
        print(local_path)
    args = f"--object_path '{local_path}' --num_renders {num_renders}"
    # get the GPU to use for rendering
    using_gpu: bool = True
    gpu_i = 0
    if isinstance(gpu_devices, int) and gpu_devices > 0:
        num_gpus = gpu_devices
        gpu_i = random.randint(0, num_gpus - 1)
    elif isinstance(gpu_devices, list):
        gpu_i = random.choice(gpu_devices)
    elif isinstance(gpu_devices, int) and gpu_devices == 0:
        using_gpu = False
    else:
        raise ValueError(
            f"gpu_devices must be an int > 0, 0, or a list of ints. Got {gpu_devices}."
        )
    save_uid = generate_random_string(10)
    with tempfile.TemporaryDirectory() as temp_dir:
        # get the target directory for the rendering job
        target_directory = os.path.join(temp_dir, save_uid)
        os.makedirs(target_directory, exist_ok=True)
        args += f" --output_dir {target_directory}"

        # check for Linux / Ubuntu or MacOS
        if platform.system() == "Linux" and using_gpu:
            args += " --engine BLENDER_EEVEE"
        elif platform.system() == "Darwin" or (
            platform.system() == "Linux" and not using_gpu
        ):
            # As far as I know, MacOS does not support BLENER_EEVEE, which uses GPU
            # rendering. Generally, I'd only recommend using MacOS for debugging and
            # small rendering jobs, since CYCLES is much slower than BLENDER_EEVEE.
            args += " --engine CYCLES"
        else:
            raise NotImplementedError(f"Platform {platform.system()} is not supported.")

        # check if we should only render the northern hemisphere
        if only_northern_hemisphere:
            args += " --only_northern_hemisphere"

        # get the command to run
        command = f"blender-3.2.2-linux-x64/blender --background --python blender_script.py -- {args}"
        if using_gpu:
            command = f"export DISPLAY=:0.{gpu_i} && {command}"

        # render the object (put in dev null)
        # command = "export DISPLAY=:0.4 && blender-3.2.2-linux-x64/blender --background --python blender_script.py -- --object_path '/tmp/tmpziytasnv/objaverse-xl-test-files/example.stl' --num_renders 12 --output_dir /tmp/tmp_tdictn9/5f6d2547-3661-54d5-9895-bebc342c753d --engine BLENDER_EEVEE"
        print(command)
        subprocess.run(
            ["bash", "-c", command],
            timeout=render_timeout,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # check that the renders were saved successfully
        png_files = glob.glob(os.path.join(target_directory, "*.png"))
        metadata_files = glob.glob(os.path.join(target_directory, "*.json"))
        npy_files = glob.glob(os.path.join(target_directory, "*.npy"))
        if (
            (len(png_files) != num_renders)
            or (len(npy_files) != num_renders)
            or (len(metadata_files) != 1)
        ):
            print(num_renders)
            print(len(png_files))
            print(len(npy_files))
            print(len(metadata_files))
            return False

        # update the metadata
        metadata_path = os.path.join(target_directory, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_file = json.load(f)
        # metadata_file["sha256"] = sha256
        # metadata_file["file_identifier"] = file_identifier
        metadata_file["save_uid"] = save_uid
        # metadata_file["metadata"] = metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_file, f, indent=2, sort_keys=True)

        # Make a zip of the target_directory.
        # Keeps the {save_uid} directory structure when unzipped
        with zipfile.ZipFile(
            f"{target_directory}.zip", "w", zipfile.ZIP_DEFLATED
        ) as ziph:
            zipdir(target_directory, ziph)

        # move the zip to the render_dir
        fs, path = fsspec.core.url_to_fs(render_dir)

        # move the zip to the render_dir
        fs.makedirs(os.path.join(path, "renders"), exist_ok=True)
        fs.put(
            os.path.join(f"{target_directory}.zip"),
            os.path.join(path, "renders", f"{save_uid}.zip"),
        )

        # log that this object was rendered successfully
        # if successful_log_file is not None:
        #     log_processed_object(successful_log_file, file_identifier, sha256)

        return True

def render (
    render_dir: str = "./.objaverse2",
    local_path : str = "",
    download_dir: Optional[str] = None,
    num_renders: int = 50,
    processes: Optional[int] = None,
    save_repo_format: Optional[Literal["zip", "tar", "tar.gz", "files"]] = None,
    only_northern_hemisphere: bool = False,
    render_timeout: int = 300,
    gpu_devices: Optional[Union[int, List[int]]] = None,
) -> None:
    if platform.system() not in ["Linux", "Darwin"]:
        raise NotImplementedError(
            f"Platform {platform.system()} is not supported. Use Linux or MacOS."
        )
    # get the gpu devices to use
    parsed_gpu_devices: Union[int, List[int]] = 0
    if gpu_devices is None:
        parsed_gpu_devices = len(GPUtil.getGPUs())
    logger.info(f"Using {parsed_gpu_devices} GPU devices for rendering.")

    if processes is None:
        processes = multiprocessing.cpu_count() * 3
    
    fs, path = fsspec.core.url_to_fs(render_dir)
    try:
        zip_files = fs.glob(os.path.join(path, "renders", "*.zip"), refresh=True)
    except TypeError:
        # s3fs may not support refresh depending on the version
        zip_files = fs.glob(os.path.join(path, "renders", "*.zip"))

    saved_ids = set(zip_file.split("/")[-1].split(".")[0] for zip_file in zip_files)
    logger.info(f"Found {len(saved_ids)} objects already rendered.")

    return render_3d(local_path=local_path,
            render_dir=render_dir,
            num_renders=num_renders,
            only_northern_hemisphere=only_northern_hemisphere,
            gpu_devices=parsed_gpu_devices,
            render_timeout=render_timeout,)

if __name__ == "__main__":
    # fire.Fire(render_objects)
    import argparse
    import sys 
# Replace 'input_file.blend' with the path to your .blend file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render_dir",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument(
    "--local_path",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument(
    "--num_renders",
        type=int,
        required=True,
        help="Path to the object file",
    )
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)
    render(render_dir=args.render_dir, local_path=args.local_path, num_renders= args.num_renders)
  
