"""Download missing datasets for the opt-in FedAvg batch protocol."""
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

TINY_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"


def tiny_complete(root):
    root = Path(root)
    try:
        classes = (root / "wnids.txt").read_text().split()
        annotations = (root / "val" / "val_annotations.txt").read_text().splitlines()
        return (len(classes) == 200 and len(annotations) == 10000
                and all(sum(1 for _ in (root / "train" / cls).rglob("*.JPEG")) == 500
                        for cls in classes)
                and sum(1 for _ in (root / "val").rglob("*.JPEG")) == 10000)
    except OSError:
        return False


def safe_extract(archive, destination):
    """Extract regular files/directories beneath the official archive root."""
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            path = Path(member.filename)
            if (not path.parts or path.parts[0] != "tiny-imagenet-200"
                    or ".." in path.parts or path.is_absolute()
                    or ((member.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError("Unsafe Tiny-ImageNet archive entry: " + member.filename)
        # Extraction verifies each file's ZIP CRC.
        zipped.extractall(destination)


def prepare_data(config):
    data = config["data_setups"]
    root = Path(data["root"]) / data["dataset_name"]
    download = config["batch_protocol"]["download_if_missing"]
    if data["dataset_name"] in ("cifar10", "cifar100"):
        from torchvision.datasets import CIFAR10, CIFAR100
        dataset = CIFAR10 if data["dataset_name"] == "cifar10" else CIFAR100
        for train in (True, False):
            dataset(str(root), train=train, download=download)
        return
    if data["dataset_name"] != "tinyimagenet":
        raise ValueError("Unsupported FedAvg batch dataset")
    if tiny_complete(root) or tiny_complete(root / "tiny-imagenet-200"):
        return
    if not download:
        raise FileNotFoundError("Tiny-ImageNet is missing/incomplete at " + str(root))
    root.parent.mkdir(parents=True, exist_ok=True)
    archive = root.parent / "tiny-imagenet-200.zip"
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        partial = archive.with_suffix(".zip.part")
        print("Downloading Tiny-ImageNet from " + TINY_URL, flush=True)
        request = urllib.request.Request(TINY_URL, headers={"User-Agent": "FedClsimb/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        if not zipfile.is_zipfile(partial):
            raise ValueError("Downloaded Tiny-ImageNet file is not a ZIP archive")
        partial.replace(archive)
    print("Extracting Tiny-ImageNet (this may take a few minutes)...", flush=True)
    with tempfile.TemporaryDirectory(prefix="tiny-extract-", dir=root.parent) as temporary:
        safe_extract(archive, temporary)
        extracted = Path(temporary) / "tiny-imagenet-200"
        if not tiny_complete(extracted):
            raise ValueError("Tiny-ImageNet archive is incomplete")
        if root.exists():
            shutil.copytree(extracted, root, dirs_exist_ok=True)
        else:
            shutil.move(str(extracted), str(root))
