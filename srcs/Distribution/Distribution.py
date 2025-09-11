import sys
import logging
from pathlib import Path
from utils import print_distribution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    data = {}
    if len(sys.argv) != 2:
        logger.error("Path folder is missing.")
        exit(1)
    pathname = Path(sys.argv[1])
    if not pathname.is_dir():
        logger.error("Path folder is not a directory.")
        exit(1)
    for subdir in pathname.glob("**"):
        if subdir.is_dir():
            jpg_files = list(subdir.glob("*.JPG"))
            if jpg_files:
                data[subdir.name] = len(jpg_files)
    print_distribution(data)

if __name__ == "__main__":
    main()