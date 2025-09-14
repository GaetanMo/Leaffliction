import sys, os
from pathlib import Path
import subprocess
from Transformation import transform, save


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Distribution")))

from Distribution import get_data


def equilibrate_data(data, path):
	max_value = max(data.values())
	values = list(data.values())
	paths = list(data.keys())
	base_dir = os.path.dirname(os.path.abspath(__file__))
	transformation_script = os.path.join(base_dir, "Transformation.py")
	relaunch = False
	for i in range(len(values)):
		if values[i] == max_value:
			continue
		new_path = os.path.join(path, paths[i])
		save_dir = os.path.abspath(new_path)
		imgs = os.listdir(new_path)
		img = 0
		while (values[i] != max_value):
			if img == len(imgs):
				break
			img_path = os.path.join(new_path, imgs[img])
			imgs_transformed = []
			if max_value - values[i] >= 6:
				imgs_transformed = [transform(img_path, j) for j in range(6)]
    			# cmd = [
				# 	"python3", transformation_script,
				# 	img_path,
				# 	"--display", "off",
				# 	"-s", save_dir
				# ]
				# result = subprocess.run(cmd, capture_output=True, text=True)
			else:
				imgs_transformed = [transform(img_path, j) for j in range(max_value - values[i])]
				# cmd = [
				# 	"python3", transformation_script,
				# 	img_path,
				# 	"--display", "off",
				# 	"-s", save_dir,
				# 	"-n", str(max_value - values[i])
				# ]
				# result = subprocess.run(cmd, capture_output=True, text=True)
				# print("stdout:", result.stdout)
				# Appel avec -n "max_value - values[i]"
			save(imgs_transformed, save_dir, img_path, False)
			values = list(get_data(path).values())
			if values[i] == max_value:
				break
			print(f"Number: {values[i]} | Goal: {max_value}")
			img += 1
			if img == len(imgs):
				relaunch = True
				break
	if relaunch:
		equilibrate_data(get_data(path), path)

def main():
    if len(sys.argv) != 2:
        print("Path folder is missing.")
        exit(1)
    pathname = Path(sys.argv[1])
    data = get_data(pathname)
    equilibrate_data(data, pathname)

if __name__ == "__main__":
    main()
