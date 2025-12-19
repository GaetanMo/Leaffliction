# Setup your .venv

all: .venv

.venv:
	@echo "\033[0;32mCreating virtual environment...\033[0m"
	@python3 -m venv .venv
	@.venv/bin/pip install --upgrade pip
	@.venv/bin/pip install -r .libRequirements.txt
	@echo "\033[0;34mTo activate the virtual environment, run 'source .venv/bin/activate'\033[0m"
	@echo "\033[0;34mTo deactivate the virtual environment, run 'deactivate'\033[0m"

fclean:
	@echo -n "\033[0;32mCleaning up...\033[0m"
	@rm -rf .venv
	@find . -type d -name "__pycache__" | xargs rm -rf
	@echo "\033[0;32m\rDone!\033[K\33[0m"

re: fclean all

distribution:
	python -m srcs.Distribution.Distribution data/images

augmentation:
	python -m srcs.Augmentation.Augmentation data/images

transformation:
	python  srcs/Transformation/Transformation.py -src data/images -dst data/images_transformed

train:
	python -m srcs.Model.train --name_tail _original

predict:
	python -m srcs.Model.predict "data/images_transformed/Apple_Black_rot/image (2)_original.JPG" "data/images_transformed/Apple_Black_rot/image (2)_original.JPG" --show

.PHONY: all fclean re .venv