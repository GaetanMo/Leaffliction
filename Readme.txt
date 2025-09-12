python3 -m venv venv
source venv/bon/activate
pip install -r requirements.txt

python -m srcs.Model.train --data_dir leaves/images --epochs 5 --batch_size 32
python -m srcs.Model.predict leaves/images/Apple_Black_rot/image\ \(119\).JPG --checkpoint Model/checkpoints/best.pt