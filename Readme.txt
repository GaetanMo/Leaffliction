python3 -m venv venv
source venv/bon/activate
pip install -r requirements.txt

python -m srcs.Model.train --data_dir leaves/images --epochs 5 --batch_size 32
python -m srcs.Model.predict leaves/images/Apple_Black_rot/image\ \(119\).JPG --checkpoint Model/checkpoints/best.pt

Model part TODO:
Save the model checkpoint + the dataset used to train (augmented img) in a zip file  
Display the original img + augmented one after predict + give the diagnosis (see the subject)  
Summary => wait for augmented img to end this