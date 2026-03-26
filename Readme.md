    make .venv

    source .venv/bin/activate

    make distribution

    make augmentation
    
        or python -m srcs.Augmentation.Augmentation [img_path] 
        (augmentation on a single image)

        or python -m srcs.Augmentation.Transformation -d off -n 6 -s {outdir} {path_to_image} 
        (use the transformation feature used for the augmentation)
    
    make transformation
    
    make train  
    1 arg needed --name_tail _original, or any tail name you want _mask _blur etc...
    make predict
    2 args needed:
    the path to the img to predict: "data/images_transformed/Apple_Black_rot/image (2)"
    the tail used during training: --name_tail "_original"
