# Abnormal Behavior and Stranger Detection System

A proposed deep learning model that integrates **Abnormal Behavior Detection** and **Stranger Detection** in surveillance video analysis.

We adopt the **UCF-11 dataset structure** as a base. All datasets used should follow this format:

<pre> root_dir/ 
    ├── class1/ 
    │   ├── videosA/ 
    │   │   ├── videoA1.mpg 
    │   │   └── videoA2.mpg 
    │   └── videosB/ 
    │   ├── videoB1.mpg 
    │   └── videoB2.mpg 
    ├── class2/ 
    │   └── ...
</pre>
---

Example Command-Lines to Use:
1. Train a New Model
<pre> 
python main.py --mode train \
               --root_dir dataset \
               --frame_count 10 \
               --batch_size 1 \
               --learning_rate 0.0001 \
               --num_epochs 1 \
               --model_save_path behavior_model.pth \
               --shuffle True
</pre>
2. Continue training an Existing Model
<pre>python main.py --mode train \
               --root_dir dataset \
               --frame_count 10 \
               --batch_size 1 \
               --learning_rate 0.0001 \
               --num_epochs 1 \
               --model_load_path behavior_model.pth \
               --model_save_path behavior_model.pth \
               --shuffle True
</pre>
3. Evaluate a Trained Model
<pre>python main.py --mode eval \
               --root_dir dataset \
               --frame_count 10 \
               --batch_size 1 \
               --save_eval_path evaluation_report.csv
</pre>


