python3 img2emb.py --dataset_dupl_word 5 --img_model_name vit_b_16 --lr 1e-5 --lr_scheduler_name None --save_model True
kaggle datasets version -p ../input/img2emb-data/ -m "upd"

python3 img2emb.py --dataset_dupl_word 5 --img_model_name vit_b_16_linear --lr 1e-5 --lr_scheduler_name None --save_model True
kaggle datasets version -p ../input/img2emb-data/ -m "upd"

python3 img2emb.py --dataset_dupl_word 5 --img_model_name regnet_y_16gf --lr 1e-5 --lr_scheduler_name None --save_model True
kaggle datasets version -p ../input/img2emb-data/ -m "upd"

python3 img2emb.py --dataset_dupl_word 5 --img_model_name regnet_y_32gf --lr 1e-5 --lr_scheduler_name None --save_model True
kaggle datasets version -p ../input/img2emb-data/ -m "upd"

python3 img2emb.py --dataset_dupl_word 5 --img_model_name regnet_y_16gf_linear --lr 1e-5 --lr_scheduler_name None --save_model True
kaggle datasets version -p ../input/img2emb-data/ -m "upd"

python3 img2emb.py --dataset_dupl_word 5 --img_model_name regnet_y_32gf_linear --lr 1e-5 --lr_scheduler_name None --save_model True
kaggle datasets version -p ../input/img2emb-data/ -m "upd"
