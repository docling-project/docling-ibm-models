# SLANet_1M

- Install PaddlePaddle with CUDA 12.3

  ```bash linenums="1"
  python -m pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
  ```
  
- Then 
  ```bash linenums="1"
  pip install -r requirements.txt
  ```

- To train: 
  ```bash linenums="1"
  python train.py -c configs/SLANet_1M.yml -o Global.use_amp=True Global.scale_loss=1024.0 Global.use_dynamic_loss_scaling=True
  ```

Pre-trained Model on PubTanNet + SynthTabNet can be found  [here](https://drive.google.com/drive/folders/1aIzP3a3Ci0n9hXD2j57Dq4uCfQlt8yoW?usp=drive_link)