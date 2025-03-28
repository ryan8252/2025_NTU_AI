# 人工智慧, NTU, Spring 2025, homework1 Basic MLLM Implementation
R13922193 渠景量


## 安裝環境
Use conda to build a environment based on the yml file (in the code folder)
```bash
conda env create -f environment.yml
```
then activate environment
```bash
conda activate ai_hw1_13922193
```
cuda version: 12.4


## 如何執行程式
### Task 1

Please go into the ```task1``` folder

- For BLIP on MSCOCO run the following command

  ```bash
  python blip_mscoco.py
  ```

  The result will be stored in ```evaluation_results_blip_mscoco.txt```.

- For BLIP on Flickr30k run the following command

  ```bash
  python blip_flickr30k.py
  ```

  The result will be stored in ```evaluation_results_blip_flickr30k.txt```

- For Phi-4 on MSCOCO run the following command

  ```bash
  python phi4_mscoco.py
  ```

  The result will be stored in ```evaluation_results_phi4_mscoco.txt```

- For Phi-4 on Flickr30k run the following command

  ```bash
  python phi4_flickr30k.py
  ```

  The result will be stored in ```evaluation_results_phi4_flickr30k.txt```

### Task 2

Please go into the ```task2``` folder

- First download the input content images into ```./content_image``` folder


- For Task 2-1 run the following command

  ```bash
  python task2-1.py
  ```

  The result will be stored in ```output2-1```

- For Task 2-2 run the following command

  ```bash
  python task2-2.py
  ```

  The result will be stored in ```output2-2```




