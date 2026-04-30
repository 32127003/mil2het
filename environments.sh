pip install --upgrade pip setuptools wheel
pip install numpy==1.26.1 pandas==2.0.3 scipy==1.15.3
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.1+cu118.html
pip install torch-geometric
pip install scikit-learn==1.7.2 umap-learn==0.5.11 pynndescent==0.6.0
pip install anndata==0.9.2 h5py==3.10.0 scanpy==1.9.6
pip install ipykernel==6.29.3
pip install esm==3.2.1.post1
pip install "accelerate>=0.32,<2"
pip install httpx
