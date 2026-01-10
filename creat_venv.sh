python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install --pre dgl -f https://data.dgl.ai/wheels-test/torch-2.4/cu121/repo.html
pip install -r requirements.txt
echo ".venv/" >> .git/info/exclude
