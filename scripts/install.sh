# create venv with all required dependencies
conda create --name aistats2020 -y

# activate venv
conda activate aistats2020

# install pip
conda install pip -y

# install conda packages
conda install scipy pandas gensim joblib sh matplotlib seaborn -y

# install pip packages
pip install tensorboardX

# install pytorch
conda install pytorch -c pytorch -y

# install rdkit
conda install rdkit -c rdkit -y

