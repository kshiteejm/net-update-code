sudo apt update
sudo apt install -y python3-pip
pip3 install --user numpy
pip3 install --user scipy
pip3 install --user torch
pip3 install --user IPython
pip3 install --user networkx
pip3 install --user graphviz
pip3 install --user matplotlib
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
echo 'export PATH=$HOME/.cargo/bin:$PATH' >> ~/.bashrc
