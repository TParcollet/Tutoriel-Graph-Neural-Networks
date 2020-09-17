# M2S3-IA - UCE1 - Les GNNs en pratique : accélération GPU via la bibliothèque PyTorch Geometric. 
## Centre d'Enseignement et de Recherche en Informatique, Avignon Université

PyTorch est une bibliothèque logicielle Python open source d'apprentissage automatique développée par Facebook permettant les calculs tensoriels nécessaires à l’apprentissage profond. Tout comme Tensorflow et Keras (développés par Google), PyTorch permet une intégration aisée des GPUs comme matériel de calcul. La manipulation de données symboliques induisant d’importantes contraintes relationnelles difficilement capturables via les outils classiques (numpy, PyTorch, Python …), une bibliothèque dédiée nommée PyTorch Geometric est apparue facilitant et accélérant la recherche pour l’apprentissage profond “géométrique”. Il s’agit ici de fournir les outils nécessaires à l’utilisation et l’analyse des jeux de données structurés, ainsi que ceux permettant l’apprentissage automatique de réseaux neuronaux de graphes allant des modèles basiques (Graph Convolutional Networks) à ceux déployés dans des environnements état-de-l’art.   

Source: https://www.youtube.com/watch?v=X_fmiIy_YyI&list=PL-Y8zK4dwCrQyASidb2mjj_itW2-YYx6-&index=9

Pré-requis: *préparer son environnement miniconda*.  
1. `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh` **(PAS NÉCESSAIRE SUR HERACLES ET ACHILES)** 
2. `chmod +x Miniconda3-latest-Linux-x86_64.sh` **(PAS NÉCESSAIRE SUR HERACLES ET ACHILES)** 
3. `./Miniconda3-latest-Linux-x86_64.sh` **(PAS NÉCESSAIRE SUR HERACLES ET ACHILES)** 
4. `source .bashrc` **(PAS NÉCESSAIRE SUR HERACLES ET ACHILES)** 
5. `conda create --name pytorch`
6. `conda activate pytorch`

Installer PyTorch et Jupyter:
1. `conda install pytorch torchvision cudatoolkit=$INSERER_CUDA_VERSION(10.1/10.2) -c pytorch`
2. `pip install jupyter`

Une fois le dépôt clôné, vous pouvez lancer votre notebook jupyter:    
`jupyter notebook --no-browser --port=XXXX` **LE PORT DOIT ÊTRE UNIQUE PAR ÉTUDIANT**

Ensuite, il ne reste plus qu'à créer un tunnel SSH pour vous connecter à votre notebook:    
`ssh -N -L YYYY:localhost:XXXX <remote_user>@<remote_host>`.    
*Note: Il faudra faire de même pour tensorboard*    

Se connecter à *localhost:XXXX* via n'importe quel naviguateur en renseignant le token d'identification.


