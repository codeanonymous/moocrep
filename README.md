# MOOCRep: A Unified Pre-trained Embedding of MOOC Entities



**Code setup and Requirements**
Recent versions of PyTorch, tensorflow, numpy, sklearn. You can install all the required packages using the following command:
```
pip install requirements.txt 
```
To obtain the embeddings using the textual content only,use the following command: 
```
python feature_extractor.py
```
To pretrain the model using  domain-oriented objective, use the following command: 
```
python KG_embeddings.py
```
To use the model for lecture recommendation task: 
```
cd recommendation 
python main.py
```

To use the model for concept pre-requisite prediction task:
```
cd pre-requisite prediction
python siamese_fc_relu.py
```

