# Temp instruction
### Install
```
    pip install SwissArmyTransformer
```
### Run CogView2
```
    cd examples/cogview2
    ./scripts/text2image_cogview2.sh
```

### Run GLM
1. Prepare input.txt. Example: "Welcome! This is the main page of SwissArmyTransformer".
2. Run the following commands:
```
    cd examples/glm
    ./scripts/generate_glm.sh config/model_glm_10B_chinese.sh
```

Output:
[CLS]Welcome! This is the main page of SwissArmyTransformer. It is a comprehensive and clear explanation of the technical problems in the transformer. It is also an introduction to the development of the SwissArmy transformers. Welcome to Swiss Army Transforters. This is the main page of Swiss army tranforter. It's a complete and clean explaination of technology problem in the Tranformer, which is an integral part of the army's technological development. It also anintroduction of the developments of the Army technicians. Well, if you have any questions, please feel free to contact the official webs
