# Toxic Comment Classification
Fullstack end-to-end lightweight toxic comment classification with result interpretation


## Train
```
python run_training.py
```

## Inference
```python
from src.toxic.inference import Toxic

model = Toxic.from_checkpoint('path_to_model')
model.infer('привет, придурок')
```
Result:
```
{
    'predicted': [
        {'class': 'insult', 'confidence': 0.99324},
        {'class': 'threat', 'confidence': 0.002},
        {'class': 'obscenity', 'confidence': 0.00225}
    ],
    'interpretation': {
        'spans': [(0, 7), (7, 16)],
        'weights': {
            'insult': [-0.34299, 0.93934],
            'threat': [-0.97362, 0.22819],
            'obscenity': [-0.99579, 0.09168]
        }
    }
}
```

## Serving
### Streamlit
```
streamlit run ui/app.py -- --model=models/model.pth
```

## Powered By
* [captum](https://github.com/pytorch/captum)
* [hydra](https://github.com/facebookresearch/hydra)
* [tokenizers](https://github.com/huggingface/tokenizers)
* [torchmetrics](https://github.com/PyTorchLightning/metrics) 
* [streamlit](https://github.com/streamlit/streamlit)