from pathlib import Path
from typing import List, Union

import torch

from src.toxic.caching import load_pretrained
from src.toxic.data.tokenizer import SentencePieceBPETokenizer
from src.toxic.interpretation import lig_explain
from src.toxic.modelling.model import Model


class Toxic:
    """Inference module"""

    def __init__(
        self,
        tokenizer: SentencePieceBPETokenizer,
        model: Model,
        labels: List[str]
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.labels = labels

    @classmethod
    def from_checkpoint(cls, name_or_path: Union[str, Path]) -> 'Toxic':
        """Loads pretrained model from checkpoint

        Args:
            name_or_path (str): Path or name or url to checkpoint.

        Returns:
            Toxic: Loaded model.
        """
        path = Path(name_or_path)
        if path.exists():
            file = path
        else:
            file = load_pretrained(name_or_path)
        data = torch.load(file, map_location='cpu')
        tokenizer = SentencePieceBPETokenizer.deserialize(data['tokenizer'])
        tokenizer.tokenizer.no_truncation()
        model = Model(data['config']).eval()
        model.load_state_dict(data['model'])
        labels = data['config'].data.labels
        return cls(tokenizer, model, labels)

    def infer(
        self,
        sentence: str,
        interpret: bool = True
    ) -> dict:
        """Makes an inference on a given sentence

        Args:
            sentence (str): Sentence.
            interpret (bool): Whether to add interpretation results to output.
                Defaults to False.
        Returns:
            dict: Inference result.
        """
        encoded = self.tokenizer.encode(sentence)
        ids = encoded['ids'].unsqueeze(0)
        with torch.no_grad():
            predicted = self.model(ids).logits.squeeze(0).sigmoid()
        result = dict()
        result['predicted'] = [
            {
                'class': label,
                'confidence': round(confidence, 5)
            }
            for label, confidence
            in zip(self.labels, predicted.tolist())
        ]
        if interpret:
            interpretation = {'spans': encoded['spans'], 'weights': {}}
            for i, confidence in enumerate(predicted):
                weights = lig_explain(
                    inputs=ids,
                    target=i,
                    forward=lambda x: self.model(x).logits,
                    embedding_layer=self.model.encoder.token_embedding,
                )
                weights = [round(weight, 5) for weight in weights.tolist()]
                label = self.labels[i]
                interpretation['weights'][label] = weights
            result['interpretation'] = interpretation
        return result
