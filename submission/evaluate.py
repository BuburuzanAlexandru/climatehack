import numpy as np
import torch

from climatehack import BaseEvaluator
from ConvLSTM2 import ConvLSTM


class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.

        In this case, it loads the trained model (in evaluation mode)."""

        self.model = ConvLSTM()
        self.model.load_state_dict(torch.load("conv_deep_lstm_fewer-channels_no-bn_epoch-21_lr1e-4.pt", map_location=torch.device('cpu'))['state_dict'])
        self.model.eval()

    def predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Makes a prediction for the next two hours of satellite imagery.

        Args:
            coordinates (np.ndarray): the OSGB x and y coordinates (2, 128, 128)
            data (np.ndarray): an array of 12 128*128 satellite images (12, 128, 128)

        Returns:
            np.ndarray: an array of 24 64*64 satellite image predictions (24, 64, 64)
        """

        assert coordinates.shape == (2, 128, 128)
        assert data.shape == (12, 128, 128)
        input = torch.from_numpy(data.astype(np.float32)).view(1, 12, 1, 128, 128) / 1024.0

        with torch.no_grad():
            prediction = (
                self.model(input)
                .view(24, 64, 64)
                .detach()
                .numpy()
            )

            assert prediction.shape == (24, 64, 64)

            return prediction


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
