from torch import nn
from tqdm import tqdm

from utils import *

LHW = 169 * 208 * 179 # brain volume
n_classes = 2

def auc(arr):
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric():
    def __init__(self, model, mode, step, substrate_fn):
        r"""Create deletion/insertion metric instance.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, img_tensor, rise, verbose=0, device = 'cpu', save_to=None):
        r"""Run metric on one image-saliency pair.

        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            save_to (str): directory to save every step plots to.

        Return:
            scores (nd.array): Array containing scores at every step.
        """
        pred, _, _, _ = self.model(img_tensor)
        top, c = torch.max(pred, 1)
        c = c.cpu().numpy()[0]
        n_steps = (LHW + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        # rise
        salient_order = np.argsort(rise.reshape(-1, LHW), axis=1)
        for i in range(n_steps+1):
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start = start.reshape(1, LHW)
                start[0, coords] = finish.reshape(1, LHW)[0, coords]
                start = start.reshape(1, 1, 169, 208, 179)

            pred, _ = self.model(start.to(device))
            pr, cl = torch.topk(pred, 2)
            pred = nn.Softmax(dim=1)(pred)

            scores[i] = pred[0, c]
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(122)
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
                if i == n_steps:
                    plt.title(title)
                    plt.xlabel(ylabel)
                    plt.ylabel(get_class_name(c))
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()


        return scores
