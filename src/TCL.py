import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class TCLModule(nn.Module):
    """
        TCL module : Multi Layer to perform Logistic Regression over time series segmented
    """
    def __init__(self, input_size, list_hidden_nodes, num_class, wd=1e-4, maxout_k=2, MLP_trainable=True, feature_nonlinearity='abs'):
        super(TCLModule, self).__init__()
        self.num_layer = len(list_hidden_nodes)
        self.maxout_k = maxout_k
        self.MLP_trainable = MLP_trainable
        self.feature_nonlinearity = feature_nonlinearity

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for ln in range(self.num_layer):
            in_dim = list_hidden_nodes[ln-1] if ln > 0 else input_size
            out_dim = list_hidden_nodes[ln]


            if ln < self.num_layer - 1:
                out_dim *= maxout_k

            layer = self._build_hidden_layer(in_dim, out_dim)
            self.hidden_layers.append(layer)

        # MLR
        self.mlr_layer = nn.Linear(list_hidden_nodes[-1], num_class)

    def _build_hidden_layer(self, in_dim, out_dim):
        layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),  # Batch normalization
            nn.ReLU(),                # ReLU activation
        )
        return layer

    def maxout(self, y):
        input_shape = y.size()
        ndim = len(input_shape)
        ch = input_shape[-1]
        assert ndim == 4 or ndim == 2
        assert ch is not None and ch % self.maxout_k == 0

        if ndim == 4:
            y = y.view(-1, input_shape[1], input_shape[2], ch // self.maxout_k, self.maxout_k)
        else:
            y = y.view(-1, ch // self.maxout_k, self.maxout_k)

        y, _ = y.max(dim=ndim)
        return y

    def forward(self, x):
        # Hidden layers
        for ln in range(self.num_layer):
            layer = self.hidden_layers[ln]
            x = layer(x)

            if ln < self.num_layer - 1:
                x = self.maxout(x)

        feats = x

        # MLR layer
        logits = self.mlr_layer(x)

        return logits, feats


class TCLLoss(nn.Module):
    """TCL loss : combination of cross entropy and LR regularization over parameters

    """
    def __init__(self, weight_decay=1e-2):
        super(TCLLoss, self).__init__()
        self.weight_decay = weight_decay
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels, model):
        # Cross entropy loss
        ce_loss = self.cross_entropy_loss(logits, labels)

        # L2 regularization
        l2_reg_terms = [torch.norm(param, p=2) for param in model.parameters()]
        l2_reg = torch.sum(torch.stack(l2_reg_terms))


        # Total loss
        total_loss = ce_loss + self.weight_decay * l2_reg

        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        correct = (predicted == labels).float()
        accuracy = torch.mean(correct)

        return total_loss, accuracy


class TCLTrainer:
    """ Training functions , criterion is usually TCLLoss, initial learning rate can be fixed or adaptive"""
    def __init__(self, model, criterion, initial_learning_rate, momentum, decay_steps, decay_factor):
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=decay_factor, last_epoch=-1, verbose=False)

    def train_step(self, inputs, labels):
        self.optimizer.zero_grad()
        logits, _ = self.model(inputs)
        loss, accuracy = self.criterion(logits, labels, self.model)
        loss.backward()
        self.optimizer.step()
        return loss.item(), accuracy.item()

    def adjust_learning_rate(self, epoch):
        self.scheduler.step(epoch)


def pca(x, num_comp=None, params=None, zerotolerance=1e-7):
    """Apply PCA whitening to data. Used as preprocessing 
    Args:
        x: data. 2D tensor [num_comp, num_data]
        num_comp: number of components
        params: (optional) dictionary of PCA parameters {'mean':?, 'W':?, 'A':?}. If given, apply this to the data
        zerotolerance: (optional)
    Returns:
        x: whitened data
        params: parameters of PCA
            mean: subtracted mean
            W: whitening matrix
            A: mixing matrix
    """
    print("PCA...")

    # Dimension
    if num_comp is None:
        num_comp = x.size(0)
    print("    num_comp={0:d}".format(num_comp))

    # From learned parameters --------------------------------
    if params is not None:
        # Use previously-trained model
        print("    use learned value")
        data_pca = x - params['mean']
        x = torch.mm(params['W'], data_pca)

    # Learn from data ----------------------------------------
    else:
        # Zero mean
        xmean = torch.mean(x, dim=1, keepdim=True)
        x = x - xmean

        # Eigenvalue decomposition
        xcov = torch.mm(x, x.t()) / x.size(1)
        d, V = torch.linalg.eigh(xcov, UPLO='U')  # Ascending order
        # Convert to descending order
        d = torch.flip(d, dims=[0])
        V = torch.flip(V, dims=[1])

        d_normalized = torch.where(d[0] > zerotolerance, d[:num_comp] / d[0], zerotolerance) ## correct 0 eigenvalues by tolerance
        

        # Calculate contribution ratio
        contratio = torch.sum(d[:num_comp]) / torch.sum(d)
        print("    contribution ratio={0:f}".format(contratio))

        # Construct whitening and dewhitening matrices
        dsqrt = torch.sqrt(d[:num_comp])
        dsqrtinv = 1 / dsqrt
        V = V[:, :num_comp]
        # Whitening
        W = torch.mm(torch.diag(dsqrtinv), V.t())  # whitening matrix
        A = torch.mm(V, torch.diag(dsqrt))  # de-whitening matrix
        x = torch.mm(W, x)

        params = {'mean': xmean, 'W': W, 'A': A}

        # Check
        datacov = torch.mm(x, x.t()) / x.size(1)
        
    return x, params
