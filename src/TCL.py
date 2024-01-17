import torch
import tqdm 
import copy
import itertools
import cuda
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

print("cuda is available =",torch.cuda.is_available())
# if torch.cuda.is_available():
#     device = 'cuda:0'
# else:
#     device = 'cpu'

device = 'cpu'
# =============================================================================
# =============================================================================
def _do_train(model, loader, optimizer, criterion, device, num_batch_to_process = None):
    # training loop
    model.to(device=device)
    train_loss = np.zeros(len(loader))
    if num_batch_to_process is None : 
        num_batch_to_process = len(loader)
        
    loader_iter = iter(loader)
    for idx_batch, (batch_x, batch_y, batch_z) in enumerate(
        tqdm.tqdm(itertools.islice(loader_iter, num_batch_to_process), 
         desc='Training batches', unit=' batches', 
         dynamic_ncols=True, position=0, leave=True)):        
        optimizer.zero_grad()
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)
        batch_z = batch_z.to(device=device, dtype=torch.int32)
        logits, _ = model(batch_x, batch_z)
        
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        train_loss[idx_batch] = loss.item()
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'
        
    return np.mean(train_loss)


# In practice we will not use this function as the training loop is pretty slow 
# We won't use validation for TCL
def _validate(model, loader, criterion, device):
    # validation loop
    val_loss = np.zeros(len(loader))
    accuracy = 0.
    with torch.no_grad():
        model.eval()

        for idx_batch, (batch_x, batch_y, batch_z) in enumerate(tqdm.tqdm(loader, desc='Validation batches',unit='iteration', dynamic_ncols=True, position=0, leave=True)):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)
            batch_z = batch_z.to(device=device, dtype=torch.int32)
            
            
            output, _ = model.forward(batch_x, batch_z)

            loss = criterion(output, batch_y)
            val_loss[idx_batch] = loss.item()

            _, top_class = output.topk(1, dim=1)
            top_class = top_class.flatten()
            accuracy += \
                torch.sum((batch_y == top_class).to(torch.float32))


    accuracy = accuracy / len(loader.dataset)
    
    return np.mean(val_loss), accuracy.item()

# =============================================================================
# =============================================================================
def train(model, loader_train, optimizer, scheduler, n_epochs,
          device,
          num_batch_to_process = None,
          file_path = 'model'):
    """Training function

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    device : str | instance of torch.device
        The device to train the model on.

    Returns
    -------
    model : instance of nn.Module
        The model that lead to the best prediction on the validation
        dataset.
    """
    # put model on cuda if not already
    device = torch.device(device) 
    model.to(device)

    # define criterion
    criterion = nn.CrossEntropyLoss()

   
   
    train_loss = np.zeros(n_epochs)
  
    
    for epoch in tqdm.tqdm(range(n_epochs), desc='Training epochs'):
        train_loss[epoch] = _do_train(model, loader_train, optimizer, criterion, device, num_batch_to_process)
        scheduler.step()
        format_str = ('Epoch: %d/%d.. Training Loss: %.6f.. ')
        print(format_str % (epoch, n_epochs, train_loss[epoch]))
        
        #save model every epoch to avoid bugs and retraining from scratch 
        file_path_epoch = file_path + '_'+str(epoch)+'.pth'
        torch.save(model, file_path_epoch)
            
    return model, train_loss 
class tcl(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class,
                 num_layers, num_patient, 
                 activation = 'Maxout',
                 pool_size = 2,
                 slope = .1, 
                 dropout_p = 0.2,
                 feature_nonlinearity='abs'
                 ):
        """Build model with a feature-MLP and multi-MLR classifier for each subject
        Args:
            input_dim: (MLP) number of channels
            hidden_dim: (MLP) size of nodes for each layer
            num_class: the dim of output (number of labels) ie number of segments
            num_layers: (MLP) number of layers
            num_patient: number of subjects/patient (number of MLR)
            activation: (MLP) (option) activation function in the middle layer
            pool_size: (MLP) pool size of max-out nonlinearity
            slope: (MLP) for ReLU and leaky_relu activation
            feature_nonlinearity:(MLP) (option) Nonlinearity of the last hidden layer (feature value)
        """
        super(tcl, self).__init__()
        self.feature_nonlinearity = feature_nonlinearity
        self.num_class = num_class
        
        # Shared feature-MLP
        self.MLP = MLP(input_dim, hidden_dim, num_layers,
                       activation = activation,
                       pool_size = pool_size, 
                       slope = slope,
                       dropout_p= dropout_p)
        
        
        self.MLP.to(device)
        
        
        if isinstance(hidden_dim, list):
            _mlr_input_dim = hidden_dim[-1]
        else:
            _mlr_input_dim = hidden_dim
        
        # MLRs (subject-specific mlr)
    
        _MLRs_list = [nn.Linear(_mlr_input_dim, num_class) for k in range(num_patient)]
        self.MLRs = nn.ModuleList(_MLRs_list)
        
        # initialize MLR for each patient
        for k in range(num_patient):
            torch.nn.init.xavier_uniform_(self.MLRs[k].weight)

    
    def forward(self, x, patient_id=None):
        """forward pass
        Args:
            x: shape(batch_size,num_channels)
            patient_id: subject id
        Returns:
            y: labels (batch_size,)
            h: features (batch_size, num_channels)
        """
        x.to(device=device)
        h = self.MLP(x)
        if self.feature_nonlinearity == 'abs':
            h = torch.abs(h) # Nonlinearity of the last hidden layer (feature value)
            
        # logis
        if patient_id is None:
            y = None
        else:
            uniq= torch.unique(patient_id)

            # Initialize an empty list to store model predictions
            #predictions = []
            predictions = torch.zeros((len(patient_id),self.num_class), device=device)

            # Iterate over unique values and apply models to corresponding rows
            for k in uniq:
                # Get indices where patient_id equals k
                indices_k = (patient_id == k).nonzero().squeeze()

                # Extract rows that a from patient k from h based on indices
                rows_to_apply_model = h[indices_k, :]
                prediction_k = self.MLRs[k](rows_to_apply_model)
                # reshape [n_segment] to [1,n_segment]
                if len(prediction_k.shape) == 1:
                    prediction_k = prediction_k.unsqueeze(0) 
                # Apply the model and append predictions to the list
                predictions[indices_k]= prediction_k

            
        return predictions, h


# =============================================================================
class MLP(nn.Module):
    @staticmethod
    def leaky_relu(x, negative_slope=0.1):
        return F.leaky_relu(x, negative_slope=negative_slope)

    @staticmethod
    def relu(x):
        return F.relu(x)

    @staticmethod
    def sigmoid(x):
        return torch.sigmoid(x)

    @staticmethod
    def identity(x):
        return x
    
    def __init__(self, input_dim, hidden_dim, 
                 num_layers,
                 activation = 'Maxout',
                 pool_size = 2,
                 slope = .1,
                 dropout_p=0.2):
        """Built feature-MLP model as feature extractor:
        Args:
            input_dim: size of input channels, here is number of components
            hidden_dim: size of nodes for each layer # last dim of hidden_dim is n_components needed
            num_layers: number of layers == len(hidden_dim)
            activation: (option) activation function in the middle layer
            pool_size: pool size of max-out nonlinearity
            slope: Leaky_relu activation
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        
        # In the following , we could have 1 hidden_dim / activation specified 
        # but several hidden units 
        # --> we use this unique value of hidden_dim / activation for every layer
        if isinstance(hidden_dim, int):
            self.hidden_dim = [hidden_dim] * (self.num_layers)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim - please choose an integer or list of integers of size hidden_dim')
        
        # Activation
        if isinstance(activation, str):
            self.activation = [activation] * (self.num_layers ) 
        elif isinstance(activation, list):
            assert len(activation) == self.num_layers , 'Incorrect length of activation list - adapt or give a string'
            self.activation = activation
        else:
            raise ValueError('Wrong argument type for activation please choose a string or list of strings of size hidden_dim')
        
        # Initialize the activation 
        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(self.leaky_relu)
            elif act == 'ReLU':
                self._act_f.append(self.relu)
            elif act == 'Maxout':
                self._act_f.append(Maxout(pool_size))
            elif act == 'sigmoid':
                self._act_f.append(self.sigmoid)
            elif act == 'none':
                self._act_f.append(self.identity)
            else:
                ValueError('Incorrect activation - activation values have to be in ["lrelu", "ReLU", "Maxout", "sigmoid","none"]')
        
        
        # MLP 
        # if Maxout we have to multiply each hidden_dim by pool size to compensate the maxout operation
        # special case for first layer as we use input_dim and not previous output
        # special case for last layer : no activation ie no compensation
        
        _layer_list = []
        for k in range(0,self.num_layers):
            activation = self.activation[k]
            if k == 0 :
                if activation == 'Maxout':
                    _layer_list.append(nn.Linear(self.input_dim, self.hidden_dim[0]*pool_size)) # compensate maxout pool size
                else : 
                    _layer_list.append(nn.Linear(self.input_dim, self.hidden_dim[0]))
            else : 
                if activation == 'Maxout':
                    _layer_list.append(nn.Linear(self.hidden_dim[k-1], self.hidden_dim[k]*pool_size)) # compensate maxout pool size
                else : 
                    _layer_list.append(nn.Linear(self.hidden_dim[k-1], self.hidden_dim[k]))
        self.layers = nn.ModuleList(_layer_list)

        # Dropout
        self.dropout = nn.Dropout(p=dropout_p)
        
        # Bacth-norm after each output of hidden layers
        self.bn = nn.ModuleList()
        for bni in range(self.num_layers-1):
            self.bn.append(nn.BatchNorm1d(self.hidden_dim[bni]))
            
        # initialize
        for k in range(len(self.layers)):
            torch.nn.init.xavier_uniform_(self.layers[k].weight)
        
    def forward(self, x):
        """forward process
        Args:
            x: input data nput [batch, dim]
        Returns:
            h: features 
        """
        #h/feature values
        h = x
        for k in range(len(self.layers)):
            # activation
            h = self._act_f[k](self.layers[k](h))
            # Dropout for first 2 layers
            if k<=1: 
                h=self.dropout(h)

        return h

class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        # Reshape the input tensor (batch_size, pool_size * hidden_dim[k] ) 
        # to have shape (batch_size, hidden_dim[k], pool_size)
        # This divides the second dimension (channels) into groups of size self._pool_size.
        reshaped_x = torch.reshape(x, (x.shape[0], x.shape[1] //self._pool_size, self._pool_size))
        #torch.reshape(x, (*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:]))
        m, _ = torch.max(reshaped_x, dim=2)
        return m



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
