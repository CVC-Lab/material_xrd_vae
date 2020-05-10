import torch
from torch import nn
from torch.nn import functional as F
from model.baselines import MLP_2layer

#######################################
## Source ： https://github.com/weixsong/NormalizingFlow
#######################################

class Transformation:

    """Base class of all normalizing flow transformations"""
    
    def __init__(self):
        self.training = None
        self.log_det = None
        
    @property
    def training(self):
        return self._training
    
    @training.setter
    def training(self, enable:bool):
        """When training is enabled, the jacobians are recorded"""
        if not enable:
            self.log_det = None
        self._training = enable

    def transform(self, zi, params):
        """Transform the latent variables using this transformation
        
        Args
        zi     -- variable which will be transfomed
        params -- parameters for this Transformation

        Returns transformed variable zi' of the same shape as the input zi
        """
        raise NotImplementedError()
    
    def det(self, zi, params):
        """Compute the jacobian of this transformation given zi and parameters"""
        raise NotImplementedError()
    
    def forward(self, zi, params):
        """Forward pass applies this Transformation with parameters on zi"""
        if self.training:
            self.log_det = torch.log( self.det( zi, params ).squeeze() + 1e-7 )
        return self.transform( zi, params )
    
    def get_num_params(self):
        """Get the number of parameters this Transformation requires"""
        return 0




class SylvesterTransformation(Transformation):
    
    """Base class for all Sylvester tranformations"""

    def __init__(self, dim:int, num_hidden:int=1, device:str='cuda', training:bool=True):
        
        self.dim = dim
        self.h = nn.Tanh()
        self.training = training
        self.num_hidden = num_hidden
        self.device = device
        self.eye_M = torch.eye(num_hidden, device=device)
        
    def get_num_params(self):
        raise NotImplementedError()
    
    def transform(self, z, params):
        """Transform the latent variables using this transformation
        
        Args
        zi     -- variable which will be transfomed
        params -- parameters for this Transformation

        Returns transformed variable zi' of the same shape as the input zi
        """
        Q, R, R_hat, b = self.unwrap_params( params, z )
        
        return z + Q.mm(R).mm(self.h( F.linear(z, R_hat.mm(Q.t()), b) ).t()).t()
    
    def unwrap_params(self, params, z):
        raise NotImplementedError()
    
    def h_deriv(self, x):
        """Derivative of the activation function"""
        ff = self.h( x )
        return 1 - ff * ff
    
    def det_triag(self, mat):
        """Determinant of an upper or lower triangular matix"""
        return torch.cumprod(torch.diagonal(mat, dim1=-2, dim2=-1), -1)
    
    # det ( I_M + diag ( h′( R_hat Q^T z + b ) ) R_hat R )
    def det(self, z, params):
        """Compute the jacobian of this transformation given zi and parameters"""
        Q, R, R_hat, b = self.unwrap_params( params, z )
        psi = self.h_deriv(F.linear(z, R_hat.mm(Q.t()), b))
        
        # a workaround for batched matrices
        psi_mat = torch.zeros((psi.shape[0], psi.shape[1], psi.shape[1]),device=self.device)
        psi_mat.as_strided(psi.size(), [psi_mat.stride(0), psi_mat.size(2) + 1]).copy_(psi)
        
        psi_mat_RR = torch.matmul(psi_mat, R_hat.mm(R))
        return self.det_triag( self.eye_M.repeat(psi_mat.shape[0], 1, 1) - psi_mat_RR ).abs()

class OrthoSylvesterTransformation(SylvesterTransformation):

    """SylvesterTransformation where column-wise orthogonality of Q is ensured
        by an iterative procedure.
    """
    
    def __init__(self, dim:int, num_hidden:int=1, device:str='cuda', training:bool=True):
        
        super().__init__(dim, num_hidden, device, training)
        
    def get_num_params(self):
        """Get the number of parameters this Transformation requires"""
        # Q, R, R_hat, b
        return self.dim * self.num_hidden + self.num_hidden * self.num_hidden + self.num_hidden + self.num_hidden
    
    def unwrap_params(self, params, z):
        """Convert an array with params into vectors and matrices of this Transformation
        Args
        params -- array with parameters
        z      -- latent variable
        
        Returns Q, R and R_hat matrices and b bias vector
        """
        Q = params[:self.dim * self.num_hidden].reshape((self.dim, self.num_hidden))
        RR = params[self.dim * self.num_hidden : (self.dim + self.num_hidden) * self.num_hidden].reshape((self.num_hidden, self.num_hidden))
        R = torch.triu(RR)
        
        R_hat = torch.tril(RR).t()
        v = params[-2 * self.num_hidden : -self.num_hidden]
        mask = torch.diag(torch.ones_like(v))
        R_hat = mask * torch.diag(v) + (1. - mask) * R_hat

        b = params[-self.num_hidden:]
        Q = self.make_ortho( Q, 1e-1 )
        return Q, R, R_hat, b
    
    def make_ortho(self, Q, eps):
        """Iteratively convert Q into column-wise orthogonal matrix"""
        # TODO: how to make sure that the convergence condition is fulfilled?
        QQ = Q.t().mm(Q)

        # check convergence condition
        _, s, _ = torch.svd(QQ - self.eye_M)
        if s[0] > 1:
            print( "[WARN] Q will not converge to orthogonal" )
            return Q
        
        # while not converged
        while torch.norm(QQ - self.eye_M) > eps:
            Q = Q.mm( self.eye_M + (self.eye_M - QQ ) / 2 )
        return Q

class HouseholderSylvesterTransformation(SylvesterTransformation):

    """SylvesterTransformation where column-wise orthogonality of Q is ensured
        by a Householder operation.
    """
    
    def __init__(self, dim:int, device:str='cuda', training:bool=True):
        
        super().__init__(dim, dim, device, training)
        
    def get_num_params(self):
        """Get the number of parameters this Transformation requires"""
        # Q, R, R_hat, b
        return self.dim + self.num_hidden * self.num_hidden + self.num_hidden + self.num_hidden
    
    def unwrap_params(self, params, z):
        """Convert an array with params into vectors and matrices of this Transformation
        Args
        params -- array with parameters
        z      -- latent variable
        
        Returns Q, R and R_hat matrices and b bias vector
        """
        v = params[:self.dim]
        Q = self.make_Q_Householder(v, z)
        
        RR = params[self.dim * self.num_hidden : (self.dim + self.num_hidden) * self.num_hidden].reshape((self.num_hidden, self.num_hidden))
        R = torch.triu(RR)
        
        R_hat = torch.tril(RR).t()
        v = params[-2 * self.num_hidden : -self.num_hidden]
        mask = torch.diag(torch.ones_like(v))
        R_hat = mask * torch.diag(v) + (1. - mask) * R_hat

        b = params[-self.num_hidden:]
        return Q, R, R_hat, b
    
    def make_Q_Householder(self, v, z):
        """Create column-wise orthogonal matrix Q using Householder operation"""
        # TODO: implement
        raise NotImplementedError()

class TriagSylvesterTransformation(SylvesterTransformation):
    
    """SylvesterTransformation where Q is either an identity or a reverse
        permutation matrix. In this implementation Q is always identity,
        and the reverse permutation is achieved by alternating R and R_hat
        between upper and lower triangular forms
    """

    def __init__(self, dim:int, num_hidden:int=1, permute:bool=False, device:str='cuda', training:bool=True):
        
        super().__init__(dim, num_hidden, device, training)
        self.permute = permute
        self.Q = torch.eye(self.dim, self.num_hidden, device=device)
        
    def get_num_params(self):
        """Get the number of parameters this Transformation requires"""
        # Q, R, R_hat, b
        return self.num_hidden * self.num_hidden + self.num_hidden + self.num_hidden
    
    def unwrap_params(self, params, z):
        """Convert an array with params into vectors and matrices of this Transformation
        Args
        params -- array with parameters
        z      -- latent variable
        
        Returns Q, R and R_hat matrices and b bias vector
        """
        RR = params[:self.num_hidden * self.num_hidden].reshape((self.num_hidden, self.num_hidden))
        R = torch.triu(RR)
        
        R_hat = torch.tril(RR)
        if self.permute:
            R = R.t()
        else:
            R_hat = R_hat.t()
            
        v = params[-2 * self.num_hidden : -self.num_hidden]
        mask = torch.diag(torch.ones_like(v))
        R_hat = mask * torch.diag(v) + (1. - mask) * R_hat

        b = params[-self.num_hidden:]
        return self.Q, R, R_hat, b

class RadialTransformation(Transformation):

    """Radial Transformation"""

    def __init__(self, dim: int, training: bool=True):

        self.dim = dim
        self.training = training

    def get_num_params(self):
        """Get the number of parameters this Transformation requires"""
        return self.dim + 2

    def unwrap_params(self, params):
        """Convert an array with params into vectors of this Transformation
        Args
        params -- array with parameters
        z      -- latent variable

        Returns z0, alpha, beta
        """
        z0 = params[:self.dim]
        alpha = params[-2]
        beta = params[-1]
        if beta < -alpha:
            beta = -alpha + torch.log( 1 + torch.exp( beta ) )
        return z0.unsqueeze(0), alpha, beta

    def transform(self, zi, params):
        """Transform the latent variables using this transformation

        Args
        zi     -- variable which will be transfomed
        params -- parameters for this Transformation

        Returns transformed variable zi' of the same shape as the input zi
        """
        z0, alpha, beta = self.unwrap_params(params)
        r = torch.norm((zi - z0), p=2, dim=1, keepdim=True)
        return zi + beta * (self.h(r, alpha) * (zi - z0))

    def h(self, r, alpha):
        """Radial function"""
        return 1 / (alpha + r)

    def h_deriv(self, r, alpha):
        """Derivative of the radial function"""
        ff = self.h(r, alpha)
        return - ff * ff

    def det(self, zi, params):
        """Compute the jacobian of this transformation given zi and parameters"""
        z0, alpha, beta = self.unwrap_params(params)
        r = torch.norm((zi - z0), p=2, dim=1, keepdim=True)
        tmp = 1 + beta * self.h(r, alpha)
        return torch.clamp(tmp.pow(self.dim - 1) *
                           (tmp + beta * self.h_deriv(r, alpha) * r),
                           min=1e-7)



class PlanarTransformation(Transformation):

    """Planar Transformation"""

    def __init__(self, dim:int, training:bool=True):

        self.dim = dim
        self.h = nn.Tanh()
        self.training = training

    def get_num_params(self):
        """Get the number of parameters this Transformation requires"""
        return self.dim * 2 + 1

    def transform(self, zi, params):
        """Transform the latent variables using this transformation

        Args
        zi     -- variable which will be transfomed
        params -- parameters for this Transformation

        Returns transformed variable zi' of the same shape as the input zi
        """
        w, u, b = self.unwrap_params(params)
        return zi + u * self.h(F.linear(zi, w, b))

    def unwrap_params(self, params):
        """Convert an array with params into vectors of this Transformation
        Args
        params -- array with parameters
        z      -- latent variable

        Returns w, u, b
        """
        w = params[:self.dim]
        u = params[self.dim:-1]
        b = params[-1]
        if torch.dot(w, u) < -1:
            dotwu = torch.dot(w, u)
            u = u + (-1 + torch.log(1 + torch.exp(dotwu)) - dotwu) \
                            * w / torch.norm(w)
        return w.unsqueeze(0), u.unsqueeze(0), b

    def h_deriv(self, x):
        """Derivative of the activation function"""
        ff = self.h(x)
        return 1 - ff * ff

    def psi(self, z, w, u, b):
        return self.h_deriv(F.linear(z, w, b)) * w

    def det(self, z, params):
        """Compute the jacobian of this transformation given zi and parameters"""
        w, u, b = self.unwrap_params(params)
        return (1 + torch.mm(self.psi(z, w, u, b), u.t())).abs()


class NormalizingFlowNet:
    
    """Normalizing flow which comprises several Transformation's of the same type"""

    def __init__( self, transformation, dim:int, K:int, num_hidden:int=20, transformations=None ):
        """Init

        Args:
        transformation -- class of the transformation to be used in this flow
        dim            -- dimension of z
        K              -- flow length (=number of chained Transformation's)

        Kwargs:
        num_hidden      -- number of hidden units, SylvesterTranformation only
        transformations -- list with transformations. If provided, these 
                            transformations will be used instead of generating new
        """

        self.K = K
        self.dim = dim
        
        if transformations is None:
            if issubclass(transformation, SylvesterTransformation):
                if issubclass(transformation, TriagSylvesterTransformation):
                    transformations = [ transformation( dim, num_hidden, i%2==0 ) for i in range( K ) ]
                else:
                    transformations = [ transformation( dim, num_hidden ) for i in range( K ) ]
            else:
                transformations = [ transformation( dim ) for i in range( K ) ]
        self.flow = transformations
        self.nParams = self.flow[0].get_num_params()
        
    def get_last_log_det(self):
        """Get log determinant of the last Transformation in the flow"""
        return self.flow[-1].log_det
    
    def get_sum_log_det(self):
        """Get summed log jacobians of all Transformation's in the flow"""
        ret = 0
        for trans in self.flow:
            ret += trans.log_det
        return ret
        
    def forward( self, z, params ):
        """Pass z through all Transformation's in the flow
        
        Args:
        z       -- variable which will be transformed
        params  -- parameters for this flow

        Returns transformed z' of the same shape as z
        """
        for i, transf in enumerate( self.flow ):
            z = transf.forward(z, params[i])
        return z

class VAE(nn.Module):


    def __init__(self, input_dim, num_latent):
        super().__init__()

        self.input_dim = input_dim
        self.num_latent = num_latent
        
        ############
        #  Encoder
        ############

        # first fully connected layer
        self.fc1 = nn.Linear(input_dim, 400)
        self.dropout = nn.Dropout(p=0.25)
        # parallel layers
        # encode mean
        self.fc21_mean = nn.Linear(400, num_latent)
        # encode variance
        self.fc22_var = nn.Linear(400, num_latent)

        ############
        #  Decoder
        ############

        # two fully connected layer, i.e. simplified reverse of the encoder
        self.fc3 = nn.Linear(num_latent, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        """Convert input into parameters
        
        Args
        x -- input tensor
        """
        raise NotImplementedError()

    def reparameterize(self, mu, logvar):
        """Use mean and variance to generate latent variables z
        
        Args
        mu      -- mean from encode()
        logvar  -- log variance from encode()

        Returns latent variables z
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        """Transform latent variables back to original space
        Reconstructs the input from latent variables

        Args
        z -- latent variables
        Returns reconstructed input of the same shape as original input
        """
        h = F.relu(self.fc3(z))
        return self.fc4(h)

    def forward(self, x):
        """Forward pass
        Transforms the input into latent variables and reconstructs the input

        Args
        x  -- input tensor

        Returns recontructed input along with mean and variance of the latent variables
        """
        raise NotImplementedError()


class VAENF(VAE):


    def __init__(self, input_dim, num_latent, flow_transform, flow_latent, flow_len):
        super().__init__(input_dim, num_latent)

        # normalizing flow
        self.flow = NormalizingFlowNet(flow_transform, flow_latent, flow_len)
        self.energy_prediction = MLP_2layer(num_latent, int(num_latent/2), 1)
        # encode flow parameters ( parallel to mean and var )
        self.fc23_flow = nn.Linear(400, self.flow.nParams * flow_len)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        # returns mean, logvar and flow params
        return (self.fc21_mean(h), self.fc22_var(h),
                self.fc23_flow(h).mean(dim=0).chunk(self.flow.K, dim=0))

    def forward(self, x, energy=False):
        mu, logvar, params = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        z = self.flow.forward(z, params)
        if energy:
            return self.decode(z), mu, logvar, self.energy_prediction(z)
        else:
            return self.decode(z), mu, logvar