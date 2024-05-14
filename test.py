import torch
from PIL import Image
from scipy.fftpack import dct, idct
import numpy as np
import torch.nn.functional as F

def dct(x, norm=None):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)
        (This code is copied from https://github.com/yuyang-long/SSA/blob/master/dct.py)

        Arguments:
            x: the input signal
            norm: the normalization, None or 'ortho'

        Return:
            the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = torch.fft.fft(v)

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
        V = Vc.real * W_r - Vc.imag * W_i
        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)

        return V

def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    (This code is copied from https://github.com/yuyang-long/SSA/blob/master/dct.py)

    Arguments:
        X: the input signal
        norm: the normalization, None or 'ortho'

    Return:
        the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape).real

def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    (This code is copied from https://github.com/yuyang-long/SSA/blob/master/dct.py)

    Arguments:
        x: the input signal
        norm: the normalization, None or 'ortho'

    Return:
        the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)

def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    (This code is copied from https://github.com/yuyang-long/SSA/blob/master/dct.py)

    Arguments:
        X: the input signal
        norm: the normalization, None or 'ortho'

    Return:
        the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)

def transform(x, **kwargs):
    """
    Use DCT to transform the input image from spatial domain to frequency domain,
    Use IDCT to transform the input image from frequency domain to spatial domain.

    Arguments:
        x: (N, C, H, W) tensor for input images
    """
    x_dct = dct_2d(x).cuda()

    # Get the amplitude and phase of the DCT coefficients
    x_amplitude = torch.abs(x_dct)
    x_dct_complex = torch.complex(x_dct, torch.zeros_like(x_dct))
    x_phase = torch.angle(x_dct_complex)

    # Perturb the amplitude and phase
    x_amplitude_perturbed = x_amplitude + torch.randn_like(x_amplitude) * 0.5
    x_phase_perturbed = x_phase + torch.randn_like(x_phase) * 0.5

    # Combine the perturbed amplitude and phase to get the perturbed DCT coefficients
    x_dct_perturbed = x_amplitude_perturbed * torch.exp(1j * x_phase_perturbed)

    # Convert the perturbed DCT coefficients back to real numbers
    x_dct_perturbed_real = torch.view_as_real(x_dct_perturbed)[..., 0]

    # Perform IDCT on the perturbed DCT coefficients
    x_idct = idct_2d(x_dct_perturbed_real)

    return x_idct


img_height = 224
img_width = 224

# Load the images
image1 = Image.open('data/images/ILSVRC2012_val_00000001.JPEG')
image2 = Image.open('data/images/ILSVRC2012_val_00000007.JPEG')

image1 = image1.resize((img_height, img_width)).convert('RGB')
image2 = image2.resize((img_height, img_width)).convert('RGB')
# Images for inception classifier are normalized to be in [-1, 1] interval.
image1 = np.array(image1).astype(np.float32)/255
image1 = torch.from_numpy(image1).permute(2, 0, 1)

image2 = np.array(image2).astype(np.float32)/255
image2 = torch.from_numpy(image2).permute(2, 0, 1)

image = torch.stack([image1, image2], dim=0)

x_trans = transform(image)

# x_trans = (x_trans.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
x_trans = (x_trans.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)

image1_trans = Image.fromarray(x_trans[0])
image2_trans = Image.fromarray(x_trans[1])

# Save the transformed images
image1_trans.save('image1_trans.jpg')
image2_trans.save('image2_trans.jpg')