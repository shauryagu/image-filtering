import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def conv(I,f):
    """Apply same-sized convolution with a filter with zero-padding"""
    #Note that this is convolution! This is filtering but with f[::-1,::-1]
    return scipy.signal.convolve2d(I, f, mode='same', boundary='fill', fillvalue=0.0)

def nnUpsample(I,factor):
    """Nearest neighbor upsample an image by the given factor"""
    return np.kron(I, np.ones((factor,factor)))


#Fill in the filters to get the data to match
x, y = np.meshgrid(np.linspace(-1,1,3), np.linspace(-1,1,3))
sdev = 0.1
filter0 = np.diag([0,1,0])
filter1 = np.exp(-1*((x**2+y**2)/2*(sdev**2)))/(2*np.pi*(sdev**2)) * 7/111
filter2 = np.exp(-1*((x**2+y**2)/2*(sdev**2)))/(2*np.pi*(sdev**2)) / 142
filter3 = np.array(([0,0,0], [1,0,0], [0,0,0]))
filter4 = np.array(([-1,0,1],[-1,0,1],[-1,0,1]))
#np.array(([-2,0,2],[-4,0,4],[-2,0,2])) * (3/8)

filters = [filter0, filter1, filter2, filter3, filter4]

np.random.seed(442)
data = (plt.imread("filtermon/442.png").astype(np.float64)[:,:,0]<0.5).astype(np.float64)


plt.figure()
plt.imshow(nnUpsample(data,10))
plt.colorbar()
plt.savefig("input.png")


for fi, f in enumerate(filters):
    c = conv(data, f)
    sol = np.load("filtermon/output_%d.npy" % fi)

    matches = False

    if np.allclose(c,sol,rtol=1e-2,atol=1e-5):
        print("Filter %d matches" % fi)
        matches = True
    else:
        print("Filter %d doesn't match" % fi)
        
    plt.figure()
    fig, axs = plt.subplots(1,2)

    im = axs[0].imshow(nnUpsample(c,10))
    axs[0].set_title("Yours (%s)" % ("Match!" if matches else "No Match"))
    plt.colorbar(im, ax=axs[0])

    im = axs[1].imshow(nnUpsample(sol,10))
    axs[1].set_title("Target")
    plt.colorbar(im, ax=axs[1])

    plt.tight_layout()
    plt.savefig("comparison_%d.pdf" % (fi))

