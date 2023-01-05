import math,os,vg,time,ray
import numpy as np 
import matplotlib.pyplot as plt
# import tensorflow as tf 
import numpy.matlib as npm
import scipy.io as sio
from scipy.spatial import distance
# import h5py

# Constants
D2R = np.pi/180.0
R2D = 180.0/np.pi

def pr2t(p,R):
    T = np.zeros((4,4))
    T[:3,:3] = R
    T[:3,-1] = p
    T[-1,-1] = 1
    return T

def t2pr(T):
    p = T[:3,-1]
    R = T[:3,:3]
    return p,R
    
def t2p(T):
    p = T[:3,-1]
    return p

def t2r(T):
    R = T[:3,:3]
    return R

def invT(T):
    [p,R] = t2pr(T)
    return pr2t(-np.matmul(R.T,p),R.T)

def rpy2r(rpy_rad):
    """
    Get a rotation matrix from rpy in radian 
    """
    r,p,y = rpy_rad[0],rpy_rad[1],rpy_rad[2]
    cos_r,sin_r = np.cos(r),np.sin(r)
    cos_p,sin_p = np.cos(p),np.sin(p)
    cos_y,sin_y = np.cos(y),np.sin(y)
    R = [[cos_y*cos_p, -sin_y*cos_r+cos_y*sin_p*sin_r, sin_y*sin_r+cos_y*sin_p*cos_r],
         [sin_y*cos_p, cos_y*cos_r+sin_y*sin_p*sin_r, -cos_y*sin_r+sin_y*sin_p*cos_r],
         [-sin_p, cos_p*sin_r, cos_p*cos_r]]
    return np.array(R)


def r2rpy(R):
    """
    Get rpy in radian from a rotation matrix.
    """
    r = math.atan2(R[2,1],R[2,2])
    p = math.atan2(-R[2,0],math.sqrt(R[2,1]*R[2,1]+R[2,2]*R[2,2]))
    y = math.atan2(R[1,0],R[0,0])
    rpy_rad = np.array([r,p,y])
    return rpy_rad


def compute_ao_angle(T_own,T_opp):
    """
    Compute angle off in radian
    It is the angle between two vectors, v1 and v2 where
        v1: p_own to p_opp (i.e., p_opp-p_own)
        v2: own heading: x-axis of R_own (i.e., R_own[:,0]) 
    """
    [p_own,R_own] = t2pr(T_own)
    [p_opp,R_opp] = t2pr(T_opp)
    v_own2opp = p_opp-p_own
    v_own_heading = R_own[:,0] # x-axis is the heading
    ao_rad = vg.angle(v_own2opp,v_own_heading,units='rad')
    return ao_rad


def compute_ao_vector(T_own,T_opp,ao_type='world'):
    """
    Compute angle off vector 
        ao_type: 'world' / 'relative'
    """
    if ao_type is 'world':
        [p_own,R_own] = t2pr(T_own)
        [p_opp,R_opp] = t2pr(T_opp)
        ao_vector = p_opp-p_own
    elif ao_type is 'relative':
        invT_own = invT(T_own)
        T_own_relative = np.matmul(invT_own,T_own)
        T_opp_relative = np.matmul(invT_own,T_opp)
        [p_own_relative,R_own] = t2pr(T_own_relative)
        [p_opp_relative,R_opp] = t2pr(T_opp_relative)
        ao_vector = p_opp_relative-p_own_relative
    else:
        ao_vector = np.zeros(3)
        raise NameError('Unknown ao_type:[%s].'%(ao_type))
    return ao_vector


class BufferClass(object):
    """
    Buffer 
    """
    def __init__(self,dim=3,max_size=500):
        self.dim = dim
        self.max_size = max_size
        self.traj = np.zeros(shape=(int(self.max_size),int(self.dim)))
        self.cnt = 0
    def append(self,x):
        if self.cnt < self.max_size:
            self.traj[self.cnt,:] = x
            self.cnt += 1
        else:
            self.traj[:-1,:] = self.traj[1:,:]
            self.traj[self.cnt-1,:] = x # put to the last
    def get(self,ret_dim=None):
        if ret_dim is None:
            return self.traj[:self.cnt,:]
        else:
            return self.traj[:self.cnt,ret_dim]
    def get_count(self):
        return self.cnt
    def get_max_count(self):
        return self.max_size
    def is_full(self):
        return (self.cnt == self.max_size)


def plot_T(ax,T_plot,arrow_len=2000,text_str=None,text_color='k',text_size=12,xyz_cols=['r','g','b']):
    """
    Plot an axis 
    """
    [p_plot,R_plot] = t2pr(T_plot)
    x_arrow = p_plot + R_plot[:,0]*arrow_len
    y_arrow = p_plot + R_plot[:,1]*arrow_len
    z_arrow = p_plot + R_plot[:,2]*arrow_len
    ax.plot3D([p_plot[0],x_arrow[0]],
                [p_plot[1],x_arrow[1]],
                [p_plot[2],x_arrow[2]],
                c=xyz_cols[0],linewidth=2)[0]
    ax.plot3D([p_plot[0],y_arrow[0]],
                [p_plot[1],y_arrow[1]],
                [p_plot[2],y_arrow[2]],
                c=xyz_cols[1],linewidth=2)[0]
    ax.plot3D([p_plot[0],z_arrow[0]],
                [p_plot[1],z_arrow[1]],
                [p_plot[2],z_arrow[2]],
                c=xyz_cols[2],linewidth=2)[0]
    if text_str is not None:
        ax.text(p_plot[0],p_plot[1],p_plot[2],
                text_str,size=text_size,c=text_color)

def plot_line(ax,p1,p2,line_color='k',line_style='-'):
    """
    Plot a line
    """
    ax.plot3D([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],
                c=line_color,linestyle=line_style,linewidth=2)[0]


def gpu_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


t_start_tictoc = time.time()
def tic():
    global t_start_tictoc
    t_start_tictoc = time.time()
def toc(toc_str=None):
    global t_start_tictoc
    t_elapsed_sec = time.time() - t_start_tictoc
    if toc_str is None:
        print ("Elapsed time is [%.4f]sec."%
        (t_elapsed_sec))
    else:
        print ("[%s] Elapsed time is [%.4f]sec."%
        (toc_str,t_elapsed_sec))



class HyperSphereInsideSamplerClass(object):
    def __init__(self,name='hyper_sphere_inside_sampler',r=1,z_dim=16):
        self.name = name
        self.r = r
        self.z_dim = z_dim

    def sampler(self,n):
        z = np.random.randn(n,self.z_dim)
        z_norm = np.linalg.norm(z,axis=1)
        z_unit = z / npm.repmat(z_norm.reshape((-1,1)),1,self.z_dim) # on the surface of a hypersphere
        u = np.power(np.random.rand(n,1),(1/self.z_dim)*np.ones(shape=(n,1)))
        z_sphere = self.r * z_unit * npm.repmat(u,1,self.z_dim) # samples inside the hypersphere
        samples = z_sphere 
        return samples

    def plot(self,n=1000,tfs=20):
        samples = self.sampler(n=n)
        plt.figure(figsize=(6,6))
        plt.plot(samples[:,0],samples[:,1],'k.')
        plt.xlim(-self.r,self.r)
        plt.ylim(-self.r,self.r)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(self.name,fontsize=tfs)
        plt.show()    

class HyperSphereSurfaceSamplerClass(object):
    def __init__(self,name='hyper_sphere_surface_sampler',r=1,z_dim=16):
        self.name = name
        self.r = r
        self.z_dim = z_dim

    def sampler(self,n):
        z = np.random.randn(n,self.z_dim)
        z_norm = np.linalg.norm(z,axis=1)
        z_unit = z / npm.repmat(z_norm.reshape((-1,1)),1,self.z_dim) # on the surface of a hypersphere
        # u = np.power(np.random.rand(n,1),(1/self.z_dim)*np.ones(shape=(n,1)))
        # z_sphere = self.r * z_unit * npm.repmat(u,1,self.z_dim) # samples inside the hypersphere
        samples = z_unit 
        return samples

    def plot(self,n=1000,tfs=20):
        samples = self.sampler(n=n)
        plt.figure(figsize=(6,6))
        plt.plot(samples[:,0],samples[:,1],'k.')
        plt.xlim(-self.r,self.r)
        plt.ylim(-self.r,self.r)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(self.name,fontsize=tfs)
        plt.show()    

class HyperCubeSamplerClass(object):
    def __init__(self,name='hyper_cube_sampler',r=1,z_dim=16):
        self.name = name
        self.r = r
        self.z_dim = z_dim

    def sampler(self,n):
        samples = -1.0+2.0*np.random.rand(n,self.z_dim)
        samples = samples * self.r # scale
        return samples

    def plot(self,n=1000,tfs=20):
        samples = self.sampler(n=n)
        plt.figure(figsize=(6,6))
        plt.plot(samples[:,0],samples[:,1],'k.')
        margin_rate = 1.1
        plt.xlim(-self.r*margin_rate,self.r*margin_rate)
        plt.ylim(-self.r*margin_rate,self.r*margin_rate)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(self.name,fontsize=tfs)
        plt.show()    
        
class GaussianSamplerClass(object):
    def __init__(self,name='GaussianSampler',z_dim=16):
        self.name = name
        self.z_dim = z_dim
        self.r = 3

    def sampler(self,n):
        samples = np.random.randn(n,self.z_dim)
        return samples

    def plot(self,n=1000,tfs=20):
        samples = self.sampler(n=n)
        plt.figure(figsize=(6,6))
        plt.plot(samples[:,0],samples[:,1],'k.')
        margin_rate = 1.1
        plt.xlim(-self.r*margin_rate,self.r*margin_rate)
        plt.ylim(-self.r*margin_rate,self.r*margin_rate)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(self.name,fontsize=tfs)
        plt.show()    


class NormalizerClass(object):
    def __init__(self,raw_data,eps=1e-8):
        self.raw_data = raw_data
        self.eps      = eps
        self.mu       = np.mean(self.raw_data,axis=0)
        self.std      = np.std(self.raw_data,axis=0)
        self.std[np.where(self.std < 1.0)] = 1.0 # small variance exception handling
        self.nzd_data = self.get_nzdval(self.raw_data)
        self.org_data = self.get_orgval(self.nzd_data)
        self.max_err  = np.max(self.raw_data-self.org_data)
    def get_nzdval(self,data):
        n = data.shape[0]
        nzd_data = (data - np.tile(self.mu,(n,1))) / np.tile(self.std+self.eps,(n,1))
        return nzd_data
    def get_orgval(self,data):
        n = data.shape[0]
        org_data = data*np.tile(self.std+self.eps,(n,1))+np.tile(self.mu,(n,1))
        return org_data

class UnitCubeNormalizerClass(object):
    """
    Set all values to be within 0 and +1 (HyperCube)
    """
    def __init__(self,raw_data):
        self.raw_data = raw_data
        self.data_min = np.min(self.raw_data,axis=0)
        self.data_max = np.max(self.raw_data,axis=0)
        self.data_range = self.data_max - self.data_min
        self.data_range[np.where(self.data_range < 1e-6)] = 1.0 # small range exception handling
        self.nzd_data = self.get_nzdval(self.raw_data)
        self.org_data = self.get_orgval(self.nzd_data)
        self.max_err  = np.max(self.raw_data-self.org_data)
    def get_nzdval(self,data):
        n = data.shape[0]
        nzd_data = (data - np.tile(self.data_min,(n,1))) / np.tile(self.data_range,(n,1))
        return nzd_data
    def get_orgval(self,data):
        n = data.shape[0]
        org_data = data*np.tile(self.data_range,(n,1))+np.tile(self.data_min,(n,1))
        return org_data
    
    
def get_synthetic_2d_point(append_rate=0.0,xres=0.05,yres=0.05,x0=0.0,y0=0.0,PERMUTE=True,
                           EMPTY_CENTER=False,EMPTY_MIDLEFT=False,EMPTY_MIDRIGHT=False,EMPTY_OUTER=False):
    # Uniformly sample within mesh grid
    xs,ys = np.meshgrid(np.arange(0,5,xres),np.arange(0,5,yres),sparse=False)
    xys = np.dstack([xs,ys]).reshape(-1, 2)
    n_cnt = 0
    x = np.zeros_like(xys)
    for i_idx in range(xys.shape[0]):
        xy = xys[i_idx,:]
        if (((1<xy[0]) and (xy[0]<4)) and ((1<xy[1]) and (xy[1]<4))) and EMPTY_CENTER:
            DO_NOTHING = True
        elif (((2.5<xy[0]) and (xy[0]<4)) and ((1<xy[1]) and (xy[1]<4))) and EMPTY_MIDLEFT:
            DO_NOTHING = True
        elif (((1<xy[0]) and (xy[0]<2.5)) and ((1<xy[1]) and (xy[1]<4))) and EMPTY_MIDRIGHT:
            DO_NOTHING = True
        elif (((4.5<xy[0]) or (xy[0]<0.5)) or ((4.5<xy[1]) or (xy[1]<0.5))) and EMPTY_OUTER:
            DO_NOTHING = True
        else:
            x[n_cnt,:] = xy # append and increase counter
            n_cnt = n_cnt + 1
    x = x[:n_cnt,:]
    # Add more samples in a small region
    n_append = (int)(n_cnt*append_rate)
    np.random.seed(0)
    x_append = np.array([x0,y0]) + np.random.rand(n_append,2) # in [0,1]x[0,1]
    x = np.vstack((x,x_append))
    n = x.shape[0]
    # Random permute
    if PERMUTE:
        perm_idxs = np.random.permutation(n) 
        x = x[perm_idxs,:]
    # Get color
    c = get_color_with_first_and_second_coordinates(x)
    # c = np.ceil(c*10)/10 # quantize colors into 10 bins
    return x,c

def get_color_with_first_and_second_coordinates(x):
    c = np.concatenate((x[:,0:1],x[:,1:2]),axis=1)
    c = (c-np.min(x,axis=0))/(np.max(x,axis=0)-np.min(x,axis=0))
    r,g,b = 1.0-c[:,1:2],c[:,0:1],0.5-np.zeros_like(c[:,0:1])
    c = np.concatenate((r,g,b),axis=1)
    return c

def plot_x_and_color_scatter(x,c,title_str='',figsize=(6,6),s=None):
    """
    Scatter Plot in 2D with Different Colors
    x: [n x 2]
    c: [n x 3]
    """
    plt.figure(figsize=figsize)
    plt.scatter(x[:,0],x[:,1],c=c,s=s)
    plt.title(title_str,fontsize=20)
    plt.axis('equal')
    plt.show()

def plot_data_latent_reconstructed_spaces(x,c,z_samples_plot,z_real_plot,z_radius,org_x_recon_plot,
                                        s=None,title_str=None,figsize=(18,5),fs=28,
                                        axis_margin_rate=0.3):
    
    # Make figure
    fig = plt.figure(figsize=figsize)
    
    if c is None:
        c = 'k'
    
    # Training Data
    plt.subplot(141)
    plt.scatter(x[:,0],x[:,1],c=c,alpha=0.8,s=s)
    plt.title('Training Data',fontsize=fs)
    plt.gca().set_aspect('equal', adjustable='box') # axis equal
    x0_min,x0_max = np.min(x[:,0]),np.max(x[:,0])
    x1_min,x1_max = np.min(x[:,1]),np.max(x[:,1])
    x0_margin,x1_margin = (x0_max-x0_min)*axis_margin_rate/2,(x1_max-x1_min)*axis_margin_rate/2
    plt.axis([x0_min-x0_margin,x0_max+x0_margin,x1_min-x1_margin,x1_max+x1_margin])

    # Latent Space Prior
    plt.subplot(142)
    h_sample,=plt.plot(z_samples_plot[:,0],z_samples_plot[:,1],'.',
                        color=[0.5,0.5,0.5],alpha=0.5,markersize=1) # first and second axes of samples
    # h_real = plt.scatter(z_real_plot[:,0],z_real_plot[:,1],c=c,alpha=1.0,s=s) 
    plt.gca().set_aspect('equal', adjustable='box') # axis equal
    # plt.legend([h_sample,h_real],['Prior','Encoded'],fontsize=fs)
    plt.xlim(-(1+axis_margin_rate)*z_radius,(1+axis_margin_rate)*z_radius)
    plt.ylim(-(1+axis_margin_rate)*z_radius,(1+axis_margin_rate)*z_radius)
    plt.title('Latent Space',fontsize=fs)

    # Encoded points
    plt.subplot(143)
    plt.scatter(z_real_plot[:,0],z_real_plot[:,1],c=c,alpha=0.9,s=s) 
    plt.gca().set_aspect('equal', adjustable='box') # axis equal
    plt.xlim(-(1+axis_margin_rate)*z_radius,(1+axis_margin_rate)*z_radius)
    plt.ylim(-(1+axis_margin_rate)*z_radius,(1+axis_margin_rate)*z_radius)
    plt.title('Encoded Data',fontsize=fs)
    
    # Reconstructed 
    plt.subplot(144)
    plt.scatter(org_x_recon_plot[:,0],org_x_recon_plot[:,1],c=c,alpha=0.8,s=s)
    plt.gca().set_aspect('equal', adjustable='box') # axis equal
    plt.axis([x0_min-x0_margin,x0_max+x0_margin,x1_min-x1_margin,x1_max+x1_margin])
    plt.title('Reconstructed Data',fontsize=fs)

    if title_str is not None:
        fig.suptitle(title_str,fontsize=fs)
    plt.show()

def get_sub_idx_from_unordered_set(K,n_sel,rand_rate=0.0):
    n_total = K.shape[0]
    remain_idxs = np.arange(n_total)
    sub_idx = np.zeros((n_sel))
    sum_K_vec = np.zeros(n_total)
    for i_idx in range(n_sel):
        if i_idx == 0:
            sel_idx = np.random.randint(n_total)
        else:
            curr_K_vec = K[(int)(sub_idx[i_idx-1]),:] 
            sum_K_vec = sum_K_vec + curr_K_vec
            k_vals = sum_K_vec[remain_idxs]
            min_idx = np.argmin(k_vals)
            sel_idx = remain_idxs[min_idx] 
            if rand_rate > np.random.rand():
                rand_idx = np.random.choice(len(remain_idxs),1,replace=False)  
                sel_idx = remain_idxs[rand_idx] 
        sub_idx[i_idx] = (int)(sel_idx)
        remain_idxs = np.delete(remain_idxs,np.argwhere(remain_idxs==sel_idx))
    sub_idx = sub_idx.astype(np.int) # make it int
    return sub_idx

def load_mat(mat_name,VERBOSE=True):
    l = sio.loadmat(mat_name) # load a mat file
    names = []
    for key in l.keys(): # Get keys as 'names'
        if (key[0] is not '_'): 
            # vars()[key] = l[key] # automatically make 
            names.append(key)
    if VERBOSE:
        print ("[%s] loaded."%(mat_name))
    for n_idx,name in enumerate(names):
        # temp = vars()[name]
        temp = l[name]
        if VERBOSE:
            print (" [%02d] size of [%s] is %s."%(n_idx,name,temp.shape))
    return l,names

def load_mat_v73(mat_name,VERBOSE=True):
    #l = sio.loadmat(mat_name) # load a mat file
    
    f = h5py.File(mat_name)
    l = {}
    for k, v in f.items():
        l[k] = np.array(v).T
        
    names = []
    for key in l.keys(): # Get keys as 'names'
        if (key[0] is not '_'): 
            # vars()[key] = l[key] # automatically make 
            names.append(key)
    if VERBOSE:
        print ("[%s] loaded."%(mat_name))
    for n_idx,name in enumerate(names):
        # temp = vars()[name]
        temp = l[name]
        if VERBOSE:
            print (" [%02d] size of [%s] is %s."%(n_idx,name,temp.shape))
    return l,names

def plot_subset_sample(append_rate=1.0,batch_size=100,n_repeat=1,k_gains=[10,100,1000],figsize=[13,3],rand_rate=0.0):
    x,c = get_synthetic_2d_point(append_rate=append_rate)
    D_temp = distance.cdist(x,x,'sqeuclidean') # sqeuclidean / euclidean
    K_temp = np.exp(-100*D_temp)
    n = x.shape[0]
    batch_idx_temp = get_sub_idx_from_unordered_set(K_temp,n//3,rand_rate=0.0)
    nzr_x = NormalizerClass(raw_data=x[batch_idx_temp,:]) # init normalizer with subset
    x_train = nzr_x.get_nzdval(x)
    n_train = x_train.shape[0]
    D_raw = distance.cdist(x_train,x_train,'sqeuclidean') # cityblock / euclidean / sqeuclidean
    # plot_x_and_color_scatter(x,c,title_str='Dataset',s=1)
    
    fig, ax = plt.subplots(2,2+len(k_gains),figsize=figsize)
    n_bin = 5
    if n_repeat == 1:
        fig.suptitle('n = %d, sample_size = %d, eps = %.2f'%(n,batch_size,rand_rate),size=15)
    else:
        fig.suptitle('n = %d, sample_size = %dx%d, eps = %.2f'%(n,batch_size,n_repeat,rand_rate),size=15)
    ax[0,0].scatter(x[:,0],x[:,1],c=c,s=0.5)
    title_str = 'Dataset'
    ax[0,0].set_title(title_str,fontsize=15)
    ax[0,0].axis('equal')
    marg = 0.3
    ax[0,0].set(xlim=(0-marg,5+marg), ylim=(0-marg,5+marg))
    # histogram
    ax[1,0].hist2d(x[:,0],x[:,1],bins=n_bin,cmap=plt.cm.jet,normed=True)
    ax[1,0].axis('equal')
    marg = 0.3
    ax[1,0].set(xlim=(0-marg,5+marg), ylim=(0-marg,5+marg))
    # 
    for i_idx in range(n_repeat):
        batch_idx_temp = np.random.permutation(n)[:batch_size]
        ax[0,1].scatter(x[batch_idx_temp,0],x[batch_idx_temp,1],c=c[batch_idx_temp,:],s=0.5)
    title_str = 'Random Sampling'
    ax[0,1].set_title(title_str,fontsize=15)
    ax[0,1].axis('equal')
    marg = 0.3
    ax[0,1].set(xlim=(0-marg,5+marg), ylim=(0-marg,5+marg))
    # histogram
    ax[1,1].hist2d(x[batch_idx_temp,0],x[batch_idx_temp,1],bins=n_bin,cmap=plt.cm.jet,normed=True)
    ax[1,1].axis('equal')
    marg = 0.3
    ax[1,1].set(xlim=(0-marg,5+marg), ylim=(0-marg,5+marg))
    for k_idx,k_gain in enumerate(k_gains): # for different k_gains
        K = np.exp(-k_gain*D_raw)
        for i_idx in range(n_repeat):
            batch_idx_temp = get_sub_idx_from_unordered_set(K,batch_size,rand_rate=rand_rate) # <= subsampling
            ax[0,k_idx+2].scatter(x[batch_idx_temp,0],x[batch_idx_temp,1],c=c[batch_idx_temp,:],s=0.5)
        title_str = 'k_gain = %d'%(k_gain)
        ax[0,k_idx+2].set_title(title_str,fontsize=13)
        ax[0,k_idx+2].axis('equal')
        marg = 0.3
        ax[0,k_idx+2].set(xlim=(0-marg,5+marg), ylim=(0-marg,5+marg))
        # histogram
        ax[1,k_idx+2].hist2d(x[batch_idx_temp,0],x[batch_idx_temp,1],bins=n_bin,cmap=plt.cm.jet,normed=True)
        ax[1,k_idx+2].axis('equal')
        marg = 0.3
        ax[1,k_idx+2].set(xlim=(0-marg,5+marg), ylim=(0-marg,5+marg))
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    plt.show()

    
def get_swiss_roll(xy):
    n = xy.shape[0]
    xyz = np.zeros(shape=(n,3))
    for i_idx in range(n):
        xi,yi = xy[i_idx,0],xy[i_idx,1]
        # w,r = 2*np.pi*0.15, 2.5
        w,r = 2*np.pi*0.2, 5.0*np.sqrt(yi)
        theta = w*yi
        xnew,ynew,znew = xi,r*np.cos(theta),r*np.sin(theta)
        xyz[i_idx,:] = np.array((xnew,ynew,znew))
    return xyz    

def cdist_safe(x,metric='sqeuclidean',n_th=10000):
    n = x.shape[0]
    D = np.inf*np.ones(shape=(n,n))
    for i_idx in range(np.ceil(n/n_th).astype(np.int)):
        fr,to = i_idx*n_th,min((i_idx+1)*n_th,n)
        x_i = x[fr:to,:]
        D_i = distance.cdist(x_i,x_i,metric=metric)
        D[fr:to,fr:to] = D_i
    return D

def get_toy_datasets_domain_A_and_B(append_rate_A=0.0,append_rate_B=0.0,n_glue=30):
    # Domain A dataset (2-D)
    xy_A,c_A = get_synthetic_2d_point(append_rate=append_rate_A,xres=0.05,yres=0.05,x0=0.0,y0=0.0,PERMUTE=True,
                                      EMPTY_CENTER=False,EMPTY_MIDLEFT=True,EMPTY_MIDRIGHT=False)
    # Domain B dataset (3-D)
    xy_B,c_B = get_synthetic_2d_point(append_rate=append_rate_B,xres=0.2,yres=0.01,x0=4.0,y0=4.0,PERMUTE=True,
                                      EMPTY_CENTER=False,EMPTY_MIDLEFT=False,EMPTY_MIDRIGHT=True)
    xyz_B = get_swiss_roll(xy_B) # <= actual 3D dataset for domain B
    # Ccorrespdondnig dataset 
    xy_AB,c_AB = get_synthetic_2d_point(append_rate=0.0,xres=0.05,yres=0.05,x0=0.0,y0=0.0,PERMUTE=True,
                                        EMPTY_CENTER=True,EMPTY_MIDLEFT=False,EMPTY_MIDRIGHT=False,EMPTY_OUTER=True)
    xyz_AB = get_swiss_roll(xy_AB)
    n_A,n_B,n_AB = xy_A.shape[0],xyz_B.shape[0],xyz_AB.shape[0]
    # Normalizers
    nzr_xy = NormalizerClass(raw_data=xy_AB)
    nzr_xyz = NormalizerClass(raw_data=xyz_AB)
    # Normalize data
    nzd_xy_A = nzr_xy.get_nzdval(xy_A)
    nzd_xyz_B = nzr_xyz.get_nzdval(xyz_B)
    nzd_xy_AB = nzr_xy.get_nzdval(xy_AB)
    nzd_xyz_AB = nzr_xyz.get_nzdval(xyz_AB)
    # Pairwise distances and kernel matrices
    D_xy_AB = cdist_safe(nzd_xy_AB)
    K_xy_AB = np.exp(-1000*D_xy_AB)
    D_xy_A = cdist_safe(nzr_xy.get_nzdval(xy_A))
    D_xyz_B = cdist_safe(nzr_xy.get_nzdval(xy_B)) # cdist_safe(nzr_xyz.get_nzdval(xyz_B))
    K_xy_A,K_xyz_B = np.exp(-1000*D_xy_A),np.exp(-1000*D_xyz_B)
    # Glue
    batch_idx_glue = np.random.permutation(xy_AB.shape[0])[:n_glue]
    xy_A_glue,xyz_B_glue,c_glue = xy_AB[batch_idx_glue,:],xyz_AB[batch_idx_glue,:],c_AB[batch_idx_glue,:]
    nxd_xy_A_glue = nzr_xy.get_nzdval(xy_A_glue)
    nxd_xyz_B_glue = nzr_xyz.get_nzdval(xyz_B_glue)
    return xy_A,c_A,xyz_B,c_B,xy_AB,xyz_AB,c_AB,xy_A_glue,xyz_B_glue,c_glue,\
            nzr_xy,nzr_xyz,nzd_xy_A,nzd_xyz_B,nzd_xy_AB,nzd_xyz_AB,nxd_xy_A_glue,nxd_xyz_B_glue,\
            D_xy_A,D_xyz_B,D_xy_AB,K_xy_A,K_xyz_B,K_xy_AB

def plot_domain_A_and_B(xy_A,c_A,xyz_B,c_B,xy_A_glue,xyz_B_glue,c_glue):
    # Plot a) Domain A, b) Domain B
    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(121)
    ax.scatter(x=xy_A[:,0],y=xy_A[:,1],c=c_A,edgecolors=c_A,s=1)
    ax.scatter(x=xy_A_glue[:,0],y=xy_A_glue[:,1],c=c_glue,edgecolors='k',s=30)
    ax.axis('scaled')
    ax.set_xlim(0,+5);ax.set_ylim(0,+5)
    ax.set_xlabel('X',fontsize=13);ax.set_ylabel('Y',fontsize=13)
    # ax.set_axis_off()
    ax.set_title('Domain A',fontsize=15)
    ax = fig.add_subplot(122,projection='3d')
    ax.scatter(xs=xyz_B[:,0],ys=xyz_B[:,1],zs=xyz_B[:,2],c=c_B,edgecolor=c_B,
               s=3.0,alpha=0.2,marker='o',cmap="rgb")
    ax.scatter(xs=xyz_B_glue[:,0],ys=xyz_B_glue[:,1],zs=xyz_B_glue[:,2],c=c_glue,edgecolor='k',
               s=30.0,alpha=1.0,marker='o',cmap="rgb")
    #ax.axis('scaled')
    ax.set_title('Domain B',fontsize=15);ax.view_init(30,10)
    ax.set_xlabel('X',fontsize=13);ax.set_ylabel('Y',fontsize=13);ax.set_zlabel('Z',fontsize=13)
    plt.tight_layout();plt.show()
    
    
def suppress_tf_warning():
    import tensorflow as tf
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # tf.logging.set_verbosity(tf.logging.ERROR)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    

def row_norm(a):
    return np.apply_along_axis(np.linalg.norm, 1, a)

def row_norm_sq(a):
    x = np.apply_along_axis(np.linalg.norm, 1, a)
    return x**2

def row_mean_square(a):
    return np.mean(np.square(a),axis=1)
    
def print_stats(x,x_str):
    x_mean,x_std = np.mean(x),np.std(x) 
    x_min,x_max = np.min(x),np.max(x)
    print (" [%s] mean:[%.3f] std:[%.3f] min:[%.3f] max:[%.3f]."%
           (x_str,x_mean,x_std,x_min,x_max))
    
def get_top_k_idx(a,k):
    """
    Get the indices of top-k values
    """
    return  (-a).argsort()[:k]
    
def get_histogram(vals,n_bin=100,bin_range=None,NORMALIZE=True,
                  SMOOTH=False,window_size=5,poly_order=3):
    """
    Get (normalized) Histogram 
    """
    if bin_range is None:
        min_val,max_val = np.min(vals),np.max(vals)
    else:
        min_val,max_val = bin_range[0],bin_range[1]
    bins = np.linspace(min_val,max_val,num=n_bin)
    intv_len = (max_val-min_val)/(n_bin-1)
    hist,_ = np.histogram(vals,bins=bins)
    if NORMALIZE:
        hist = hist / np.sum(hist) / intv_len
    bin_mids = bins[:-1] + 0.5*intv_len
    if SMOOTH:
        from scipy.signal import savgol_filter
        hist = savgol_filter(hist,window_size,poly_order)
        hist[0] = 0.0
    return hist,bin_mids

def get_percentile(vals,rate=0.5,n_bin=100):
    """
    Get percentile (100*rate) of given values
    """
    hist,bin_mids = get_histogram(vals,n_bin,NORMALIZE=True)
    cum_hist = np.cumsum(hist/np.sum(hist)) 
    percentile_idx = np.searchsorted(cum_hist,rate,side="left") 
    percentile_val = bin_mids[percentile_idx]
    return percentile_val
    
def get_maximally_distinguishing_threhold(vals_a,vals_b,n_bin=1000):
    """
    Get the maximally distinguishing threshold of two values
    """
    vals = np.concatenate((vals_a,vals_b))
    min_val,max_val = np.min(vals),np.max(vals)
    bins = np.linspace(min_val,max_val,num=n_bin)
    intv_len = (max_val-min_val)/(n_bin-1)
    bin_mids = bins[:-1] + 0.5*intv_len
    hist_a,_ = np.histogram(vals_a,bins=bins)
    hist_b,_ = np.histogram(vals_b,bins=bins)
    cum_hist_a = np.cumsum(hist_a/np.sum(hist_a)) 
    cum_hist_b = np.cumsum(hist_b/np.sum(hist_b))
    
    scores = np.zeros(n_bin-1)
    for i_idx in range(n_bin-1):
        p_a_m = cum_hist_a[i_idx]
        p_a_p = 1-p_a_m
        p_b_m = cum_hist_b[i_idx]
        p_b_p = 1-p_b_m
        # scores[i_idx] = max(p_a_m+p_b_p, p_a_p+p_b_m)/2.0
        scores[i_idx] = abs(p_a_m-p_b_m)
    max_idx = np.argmax(scores)
    mdt = bin_mids[max_idx]
    mdp = scores[max_idx] # maximally distinguished prob
    return mdt,mdp

def r_sq(x1,x2,x_range=1.0,invlen=5.0):
    """
    Scaled pairwise dists 
    """
    x1_scaled,x2_scaled = invlen*x1/x_range,invlen*x2/x_range
    D_sq = distance.cdist(x1_scaled,x2_scaled,'sqeuclidean') 
    return D_sq
    
def k_m52(x1,x2,x_range=1.0,gain=1.0,invlen=5.0):
    """
    Automatic relevance determination (ARD) Matern 5/2 kernel
    """
    R_sq = r_sq(x1,x2,x_range=x_range,invlen=invlen)
    K = gain*(1+np.sqrt(5*R_sq)+(5.0/3.0)*R_sq)*np.exp(-np.sqrt(5*R_sq))
    return K
    
def gp_m52(x,y,x_test,gain=1.0,invlen=5.0,eps=1e-8):
    """
    Gaussian process with ARD Matern 5/2 Kernel
    """
    x_range = np.max(x,axis=0)-np.min(x,axis=0)
    k_test = k_m52(x_test,x,x_range=x_range,gain=gain,invlen=invlen)
    K = k_m52(x,x,x_range=x_range,gain=gain,invlen=invlen)
    n = x.shape[0]
    inv_K = np.linalg.inv(K+eps*np.eye(n))
    mu_y = np.mean(y)
    # print(np.matmul(k_test,inv_K), y-mu_y)
    mu_test = np.matmul(np.matmul(k_test,inv_K),y-mu_y)+mu_y
    var_test = (gain-np.diag(np.matmul(np.matmul(k_test,inv_K),k_test.T))).reshape((-1,1))
    return mu_test,var_test
    
def cos_exp_square(x):
    """
    An example function f(x) = -cos(2*pi*x)*exp(-x^2)
    """
    y = -np.cos(2*np.pi*x)*np.exp(-x**2)
    return y

def cos_exp_square_nd(x):
    """
    f(x) = -cos(2*pi*x)*exp(-x^2)
    where x is a d-dimensional vector 
    """
    x_rownorm = np.linalg.norm(x,axis=1).reshape((-1,1))
    y = -np.cos(2*np.pi*x_rownorm)*np.exp(-x_rownorm**2)
    return y

def cos_exp_abs_nd(x):
    """
    f(x) = -cos(2*pi*x)*exp(-|x|)
    where x is a d-dimensional vector 
    """
    x_rownorm = np.linalg.norm(x,axis=1).reshape((-1,1))
    y = -np.cos(2*np.pi*x_rownorm)*np.exp(-np.abs(x_rownorm)**1)
    return y

def Phi(x):
    """
    CDF of Gaussian
    """
    from scipy.special import erf
    return (1.0 + erf(x / math.sqrt(2.0))) / 2.0

def acquisition_function(x_bo,y_bo,x_test,SCALE_Y=True,gain=1.0,invlen=5.0,eps=1e-6):
    """
    Acquisition function of Bayesian Optimization with Expected Improvement
    """
    from scipy.stats import norm
    
    if SCALE_Y:
        y_bo_scaled = np.copy(y_bo)
        y_bo_mean = np.mean(y_bo_scaled)
        y_bo_scaled = y_bo_scaled - y_bo_mean
        y_min,y_max = np.min(y_bo_scaled), np.max(y_bo_scaled)
        y_range = y_max - y_min
        y_bo_scaled = 2.0 * y_bo_scaled / y_range
    else:
        y_bo_scaled = np.copy(y_bo)
    
    mu_test,var_test = gp_m52(x_bo,y_bo_scaled,x_test,gain=gain,invlen=invlen,eps=eps)
    gamma = (np.min(y_bo_scaled) - mu_test)/np.sqrt(var_test)
    a_ei = 2.0 * np.sqrt(var_test) * (gamma*Phi(gamma) + norm.pdf(mu_test,0,1))
    
    if SCALE_Y:
        mu_test = 0.5 * y_range * mu_test + y_bo_mean
    
    return a_ei,mu_test,var_test
    
    
def x_sampler(n_sample,x_minmax):
    """
    Sample x as a list from the input domain 
    """
    x_samples = []
    for _ in range(n_sample):
        x_samples.append(
            x_minmax[:,0]+(x_minmax[:,1]-x_minmax[:,0])*np.random.rand(1,x_minmax.shape[0]))
    return x_samples
def get_best_xy(x_data,y_data):
    """
    Get the current best solution
    """
    min_idx = np.argmin(y_data)
    return x_data[min_idx,:].reshape((1,-1)),y_data[min_idx,:].reshape((1,-1))
    
def run_bocd(
    func_eval,x_minmax,n_random=1,n_bo=1,n_cd=1,n_data_max=100,n_worker=10,seed=0,
    n_sample_for_bo=2000,save_folder=''):
    
    """
    Run BO-CD
    """
    @ray.remote
    def func_eval_ray(x):
        """
        Eval with Ray
        """
        y = func_eval(x)
        return y
    
    np.random.seed(seed=seed) # fix seed 
    # First start
    x_dim = x_minmax.shape[0]
    x_evals = x_sampler(n_sample=n_worker,x_minmax=x_minmax)
    evals = [func_eval_ray.remote(x=x_eval) for x_eval in x_evals]
    y_evals = ray.get(evals)
    x_data,y_data = np.asarray(x_evals)[:,0,:],np.asarray(y_evals)[:,0,:]
    iclk_bocd = time.time()
    
    
    print ( "\nStart Bayesian Optimization with Coordinate Descent with [%d] Workers."%(n_worker) )
    print ( " x_dim:[%d] n_random:[%d] n_bo:[%d] n_cd:[%d]."%(x_dim,n_random,n_bo,n_cd) )
    print ( " seed:[%d] n_sample_for_bo:[%d]."%(seed,n_sample_for_bo) )
    if save_folder:
        print ( " Optimization results will be saved to [%s]."%(save_folder) )
    print ( "\n" )
    
    while True:
        # Random sample
        iclk_random = time.time()
        for rd_idx in range(n_random):
            x_evals = x_sampler(n_sample=n_worker,x_minmax=x_minmax)
            evals = [func_eval_ray.remote(x=x_eval) for x_eval in x_evals] # EVALUATE
            y_evals = ray.get(evals)
            x_random,y_random = np.asarray(x_evals)[:,0,:],np.asarray(y_evals)[:,0,:]
            x_data = np.concatenate((x_data,x_random),axis=0)
            y_data = np.concatenate((y_data,y_random),axis=0)
        esec_random = time.time() - iclk_random
        esec_bocd = time.time() - iclk_bocd
        # Plot Random samples
        if n_random > 0:
            x_sol,y_sol = get_best_xy(x_data,y_data)
            print("[%.1f]sec [%d/%d] RS took [%.1f]sec. Current best x:%s best y:[%.3f]"%
                  (esec_bocd,x_data.shape[0],n_data_max,esec_random,x_sol,y_sol))
        # Terminate Condition
        if x_data.shape[0] >= n_data_max: break

        # Bayesian Optimization
        iclk_bo = time.time()
        for bo_idx in range(n_bo):
            # Constant liar model for parallelizing BO
            x_evals,x_data_copy,y_data_copy = [],np.copy(x_data),np.copy(y_data)
            for _ in range(n_worker):
                x_checks = np.asarray(x_sampler(n_sample_for_bo,x_minmax=x_minmax))[:,0,:]
                a_ei,mu_checks,_ = acquisition_function(
                    x_data_copy,y_data_copy,x_checks,gain=1.0,invlen=5.0,eps=1e-6) # get the acquisition values 
                max_idx = np.argmax(a_ei) # select the one with the highested value 
                # As we cannot get the actual y_eval from the real evaluation, we use the constant liar model
                # that uses the GP mean to approximate the actual evaluation value. 
                x_liar,y_liar = x_checks[max_idx,:].reshape((1,-1)),mu_checks[max_idx].reshape((1,-1))
                # Append
                x_data_copy = np.concatenate((x_data_copy,x_liar),axis=0)
                y_data_copy = np.concatenate((y_data_copy,y_liar),axis=0)
                x_evals.append(x_liar) # append the inputs to evaluate 
                
            # Evaluate k candidates in one scoop 
            evals = [func_eval_ray.remote(x=x_eval) for x_eval in x_evals] # EVALUATE
            x_bo = np.asarray(x_evals)[:,0,:]
            y_bo = np.asarray(ray.get(evals))[:,0,:]
            # Concatenate BO results
            x_data = np.concatenate((x_data,x_bo),axis=0)
            y_data = np.concatenate((y_data,y_bo),axis=0)
        esec_bo = time.time() - iclk_bo
        esec_bocd = time.time() - iclk_bocd
        # Plot BO
        if n_bo > 0:
            x_sol,y_sol = get_best_xy(x_data,y_data)
            print("[%.1f]sec [%d/%d] BO took [%.1f]sec. Current best x:%s best y:[%.3f]"%
                  (esec_bocd,x_data.shape[0],n_data_max,esec_bo,x_sol,y_sol))
        # Terminate Condition
        if x_data.shape[0] >= n_data_max: break

        # Coordinate Descent 
        iclk_cd = time.time()
        for cd_idx in range(n_cd):
            x_sol,y_sol = get_best_xy(x_data,y_data)
            for d_idx in range(x_dim): # for each dim 
                x_minmax_d = x_minmax[d_idx,:]
                x_sample_d = x_minmax_d[0]+(x_minmax_d[1]-x_minmax_d[0])*np.random.rand(n_worker)
                x_sample_d[0] = x_sol[0,d_idx]
                x_temp,x_evals = x_sol,[]
                for i_idx in range(n_worker):
                    x_temp[0,d_idx] = x_sample_d[i_idx]
                    x_evals.append(np.copy(x_temp.reshape((1,-1))))
                # Evaluate k candidates in one scoop
                evals = [func_eval_ray.remote(x=x_eval) for x_eval in x_evals] # EVALUATE
                y_evals = ray.get(evals)
                # Update the current coordinate
                min_idx = np.argmin(np.asarray(y_evals)[:,0,0])
                x_sol[0,d_idx] = x_sample_d[min_idx]
                # Concatenate CD results
                x_cd,y_cd = np.asarray(x_evals)[:,0,:],np.asarray(y_evals)[:,0,:]
                x_data = np.concatenate((x_data,x_cd),axis=0)
                y_data = np.concatenate((y_data,y_cd),axis=0)
        esec_cd = time.time() - iclk_cd
        esec_bocd = time.time() - iclk_bocd
        # Plot CD
        if n_cd > 0:
            x_sol,y_sol = get_best_xy(x_data,y_data)
            print("[%.1f]sec [%d/%d] CD took [%.1f]sec. Current best x:%s best y:[%.3f]"%
                  (esec_bocd,x_data.shape[0],n_data_max,esec_cd,x_sol,y_sol))
        
        # Save intermediate resutls
        if save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                print ( "[%s] created."%(save_folder) )
            # Save
            npz_path = os.path.join(save_folder,'bocd_result.npz')
            np.savez(npz_path, x_data=x_data,y_data=y_data,
                     x_minmax=x_minmax,n_random=n_random,n_bo=n_bo,n_cd=n_cd,
                     n_data_max=n_data_max,n_worker=n_worker,seed=seed,
                     n_sample_for_bo=n_sample_for_bo)
            print ( "[%s] saved."%(npz_path) )
            
        # Terminate Condition
        if x_data.shape[0] >= n_data_max: break
            
    return x_data,y_data
    
def sample_from_best_voronoi_cell(x_data,y_data,x_minmax,n_sample,
                                  max_try_sbv=5000,center_coef=0.1):
    """
    Sample from the Best Voronoi Cell for Voronoi Optimistic Optimization (VOO)
    """
    x_dim = x_minmax.shape[0]
    idx_min_voo = np.argmin(y_data) # index of the best x
    x_evals = []
    for _ in range(n_sample):
        n_try,x_tried,d_tried = 0,np.zeros((max_try_sbv,x_dim)),np.zeros((max_try_sbv,1))
        x_sol,y_sol = get_best_xy(x_data,y_data)
        while True:
            # if n_try < (max_try_sbv/2):
            if np.random.rand() < 0.5:
                x_sel = x_sampler(n_sample=1,x_minmax=x_minmax)[0] # random sample
            else:
                # Gaussian sampling centered at x_sel
                x_sel = x_sol + center_coef*np.random.randn(*x_sol.shape)*np.sqrt(
                    1e-6+x_minmax[:,1].reshape((1,-1))-x_minmax[:,0].reshape((1,-1))
                    )
            dist_sel = r_sq(x_data,x_sel)
            idx_min_sel = np.argmin(dist_sel)
            if idx_min_sel == idx_min_voo: 
                break
            # Sampling the best vcell might took a lot of time 
            x_tried[n_try,:] = x_sel
            d_tried[n_try,:] = r_sq(x_data[idx_min_voo,:].reshape((1,-1)),x_sel)
            n_try += 1 # increase tick
            if n_try >= max_try_sbv:
                idx_min_tried = np.argmin(d_tried) # find the closest one 
                x_sel = x_tried[idx_min_tried,:].reshape((1,-1))
                break
        x_evals.append(x_sel) # append 
    return x_evals


def run_bavoo(
    func_eval,x_minmax,n_random=1,n_bo=1,n_voo=1,n_cd=1,
    n_data_max=100,n_worker=10,seed=0,
    n_sample_for_bo=2000,max_try_sbv=5000,center_coef=0.1,
    save_folder='',
    saved_data=''):
    
    """
    Run BAyesian-VOO
    """
    @ray.remote
    def func_eval_ray(x):
        """
        Eval with Ray
        """
        y = func_eval(x)
        return y
    
    np.random.seed(seed=seed) # fix seed 
    # First start
    x_dim = x_minmax.shape[0]    
    if saved_data:
        x_data, y_data = saved_data['x_data'], saved_data['y_data']
    else:
        x_evals = x_sampler(n_sample=n_worker,x_minmax=x_minmax)
        evals = [func_eval_ray.remote(x=x_eval) for x_eval in x_evals]
        y_evals = ray.get(evals)
        x_data, y_data = np.asarray(x_evals)[:,0,:],np.asarray(y_evals)[:,0,:]
    iclk_total = time.time()
    
    
    print ( "\nStart Bayesian VOO with [%d] Workers."%(n_worker) )
    print ( " x_dim:[%d] n_random:[%d] n_bo:[%d] n_cd:[%d]."%(x_dim,n_random,n_bo,n_cd) )
    print ( " seed:[%d] n_sample_for_bo:[%d]."%(seed,n_sample_for_bo) )
    if save_folder:
        print ( " Optimization results will be saved to [%s]."%(save_folder) )
    print ( "\n" )
    
    while True:
        # Random sample
        iclk_random = time.time()
        for rd_idx in range(n_random):
            x_evals = x_sampler(n_sample=n_worker,x_minmax=x_minmax)
            evals = [func_eval_ray.remote(x=x_eval) for x_eval in x_evals] # EVALUATE
            y_evals = ray.get(evals)
            x_random,y_random = np.asarray(x_evals)[:,0,:],np.asarray(y_evals)[:,0,:]
            x_data = np.concatenate((x_data,x_random),axis=0)
            y_data = np.concatenate((y_data,y_random),axis=0)
        esec_random = time.time() - iclk_random
        esec_total = time.time() - iclk_total
        # Plot Random samples
        if n_random > 0:
            x_sol,y_sol = get_best_xy(x_data,y_data)
            print("[%.1f]sec [%d/%d] RS took [%.1f]sec. Current best x:%s best y:[%.3f]"%
                  (esec_total,x_data.shape[0],n_data_max,esec_random,x_sol,y_sol))
        # Terminate Condition
        if x_data.shape[0] >= n_data_max: break

        # Bayesian Optimization
        iclk_bo = time.time()
        for bo_idx in range(n_bo):
            # Constant liar model for parallelizing BO
            x_evals,x_data_copy,y_data_copy = [],np.copy(x_data),np.copy(y_data)
            for _ in range(n_worker):
                x_checks = np.asarray(x_sampler(n_sample_for_bo,x_minmax=x_minmax))[:,0,:]
                a_ei,mu_checks,_ = acquisition_function(
                    x_data_copy,y_data_copy,x_checks,gain=1.0,invlen=5.0,eps=1e-6) # get the acquisition values 
                max_idx = np.argmax(a_ei) # select the one with the highested value 
                # As we cannot get the actual y_eval from the real evaluation, we use the constant liar model
                # that uses the GP mean to approximate the actual evaluation value. 
                x_liar,y_liar = x_checks[max_idx,:].reshape((1,-1)),mu_checks[max_idx].reshape((1,-1))
                # Append
                x_data_copy = np.concatenate((x_data_copy,x_liar),axis=0)
                y_data_copy = np.concatenate((y_data_copy,y_liar),axis=0)
                x_evals.append(x_liar) # append the inputs to evaluate 
                
            # Evaluate k candidates in one scoop 
            evals = [func_eval_ray.remote(x=x_eval) for x_eval in x_evals] # EVALUATE
            x_bo = np.asarray(x_evals)[:,0,:]
            y_bo = np.asarray(ray.get(evals))[:,0,:]
            # Concatenate BO results
            x_data = np.concatenate((x_data,x_bo),axis=0)
            y_data = np.concatenate((y_data,y_bo),axis=0)
        esec_bo = time.time() - iclk_bo
        esec_total = time.time() - iclk_total
        # Plot BO
        if n_bo > 0:
            x_sol,y_sol = get_best_xy(x_data,y_data)
            print("[%.1f]sec [%d/%d] BO took [%.1f]sec. Current best x:%s best y:[%.3f]"%
                  (esec_total,x_data.shape[0],n_data_max,esec_bo,x_sol,y_sol))
        # Terminate Condition
        if x_data.shape[0] >= n_data_max: break
        
        # Voronoi Optimistic Optimization
        iclk_voo = time.time()
        for voo_idx in range(n_voo):
            # Get input points to eval from sampling the best Voronoi cell 
            x_evals = sample_from_best_voronoi_cell(
                x_data,y_data,x_minmax,n_sample=n_worker,max_try_sbv=max_try_sbv,
                center_coef=center_coef)
            # Evaluate
            evals = [func_eval_ray.remote(x=x_eval) for x_eval in x_evals] # EVALUATE
            y_evals = ray.get(evals)
            x_sbv,y_sbv = np.asarray(x_evals)[:,0,:],np.asarray(y_evals)[:,0,:]
            x_data = np.concatenate((x_data,x_sbv),axis=0)
            y_data = np.concatenate((y_data,y_sbv),axis=0)
        esec_voo = time.time() - iclk_voo
        esec_total = time.time() - iclk_total
        # Plot VOO
        if n_voo > 0:
            x_sol,y_sol = get_best_xy(x_data,y_data)
            print("[%.1f]sec [%d/%d] VOO took [%.1f]sec. Current best x:%s best y:[%.3f]"%
                  (esec_total,x_data.shape[0],n_data_max,esec_voo,x_sol,y_sol))
        

        # Coordinate Descent 
        iclk_cd = time.time()
        for cd_idx in range(n_cd):
            x_sol,y_sol = get_best_xy(x_data,y_data)
            for d_idx in range(x_dim): # for each dim 
                x_minmax_d = x_minmax[d_idx,:]
                x_sample_d = x_minmax_d[0]+(x_minmax_d[1]-x_minmax_d[0])*np.random.rand(n_worker)
                x_sample_d[0] = x_sol[0,d_idx]
                x_temp,x_evals = x_sol,[]
                for i_idx in range(n_worker):
                    x_temp[0,d_idx] = x_sample_d[i_idx]
                    x_evals.append(np.copy(x_temp.reshape((1,-1))))
                # Evaluate k candidates in one scoop
                evals = [func_eval_ray.remote(x=x_eval) for x_eval in x_evals] # EVALUATE
                y_evals = ray.get(evals)
                # Update the current coordinate
                min_idx = np.argmin(np.asarray(y_evals)[:,0,0])
                x_sol[0,d_idx] = x_sample_d[min_idx]
                # Concatenate CD results
                x_cd,y_cd = np.asarray(x_evals)[:,0,:],np.asarray(y_evals)[:,0,:]
                x_data = np.concatenate((x_data,x_cd),axis=0)
                y_data = np.concatenate((y_data,y_cd),axis=0)
        esec_cd = time.time() - iclk_cd
        esec_total = time.time() - iclk_total
        # Plot CD
        if n_cd > 0:
            x_sol,y_sol = get_best_xy(x_data,y_data)
            print("[%.1f]sec [%d/%d] CD took [%.1f]sec. Current best x:%s best y:[%.3f]"%
                  (esec_total,x_data.shape[0],n_data_max,esec_cd,x_sol,y_sol))
        
        # Save intermediate resutls
        if save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                print ( "[%s] created."%(save_folder) )
            # Save
            npz_path = os.path.join(save_folder,'bavoo_result.npz')
            np.savez(npz_path, x_data=x_data,y_data=y_data,
                     x_minmax=x_minmax,n_random=n_random,n_bo=n_bo,n_cd=n_cd,
                     n_data_max=n_data_max,n_worker=n_worker,seed=seed,
                     n_sample_for_bo=n_sample_for_bo)
            print ( "[%s] saved."%(npz_path) )
            
        # Terminate Condition
        if x_data.shape[0] >= n_data_max: break
            
    return x_data,y_data
