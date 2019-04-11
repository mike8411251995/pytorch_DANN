from models import models

# utility params
fig_mode = None
embed_plot_epoch=10

# model params
use_gpu = True
dataset_mean = (0.5, 0.5, 0.5)
dataset_std = (0.5, 0.5, 0.5)

batch_size = 512
epochs = 1000
gamma = 10
theta = 1

# path params
data_root = './data'

mnist_path = data_root + '/mnist'
mnistm_path = data_root + '/mnistm'
svhn_path = data_root + '/svhn'
syndig_path = data_root + '/SynthDigits'

save_dir = './experiment'


# specific dataset params
extractor_dict = {'mnist_mnistm': models.Extractor(),
                  'svhn_mnist': models.SVHN_Extractor(),
                  'SynDig_svhn': models.SVHN_Extractor()}

class_dict = {'mnist_mnistm': models.Class_classifier(),
              'svhn_mnist': models.SVHN_Class_classifier(),
              'SynDig_svhn': models.SVHN_Class_classifier()}

domain_dict = {'mnist_mnistm': models.Domain_classifier(),
               'svhn_mnist': models.SVHN_Domain_classifier(),
               'SynDig_svhn': models.SVHN_Domain_classifier()}
