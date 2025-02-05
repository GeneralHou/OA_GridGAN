import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

opt = TestOptions().parse(save=False)
opt.nThreads = 0   
opt.batchSize = 1  
opt.serial_batches = True  
opt.no_flip = True  

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))


model = create_model(opt)
if opt.data_type == 16:
    model.half()
elif opt.data_type == 8:
    model.type(torch.uint8)

if opt.verbose:
    print(model)
    
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['label_h'] = data['label_h'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['label_h'] = data['label_h'].uint8()

    minibatch = 1
    generated = model.inference(data['label'], data['label_h'], data['image'])
        
    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
