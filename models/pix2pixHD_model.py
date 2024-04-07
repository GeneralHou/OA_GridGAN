import torch
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.input_nc
        input_hc = opt.input_hc

        
        
        netG_input_nc = input_nc + input_hc
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)        

        
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + input_hc + opt.output_nc
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  

        
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:  
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
                
        
            
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake')

            
            
            if opt.niter_fix_global > 0:  
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, height_map, real_image=None, infer=False):
        input_label = Variable(label_map.data.cuda(), volatile=infer)
        input_label_h = Variable(height_map.data.cuda(), volatile=infer)

        
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        return input_label, input_label_h, real_image

    def discriminate(self, input_label, input_label_h, test_image, use_pool=False):
        input_concat = torch.cat((input_label, input_label_h, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, label_h, image, infer=False):
        input_label, input_label_h, real_image = self.encode_input(label, label_h, image)

        
        input_concat = torch.cat((input_label, input_label_h), dim=1)
        fake_image = self.netG.forward(input_concat)

        
        pred_fake_pool = self.discriminate(input_label, input_label_h, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        
        pred_real = self.discriminate(input_label, input_label_h, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        
        pred_fake = self.netD.forward(torch.cat((input_label, input_label_h, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
        
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        
        
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake), None if not infer else fake_image]

    
    def inference(self, label, label_h, image=None):
        
        image = Variable(image) if image is not None else None
        input_label, input_label_h, real_image = self.encode_input(Variable(label), Variable(label_h), image, infer=True)

        
        input_concat = torch.cat((input_label, input_label_h), dim=1)

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        
        params = list(self.netG.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label = inp
        return self.inference(label)

        
