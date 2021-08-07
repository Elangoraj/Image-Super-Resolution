# Import libraries
from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset
import numpy as np
from sklearn.metrics import r2_score
from skimage.metrics import normalized_root_mse
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from PIL import ImageFont, ImageChops
import PIL.ImageOps  
import scipy.stats
import cv2
import imquality.brisque as brisque
import pandas as pd
from niqe import niqe
from piqe import piqe
import time
import os
import json
start_time = time.time()

class evaluation:

    def __init__(self):
        self.psnr_out = []


    def show_from_cv(self, img, title=None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(img)
        if title is not None:
            plt.title(title)

    def run(self):

        print('Run started')
        data_folder = base_path+'Dataset/test'
        test_data_names = ["England_960x720"] # full_960x720   England_960x720


        # Model checkpoints
        #srgan_checkpoint = "./checkpoint_srgan.pth.tar"
        if scaling_factor == 4 and network == 'RESNET':
            model_path = base_path+"checkpoint_srresnet_4X.pth.tar"
            model = torch.load(model_path)['model'].to(device)
        elif scaling_factor == 2 and network == 'RESNET':
            model_path = base_path+"checkpoint_srresnet_2X.pth.tar"
            model = torch.load(model_path)['model'].to(device)
       
        model.eval()

        ssim_out = []
        nrmse_out = []
        p_corr_out = []
        s_corr_out = []
        r2_out = []
        brisque_out = []
        niqe_out = []
        piqe_out = []

        brisque_overall = []
        piqe_overall = []
        niqe_overall = []

        tit_idx = []

        # Evaluate
        for test_data_name in test_data_names:
            print("\nFor %s:\n" % test_data_name)

            # Custom dataloader
            test_dataset = SRDataset(data_folder,
                                     split='test',
                                     crop_size=0,
                                     scaling_factor=scaling_factor,
                                     lr_img_type='imagenet-norm',
                                     hr_img_type='[-1, 1]',
                                     test_data_name=test_data_name)

            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                                                      pin_memory=True)
             
            idx = np.random.randint(len(test_loader), size=len(test_loader))

            #print(idx)

            dfol = data_folder+'/'+test_data_names[0]+"_test_images.json"
            f = open(dfol,)
            fn = json.load(f)

            # Keep track of the PSNRs and the SSIMs across batches
            PSNRs = AverageMeter()
            SSIMs = AverageMeter()


            # Prohibit gradient computation explicitly because I had some problems with memory
            with torch.no_grad():
                # Batches
                # print('C1')
        
                for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                    
                    full_image_name = os.path.basename(fn[i])

                    image_name = os.path.splitext(full_image_name)[0]

                    print('Image ', i, ' -', image_name)
                    
                    # Move to default device
                
                    lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
                    hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]
                    # Forward prop.
                    sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]

                    #if i in idx:
                    sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='rgb') 
                    #print('RGB score ',brisque.score(sr_imgs_c),'before - ',brisque.score(sr_imgs_y.cpu().numpy()))
                    hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='rgb') 
                    lr_imgs_c = convert_image(lr_imgs, source='[-1, 1]', target='rgb') 
                    
                    
                    #hr_imgs_y = hr_imgs_c
                    #sr_imgs_y = sr_imgs_c

                    psnr = peak_signal_noise_ratio(hr_imgs_y, sr_imgs_y,
                                                   data_range=255.)
                    ssim = structural_similarity(hr_imgs_y, sr_imgs_y, multichannel=True,
                                                 data_range=255.)

                    self.psnr_out.append(psnr)
                    ssim_out.append(ssim)

                    PSNRs.update(psnr, lr_imgs.size(0))
                    SSIMs.update(ssim, lr_imgs.size(0))

                    
                    p_corr_out.append(np.corrcoef(hr_imgs_y.ravel(), sr_imgs_y.ravel())[0][1])
                    s_corr_out.append(scipy.stats.spearmanr(hr_imgs_y.ravel(),sr_imgs_y.ravel())[0])
                    
                    
                    r2_out.append(r2_score(hr_imgs_y.ravel(), sr_imgs_y.ravel()))
                    nrmse_out.append(normalized_root_mse(hr_imgs_y.ravel(), sr_imgs_y.ravel()))

                    brisque_overall.append(-(brisque.score(hr_imgs_y)-brisque.score(sr_imgs_y)))
                    piqe_overall.append(-((piqe(hr_imgs_y)[0])-(piqe(sr_imgs_y)[0])))
                    niqe_overall.append(-(niqe(hr_imgs_y)-niqe(sr_imgs_y)))


                    #print('LR Shape', lr_imgs_c.shape)
                    #Image.fromarray(lr_imgs_c).show()
                        
                    im1 = Image.fromarray(hr_imgs_y)
                    im2 = Image.fromarray(sr_imgs_y)

                    diff = PIL.ImageOps.invert(ImageChops.difference(im1,im2))
                    

                    def get_concat_h(im1, im2, im3):
                        dst = Image.new('RGB', (im1.width + im2.width + im3.width, im1.height))
                        dst.paste(im1, (0, 0))
                        dst.paste(im2, (im1.width, 0))
                        dst.paste(im3, (im1.width + im2.width, 0))
                        return dst
                    
                    tp = get_concat_h(im1, im2, diff)
  
                    pth = base_path+'output_'+network+'/image_results/'+image_name+'.jpg'

                    tp.save(pth, "jpeg")

                    #tp.show()

                    fig = plt.figure()
                    plt.hist(hr_imgs_y.ravel(), bins = 32, alpha = 0.3, label='High Resolution')
                    plt.hist(sr_imgs_y.ravel(), bins = 32, alpha = 0.3, label='Super Resolution')
                    plt.legend(loc='upper right')
                        
                    plt.title(image_name)
                    pth = base_path+'output_'+network+'/histograms/'+image_name+'.jpg'
                    fig.savefig(pth, dpi=250)
                    #Image.open(pth).show()

                    tit_idx.append(image_name)
            
                    

            # Print average PSNR and SSIM
            print('Overall PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
            print('Overall SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))

        #v = plt.figure()
        #print(self.psnr_out)
        

        error_tuples = list(zip(tit_idx,ssim_out,self.psnr_out,nrmse_out,p_corr_out,r2_out,s_corr_out,brisque_overall,piqe_overall,niqe_overall))
        error_df = pd.DataFrame(error_tuples, columns=['Image_name','SSIM','PSNR','NRMSE','p_corr_out','r2_out','s_corr_out','brisque_overall','piqe_overall','niqe_overall'])
        
        error_df.to_csv(base_path+'output_'+network+'/'+network+'_errors_'+str(scaling_factor)+'X.csv')

        print("--- %s minutes---" % round((time.time() - start_time)/60,2))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = './'
network = 'RESNET'
scaling_factor = 4


if __name__ == '__main__':
    global eval_obj 
    eval_obj = evaluation()
    
    eval_obj.run()
