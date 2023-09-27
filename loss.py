import torch
from torchvision import transforms
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from perceptual_similarity import perceptual_loss as ps

from collections import namedtuple

D_Output = namedtuple("D_Output", ["D_real", "D_gen", "D_real_logits", "D_gen_logits"])

def _non_saturating_loss(D_real_logits, D_gen_logits, D_real=None, D_gen=None):

    D_loss_real = F.binary_cross_entropy_with_logits(input=D_real_logits,
        target=torch.ones_like(D_real_logits))
    D_loss_gen = F.binary_cross_entropy_with_logits(input=D_gen_logits,
        target=torch.zeros_like(D_gen_logits))
    D_loss = D_loss_real + D_loss_gen

    G_loss = F.binary_cross_entropy_with_logits(input=D_gen_logits,
        target=torch.ones_like(D_gen_logits))

    return D_loss, G_loss

def _least_squares_loss(D_real, D_gen, D_real_logits=None, D_gen_logits=None):
    D_loss_real = torch.mean(torch.square(D_real - 1.0))
    D_loss_gen = torch.mean(torch.square(D_gen))
    D_loss = 0.5 * (D_loss_real + D_loss_gen)

    G_loss = 0.5 * torch.mean(torch.square(D_gen - 1.0))
    
    return D_loss, G_loss

def gan_loss(gan_loss_type, disc_out, mode='generator_loss'):

    if gan_loss_type == 'non_saturating':
        loss_fn = _non_saturating_loss
    elif gan_loss_type == 'least_squares':
        loss_fn = _least_squares_loss
    else:
        raise ValueError('Invalid GAN loss')

    D_loss, G_loss = loss_fn(D_real=disc_out.D_real, D_gen=disc_out.D_gen,
        D_real_logits=disc_out.D_real_logits, D_gen_logits=disc_out.D_gen_logits)
        
    if mode == 'generator_discriminator_loss':
        loss = [D_loss, G_loss]
    elif mode == 'generator_loss':
        loss = G_loss 
    else:
        loss = D_loss
    
    return loss

class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()

    def forward(self, img1, img2):
        if img1.shape[1] == 3:
            img1 = TF.rgb_to_grayscale(img1)
        if img2.shape[1] == 3:
            img2 = TF.rgb_to_grayscale(img2)
        laplacian1 = self.laplacian(img1)
        laplacian2 = self.laplacian(img2)
        laplacian_loss = F.mse_loss(laplacian1, laplacian2, reduction="none")
        return laplacian_loss

    def laplacian(self, img):
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
        img = F.pad(img, (1, 1, 1, 1), mode='reflect')
        laplacian = F.conv2d(img, laplacian_kernel)
        return laplacian
    
class ContentOrientedLoss(nn.Module):
    def __init__(self, args, discriminator):
        super(ContentOrientedLoss, self).__init__()
        self.args = args
        self.vgg_perceptual_loss = ps.PerceptualLoss(model='net-lin', net='vgg', use_gpu=torch.cuda.is_available(), gpu_ids=[0])
        self.laplacian = LaplacianLoss()
        self.Discriminator = discriminator

    def forward(self, orig_imgs, recon_imgs, face_masks, structure_masks):
        weighted_rate = self.rate_loss(orig_imgs)
        if self.args.normalize_input_image is True:
            # [-1.,1.] -> [0.,1.]
            orig_imgs = (orig_imgs * 0.5) + 0.5 
            recon_imgs = (recon_imgs * 0.5) + 0.5
        content_oiented_loss = self.content_oriented_loss(orig_imgs, recon_imgs, face_masks, structure_masks)
        rate_content_oriented_loss = weighted_rate + content_oiented_loss
        # compute discriminator loss
        face_masks, structure_masks, texture_masks = self.process_masks(face_masks, structure_masks)
        replaced_recon_imgs = self.real_value_replacement(orig_imgs, recon_imgs, texture_masks)
        D_loss = self.gan_loss(orig_imgs, replaced_recon_imgs, "discriminator_loss") 
        return rate_content_oriented_loss, D_loss

    def rate_loss(self, imgs):
        return torch.zeros((1,), device=imgs.device)

    def content_oriented_loss(self, original_img, reconstructed_img, face_mask, structure_mask):
        face_mask, structure_mask, texture_mask = self.process_masks(face_mask, structure_mask)
        L_tex = self.loss_texture(original_img, reconstructed_img, texture_mask)
        L_struc = self.loss_structure(original_img, reconstructed_img, texture_mask)
        L_face = self.loss_face(original_img, reconstructed_img, face_mask)
        w_L_tex = L_tex
        w_L_struc = L_struc * self.args.epsilon
        w_L_face = L_face * self.args.gamma
        return  L_tex + L_struc + L_face

    def loss_face (self, original_img, reconstructed_img, face_mask): 
        return torch.mean(face_mask * F.mse_loss(original_img, reconstructed_img, reduction="none"))
    
    def loss_structure (self, original_img, reconstructed_img, structure_mask):
        return torch.mean(structure_mask * self.laplacian(original_img, reconstructed_img))
    
    def loss_texture (self, original_img, reconstructed_img, texture_mask): 
        replaced_reconstruction = self.real_value_replacement(original_img, reconstructed_img, texture_mask)
        loss = self.args.alpha * torch.mean(texture_mask * F.l1_loss(original_img, reconstructed_img, reduction="none")) + self.args.beta * self.lpips(original_img, replaced_reconstruction) + self.args.delta * self.gan_loss(original_img, replaced_reconstruction, "generator_loss")
        return loss

    def lpips(self, original_img, reconstructed_img):
        LPIPS_loss = self.vgg_perceptual_loss.forward(original_img, reconstructed_img, normalize=True)
        return torch.mean(LPIPS_loss)
    
    def discriminator_forward(self, orig_imgs, recon_imgs, train_generator):
        """ Train on gen/real batches simultaneously. """
        # Alternate between training discriminator and compression models
        if train_generator is False:
            recon_imgs = recon_imgs.detach()

        D_in = torch.cat([orig_imgs, recon_imgs], dim=0)

        D_out = self.Discriminator(D_in)
        D_out = torch.squeeze(D_out)
        D_real, D_gen = torch.chunk(D_out, 2, dim=0)
        
        return D_Output(D_real, D_gen, D_real, D_gen)

    def gan_loss(self, original_img, replaced_reconstruction, mode):
        disc_out = self.discriminator_forward(original_img, replaced_reconstruction, train_generator=(mode=="generator_loss"))
        D_loss, G_loss = gan_loss(gan_loss_type=self.args.gan_loss_type, disc_out=disc_out, mode='generator_discriminator_loss')
        if mode == 'generator_discriminator_loss':
            loss = [D_loss, G_loss]
        elif mode == 'generator_loss':
            loss = G_loss 
        else:
            loss = D_loss
        return loss

    def real_value_replacement(self, original_img, reconstructed_img, texture_mask):
        return (1-texture_mask) * original_img + texture_mask * reconstructed_img

    def process_masks(self, face_mask, structure_mask):
        structure_mask = ~face_mask & structure_mask
        texture_mask = torch.ones_like(face_mask) & ~face_mask & ~structure_mask
        return face_mask, structure_mask, texture_mask


