#@title 1) Download Required Python Packages
 
print("Download CLIP...")
!git clone https://github.com/openai/CLIP                 &> /dev/null
 
print("Installing VQGAN...")
!git clone https://github.com/CompVis/taming-transformers &> /dev/null
!pip install ftfy regex tqdm omegaconf pytorch-lightning  &> /dev/null
!pip install kornia                                       &> /dev/null
!pip install einops                                       &> /dev/null
!pip install wget                                         &> /dev/null
 
print("Installing Extra Libraries...")
!pip install stegano                                      &> /dev/null
!apt install exempi                                       &> /dev/null
!pip install python-xmp-toolkit                           &> /dev/null
!pip install imgtag                                       &> /dev/null
!pip install pillow==7.1.2                                &> /dev/null
 
!pip install imageio-ffmpeg                               &> /dev/null
!mkdir steps
print("Installing Finished!!")
