#@title 4) Art Generator Parameters
text = "bubble 3D shinny sun| rendered in unreal engine" #@param {type:"string"}
textos = text
height =  250#@param {type:"number"}
width =  500#@param {type:"number"}
ancho=width
alto=height
model = "vqgan_imagenet_f16_16384" #@param ["vqgan_imagenet_f16_16384", "vqgan_imagenet_f16_1024", "wikiart_1024", "wikiart_16384", "coco", "faceshq", "sflckr", "ade20k", "ffhq", "celebahq", "gumbel_8192"]
modelo=model
interval_image =  3#@param {type:"number"}
intervalo_imagenes = interval_image
initial_image = "/content/Sem ti\u0301tulo-6.png"#@param {type:"string"}
imagen_inicial= initial_image
objective_image = "/content/Sem ti\u0301tulo-6.png"#@param {type:"string"}
imagenes_objetivo = objective_image
seed = -1#@param {type:"number"}
max_iterations = 18#@param {type:"number"}
max_iteraciones = max_iterations
input_images = ""

nombres_modelos={"vqgan_imagenet_f16_16384": 'ImageNet 16384',"vqgan_imagenet_f16_1024":"ImageNet 1024", 
                 "wikiart_1024":"WikiArt 1024", "wikiart_16384":"WikiArt 16384", "coco":"COCO-Stuff", "faceshq":"FacesHQ", "sflckr":"S-FLCKR", "ade20k":"ADE20K", "ffhq":"FFHQ", "celebahq":"CelebA-HQ", "gumbel_8192": "Gumbel 8192"}
nombre_modelo = nombres_modelos[modelo]     

if modelo == "gumbel_8192":
    is_gumbel = True
else:
    is_gumbel = False

if seed == -1:
    seed = None
if imagen_inicial == "None":
    imagen_inicial = None
elif imagen_inicial and imagen_inicial.lower().startswith("http"):
    imagen_inicial = download_img(imagen_inicial)


if imagenes_objetivo == "None" or not imagenes_objetivo:
    imagenes_objetivo = []
else:
    imagenes_objetivo = imagenes_objetivo.split("|")
    imagenes_objetivo = [image.strip() for image in imagenes_objetivo]

if imagen_inicial or imagenes_objetivo != []:
    input_images = True

textos = [frase.strip() for frase in textos.split("|")]
if textos == ['']:
    textos = []


args = argparse.Namespace(
    prompts=textos,
    image_prompts=imagenes_objetivo,
    noise_prompt_seeds=[],
    noise_prompt_weights=[],
    size=[ancho, alto],
    init_image=imagen_inicial,
    init_weight=0.,
    clip_model='ViT-B/32',
    vqgan_config=f'{modelo}.yaml',
    vqgan_checkpoint=f'{modelo}.ckpt',
    step_size=0.1,
    cutn=64,
    cut_pow=1.,
    display_freq=intervalo_imagenes,
    seed=seed,
)
