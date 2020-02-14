from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os.path
import glob

# def convertjpg(jpgfile,outdir,width=182,height=268):
#     img=Image.open(jpgfile)   
#     new_img=img.resize((width,height),Image.BILINEAR)   
#     new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
# for jpgfile in glob.glob("C:/code/合照/*.jpg"):
#     try:
#         convertjpg(jpgfile,"newhezhao")
#     except OSError:
#         print(os.path.basename(jpgfile)+"写入失败！")


class Config:
    corp_size = (20, 30)
    filter_size = 300
    num = 100
    mo_pic_path = 'hezhao1.jpg'
    sub_pic_path = 'poster_downloads'
    save_path = 'hezhao_result2.jpg'

picture = Image.open(Config.mo_pic_path)
width, height = picture.size
to_width = Config.corp_size[0] * Config.num
to_height = ((to_width / width) * height // Config.corp_size[1]) * Config.corp_size[1]
picture = picture.resize((int(to_width), int(to_height)), Image.ANTIALIAS)

def pic_code(image: np.ndarray):
    width, height = image.shape
    avg = image.mean()
    one_hot = np.array([1 if image[i, j] > avg else 0 for i in range(width) for j in range(height)])
    return one_hot

def rgb_mean(rgb_pic):
    """
    if picture is RGB channel, calculate average [R, G, B].
    """
    r_mean = np.mean(rgb_pic[:, :, 0])
    g_mean = np.mean(rgb_pic[:, :, 1])
    b_mean = np.mean(rgb_pic[:, :, 2])
    val = np.array([r_mean, g_mean, b_mean])
    return val

def mapping_table(pic_folder):
    suffix = ['jpg', 'jpeg', 'JPG', 'JPEG', 'gif', 'GIF', 'png', 'PNG']
    if not os.path.isdir(pic_folder):
        raise OSError('Folder [{}] is not exist, please check.'.format(pic_folder))

    pic_list = os.listdir(pic_folder)
    item_num = len(pic_list)
    means, codes, pic_dic = {}, {}, {}
    for idx, pic in tqdm(enumerate(pic_list), desc='CODE'):
        if pic.split('.')[-1] in suffix:
            path = os.path.join(pic_folder, pic)
            try:
                img = Image.open(path).convert('RGB').resize(Config.corp_size, Image.ANTIALIAS)
                codes[idx] = pic_code(np.array(img.convert('L').resize((8, 8), Image.ANTIALIAS)))
                means[idx] = rgb_mean(np.array(img))
                pic_dic[idx] = np.array(img)
            except OSError as e:
                print(e)
    return codes, means, pic_dic

codes, means, pic_dic = mapping_table(Config.sub_pic_path)
def structure_similarity(section, candidate):
    section = Image.fromarray(section).convert('L')
    one_hot = pic_code(np.array(section.resize((8, 8), Image.ANTIALIAS)))
    candidate = [(key_, np.equal(one_hot, codes[key_]).mean()) for key_, _ in candidate]
    most_similar = max(candidate, key=lambda item: item[1])
    return pic_dic[most_similar[0]]

def color_similarity(pic_slice, top_n=Config.filter_size):
    slice_mean = rgb_mean(pic_slice)
    diff_list = [(key_, np.linalg.norm(slice_mean - value_)) for key_, value_ in means.items()]
    filter_ = sorted(diff_list, key=lambda item: item[1])[:top_n]
    return filter_

def merge(picture):
    width, height = picture.size
    w_times, h_times = int(width / Config.corp_size[0]), int(height / Config.corp_size[1])
    picture = np.array(picture)

    for i in tqdm(range(w_times), desc='MERGE'):
        for j in range(h_times):
            section = picture[j * Config.corp_size[1]:(j + 1) * Config.corp_size[1],
                              i * Config.corp_size[0]:(i + 1) * Config.corp_size[0], :]
            candidate = color_similarity(section)
            most_similar = structure_similarity(section, candidate)
            picture[j * Config.corp_size[1]:(j + 1) * Config.corp_size[1],
                    i * Config.corp_size[0]:(i + 1) * Config.corp_size[0], :] = most_similar

    picture = Image.fromarray(picture)
    picture.save(Config.save_path)
    return picture

result_picture = merge(picture)

plt.figure(figsize=(15,15)) 
plt.subplot(1,2,1), plt.title('原图')
plt.imshow(picture)
plt.axis('off') 
plt.subplot(1,2,2), plt.title('拼图')
plt.imshow(result_picture)
plt.axis('off') 
plt.show()