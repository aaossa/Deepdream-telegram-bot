import numpy as np
import tensorflow as tf
from gc import collect
from PIL import Image

MODEL_FILENAME = "tensorflow_inception_graph.pb"
MODEL_ZIPNAME = "inception5h.zip"
MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"


class DeepDreamer:

    def __init__(self):
        model_filename = 'tensorflow_inception_graph.pb'
        self.__graph = tf.Graph()
        self.__sess = tf.InteractiveSession(graph=self.__graph)
        with tf.gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        self.__t_input = tf.placeholder(np.float32, name='input')
        imagenet_mean = 117.0
        t_preprocessed = tf.expand_dims(self.__t_input-imagenet_mean, 0)
        tf.import_graph_def(graph_def, {'input': t_preprocessed})
        # print([n.name for n in tf.get_default_graph().as_graph_def().node])

    def dream(self, image_bytes, option=1, **kwargs):
        image = Image.open(image_bytes)
        t_obj = self.__select_layer(option)
        output = self.__dream(t_obj, image, **kwargs)
        return Image.fromarray(np.uint8(np.clip(output, 0, 1)*255))

    def __select_layer(self, option):
        layers = {
            1: tf.square(self.__T('mixed4c')),
            2: self.__T('mixed4d_3x3_bottleneck_pre_relu')[:, :, :, 139],
        }
        return layers[option]

    def __dream(self, t_obj, image, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
        t_score = tf.reduce_mean(t_obj)
        t_grad = tf.gradients(t_score, self.__t_input)[0]

        # Split image into octaves
        image = np.float32(image)
        octaves = []
        for i in range(octave_n-1):
            hw = image.shape[:2]
            lo = self.__resize(image, np.int32(np.float32(hw)/octave_scale))
            hi = image-self.__resize(lo, hw)
            image = lo
            octaves.append(hi)

        # Generate details octave by octave
        for octave in range(octave_n):
            if octave > 0:
                hi = octaves[-octave]
                image = self.__resize(image, hi.shape[:2])+hi
            for i in range(iter_n):
                g = self.__calc_grad_tiled(image, t_grad)
                image += g*(step / (np.abs(g).mean()+1e-7))
        collect()
        return image/255.0

    def __calc_grad_tiled(self, image, t_grad, tile_size=512):
        sz = tile_size
        h, w = image.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(image, sx, 1), sy, 0)
        grad = np.zeros_like(image)
        for y in range(0, max(h-sz//2, sz), sz):
            for x in range(0, max(w-sz//2, sz), sz):
                sub = img_shift[y:y+sz, x:x+sz]
                g = self.__sess.run(t_grad, {self.__t_input: sub})
                grad[y:y+sz, x:x+sz] = g
        collect()
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    def __resize(self, img, size):
        placeholders = list(map(tf.placeholder, [np.float32, np.int32]))
        img_prime = tf.expand_dims(placeholders[0], 0)
        out = tf.image.resize_bilinear(img_prime, placeholders[1])[0, :, :, :]
        return out.eval(dict(zip(placeholders, (img, size))), session=self.__sess)

    def __T(self, tensor):
        return self.__graph.get_tensor_by_name("import/{}:0".format(tensor))


def __download_from_url(url, filename):
    import requests
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                f.flush()


def __unzip_file(filename):
    import zipfile
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall()


def download_model():
    from os.path import exists
    # Check zip file presence
    if exists(MODEL_ZIPNAME):
        print("Model already downloaded.")
    else:
        print("Downloading model...")
        __download_from_url(MODEL_URL, MODEL_ZIPNAME)
    # Check unziped file presence
    if exists(MODEL_FILENAME):
        print("Model already unzipped.")
    else:
        print("Unziping file...")
        __unzip_file(MODEL_ZIPNAME)
    print("Ready!")


download_model()
