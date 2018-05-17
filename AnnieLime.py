import sys,os,time,csv,getopt,cv2,argparse
import numpy, ctypes, array
import numpy as np
#import matplotlib as plt
from datetime import datetime
from ctypes import cdll, c_char_p
from skimage.transform import resize
from numpy.ctypeslib import ndpointer
from lime import lime_image
from skimage.segmentation import mark_boundaries
import ntpath
import scipy.misc
from PIL import Image


AnnInferenceLib = ctypes.cdll.LoadLibrary('/home/rajy/work/inceptionv4/build/libannmodule.so')
inf_fun = AnnInferenceLib.annRunInference
inf_fun.restype = ctypes.c_int
inf_fun.argtypes = [ctypes.c_void_p,
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_size_t]
hdl = 0

def PreprocessImage(img, dim):
    imgw = img.shape[1]
    imgh = img.shape[0]
    imgb = np.empty((dim[0], dim[1], 3))    #for inception v4
    imgb.fill(1.0)
    if imgh/imgw > dim[1]/dim[0]:
        neww = int(imgw * dim[1] / imgh)
        newh = dim[1]
    else:
        newh = int(imgh * dim[0] / imgw)
        neww = dim[0]
    offx = int((dim[0] - neww)/2)
    offy = int((dim[1] - newh)/2)
    imgc = img.copy()*(2.0/255.0) - 1.0

    #print('INFO:: newW:%d newH:%d offx:%d offy: %d' % (neww, newh, offx, offy))
    imgb[offy:offy+newh,offx:offx+neww,:] = resize(imgc,(newh,neww),1.0)
    #im = imgb[:,:,(2,1,0)]
    return imgb

def runInference(img):
    global hdl
    imgw = img.shape[1]
    imgh = img.shape[0]
    #proc_images.append(im)
    out_buf = bytearray(1000*4)
    #out_buf = memoryview(out_buf)
    out = np.frombuffer(out_buf, dtype=numpy.float32)
    #im = im.astype(np.float32)
    inf_fun(hdl, np.ascontiguousarray(img, dtype=np.float32), (img.shape[0]*img.shape[1]*3*4), np.ascontiguousarray(out, dtype=np.float32), len(out_buf))
    return out

def predict_fn(images):
    results = np.zeros(shape=(len(images), 1000))
    for i in range(len(images)):
        results[i] = runInference(images[i])    
    return results

def lime_explainer(image, preds):
    for x in preds.argsort()[0][-5:]:
        print (x, names[x], preds[0,x])
        top_indeces.append(x)
    tmp = datetime.now()
    explainer = lime_image.LimeImageExplainer()
    # Hide color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels
    explanation = explainer.explain_instance(image, predict_fn, top_labels=5, hide_color=0, num_samples=1000)
    #to see the explanation for the top class
    temp, mask = explanation.get_image_and_mask(top_indeces[4], positive_only=True, num_features=5, hide_rest=True)
    im_top1 = mark_boundaries(temp / 2 + 0.5, mask)
    #print "iminfo",im_top1.shape, im_top1.dtype
    im_top1 = im_top1[:,:,(2,1,0)] #BGR to RGB
    temp1, mask1 = explanation.get_image_and_mask(top_indeces[3], positive_only=True, num_features=100, hide_rest=True)
    im_top2 = mark_boundaries(temp1 / 2 + 0.5, mask1)
    im_top2 = im_top2[:,:,(2,1,0)] #BGR to RGB
    del top_indeces[:]
    return im_top1, im_top2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest='image', type=str,
                        default='./images/dog.jpg', help='An image path.')
    parser.add_argument('--video', dest='video', type=str,
                        default='./videos/car.avi', help='A video path.')
    parser.add_argument('--imagefolder', dest='imagefolder', type=str,
                        default='./', help='A directory with images.')
    parser.add_argument('--resultsfolder', dest='resultfolder', type=str,
                        default='./', help='A directory with images.')
    parser.add_argument('--labels', dest='labelfile', type=str,
                        default='./labels.txt', help='file with labels')
    args = parser.parse_args()

    imagefile = args.image
    videofile = args.video
    imagedir  = args.imagefolder
    outputdir = args.resultfolder
    synsetfile = args.labelfile
    images = []
    proc_images = []
    AnnInferenceLib.annCreateContext.argtype = [ctypes.c_char_p]
    data_folder = "/home/rajy/work/inceptionv4"
    b_data_folder = data_folder.encode('utf-8')
    global hdl
    hdl = AnnInferenceLib.annCreateContext(b_data_folder)
    top_indeces = []
    #read synset names
    if synsetfile:
        fp = open(synsetfile, 'r')
        names = fp.readlines()
        names = [x.strip() for x in names]
        fp.close()

    if sys.argv[1] == '--image':
        # image preprocess
        img = cv2.imread(imagefile)
        dim = (299,299)
        imgb = PreprocessImage(img, dim)
        images.append(imgb)
        #proc_images.append(imgb)
        start = datetime.now()
        preds = predict_fn(images)
        end = datetime.now()
        elapsedTime = end-start
        print ('total time for inference in milliseconds', elapsedTime.total_seconds()*1000)
        if False:
            for x in preds.argsort()[0][-5:]:
                print (x, names[x], preds[0,x])
                top_indeces.append(x)
            image0 = images[0]
            tmp = datetime.now()
            explainer = lime_image.LimeImageExplainer()
            # Hide color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels
            explanation = explainer.explain_instance(image0, predict_fn, top_labels=5, hide_color=0, num_samples=1000)
            elapsedTime = datetime.now()-tmp
            print ('total time for lime is " milliseconds', elapsedTime.total_seconds()*1000)

            #to see the explanation for the top class
            temp, mask = explanation.get_image_and_mask(top_indeces[4], positive_only=True, num_features=5, hide_rest=True)
            im_top1 = mark_boundaries(temp / 2 + 0.5, mask)
            #print "iminfo",im_top1.shape, im_top1.dtype
            im_top1_save = im_top1[:,:,(2,1,0)] #BGR to RGB

            infile = ntpath.basename(imagefile)
            inname,ext = infile.split('.')

            cv2.imshow('top1', im_top1)
            scipy.misc.imsave(outputdir + inname + '_top1.jpg', im_top1_save)
            #scipy.imsave(outputdir + inname + '_1.jpg', im_top1)
            #im_top1_norm.save(outputdir + inname + '_1.jpg')
            temp1, mask1 = explanation.get_image_and_mask(top_indeces[3], positive_only=True, num_features=100, hide_rest=True)
            #temp, mask = explanation.get_image_and_mask(top_indeces[3], positive_only=True, num_features=1000, hide_rest=False, min_weight=0.05)        
            #cv2.imshow('top2', mark_boundaries(temp1 / 2 + 0.5, mask1))
            im_top2 = mark_boundaries(temp1 / 2 + 0.5, mask1)
            im_top2 = im_top2[:,:,(2,1,0)] #BGR to RGB
            scipy.misc.imsave(outputdir + inname + '_top2.jpg', im_top2)
        else:    
            im_top1, im_top2 = lime_explainer(images[0], preds)
            infile = ntpath.basename(imagefile)
            inname,ext = infile.split('.')
            #cv2.imshow('top1', im_top1)
            scipy.misc.imsave(outputdir + inname + '_top1.jpg', im_top1)
            scipy.misc.imsave(outputdir + inname + '_top2.jpg', im_top2)        
        #cv2.destroyAllWindows()
        AnnInferenceLib.annReleaseContext(ctypes.c_void_p(hdl))
        exit()
    elif sys.argv[1] == '--imagefolder':
        count = 0
        start = datetime.now()
        for image in sorted(os.listdir(imagedir)):
            print('Processing Image ' + image)
            img = cv2.imread(imagedir + image)
            dim = (299,299)
            imgb = PreprocessImage(img, dim)
            images.append(imgb)
            #proc_images.append(imgb)
            preds = predict_fn(images)
            im_top1, im_top2 = lime_explainer(images[0], preds)
            inname,ext = image.split('.')
            #cv2.imshow('top1', im_top1)
            scipy.misc.imsave(outputdir + inname + '_top1.jpg', im_top1)
            scipy.misc.imsave(outputdir + inname + '_top2.jpg', im_top2)
            images.remove(imgb)       
            count += 1
        end = datetime.now()
        elapsedTime = end-start
        print ('total time is " milliseconds', elapsedTime.total_seconds()*1000)
        AnnInferenceLib.annReleaseContext(ctypes.c_void_p(hdl))
        exit()

