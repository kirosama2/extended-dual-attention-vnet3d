
from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import cv2
import os
from glob import glob


def load_itk(filename):
    """
    load mhd files and normalization 0-255
    :param filename:
    :return:
    """
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    # Reads the image using SimpleITK
    itkimage = rescalFilt.Execute(sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32))
    return itkimage


def removesmallConnectedCompont(sitk_maskimg, rate=0.5):
    cc = sitk.ConnectedComponent(sitk_maskimg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, sitk_maskimg)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size

    not_remove = []
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if size > maxsize * rate:
            not_remove.append(l)
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage != maxlabel] = 0
    for i in range(len(not_remove)):
        outmask[labelmaskimage == not_remove[i]] = 255
    return outmask


def getLargestConnectedCompont(sitk_maskimg):
    cc = sitk.ConnectedComponent(sitk_maskimg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, sitk_maskimg)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size

    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 255
    outmask[labelmaskimage != maxlabel] = 0
    return outmask


def morphologicaloperation(sitk_maskimg, kernelsize, name='open'):
    if name == 'open':
        morphoimage = sitk.BinaryMorphologicalOpening(sitk_maskimg, [kernelsize, kernelsize, kernelsize])
        labelmaskimage = sitk.GetArrayFromImage(morphoimage)
        outmask = labelmaskimage.copy()
        outmask[labelmaskimage == 1.0] = 255
        return outmask
    if name == 'close':
        morphoimage = sitk.BinaryMorphologicalClosing(sitk_maskimg, [kernelsize, kernelsize, kernelsize])
        labelmaskimage = sitk.GetArrayFromImage(morphoimage)
        outmask = labelmaskimage.copy()
        outmask[labelmaskimage == 1.0] = 255
        return outmask
    if name == 'dilate':
        morphoimage = sitk.BinaryDilate(sitk_maskimg, [kernelsize, kernelsize, kernelsize])
        labelmaskimage = sitk.GetArrayFromImage(morphoimage)
        outmask = labelmaskimage.copy()
        outmask[labelmaskimage == 1.0] = 255
        return outmask
    if name == 'erode':
        morphoimage = sitk.BinaryErode(sitk_maskimg, [kernelsize, kernelsize, kernelsize])
        labelmaskimage = sitk.GetArrayFromImage(morphoimage)
        outmask = labelmaskimage.copy()
        outmask[labelmaskimage == 1.0] = 255
        return outmask


def gettestiamge():
    src = load_itk("D:\Data\LIST\LITS-Challenge-Test-Data\\test-volume-" + str(51) + ".nii")
    srcimg = sitk.GetArrayFromImage(src)
    for i in range(np.shape(srcimg)[0]):
        image = srcimg[i]
        image = np.clip(image, 0, 255).astype('uint8')
        cv2.imwrite("D:\Data\LIST\LITS-Challenge-Test-Data\\" + str(51) + "\\" + str(i) + ".bmp", image)
