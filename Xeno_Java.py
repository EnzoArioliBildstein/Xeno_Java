    #---------------------------------------------------------------------------------------------------------------- IMPORT
import sys
import cv2
import numpy as np
import os
import time
from numpy.lib.npyio import load
from numpy.linalg.linalg import transpose
import sounddevice as sd
from scipy.interpolate import griddata
import argparse
import concurrent.futures
import matplotlib.pyplot as plt
from threading import Thread
import scipy.fftpack
import math
import open3d as o3d
from scipy.spatial.qhull import ConvexHull,Delaunay,Voronoi
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import hashlib

    #---------------------------------------------------------------------------------------------------------------- RAGGED ARRAY... FUCK OFF
np.warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)
np.seterr(divide='ignore',invalid='ignore')
np.set_printoptions(suppress=True)
    #---------------------------------------------------------------------------------------------------------------- DEF
def callback(indata,frames,time,status):
    global recording_raw
    global windowSamples
    global max_freg
    recording_raw=indata
    windowSamples=np.concatenate((windowSamples,indata[:,0])) # append new samples
    windowSamples=windowSamples[len(indata[:,0]):] # remove old samples
    magnitudeSpec=abs(scipy.fftpack.fft(windowSamples)[:len(windowSamples)//2])
    maxInd=np.argmax(magnitudeSpec)
    max_freg_raw=maxInd*(fs/frame_annalyse)
    if max_freg_raw > freg_auditible_limite :
        max_freg=freg_auditible_limite
    else:
        max_freg=max_freg_raw
def motion(img,img_copy):
    img_1d=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    img_1d=cv2.blur(img_1d,(10,10))
    img_add=np.array(((np.round(np.copy(img_1d)/val)==np.round(img_copy/val))!=0),dtype=np.uint8)
    img_copy=np.copy(img_1d)
    return img_add,img_copy,img_1d
def Sound_Annalyse(recording_raw):
    recording=abs(recording_raw)
    max_sound=np.max(recording)
    return max_sound
def start_interpolation (frameHeight_tem,frameWidth_tem):
    y=np.linspace(0,frameHeight_tem,frameHeight_tem)
    x=np.linspace(0,frameWidth_tem,frameWidth_tem)
    X,Y=np.meshgrid(x,y)
    return X,Y
def intercorps (key_old,key):
    if np.size(key)==1 or np.size(key_old)==1   :
        cal=np.zeros((0))
    else :
        nbr_pers=np.shape(key)[0]
        nbr_pers_old=np.shape(key_old)[0]
        key_old_cal=np.tile(np.swapaxes(key_old,0,1),(nbr_pers,1,1,1))
        key_cal=np.tile([key],(np.shape(key_old)[0])).reshape(nbr_pers,25,nbr_pers_old,3)
        cal=(np.abs(key_old_cal[:,:,:,0]- key_cal[:,:,:,0]))+(np.abs(key_old_cal[:,:,:,1]- key_cal[:,:,:,1]))+(np.abs(key_old_cal[:,:,:,2]- key_cal[:,:,:,2]))  
        cal=np.average(cal,axis=1)
        cal=np.argmin(cal,axis=1)
        cal=np.concatenate((key,key_old[cal]),axis=2)
    return cal
def intercorps_link (cal) :
    interlink=np.zeros((frameHeight,frameWidth))
    cal=cal.astype(int)
    if np.size(cal)!=1 :
        nbr_pers=np.shape(cal)[0]
        for i in range (0,nbr_pers):
            for o in range (0,25):
                if (cal[i,o,0]*cal[i,o,1])!=0 and (cal[i,o,3]*cal[i,o,4])!=0:
                    interlink=cv2.line(interlink,(cal[i,o,0],cal[i,o,1]),(cal[i,o,3],cal[i,o,4]),255,2)
    return interlink
def prolongation_3d (cal,list_futur,amplitude,dispartition_max) :
    if np.size(cal)!=0 :
        list_futur_add=cal[:,[[0,1],[0,15],[0,16],[1,2],[1,5],[1,8],[2,3],[3,4],[5,6],[6,7],[8,9],[8,12],[9,10],[10,11],[11,22],[11,24],[12,13],[13,14],[14,19],[14,21],[15,17],[16,18],[19,20],[22,23]]]
        list_futur_add=list_futur_add.reshape(-1,12)
        list_futur_add=list_futur_add[np.any(np.isnan(list_futur_add[:,[0,2,3,5,6,8,9,11]]),axis=1)==False]
        list_futur_add=list_futur_add.reshape(-1,12)
        list_futur_add_1=list_futur_add[:,[3,4,5,9,10,11]]
        list_futur_add_2=list_futur_add[:,[0,1,2,6,7,8]]-list_futur_add_1
        v_c=(list_futur_add_2[:,0:3]+list_futur_add_2[:,3:6])/2
        center=(list_futur_add_1[:,0:3]+list_futur_add_1[:,3:6])/2
        list_futur_add=np.concatenate((list_futur_add_2,v_c,list_futur_add_1,center,np.zeros((np.shape(list_futur_add)[0],2))+amplitude),axis=1)
        list_futur=np.concatenate((list_futur,list_futur_add))
    return list_futur
def center_cal (cal_1,cal_2) :
    center=(cal_1[0:3]+cal_2[0:3])/2
    center_vector=((cal_1[0:3]-cal_1[3:6])+(cal_2[0:3]-cal_2[3:6]))/2
    return center,center_vector
def mesh_body_cal (cal,mesh_body,mesh_color_organ) :
    if np.size(cal)!=0 :
        mesh_all_organe=cal[:,:,3:6][:,[[1,2,3],[1,5,6],[0,1,16],[0,1,15],[0,15,17],[0,16,18],[2,3,4],[5,6,7],[8,9,10],[8,12,13],[8,9,1],[8,12,1],[9,10,11],[12,13,14],[11,24,22],[11,22,23],[14,19,21],[14,19,20]]]
        mesh_all_organe=mesh_all_organe.reshape(-1,9)
        proc_mesh_color_organ_cal=executor.submit(mesh_color_organ_cal,mesh_color_organ,cal)
        delter=np.where((np.sum(mesh_all_organe[:,[0,1,2]]*mesh_all_organe[:,[3,4,5]]*mesh_all_organe[:,[6,7,8]],axis=1)==0))
        mesh_all_organe=np.delete(mesh_all_organe,delter,0)
        mesh_all_organ_color=proc_mesh_color_organ_cal.result()
        mesh_all_organ_color=np.delete(mesh_all_organ_color.reshape(-1,3),delter,0).reshape(-1,1)
        mesh_all_organe=mesh_all_organe.reshape(-1,3)
    else :
        mesh_all_organe=np.zeros((0,3))
        mesh_all_organ_color=np.zeros((0,1))
    return mesh_all_organe,mesh_all_organ_color
def mesh_color_organ_cal (mesh_color_organ,cal):
    color=np.tile(np.tile(mesh_color_organ[:,np.newaxis],(3,1)),(cal.shape[0])).astype(int)
    return color
def round_3d (list_futur,amplitude,dispartition_max,prolongation_max) :
    list_futur=np.delete(list_futur,(np.where((list_futur[:,19]) <=0)[0]),0)
    proc_mesh_color=executor.submit(color_people,list_futur,amplitude,2)
    ply_vertex,orgine_boom,boom_value=list_futur[:,0:9],list_futur[:,9:18],(list_futur[:,18]-list_futur[:,19])
    ply_vertex=(ply_vertex*(boom_value*prolongation_max+1)[:,np.newaxis]+orgine_boom)
    ply_vertex=np.concatenate((ply_vertex,list_futur[:,9:15]),axis=1).reshape(-1,3)
    color_mesh=proc_mesh_color.result()
    return ply_vertex,color_mesh,list_futur
def color_people (A,B,p):
    if np.size(A)!=0 :
        A=(np.tile(A[:,19][:,np.newaxis],5).reshape(-1)).astype(int)
        B=(np.zeros_like(A)*B).astype(int)
        mask=(((A[:,None] & (1 << np.arange(4-1,-1,-1)))!=0))
        mask_B=(((B[:,None] & (1 << np.arange(4-1,-1,-1)))!=0))
        mask=np.concatenate((mask,mask_B),axis=1)
        mask=np.packbits(mask,axis=1)
        mask_2=np.zeros((np.shape(mask)[0],3))
        mask_2[:,p]=mask.T
    else :
        mask_2=np.zeros((0,3))
    return mask_2
def floor_ta_les_pieds_au_sol (ply_vertex,floor) :
    if ply_vertex.shape[0]!=0:
        min_foot=np.nanmin(ply_vertex[:,:,[2,5]])
        foot_dif=min_foot-np.max(floor[floor<=min_foot],initial=floor[0])
        ply_vertex[:,:,[2,5]]=ply_vertex[:,:,[2,5]]-foot_dif
    return ply_vertex
def process_3d (cal,list_link,max_sound,list_futur_pbr,list_futur,num,dispartition_max,prolongation_max,mesh_color_organ,img_add,laser_vect,mesh_java,laser_org,laser_to_mesh_false,laser_6d,normal_imporant_on_dist,laser_speed_attributif,star_ray_pres,floor,lood_geo):
    amplitude=(1-math.exp(-10*max_sound))*(dispartition_max-1)+1
    #cal=floor_ta_les_pieds_au_sol(cal,floor)
    proc_color_img_add_for_ply=executor.submit(color_img_add_for_ply,img_add,amplitude)
    if np.size(list_futur)==0 :
        list_futur=np.zeros((1,20))
        list_futur_pbr=True
    proc_prolongation_3d=executor.submit(prolongation_3d,cal,list_futur,amplitude,dispartition_max)
    proc_mesh_body_cal=executor.submit(mesh_body_cal,cal,mesh_body,mesh_color_organ)
    list_futur=proc_prolongation_3d.result()
    list_futur=list_futur.reshape(-1,20)
    proc_round_3d=executor.submit(round_3d,list_futur,amplitude,dispartition_max,prolongation_max)
    b_color_body,cd_color_body=proc_color_img_add_for_ply.result()
    mesh_all_organe,a_color_organ=proc_mesh_body_cal.result()
    proc_color_continant_noir_rapide=executor.submit(color_continant_noir_rapide,a_color_organ,b_color_body,cd_color_body,np.size(a_color_organ),1)
    ply_vertex,color_mesh,list_futur=proc_round_3d.result()
    proc_hull_process=executor.submit(hull_process,ply_vertex,(mesh_all_organe.reshape(-1,3)),int(np.floor(amplitude)),laser_vect,mesh_java,laser_org,laser_to_mesh_false,laser_6d,normal_imporant_on_dist,laser_speed_attributif,star_ray_pres)
    if list_futur_pbr :
        list_futur=list_futur[1:]
        list_futur_pbr=False   
    list_futur[:,19]=list_futur[:,19] -1
    body_color_mesh=proc_color_continant_noir_rapide.result()
    pts_hull_w,mesh_hull_w,RASPUTIN,pts_laser,mesh_laser=proc_hull_process.result()
    pcd,name,triangle,vertices,mesh_color=face_maker(ply_vertex,list_futur,mesh_all_organe,body_color_mesh,color_mesh,num,pts_hull_w,mesh_hull_w,RASPUTIN,pts_laser,mesh_laser,lood_geo)
    return  list_futur,pcd,name,list_futur_pbr,triangle,vertices,mesh_color,pts_hull_w
def face_maker (ply_vertex,list_futur,mesh_all_organe,body_color_mesh,color_mesh,num,pts_hull_w,mesh_hull_w,RASPUTIN,pts_laser,mesh_laser,load_geo) :
    proc_color_retour_proletaire=executor.submit(color_retour_proletaire,RASPUTIN,0)
    pcd=o3d.geometry.TriangleMesh()
    proc_trianglum_meshe_cal=executor.submit(triangle_cal,ply_vertex,mesh_all_organe,pcd,mesh_hull_w,mesh_laser,load_geo[2])
    proc_cal_pcd_vertrice=executor.submit(cal_pcd_vertrice,pcd,ply_vertex,mesh_all_organe,pts_hull_w,pts_laser,load_geo[0])
    proc_ply_name_write=executor.submit(ply_name_writer,num)
    color_mesh_RASPUTIN=proc_color_retour_proletaire.result()
    laser_color=np.zeros_like(pts_laser)+255
    mesh_color=np.concatenate((color_mesh,body_color_mesh,color_mesh_RASPUTIN,laser_color,load_geo[1]))/255
    name=proc_ply_name_write.result()
    pcd.vertex_colors=o3d.utility.Vector3dVector(mesh_color)
    pcd.triangles,triangle=proc_trianglum_meshe_cal.result()
    pcd.vertices,vertices=proc_cal_pcd_vertrice.result()
    return pcd,name,triangle,vertices,mesh_color
def cal_pcd_vertrice (pcd,ply_vertex,mesh_all_organe,pts_hull_w,pts_laser,load_geo) :
    ply_vertex=np.concatenate((ply_vertex,mesh_all_organe,pts_hull_w,pts_laser,load_geo))
    pcd.vertices=o3d.utility.Vector3dVector(ply_vertex)
    return pcd.vertices,ply_vertex
def ply_name_writer (num) :
    name=str(num)[-10:]+"ARC"+".ply"
    return name
def triangle_cal (ply_vertex,mesh_all_organe,pcd,mesh_hull_w,mesh_laser,load_geo) :
    mesh_nbr=int(np.shape(ply_vertex)[0]/5)
    k=np.repeat(np.linspace(0,mesh_nbr*5-5,mesh_nbr),12)
    triangle=(np.tile(np.array(((0,3,4,2,1,4,0,1,4,2,3,4))),mesh_nbr)+k).reshape(int(mesh_nbr*4),3)
    chiffre_chem=np.max([triangle],initial=0)
    proc_triangle_cal_paralle_proc=executor.submit(triangle_cal_paralle_proc,chiffre_chem,mesh_hull_w,mesh_all_organe,mesh_laser)
    mesh_all_organe=np.arange(int(chiffre_chem+1),int(chiffre_chem+np.shape(mesh_all_organe)[0])+1).reshape(-1,3)
    mesh_hull_w,mesh_laser=proc_triangle_cal_paralle_proc.result()
    if np.shape (load_geo)[0]==0 :
        mpo = 0
    else :
        mpo = np.max(mesh_laser+1)
    triangle=np.concatenate((triangle,mesh_all_organe,mesh_hull_w,mesh_laser,(load_geo+mpo)))
    pcd.triangles=o3d.utility.Vector3iVector(triangle)
    return pcd.triangles,triangle
def triangle_cal_paralle_proc (chiffre_chem,mesh_hull_w,mesh_all_organe,mesh_laser) :
    chiffre_un_peu_chem=int(chiffre_chem+np.shape(mesh_all_organe)[0])+1
    mesh_hull_w=mesh_hull_w+chiffre_un_peu_chem
    mesh_laser=mesh_laser+mesh_hull_w.shape[0]*3+chiffre_un_peu_chem-1
    return mesh_hull_w,mesh_laser
def interpolation (key,frameHeight,frameWidth,Reduce_interpolotion):
    if np.size(key.shape)!=0:
        frameHeight_tem,frameWidth_tem=int(frameHeight/Reduce_interpolotion),int(frameWidth/Reduce_interpolotion)
        proc_start_interpolation=executor.submit(start_interpolation,frameHeight_tem,frameWidth_tem)
        nbr_pers=key.shape[0]
        key_l=np.zeros((nbr_pers))
        for i in range (0,nbr_pers):
            key_l[i]=np.average(key[i,:,2]*228+27)
            minmax_point=np.array([(np.max(key[i,:,0]),np.max(key[i,:,1]),key_l[i]),(np.max(key[i,:,0]),np.min(key[i,:,1][np.nonzero(key[i,:,1])]),key_l[i]),(np.min(key[i,:,0][np.nonzero(key[i,:,0])]),np.max(key[i,:,1]),key_l[i]),(np.min(key[i,:,0][np.nonzero(key[i,:,0])]),np.min(key[i,:,1][np.nonzero(key[i,:,1])]),key_l[i])])
            if i==0 :
                point_interpolation=minmax_point
            else :
                point_interpolation=np.concatenate((point_interpolation,minmax_point),axis=0)
        point_interpolation=(point_interpolation/Reduce_interpolotion)
        key=key[key_l.argsort()]
        X,Y=proc_start_interpolation.result()
        interpolation_map=griddata((point_interpolation[:,0],point_interpolation[:,1]),point_interpolation[:,2],(X,Y),method='linear')
    else :
        interpolation_map=np.zeros((frameHeight,frameWidth))
    return key,interpolation_map
def Turple_index (key,frameHeight,frameWidth,max_sound,img_add) :  
    color_sk=int(254-(max_sound**0.5*254))
    epaisseur=int((np.shape(np.where(img_add==0))[1])/(frameHeight*frameWidth)*9)+1
    key_output=np.zeros((frameHeight,frameWidth))
    if np.size(key.shape)!=0:
        key=key.astype(int)
        nbr_pers=key.shape[0]
        for i in range (0,nbr_pers):
            for t in list_link :
                if key[i,t[0],0]*key[i,t[0],1]!=0 and  key[i,t[1],0]*key[i,t[1],1]!=0 :
                    key_output=cv2.line(key_output,(key[i,t[0],0],key[i,t[0],1]),(key[i,t[1],0],key[i,t[1],1]),color_sk,epaisseur)
                    if key[i,t[0],1] < frameHeight and key[i,t[0],0] < frameWidth :
                       key_output[key[i,t[0],1],key[i,t[0],0]]=t[0]
                    if key[i,t[1],1] < frameHeight and key[i,t[1],0] < frameWidth :
                        key_output[key[i,t[1],1],key[i,t[1],0]]=t[1]
    return key_output
def Output_def(key,img_1d,img_add,frameHeight,frameWidth,max_freg,max_sound,interpolation_map,angle_map,cal):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        proc_Turple_index=executor.submit(Turple_index,key,frameHeight,frameWidth,max_sound,img_add)
        proc_intercorps_link=executor.submit(intercorps_link,cal)
        output=np.zeros((frameHeight,frameWidth,3),dtype=np.uint8)
        color_motion=(((max_freg/freg_auditible_limite)**0.25)*254)+1
        output[:,:,0][img_add==0]=color_motion
        output[:,:,0][angle_map==1]=1
        output[:,:,2]=interpolation_map
        key_output=proc_Turple_index.result()
        interlink=proc_intercorps_link.result()
        output[:,:,1]=key_output
        output[:,:,1][interlink==255]=255
        return output
def color_continant_noir_rapide (A,B,C,lng,p):
    if np.size(A)!=0 :
        mask=np.ones((lng,2)).astype(int)
        mask_2=np.ones((lng)).astype(int)
        mask=(mask*np.concatenate((A,np.tile(B,(lng,1))),axis=1)).reshape(-1,1).astype(int)
        mask_2=(mask_2*np.array((np.tile(C,(lng))))).reshape(-1,1).astype(int)
        mask=(((mask[:,None] & (1 << np.arange(2-1,-1,-1)))!=0))
        mask_2=(((mask_2[:,None] & (1 << np.arange(4-1,-1,-1)))!=0))
        mask=np.concatenate((mask_2.reshape(-1,4),mask.reshape(-1,4)),axis=1)
        mask=np.packbits(mask,axis=1)
        mask_2=np.zeros((np.shape(mask)[0],3))
        mask_2[:,p]=mask.T
    else :
        mask_2=np.zeros((0,3))
    return mask_2
def color_retour_proletaire (RASPUTIN,p):
    if np.size(RASPUTIN)!=0 :
        RASPUTIN=np.tile(RASPUTIN[:,np.newaxis],(3,1)).reshape(-1,8)
        lng=np.shape(RASPUTIN)[0]
        mask=np.ones((lng,8)).astype(int)
        mask=(mask*RASPUTIN).reshape(-1,1)
        mask=mask.reshape(-1,8)
        mask=np.packbits(mask,axis=1)
        mask_2=np.zeros((np.shape(mask)[0],3))
        mask_2[:,p]=mask.T
    else :
        mask_2=np.zeros((0,3))
    return mask_2
def dimension_W (pts_2,pts) :
    if np.size(pts)!=0 or np.size(pts_2)!=0 :
        proc_hull_bis=executor.submit(hull_bis_processus,pts_2)
        try :
            hull=ConvexHull(pts)
        except :
            pts=pts+np.random.uniform(0,1,size=np.shape(pts))
            hull=ConvexHull(pts)
        hull_2=proc_hull_bis.result()
        vectrice=np.concatenate((pts[hull.vertices],pts_2[hull_2.vertices]))
        pts=vectrice
        tri=Delaunay(vectrice)
        a=np.unique(np.sort(np.concatenate((tri.vertices.reshape(-1,2),tri.vertices[:,[0,3,1,2]].reshape(-1,2),tri.vertices[:,[0,2,1,3]].reshape(-1,2))),axis=1),axis=0)
        j=np.tile(hull.equations,(a.shape[0],1))
        b=np.tile(np.sum(pts[a],axis=1)/2,(hull.equations.shape[0])).reshape(-1,3)
        b=np.sort(np.delete(a,np.all((np.sum(b*j[:,0:3],axis=1)+j[:,3].T).reshape(-1,hull.equations.shape[0]) < 0,axis=1),axis=0),axis=1)
        c=np.sum((b < np.shape(hull.vertices)[0]).astype(int),axis=1)
        b=pts[b]
        max_w,min_w,vertical_w=np.unique(b[c==2].reshape(-1,3),axis=0),np.unique(b[c==0].reshape(-1,3),axis=0),b[c==1].reshape(-1,6)  
        vertical_w[:,0:3]=vertical_w[:,0:3]-vertical_w[:,3:6]
        equations=hull.equations
    else :
        if np.size(pts)==0 and np.size(pts_2)==0 :
            min_w,max_w=np.zeros((0,3)),np.zeros((0,3))
            vertical_w=np.zeros((0,6))
            equations=np.zeros((0,4))
        else :
            if np.size(pts)==0 :
                vertrice=pts_2
            else:
                vertrice=pts
            try:
                hull=ConvexHull(vertrice)
            except :
                vertrice=vertrice+np.random.uniform(-1,1,size=np.shape(vertrice))
                hull=ConvexHull(vertrice)
            min_w=np.unique(vertrice[hull.vertices].reshape(-1,3),axis=0)
            max_w=min_w
            vertical_w=np.concatenate((np.zeros_like(max_w)[:,np.newaxis],min_w[:,np.newaxis]),axis=1)
            equations=np.zeros((0,4))
    return max_w,min_w,vertical_w,equations
def hull_bis_processus (pts) :
    try :
        hull=ConvexHull(pts)
    except :
        pts=pts+np.random.uniform(0,1,size=np.shape(pts))
        hull=ConvexHull(pts)
    return hull
def dimension_W_pli (vertical_w,hyperplan,num_de_pli,min_w) :
    tri=ConvexHull(vertical_w[:,0:3]*0.5+vertical_w[:,3:6])
    face=np.unique(np.sort(tri.simplices).reshape(-1,3),axis=0)
    pre_center=np.sum(tri.points[face.flatten()].reshape(-1,3,3),axis=1)/3
    fac_pts_cal=vertical_w[face].reshape(-1,3,2,3)
    pts=np.tile(fac_pts_cal,(num_de_pli,1,1,1,1))
    pts[:,:,:,0,:]=pts[:,:,:,0,:]*(np.tile(((0.5**np.arange(0,num_de_pli))*0.4+0.1),(int(np.shape(face)[0]*9),1)).T).reshape(pts[:,:,:,0,:].shape)
    pts=np.sum(pts,axis=3).reshape(-1,3)
    mesh=np.arange(0,np.shape(pts)[0]).reshape(-1,3)
    return pts,mesh
    
def cal_amplitude (min_w,morph_this_voro,num_de_pli) :
    hull=ConvexHull(min_w)
    min_w_certer=np.average(min_w[hull.simplices],axis=1)
    min_w_center_ckd_tree=cKDTree(min_w_certer)
    amplitude,region=min_w_center_ckd_tree.query(morph_this_voro)
    return amplitude
def hull_process (pts,pts_2,num_de_pli,laser_vect,mesh_java,laser_org,laser_to_mesh_false,laser_6d,normal_imporant_on_dist,laser_speed_attributif,star_ray_pres): 
    if np.size(pts)!=0 and np.size(pts_2)!=0 :
        pts_2=pts_2[np.any((np.isnan(pts_2)),axis=1)==False]
        proc_dimension_w=executor.submit(dimension_W,pts,pts_2)
        max_w,min_w,vertical_w,hyperplan=proc_dimension_w.result()
        if np.size(vertical_w!=1) :
            proc_dimension_W_pli=executor.submit(dimension_W_pli,vertical_w,hyperplan,num_de_pli,min_w)
            pts,mesh=proc_dimension_W_pli.result()
            proc_laser_attack=executor.submit(laser_attack,laser_vect,mesh_java,pts[mesh],laser_org,laser_to_mesh_false,laser_6d,normal_imporant_on_dist,laser_speed_attributif,star_ray_pres)
            proc_dimension_EBA=executor.submit(EBA_DANCE,mesh.shape[0])
            RASPUTIN=proc_dimension_EBA.result()
            pts_laser,mesh_laser=proc_laser_attack.result()
    else : 
        pts,mesh,RASPUTIN,pts_laser,mesh_laser=np.zeros((0,3)),np.zeros((0,3)),np.zeros((0,8)),np.zeros((0,3)),np.zeros((0,3))
    return pts,mesh,RASPUTIN,pts_laser,mesh_laser
def cal_scale_moyenne (key,list_link_size,frameHeight,focal_estimation,frame_center,capteur_estimation,scale_z,num) :
    key_2=np.copy(key)
    if np.shape(key_2[key_2!=np.array(None)])[0]!=0 :
        key_2[:,:,2]=1-key_2[:,:,2]
        key_2[:,:,0:2],key_2[:,:,2]=key_2[:,:,0:2]-(1-frame_center[np.newaxis,:])*key_2[:,:,2][:,:,np.newaxis]/focal_estimation,key_2[:,:,2]
        list_key_2=key_2[:,[[0,1],[0,15],[0,16],[1,2],[1,5],[1,8],[2,3],[3,4],[5,6],[6,7],[8,9],[8,12],[9,10],[10,11],[11,22],[11,24],[12,13],[13,14],[14,19],[14,21],[15,17],[16,18],[19,20],[22,23]]]
        Fov_dist=np.concatenate(((list_link_size/(np.sum(np.subtract(list_key_2[:,:,0,0:2],list_key_2[:,:,1,0:2])**2,axis=2)**0.5))[:,:,np.newaxis],list_key_2[:,:,1,2][:,:,np.newaxis]),axis=2).reshape(-1,2)
        Fov_dist=Fov_dist[Fov_dist[:,0]!=np.inf]
        Fov_dist=np.average(1/(Fov_dist[0]/frameHeight*capteur_estimation[0]/Fov_dist[1]))*0.75
        if scale_z!=0:
            scale_z=Fov_dist*(1/((num+1)**0.2))+scale_z*(1-1/(num+1)**0.2)
        else :
            scale_z=Fov_dist
    return scale_z
def mise_en_context (pts,focal_estimation,frame_center,scale_z,applic_prisme,contextualisation_bottom,contextualisation_top,size,normal_camera,distance_deep) :
    if np.size(pts) > 2 :
        save_pts=pts
        pts[pts==0]=np.nan
        pts=pts.reshape(-1,3)
        proc_placement_z=executor.submit(placement_z,np.copy(pts)[:,2],scale_z,distance_deep)
        #pts[:,0:2]=(pts[:,0:2]-frame_center[np.newaxis,:])*(1-pts[:,2][:,np.newaxis])/focal_estimation*(pts[:,0]!=0)[:,np.newaxis]
        pts[:,0:2]=(pts[:,0:2]/size[np.newaxis,:])-np.array(([0.5,0.5]))
        pts=(np.concatenate((pts,pts),axis=1)*np.concatenate((contextualisation_bottom[3:5],[0],contextualisation_top[3:5],[0]))).reshape(-1,3)
        pts[:,[0,1,2]]=pts[:,[0,2,1]]
        pts=applic_prisme.apply(pts)
        pts[:,2]=-pts[:,2]
        pts=(pts.reshape(-1,6)+np.concatenate((contextualisation_bottom[0:3],contextualisation_top[0:3]))).reshape(-1,2,3)
        pts_z=proc_placement_z.result()
        pts=np.sum(pts*pts_z,axis=1)
        pts[:,1][np.isnan(pts[:,1])]=0
        pts=pts.reshape(save_pts.shape)  
    return pts
def placement_z (pts_z,scale_z,distance_deep) :
    min_pts_z=np.nanmin(pts_z)
    deepmax=1/(min_pts_z+scale_z/distance_deep)
    pts_z=((pts_z-min_pts_z)/(1-min_pts_z))*deepmax+min_pts_z
    pts_z=np.concatenate((1-pts_z[:,np.newaxis],pts_z[:,np.newaxis]),axis=1).reshape(-1,2,1)
    return pts_z
def save_ply (pcd,name_ply,save_ply_question,path) :
    if save_ply_question :
        o3d.io.write_triangle_mesh(str(str(path)+("0000000000000000"+name_ply)[-17:]),pcd,write_ascii=True)
        ok_save_ply="True"
    else:
        ok_save_ply="False"
    return ok_save_ply
def save_img (num,img_1d,output,path,save_img_question):
    if save_img_question :
        cv2.imwrite(str(str(path)+str("0000000000000000"+str(num))[-10:]+"ORG"+".jpg"),img_1d)
        cv2.imwrite(str(str(path)+str("0000000000000000"+str(num))[-10:]+"KEY"+".png"),output)
        ok_save_img="True"
    else :
        ok_save_img="False"
    return ok_save_img
def color_img_add_for_ply (img_add,amplitude) :
    color_b=np.floor(np.size(np.where(img_add==1)[0])/np.size(img_add)*4)
    color_cd=np.min(np.floor(np.array((amplitude,16))))
    return color_b,color_cd
def EBA_DANCE (size) :
    RASPUTIN=np.random.randint(2,size=(size,8))
    return RASPUTIN 
def clearning_Z(key):
    if np.size(key)!=1 :
        key_3=np.copy(key)
        key_3[key_3==0 ]=np.nan
        key_3[:,:,2]=(key_3[:,:,2]+4*np.nanmean((key_3[:,:,2]),axis=1)[:,np.newaxis])/5
        key_3[np.isnan(key_3)]=0
    else :
        key_3=key
    return key_3
def Möller_Trumbore_algorithm_WT_DUP_MESH(ray_near,ray_dir,mesh_java):
    eps=0.00001
    ray_dir=(1/np.sum(np.abs(ray_dir),axis=1))[:,np.newaxis]*ray_dir
    edge=np.array((mesh_java[:,1]-mesh_java[:,0],mesh_java[:,2]-mesh_java[:,0]))
    pvec=np.cross(ray_dir,edge[1])
    det=np.sum(edge[0]*pvec,axis=1)
    proc_Speed_up_Möller_Trumbore_algorithm_5=executor.submit(Speed_up_Möller_Trumbore_algorithm_5,det)
    tvec=ray_near-mesh_java[:,0].reshape(-1,3)
    inv_det=proc_Speed_up_Möller_Trumbore_algorithm_5.result()
    proc_Speed_up_Möller_Trumbore_algorithm=executor.submit(Speed_up_Möller_Trumbore_algorithm,tvec,pvec,det,eps,inv_det)
    qvec=np.cross(tvec,edge[0])
    v=np.sum(ray_dir*qvec,axis=1)*inv_det
    false_value,u=proc_Speed_up_Möller_Trumbore_algorithm.result()
    false_value[np.concatenate(((v < 0)[:,np.newaxis],(u+v > 1)[:,np.newaxis]),axis=1).any(axis=1)]=True
    t=np.sum(edge[1]*qvec,axis=1)*inv_det
    false_value [t < eps]=True
    false_value=false_value==False
    return false_value,t,ray_dir
def Möller_Trumbore_algorithm(ray_near,ray_dir,mesh_java):
    eps=0.00001
    proc_Speed_up_Möller_Trumbore_algorithm_2=executor.submit(Speed_up_Möller_Trumbore_algorithm_2,ray_dir,mesh_java)
    proc_Speed_up_Möller_Trumbore_algorithm_3=executor.submit(Speed_up_Möller_Trumbore_algorithm_3,ray_near,mesh_java)
    proc_Speed_up_Möller_Trumbore_algorithm_4=executor.submit(Speed_up_Möller_Trumbore_algorithm_4,mesh_java,ray_dir)
    ray_dir_tile=proc_Speed_up_Möller_Trumbore_algorithm_2.result()
    edge=proc_Speed_up_Möller_Trumbore_algorithm_4.result()
    pvec=np.cross(ray_dir_tile,edge[1])
    det=np.sum(edge[0]*pvec,axis=1)
    proc_Speed_up_Möller_Trumbore_algorithm_5=executor.submit(Speed_up_Möller_Trumbore_algorithm_5,det)
    ray_near_tile=proc_Speed_up_Möller_Trumbore_algorithm_3.result()
    tvec=ray_near_tile-np.tile(mesh_java[:,0][:,np.newaxis,:],(1,ray_dir.shape[0],1)).reshape(-1,3)
    inv_det=proc_Speed_up_Möller_Trumbore_algorithm_5.result()
    proc_Speed_up_Möller_Trumbore_algorithm=executor.submit(Speed_up_Möller_Trumbore_algorithm,tvec,pvec,det,eps,inv_det)
    qvec=np.cross(tvec,edge[0])
    v=np.sum(ray_dir_tile*qvec,axis=1)*inv_det
    false_value,u=proc_Speed_up_Möller_Trumbore_algorithm.result()
    false_value[np.concatenate(((v < 0)[:,np.newaxis],(u+v > 1)[:,np.newaxis]),axis=1).any(axis=1)]=True
    t=np.sum(edge[1]*qvec,axis=1)*inv_det
    false_value [t < eps]=True
    false_value=false_value==False
    t=np.linalg.norm(1/np.sum(np.abs(ray_dir_tile),axis=1)[:,np.newaxis]*ray_dir_tile,axis=1)*t
    t=t*false_value
    return false_value,t
def Speed_up_Möller_Trumbore_algorithm(tvec,pvec,det,eps,inv_det):
    u=np.sum(tvec*pvec,axis=1)*inv_det
    false_value=np.abs(np.copy(det))<eps
    false_value[np.concatenate(((u < 0)[:,np.newaxis],(u > 1)[:,np.newaxis]),axis=1).any(axis=1)]=True
    return false_value,u
def Speed_up_Möller_Trumbore_algorithm_2(ray_dir,mesh_java) :
    ray_dir=(1/np.sum(np.abs(ray_dir),axis=1))[:,np.newaxis]*ray_dir
    ray_dir_tile=np.tile(ray_dir[np.newaxis,:],(np.shape(mesh_java)[0],1,1)).reshape(-1,3)
    return ray_dir_tile
def Speed_up_Möller_Trumbore_algorithm_3 (ray_near,mesh_java):
    ray_near_tile=np.tile(ray_near[np.newaxis,:],(np.shape(mesh_java)[0],1,1)).reshape(-1,3)
    return ray_near_tile
def Speed_up_Möller_Trumbore_algorithm_4 (mesh_java,ray_dir):
    edge=np.tile(np.array((mesh_java[:,1]-mesh_java[:,0],mesh_java[:,2]-mesh_java[:,0]))[:,:,np.newaxis,:],(1,ray_dir.shape[0],1)).reshape(2,-1,3)
    edge=np.around(edge,decimals=4)
    return edge
def Speed_up_Möller_Trumbore_algorithm_5 (det):
    inv_det=1/det
    return inv_det
def laser_attack(ray_dir,mesh_java,face_pts,laser_org,laser_to_mesh_false,laser_6d,normal_imporant_on_dist,laser_speed_attributif,star_ray_pres) :
    proc_hyperplan_calcule_3=executor.submit(hyperplan_calcule_3,face_pts)
    face_equation=proc_hyperplan_calcule_3.result()
    proc_find_best_laser=executor.submit(find_best_laser,face_pts,face_equation,normal_imporant_on_dist,laser_6d)
    best_laser,laser_need=proc_find_best_laser.result()
    ray_dir=face_pts.reshape(-1,3)-best_laser
    proc_decompose_java=executor.submit(Decompose_java,mesh_java,laser_need,laser_to_mesh_false,laser_speed_attributif,star_ray_pres,ray_dir)
    mesh_try,vecteur_echec=proc_decompose_java.result()
    mesh_try=mesh_java[mesh_try]
    false_value,t,Vector_on_one=Möller_Trumbore_algorithm_WT_DUP_MESH(best_laser,ray_dir,mesh_try)
    t[np.concatenate((false_value[:,np.newaxis]==False,t[:,np.newaxis] < 0),axis=1).any(axis=1)]=np.nan
    t=t.reshape(ray_dir.shape[0],-1)
    Vect_Laser=Vector_on_one*np.nanmin(t,axis=1)[:,np.newaxis]
    pts_laser,mesh_laser=laser_materialisation(best_laser,Vect_Laser,vecteur_echec,mesh_try)
    return pts_laser,mesh_laser
def find_best_laser_little_m_speed (face_pts,face_equation,normal_imporant_on_dist):
    normal=np.sum((np.abs(face_equation[:,0:3]).reshape(-1,3)),axis=1)[:,np.newaxis]*face_equation[:,0:3].reshape(-1,3)/normal_imporant_on_dist
    pts_6d=np.concatenate((face_pts.reshape(-1,3),np.tile(normal[:,np.newaxis],(3,1)).reshape(-1,3)),axis=1)
    return pts_6d
def find_best_laser (face_pts,face_equation,normal_imporant_on_dist,laser_6d):
    proc_find_best_laser_little_m_speed=executor.submit(find_best_laser_little_m_speed,face_pts,face_equation,normal_imporant_on_dist)
    k=cKDTree(laser_6d)
    pts_6d=proc_find_best_laser_little_m_speed.result()
    amplitude,region=k.query(pts_6d,workers=-1)
    laser_need=region
    best_laser=laser_org[region]
    return best_laser,laser_need
def Decompose_java (mesh_java,laser_need,laser_to_mesh_false,laser_speed_attributif,star_ray_pres,ray_dir):
    mesh_try=mesh_java[np.any(laser_to_mesh_false[np.unique(np.sort(laser_need.flatten()))],axis=0)]
    sel_mesh_by_vect=star_ray_pres/np.sum(np.abs(ray_dir),axis=1)
    sel_mesh_by_vect=np.round(np.copy(ray_dir)*sel_mesh_by_vect[:,np.newaxis]).astype(int)
    sel_mesh_by_vect=sel_mesh_by_vect+star_ray_pres+1
    proc_laser_echec=executor.submit(laser_echec,laser_need,sel_mesh_by_vect,star_ray_pres,laser_speed_attributif)
    mesh_try=np.array(((laser_speed_attributif[0][laser_need,sel_mesh_by_vect[:,0],sel_mesh_by_vect[:,1],sel_mesh_by_vect[:,2]])),dtype=np.uint8)
    vecteur_echec=proc_laser_echec.result()
    return mesh_try,vecteur_echec
def laser_echec (laser_need,sel_mesh_by_vect,star_ray_pres,laser_speed_attributif):
    proc_laser_echec_speed=executor.submit(laser_echec_speed,sel_mesh_by_vect,star_ray_pres)
    power_echec=laser_speed_attributif[1][laser_need,sel_mesh_by_vect[:,0],sel_mesh_by_vect[:,1],sel_mesh_by_vect[:,2]]
    vecteur_echec=proc_laser_echec_speed.result()*power_echec[:,np.newaxis]
    return vecteur_echec
def laser_echec_speed (sel_mesh_by_vect,star_ray_pres) :
    vecteur_echec=(1/np.sum(np.abs(sel_mesh_by_vect-star_ray_pres-1),axis=1))[:,np.newaxis]*(sel_mesh_by_vect-star_ray_pres-1)
    return vecteur_echec
def laser_materialisation (best_laser,Vect_Laser,laser_echec,mesh_try) :
    laser_nan=np.any(np.concatenate((np.isnan(Vect_Laser)[:,np.newaxis],(np.abs(Vect_Laser)==np.inf)[:,np.newaxis]),axis=1),axis=1)
    Vect_Laser[laser_nan]=laser_echec[laser_nan]
    mesh_try=mesh_try[(np.any(np.isnan(Vect_Laser),axis=1)==False)]
    proc_hyperplan_calcule_3=executor.submit(hyperplan_calcule_3,mesh_try)
    best_laser=best_laser[(np.any(np.isnan(Vect_Laser),axis=1)==False)]
    Vect_Laser=Vect_Laser[(np.any(np.isnan(Vect_Laser),axis=1)==False)]
    best_laser_projet=best_laser+Vect_Laser
    hyperplan_mesh=proc_hyperplan_calcule_3.result()
    hyperplan_mesh_normal=1/np.sum(np.abs(hyperplan_mesh[:,0:3]),axis=1)[:,np.newaxis]*hyperplan_mesh[:,0:3]
    proc_laser_materialisation_speed_up_3=executor.submit(laser_materialisation_speed_up_3,best_laser)
    proc_laser_materialisation_speed_up=executor.submit(laser_materialisation_speed_up,Vect_Laser)
    Vect_Laser_tile=np.tile(Vect_Laser[:,np.newaxis,:],(1,3,1)).reshape(-1,3)
    proc_laser_materialisation_speed_up_2=executor.submit(laser_materialisation_speed_up_2,hyperplan_mesh_normal)
    m=np.tile((np.concatenate((np.sin(np.array([[0],[2/3],[4/3]])*np.pi),np.cos(np.array([[0],[2/3],[4/3]])*np.pi),np.ones((3,1)))).reshape(3,3).T)[np.newaxis,:,:],(best_laser.shape[0],1,1)).reshape(-1,3)*0.1*(np.sum(np.abs(Vect_Laser_tile),axis=1))[:,np.newaxis]
    Vect_Laser_tile=proc_laser_materialisation_speed_up.result()
    u=proc_laser_materialisation_speed_up_2.result()
    pts_laser=np.concatenate(((np.tile(best_laser_projet[:,np.newaxis,:],(1,3,1)).reshape(-1,3)+u*m+np.cross(Vect_Laser_tile,u)).reshape(-1,9),best_laser),axis=1).reshape(-1,3)
    pts_laser=np.concatenate((((np.tile(best_laser_projet[:,np.newaxis,:],(1,3,1))).reshape(-1,3)+(np.random.rand(best_laser_projet.shape[0]*3,3)*0.1)).reshape(-1,9),best_laser),axis=1).reshape(-1,3)
    mesh_laser=proc_laser_materialisation_speed_up_3.result()
    return pts_laser,mesh_laser
def laser_materialisation_speed_up (Vect_Laser):
    Vect_Laser_tile=np.tile(Vect_Laser[:,np.newaxis,:],(1,3,1)).reshape(-1,3)
    return Vect_Laser_tile
def laser_materialisation_speed_up_2 (Vect_Laser):
    u=np.tile(np.cross(Vect_Laser,np.array([[0,0,1]]))[:,np.newaxis],(3,1)).reshape(-1,3)
    return u
def laser_materialisation_speed_up_3 (best_laser):
    mesh_laser=(np.tile(np.array(([[1,2,4,2,3,4,1,3,4]])),(best_laser.shape[0],1))+((np.arange(best_laser.shape[0])[:,np.newaxis])*4)).reshape(-1,3)
    return mesh_laser
def Laser_org_to_mesh (laser_dir,mesh_dir,star_ray_pres,dir_file,steep) :
    mesh_file=o3d.io.read_triangle_mesh(mesh_dir)
    mesh=np.asarray(mesh_file.vertices)[np.asarray(mesh_file.triangles)]
    mesh=np.concatenate((np.zeros((1,3,3)),mesh),axis=0)
    hash_mesh=hashlib.sha1(mesh).hexdigest()
    laser_file=o3d.io.read_triangle_mesh(laser_dir)
    laser=np.asarray(laser_file.vertices)[np.asarray(laser_file.triangles)]
    hash_new_file=dir_file+str(star_ray_pres)+str(hashlib.sha1(laser).hexdigest()+hash_mesh)+".npy"
    hash_new_file_2=dir_file+str(star_ray_pres)+hash_mesh+str(hashlib.sha1(laser).hexdigest())+".npy"
    laser_file.compute_vertex_normals()
    laser_vect=np.asarray(laser_file.triangle_normals)
    laser_vect_difraction=np.copy(laser_vect)
    laser_vect_difraction[laser_vect==0]=1
    laser_org=laser[:,0]
    laser_6d=np.concatenate((laser_org,laser_vect),axis=1)
    if os.path.isfile(hash_new_file) and os.path.isfile(hash_new_file_2) :
        laser_to_mesh_false=np.load(hash_new_file)
        laser_speed_attributif=np.load(hash_new_file_2)
        print("Load Laser to Mesh OK")
    else :
        print("Load Laser to Mesh NAN")
        laser_star_copy=np.mgrid[-star_ray_pres-1:star_ray_pres+1,-star_ray_pres-1:star_ray_pres+1,-star_ray_pres-1:star_ray_pres+1].reshape(3,-1).T
        laser_star=1/np.sum(np.abs(laser_star_copy),axis=1)[:,np.newaxis]*laser_star_copy
        #laser_star=laser_star[np.sum(np.abs(laser_star),axis=1)>=star_ray_pres-2]
        #laser_star=laser_star[np.sum(np.abs(laser_star),axis=1)<=star_ray_pres+2]
        laser_speed_attributif=np.zeros((laser_org.shape[0],star_ray_pres*2+3,star_ray_pres*2+3,star_ray_pres*2+3))
        size_laser_attributif=np.zeros((laser_org.shape[0],star_ray_pres*2+3,star_ray_pres*2+3,star_ray_pres*2+3))
        laser_to_mesh_false_2=np.zeros((laser_org.shape[0],mesh.shape[0]))
        for i in range(0,steep):
            print("Steep laser_org : "+str(i/steep*100)+"%")
            part_of_laser_org=laser_org[(int(np.floor(laser_org.shape[0]/steep*i))):(int(np.floor(laser_org.shape[0]/steep*(i+1))))]
            part_of_laser_vect_dif=laser_vect_difraction[(int(np.floor(laser_org.shape[0]/steep*i))):(int(np.floor(laser_org.shape[0]/steep*(i+1))))]
            org_tuile=int(np.floor(laser_org.shape[0]/steep*i))
            laser_org_tile=np.tile(np.copy(part_of_laser_org)[np.newaxis,:,:],(laser_star.shape[0],1,1)).reshape(-1,3)
            laser_vect_difraction_tile=np.tile(part_of_laser_vect_dif[np.newaxis,:],(laser_star.shape[0],1,1))
            laser_star_tile=(np.tile(np.copy(laser_star)[:,np.newaxis,:],(1,part_of_laser_org.shape[0],1))*laser_vect_difraction_tile).reshape(-1,3)
            laser_to_mesh_false,t=Möller_Trumbore_algorithm(laser_org_tile,laser_star_tile,mesh)     
            proc_approximation_mesh_laser=executor.submit(approximation_mesh_laser,t,laser_star_copy,part_of_laser_org,star_ray_pres,laser_to_mesh_false,mesh,size_laser_attributif,laser_speed_attributif,org_tuile)
            t[np.concatenate(((t <=0.1)[:,np.newaxis],(np.abs(t)==np.nan)[:,np.newaxis],(laser_to_mesh_false==False)[:,np.newaxis]),axis=1).any(axis=1)]=-np.inf
            t=t.reshape(mesh.shape[0],laser_star.shape[0],part_of_laser_org.shape[0]) 
            t_2=(np.argmin(abs(t),axis=0)).astype(int).T
            t_2=np.concatenate(((np.tile(np.arange(t_2.shape[0])[:,np.newaxis],(1,laser_star.shape[0])).flatten())[:,np.newaxis],(t_2.flatten())[:,np.newaxis]),axis=1)[(np.all((t==np.inf),axis=0).T.flatten())==False]
            laser_to_mesh_false_2[(int(np.floor(laser_org.shape[0]/steep*i))):(int(np.floor(laser_org.shape[0]/steep*(i+1))))][t_2[:,0],t_2[:,1]]=True
            laser_speed_attributif,size_laser_attributif=proc_approximation_mesh_laser.result() 
        size_laser_attributif[laser_speed_attributif==0]=np.nan
        laser_speed_attributif[laser_speed_attributif==0]=np.nan
        laser_speed_attributif=np.concatenate((laser_speed_attributif[np.newaxis,:,:,:,:],size_laser_attributif[np.newaxis,:,:,:,:]),axis=0)
        np.save(hash_new_file,laser_to_mesh_false_2)
        np.save(hash_new_file_2,laser_speed_attributif)
        print("Save Laser to Mesh OK")
    return laser_to_mesh_false,mesh,laser_org,laser_vect,laser_6d,laser_speed_attributif
def approximation_mesh_laser (t,laser_star,part_of_laser_org,star_ray_pres,laser_to_mesh_false,mesh,size_laser_attributif,laser_speed_attributif,org_tuile) :
    t[np.concatenate(((t <=0.1)[:,np.newaxis],(np.abs(t)==np.nan)[:,np.newaxis],(t==-np.inf)[:,np.newaxis],(laser_to_mesh_false==False)[:,np.newaxis]),axis=1).any(axis=1)]=np.inf
    pop=t.reshape(mesh.shape[0],laser_star.shape[0],part_of_laser_org.shape[0])
    pop_index=np.argmin(pop,axis=0)
    pop_dist=np.min(pop,axis=0)
    laser_star_attributif=np.copy(laser_star)
    laser_star_attributif=laser_star_attributif+star_ray_pres+1
    attribution=np.concatenate((np.tile((np.arange(part_of_laser_org.shape[0])+org_tuile)[:,np.newaxis],(1,laser_star.shape[0])).flatten()[:,np.newaxis],np.tile(laser_star_attributif,(part_of_laser_org.shape[0],1))),axis=1)
    laser_speed_attributif[attribution[:,0],attribution[:,1],attribution[:,2],attribution[:,3]]=pop_index.flatten()
    size_laser_attributif[attribution[:,0],attribution[:,1],attribution[:,2],attribution[:,3]]=pop_dist.flatten()
    return laser_speed_attributif,size_laser_attributif
def hyperplan_calcule_3 (a) :
    uv=np.array((a[:,1,0]-a[:,0,0],a[:,1,1]-a[:,0,1],a[:,1,2]-a[:,0,2],a[:,2,0]-a[:,0,0],a[:,2,1]-a[:,0,1],a[:,2,2]-a[:,0,2]))
    u_cross_v=np.array([uv[1]*uv[5]-uv[2]*uv[4],uv[2]*uv[3]-uv[0]*uv[5],uv[0]*uv[4]-uv[1]*uv[3]]).T
    hyperplan=np.concatenate((u_cross_v,(-np.sum(a[:,0]*u_cross_v,axis=1))[:,np.newaxis]),axis=1)
    u_cross_v=np.array([uv[1]*uv[5]-uv[2]*uv[4],uv[2]*uv[3]-uv[0]*uv[5],uv[0]*uv[4]-uv[1]*uv[3]]).T
    hyperplan=np.concatenate((u_cross_v,(-np.sum(a[:,0]*u_cross_v,axis=1))[:,np.newaxis]),axis=1)
    return hyperplan
def rotation_matrix_from_vectors(vec1,vec2):
    a,b=(vec1/np.linalg.norm(vec1)).reshape(3),(vec2/np.linalg.norm(vec2)).reshape(3)
    v=np.cross(a,b)
    c=np.dot(a,b)
    s=np.linalg.norm(v)
    kmat=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    rotation_matrix=np.eye(3)+kmat+kmat.dot(kmat)*((1-c)/(s*2))
    return rotation_matrix
def visualisation (pcd_line,triangle,point,mesh_color,pts_hull_w,round_two,pi_step,pi_step_d):
    pcd_line=o3d.geometry.LineSet()
    if np.size(triangle)!=0 :
        proc_mesh_to_line_color=executor.submit(mesh_to_line_color,mesh_color,triangle)
        proc_mesh_to_line_triangle=executor.submit(mesh_to_line_triangle,triangle)
        point[:,[0,1,2]]=point[:,[0,2,1]]
        point[:,2]=-point[:,2]
        pi_step=pi_step-2*np.pi+pi_step_d if pi_step > 2*np.pi else pi_step+pi_step_d
        pcd_line.points=o3d.utility.Vector3dVector(point)
        pcd_line.lines=proc_mesh_to_line_triangle.result()
        pcd_line.colors=proc_mesh_to_line_color.result()
    return pcd_line, pi_step
def mesh_to_line_color (mesh_color,triangle):
    line_color=1-mesh_color[triangle.astype(int),:].reshape(-1,3)
    line_color[np.sum(line_color,axis=1)==0]=np.array(((1,0,0)))
    return o3d.utility.Vector3dVector(line_color)
def mesh_to_line_triangle (triangle):
    line_list=triangle[:,[0,1,1,2,2,0]].reshape(-1,2)
    return o3d.utility.Vector2iVector(line_list)
def soft_transposition (view_control,pts_hull_w,round_three,better_zoom):
    extrainsic_matrix=view_control.convert_to_pinhole_camera_parameters().extrinsic
    if round_three :
        camera_world=np.sum(-extrainsic_matrix[0:3,0:3]*extrainsic_matrix[0:3,3][:,np.newaxis],axis=0)
        center_point_aim=np.nanmean(pts_hull_w,axis=0)
        #center_point_aim=np.zeros((3))
        centre_porjeter=center_point_aim-(((center_point_aim*(-extrainsic_matrix[2,0:3].flatten()))-np.sum(camera_world.flatten()*(-extrainsic_matrix[2,0:3].flatten())))*(-extrainsic_matrix[2,0:3].flatten()))
        vis_line=o3d.geometry.LineSet()
        vis_line.points=o3d.utility.Vector3dVector(np.array((center_point_aim,centre_porjeter,np.zeros((3)))))
        vis_line.lines=o3d.utility.Vector2iVector(np.array((0,1,0,2)).reshape(-1,2))
        vis_line.colors=o3d.utility.Vector3dVector(np.array((1,0,0,1,0,0)).reshape(-1,3))
        X_transposer=np.sum(camera_world*extrainsic_matrix[1,0:3].flatten())-np.sum(centre_porjeter*extrainsic_matrix[1,0:3].flatten())
        Y_transposer=np.sum(camera_world*extrainsic_matrix[0,0:3].flatten())-np.sum(centre_porjeter*extrainsic_matrix[0,0:3].flatten())
        transposer=np.array((-Y_transposer,X_transposer))
        klklk=camera_world
        vis_line.points=o3d.utility.Vector3dVector(np.array((center_point_aim,centre_porjeter,np.zeros((3)),klklk)))
        vis_line.lines=o3d.utility.Vector2iVector(np.array((0,1,0,2,1,3)).reshape(-1,2))
        vis_line.colors=o3d.utility.Vector3dVector(np.array((1,0,1,1,0,1,1,0,1)).reshape(-1,3))
    else :
        transposer=np.array((0,0))
        vis_line=o3d.geometry.LineSet()
    transposer=np.nan_to_num(transposer)   
    return transposer[0] , transposer[1] ,vis_line, extrainsic_matrix
def evaluated_hull_w_for_Z (pts_hull_w,size_capteur_vis,view_control,fake_zoom_speed,best_deep) :
    if np.size(pts_hull_w)!=0 :
        extrainsic_matrix=view_control.convert_to_pinhole_camera_parameters().extrinsic
        intrainsic_matrix=view_control.convert_to_pinhole_camera_parameters().intrinsic.intrinsic_matrix
        camera_world=np.sum(-extrainsic_matrix[0:3,0:3]*extrainsic_matrix[0:3,3][:,np.newaxis],axis=0)
        max_min_visu=np.array((np.nanmax(pts_hull_w,axis=0),np.nanmin(pts_hull_w,axis=0))).flatten()
        max_min_visu=max_min_visu[[0,1,2,0,1,5,0,4,5,3,1,2,3,1,5,3,4,5]].reshape(-1,3)
        max_min_regeo=np.zeros_like((max_min_visu))
        max_min_regeo[:,0]=(np.sum(camera_world*extrainsic_matrix[0,0:3].flatten())-np.sum(max_min_visu*extrainsic_matrix[0,0:3].flatten(),axis=1)).T
        max_min_regeo[:,1]=(np.sum(camera_world*extrainsic_matrix[1,0:3].flatten())-np.sum(max_min_visu*extrainsic_matrix[1,0:3].flatten(),axis=1)).T
        max_min_regeo[:,2]=(np.sum(camera_world*extrainsic_matrix[2,0:3].flatten())-np.sum(max_min_visu*extrainsic_matrix[2,0:3].flatten(),axis=1)).T
        max_min_regeo[:,0]=np.abs( intrainsic_matrix[0,0]*max_min_regeo[:,0]/max_min_regeo[:,2])/(intrainsic_matrix[0,2] *7)
        max_min_regeo[:,1]=np.abs( intrainsic_matrix[1,1]*max_min_regeo[:,1]/max_min_regeo[:,2])/(intrainsic_matrix[1,2] *7)
        max_min_projeter=np.array((np.nanmax(max_min_regeo[:,0:2],axis=0),np.nanmin(max_min_regeo[:,0:2],axis=0)))
        best_deep=(np.max(max_min_projeter)+(fake_zoom_speed*best_deep))/(fake_zoom_speed+1)
    else :
        best_deep=1
    return best_deep
def height_reboot_def (view_control):
    intrainsic_matrix=view_control.convert_to_pinhole_camera_parameters().intrinsic.intrinsic_matrix
    reboot_height=True if np.any(intrainsic_matrix[0:2,2])==0 else False
    return reboot_height
def resize_cv2_output (img,extrinsic,best_zoom_print,Y_transposer_print,X_transposer_print) :
    img=cv2.resize(img, (1920,1080))
    extrinsic_print=np.concatenate((extrinsic,np.array((X_transposer_print,Y_transposer_print,best_zoom_print,0))[np.newaxis,:]),axis=0)
    
    for i in range (1,6) :
        img=cv2.putText(img,str(extrinsic_print[i-1]),(0,int(i*(1080/6))),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255, 0, 255),2, cv2.LINE_AA,False)
    return img
def save_poly_visual (vis,name,save_poly_visu_question) :
    save_poly_done=False
    if save_poly_visu_question :
        name =str("0000000000000000"+str(name))[-10:]+"gen"+".png"
        vis.capture_screen_image(("C:/Users/enzoa/.VisualSt/python/openpose/python/output_p/"+name))
        save_poly_done=True
    return save_poly_done
def pt_ref_def_visual (pcd) :
    pt_ref=np.copy(np.asarray(pcd.vertices))[np.all(np.asarray(pcd.vertex_colors)!=1,axis=1)]
    return pt_ref
def body_to_point (cal,choise_last):
    mean_of_extrinsic_man=(np.nanmean(choise_last) != 0)
    if (np.size(cal) != 0)  or mean_of_extrinsic_man :
        if np.size(choise_last) != 0 and mean_of_extrinsic_man :
            test_new_choise=np.nanmean(np.sum(np.tile(choise_last[np.newaxis,:],(np.shape(cal)[0],1,1)) * cal[:,:,0:3],axis=2),axis=1)
            if np.min(test_new_choise) < ((frameHeight+frameHeight)/4) :
                choise_last=cal[:,:,0:3][np.argmin(test_new_choise)]
        else:
            max_mix_test=np.argmax(np.sum((np.nanmax(cal[:,:,0:3],axis=1)-np.nanmin(cal[:,:,0:3],axis=1)),axis=1))
            choise_last=cal[:,:,0:3][max_mix_test]
    else :
        choise_last=np.zeros((25,3))
    proc_position_body = executor.submit(position_body,choise_last)
    proc_body_to_left=executor.submit(body_to_left,choise_last)
    proc_body_to_up=executor.submit(body_to_up,choise_last)
    fps_extrinsic=np.zeros((3,3))
    fps_extrinsic[0]=proc_body_to_left.result()
    fps_extrinsic[1]=proc_body_to_up.result()
    front=hyperplan_calcule_3(fps_extrinsic[np.newaxis,:,:])
    fps_extrinsic[1],fps_extrinsic[2]=0,front[:,0:3]
    fps_extrinsic[2]=((fps_extrinsic[2]**2/np.sum(fps_extrinsic[2]**2))**0.5)*((0>fps_extrinsic[2])*(-2)+1)
    fps_extrinsic[1]=(hyperplan_calcule_3(fps_extrinsic[np.newaxis,:,:])[:,0:3])
    best_body = proc_position_body.result()
    new_extrinsic = auto_gerorefeancement(best_body,fps_extrinsic)
    print(new_extrinsic)
    return fps_extrinsic, choise_last,new_extrinsic,best_body
def body_to_left (cal):
    plante=cal[[17,15,16,18,7,6,6,5,5,1,1,2,2,3,3,4,0,15,16,0,8,12,9,8,9,10,11,10,13,14,13,12,20,19,19,14,14,22,24,11,11,22,22,23]].reshape(2,22,3)
    plante[np.isnan(plante)] =0
    plante=np.sum((plante[1]-plante[0] * np.array((5,5,20,20,20,20,20,20,80,80,10,10,2,2,2,2,1,1,1,1,1,1))[:,np.newaxis]),axis=0)
    plante[2]=0
    plante=((plante**2/np.sum(plante**2))**0.5)*((0>plante)*(-2)+1)
    return plante
def body_to_up (cal):
    herbe=cal[[8,1,1,0,0,15,15,17,0,16,16,18,4,3,3,2,7,6,6,5,5,1,2,1,9,8,12,8,10,9,13,12,11,10,14,13,24,11,22,11,22,23,21,14,19,14,19,20]].reshape(2,24,3)
    herbe[np.isnan(herbe)] = 0 
    herbe= np.sum((herbe[1]-herbe[0] * (np.array((80,80,2,2,2,2,10,20,10,220,10,10,2,2,20,20,80,80,1,1,1,1,1,1))[:,np.newaxis])),axis=0)
    herbe=((herbe**2/np.sum(herbe**2))**0.5)*((0>herbe)*(-2)+1)
    return herbe
def position_body (best_body) :
    copy_of_best_body =np.copy(best_body)
    best_body = best_body[[0,15,16,1,17,18,2,5,6,3,4,7,8,9,12,13,10,11,14,22,24,21,19,20,23]]
    best_body = best_body*np.tile((np.array((25,24,24,22,21,21,19,19,17,17,15,15,13,12,12,10,10,8,8,6,6,4,4,2,2))**2)[:,np.newaxis],(1,3))
    best_body[np.isnan(best_body)]=0
    best_body = np.sum(best_body,axis=0)/np.sum((np.isnan(copy_of_best_body[:,0])==False).astype(int)*(np.array((25,24,24,22,21,21,19,19,17,17,15,15,13,12,12,10,10,8,8,6,6,4,4,2,2))**2))
    return best_body
def auto_gerorefeancement (best_body,fps_extrinsic):
    new_extrinsic = np.zeros((4,4))
    new_extrinsic [0:3,0:3] = fps_extrinsic
    new_extrinsic [0:3,3] = np.sum(fps_extrinsic*np.tile(best_body[np.newaxis,:],(3,1)),axis=1).T
    new_extrinsic[3,3] = 1
    return new_extrinsic
    #---------------------------------------------------------------------------------------------------------------- INPUT
    #--------------------------------------OPEN POSE
dir_path=os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/../bin/python/openpose/Release');
os.environ['PATH']=os.environ['PATH']+';'+dir_path+'/../x64/Release;'+dir_path+'/../bin;'
parser=argparse.ArgumentParser()
args=parser.parse_known_args()
params=dict()
params["model_folder"]="C:/Users/enzoa/.VisualSt/python/openpose/models"
params["disable_blending"]=True
params["part_to_show"]=3
params["render_pose"]=0
import pyopenpose as op
opWrapper=op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
#--------------------------------------OUTPUT
num=0
path="C:/Users/enzoa/.VisualSt/python/openpose/python/Output/"
#--------------------------------------CAMERA
list_link=[[0,1],[0,15],[0,16],[1,2],[1,5],[1,8],[2,3],[3,4],[5,6],[6,7],[8,9],[8,12],[9,10],[10,11],[11,22],[11,24],[12,13],[13,14],[14,19],[14,21],[15,17],[16,18],[19,20],[22,23]]
list_link_size=np.array((23,3.5,3.5,18,18,60,20,22,20,22,11,11,38,42,17,5,38,42,17,5,10,10,5,5))
mesh_body=[[1,2,3],[1,5,6],[0,1,16],[0,1,15],[0,15,17],[0,16,18],[2,3,4],[5,6,7],[8,9,10],[8,12,13],[8,9,1],[8,12,1],[9,10,11],[12,13,14],[11,24,22],[11,22,23],[14,19,21],[14,19,20]]
mesh_color_organ=np.array((3,3,2,2,2,2,0,0,3,3,3,3,3,3,1,1,1,1))
#cap=cv2.VideoCapture(3)
cap=cv2.VideoCapture(3)
frameWidth=640
frameHeight=480
frame_center=np.array((frameHeight,frameHeight))/2
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,150)
cap.set(3,frameWidth)
img_copy=np.zeros((frameHeight,frameWidth))
val=25
max_sound_epaiseur_max=10
#----------------------------------------------SOUND
fs=44800
sd.default.device='CABLE Output (VB-Audio Virtual , MME'
frame_annalyse=3000
freg_auditible_limite=2000
windowSamples=[0 for _ in range(frame_annalyse)]
#---------------------------------------------------------------------------------------------------------------- 3D Transform
#----------------------------------------------ANGLE MAP
floor=np.array(([-2.39,0.44,3.78,7.05]))
observateur=np.array((0,0,0,90,270,0))
choise_last=np.zeros((3,25))
rotation_contexte=R.from_euler('zyx',observateur[3:6],degrees=True)
rotation_de_observateur_0=R.apply(rotation_contexte,observateur[0:3]) #ok gros ca a pas de sens ne cherche pas mais on en a besoin por vire 0
rotation_de_observateur_0[[0,2]]=rotation_de_observateur_0[[0,2]]
capteur_estimation=[24,36] #mm width=1  height=0
focal_estimation=35 #mm
crop_estimation=0 # width=0  height=1
frameWidth=640
frameHeight=480
angle_serie=np.zeros(((int(frameHeight)*crop_estimation)+(int(frameWidth)*(crop_estimation==0))))
lng_usg=(int(capteur_estimation[0])*crop_estimation)+(int(capteur_estimation[1])*(crop_estimation==0))
angle_cut=15
angle_map=np.zeros((int(frameHeight/2),int(frameWidth/2)))
for i in range (0,int(90/angle_cut)):
    try :
        alpha=i*angle_cut
        beta=(math.tan(math.radians(alpha/2)))*(focal_estimation*2)*(np.size(angle_serie)/lng_usg)
        angle_serie[int(beta)]=1
    except :
        break
angle_map[0,:]=angle_serie[:np.shape(angle_map)[1]]
angle_map[:,0]=angle_serie[:np.shape(angle_map)[0]]
angle_map=np.concatenate((angle_map[:,::-1],angle_map),axis=1)
angle_map=np.concatenate((angle_map[::-1],angle_map),axis=0)
#----------------------------------------------SAVE PLY
dispartition_max=15 #base bin 4 add 0
prolongation_max=0.005
#----------------------------------------------Save
save_ply_question=False
save_img_question=False
save_npy_question=False
save_csv_question=False
save_poly_visu_question=False
Visual_Question=True
#----------------------------------------------Interpolation
Reduce_interpolotion=1
with concurrent.futures.ThreadPoolExecutor() as executor:
#---------------------------------------------- Mesh_Java_laser_pre_calcule
    star_ray_pres=3
    steep=50
    dir_file="C:/Users/enzoa/.VisualSt/python/openpose/python/"
    laser_dir="C:/Users/enzoa/.VisualSt/python/openpose/python/rolex_laser.ply"
    mesh_dir="C:/Users/enzoa/.VisualSt/python/openpose/python/rolex.ply"
    laser_to_mesh_false,mesh_java,laser_org,laser_vect,laser_6d,laser_speed_attributif=Laser_org_to_mesh(laser_dir,mesh_dir,star_ray_pres,dir_file,steep)
    normal_imporant_on_dist=1000
    laser_6d[1]=laser_6d[1]/normal_imporant_on_dist
#---------------------------------------------- Mise en contexte force
    size=np.array([frameHeight,frameWidth])
    height_contextualisation=1
    degret=math.degrees(math.atan((capteur_estimation[0]/2)/focal_estimation))
    normal_camera=-np.array(([10,0,0]))
    distance_deep=np.sum((normal_camera**2))**0.5
    org_pt=np.array(([0,0,1]))
    contextualisation_top=np.array(([height_contextualisation,size[0]/size[1]*height_contextualisation]))
    contextualisation_top=np.concatenate((org_pt.flatten(),contextualisation_top))
    contextualisation_bottom=np.array(([math.tan(math.radians(degret))*distance_deep,math.tan(math.radians(degret*size[0]/size[1]))*distance_deep]))
    contextualisation_bottom=np.concatenate((org_pt+normal_camera,contextualisation_bottom*2+contextualisation_top[3:5]))
    rotation_matrix=rotation_matrix_from_vectors(np.array([0,1,0]),normal_camera)
    applic_prisme=R.from_matrix(rotation_matrix)
    #---------------------------------------------------------------------------------------------------------------- WHILE
    roding_turn=True
    round_zero=True
    round_two=False
    round_three=False
    height_reboot=False
    scale_moyenne=0
    scale_z=0
    userWantsToExit=False
    cal_for_ply=np.zeros((3,0))
    G=time.time()
    list_futur=np.zeros((1,20))
    list_futur_pbr=True
    lood_geo=[np.zeros((0,3)),np.zeros((0,3)),np.zeros((0,3))]
    with sd.InputStream(channels=1,callback=callback,blocksize=int(frame_annalyse),samplerate=fs):
        time.sleep(0.33)
        while not userWantsToExit:
            success,img=cap.read()
            if round_zero==True:
                key_old=np.array((0))
            else :
                key_old=np.copy(key)
            proc_sound=executor.submit(Sound_Annalyse,recording_raw)
            proc_motion=executor.submit(motion,img,img_copy)
            # Process Image
            datum=op.Datum()
            datum.cvInputData=img
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            # Display Image
            key=np.array(datum.poseKeypoints)
            key=clearning_Z(key)
            proc_clearning_Z=executor.submit(clearning_Z,key)
            proc_intercorps=executor.submit(intercorps,key_old,key)
            proc_interpolation=executor.submit(interpolation,key,frameHeight,frameWidth,Reduce_interpolotion)
            proc_cal_scale_moyenne=executor.submit(cal_scale_moyenne,key,list_link_size,frameHeight,focal_estimation,frame_center,capteur_estimation,scale_z,num)
            max_sound=proc_sound.result()
            img_add,img_copy,img_1d=proc_motion.result()
            cal=proc_intercorps.result()
            scale_z=proc_cal_scale_moyenne.result()
            proc_mise_en_contexe=executor.submit(mise_en_context,np.copy(cal),focal_estimation,frame_center,scale_z,applic_prisme,contextualisation_bottom,contextualisation_top,size,normal_camera,distance_deep)
            key,interpolation_map=proc_interpolation.result()
            if roding_turn==False :
                if round_two :
                    ok_save_img=proc_save_img.result()
                    print("img : "+str(ok_save_img))
                    ok_save_ply=proc_save_ply.result()
                    print("ply : "+str(ok_save_ply))
                    
                    h=time.time()
                    if round_three :
                        vis.clear_geometries()
                    pcd_line,pi_step=proc_visualisation.result()
                    X_transposer,Y_transposer,vis_line,extrinsic=proc_soft_transposition.result()
                    if num > 4 :
                        vis.add_geometry(pcd_line, reset_bounding_box=False)
                        vis.add_geometry(vis_line, reset_bounding_box=False)
                        ok_save_poly_visual=save_poly_visual(vis,num-1,save_poly_visu_question)
                        print("png : "+str(ok_save_ply))
                    else : 
                        vis.add_geometry(pcd_line)
                        
                    better_zoom=proc_evaluated_hull_w_for_Z.result()
                    #view_control.translate(-X_transposer,Y_transposer)
                    fps_extrinsic, choise_last,new_extrinsic,xyz_body = proc_body_to_point.result()
                    camera_params=view_control.convert_to_pinhole_camera_parameters()
                    mpa = np.array(([[0,0,0],[0,0,1],[0,1,0],[1,0,0]]))
                    lood_geo = [np.tile(np.array(xyz_body)[np.newaxis,:],(4,1))+mpa,np.tile(np.array(([[255,255,255]])),(4,1)),np.array(([[0,1,2],[1,2,3],[0,2,3],[0,1,3]]))]
                    camera_params.extrinsic=(new_extrinsic)
                    print(camera_params.extrinsic)
                    #view_control.set_zoom(better_zoom)
                    height_reboot=proc_height_reboot_def.result()
                    print(camera_params.extrinsic)
                    view_control.convert_from_pinhole_camera_parameters(camera_params)
                    vis.update_renderer()
                    vis.poll_events()
                    round_three=True
                    #extrinsic =view_control.convert_to_pinhole_camera_parameters().extrinsic
                else : 
                    extrinsic=np.zeros((4,4))
                if round_zero==False:
                    num=num+1
                    output=proc_Output_def.result()
                    extrinsic=np.copy(extrinsic)
                    try : 
                        best_zoom_print,X_transposer_print,Y_transposer_print=np.copy(better_zoom), X_transposer,Y_transposer
                    except : 
                        best_zoom_print,Y_transposer_print,X_transposer_print=0,0,0
                    proc_resize_cv2_output=executor.submit(resize_cv2_output,output,extrinsic,best_zoom_print,Y_transposer_print,X_transposer_print)
                    list_futur,pcd,name_ply,list_futur_pbr,triangle,vertices,mesh_color,pts_hull_w=proc_process_3d.result()
                    proc_pt_ref_def_visual=executor.submit(pt_ref_def_visual,pcd)
                    proc_save_img=executor.submit(save_img,num,img_1d,output,path,save_img_question)
                    proc_save_ply=executor.submit(save_ply,pcd,name_ply,save_ply_question,path)
                    output=proc_resize_cv2_output.result()
                    cv2.imshow('output',output)
                    print("loop_time : "+str((time.time()-G))+"\n"+"loop_num : "+str(num))
                    print("\n"+"---------------------------------"+"\n")  
                    if round_two==False :
                        vis=o3d.visualization.Visualizer()
                        size_of_screen=np.array((int(1920/2),int(1080/2)))
                        vis.create_window(width=size_of_screen[0],height=size_of_screen[1])
                        vis.set_full_screen(False)
                        view_control=vis.get_view_control()
                        render_option=vis.get_render_option()
                        render_option.point_size=0.1
                        render_option.background_color=(0.109, 0.125,0.156)
                        pi_step=0
                        pi_step_d=0.04
                        vecteur_Z=0.3
                        view_control.change_field_of_view(step=-90.0)
                        pcd_line=np.zeros((0))
                        fake_zoom_speed=5
                        height_reboot=True
                        render_option.line_width=0.1
                    if height_reboot :
                        better_zoom=1
                        best_deep=1
                        view_control.set_zoom(1)
                        height_reboot=False
                    pt_ref_visual=proc_pt_ref_def_visual.result()
                    proc_visualisation=executor.submit(visualisation,pcd_line,triangle,vertices,mesh_color,pt_ref_visual,round_two,pi_step,pi_step_d)
                    proc_soft_transposition=executor.submit(soft_transposition,view_control,pt_ref_visual,round_three,better_zoom)
                    proc_evaluated_hull_w_for_Z=executor.submit(evaluated_hull_w_for_Z,pt_ref_visual,np.max(size_of_screen),view_control,fake_zoom_speed,better_zoom)
                    proc_height_reboot_def=executor.submit(height_reboot_def, view_control)
                    round_two=True
                G=time.time()
                cal_for_ply=proc_mise_en_contexe.result()
                proc_body_to_point = executor.submit(body_to_point,cal_for_ply,choise_last)
                proc_process_3d=executor.submit(process_3d,np.copy(cal_for_ply),list_link,max_sound,list_futur_pbr,list_futur,num,dispartition_max,prolongation_max,mesh_color_organ,img_add,laser_vect,mesh_java,laser_org,laser_to_mesh_false,laser_6d,normal_imporant_on_dist,laser_speed_attributif,star_ray_pres,floor,lood_geo)
                proc_Output_def=executor.submit(Output_def,key,img_1d,img_add,frameHeight,frameWidth,max_freg,max_sound,interpolation_map,angle_map,cal_for_ply)
                round_zero=False
            roding_turn=False
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        video.release()
        cv2.destroyWindow()  