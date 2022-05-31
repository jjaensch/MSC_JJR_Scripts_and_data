# -*- coding: utf-8 -*-
"""
Created on Sun May 29 23:10:10 2022
@author: Jakob Jaensch Rasmussen
"""
#%%
def evaluate(cfg, # model.cfg file
             weights, # model.weights file
             names, # model.names file
             img_name, # string name of image to evaluate
             valid_txt, # text file housing validation bounding box coordiantes
             input_size, # YOLOv4 input size for SPP resizing
             spatial_res, # Spatial resolution of imagery, Unit cm/pixel
             conf_lvl, # confidence level for detections
             NMS_lvl, # Non Max Suppression threshhold (NMS) for chosing best detections
             iou, # IoU threshhold for determining TP, FP and FN
             color, # color of detections, color = [b:blue, r:red, other:green]
             fig, # Figure size when visualizing coverages, validation bounding boxes and detections
             print_mask, # Visualize bianry masks of coverages [yes,no] 
             print_bboxes): # Visualize bounding boxes individually for each detected class and class within the validation textfile [yes,no]
    
    ######################
    #### Dependencies ####
    import cv2
    import matplotlib.pyplot as plt
    #%matplotlib inline #required when using notebooks 
    import numpy as np
    import pandas as pd
    import shapely
    from shapely.geometry import Polygon,Point

    ###################
    ###################
    ### Preprocessing
    #
    #ensure spatial resolution is a float value
    spatial_res = float(spatial_res)  
    #
    #import image for inference
    img = cv2.imread(img_name) 
    img_height= img.shape[0]
    img_width = img.shape[1]  
    #
    #read and transform validation textfile, conversion from image coordiantes to pixel position
    valid_df = pd.read_csv(valid_txt, sep=" ", header=None, 
                 names=["class", "x_center", "y_center","width","height"])
    valid_df = valid_df.reset_index()
    valid_df["x_center"] = round((img_width/100) * valid_df["x_center"] * 100,0)
    valid_df["y_center"] = round((img_height/100) * valid_df["y_center"] * 100,0)
    valid_df["width"] = round((img_width/100) * valid_df["width"] * 100,0)
    valid_df["height"] = round((img_height/100) * valid_df["height"] * 100,0)
    #
    #Read class names
    with open(names, 'r') as names_file:
        classes = names_file.read().splitlines()
    #
    #set color for boxes and text
    if color == 'b':
        c = (255, 0, 0)
    elif color == 'r':
        c = (0, 0, 255)
    else:
        c = (0, 255, 0)
    
    #########################
    ### Prepare the detector
    #
    #load in CFG and weights via darknet
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    #
    #Set the detection model for openCV
    model = cv2.dnn_DetectionModel(net) # Couple model with CFG and weights
    #
    #Turn image into RGB and shape input size of the model
    model.setInputParams(scale=1 / 255, size=(input_size, input_size), swapRB=True) 
   
    ####################
    ### Detect objects
    #
    classIds, scores, boxes = model.detect(img, confThreshold=conf_lvl, #Threshhold for boxes
                                           nmsThreshold=NMS_lvl) #Threshhold for sleecting best box on multi detection
    #        
    #Append all classes detected to a list
    id_names = []
    for (classId, score, box) in zip(classIds, scores, boxes):
        if classId not in id_names:
            id_names.append(classId)
    #
    #Check if anything is detected
    if len(id_names) > 0:
        # If any objects is detected:
        #
        ##################################
        ### Iterate for each class present
        for id in id_names:
            #
            # Prepare images for visualization
            detections = img.copy() #image for visualizing detection bounding boxes
            valid = img.copy() #image for visualizing validations
            test = img.copy() #image for detections that will be converted into a binary mask
            valid_area = img.copy() #image for validations that will be converted into a binary mask
            bbox_overlap = img.copy() #image for visualizing detection bounding boxes together with validation bounding boxes for the given class
            #
            #Prepare storage for TP FP evalaution
            pred = [] #List for Prediction coordinates for TP evaluation
            ground = [] #List for Groundtruth coordinates for TP evaluation
            #
            ### Iterate in YOLO detections: 
            for (classId, score, box) in zip(classIds, scores, boxes):
                if classId == id:
                    #
                    ### for the image visualizing detections:
                    # draw rectangles for YOLO detections
                    cv2.rectangle(detections, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                                  color= c, thickness=2)
                    # Define text for bounding box with score and class in detections
                    text = '%s: %.2f' % (classes[classId],
                                         score)
                    # Input text on bounding box
                    cv2.putText(detections, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                color= c, thickness=2)
                    #
                    ### for the image visualizing detections and valdiation boxes for given class:
                    cv2.rectangle(bbox_overlap, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                                  color= c, thickness=2)
                    # Define text for bounding box with score and class in detections
                    text = '%s: %.2f' % (classes[classId],
                                         score)
                    # Input text on bounding box
                    cv2.putText(bbox_overlap, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                color= c, thickness=2)                
                    #
                    ### For the detection mask: 
                    # draw bounding boxes for detections, but filled
                    cv2.rectangle(test, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                                  (0,0,0),-1) 
                    #
                    # Convert detection to binary mask
                    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
                    th, test_mask = cv2.threshold(test_gray, 0, 1, cv2.THRESH_BINARY)
                    #
                    ### TP FP evaluation
                    # Get bounding box coordinates
                    x1 = box[0]
                    y1 = box[1]
                    x2 = box[0] + box[2]
                    y2 = box[1] + box[3]
                    # append bounding box to predictions storage list
                    pred.append([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
            #
            ### Iterate in validation data frame:  
            # check if detected class is in valid txt
            if id in valid_df['class'].unique():
                #if so:
                for index, row in valid_df.iterrows():
                    if row['class'] == id:
                        #
                        ### for the image visualizing validation:
                        # draw rectangles for valdiation boxes
                        cv2.rectangle(valid, 
                                      (int(row['x_center'] - int(row['width']*0.5)), int(row['y_center'] + int(row['height']*0.5))), 
                                      (int(row['x_center'] + int(row['width']*0.5)), int(row['y_center'] - int(row['height']*0.5))),
                                      (0, 255, 255), 2)
                        # Define class name for box 
                        text = classes[id] 
                        # Input text
                        cv2.putText(valid, text, (int(row['x_center'] - int(row['width']*0.5)), int(row['y_center'] + int(row['height']*0.5))), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                    (0, 255, 255), thickness=2)

                        #
                        ### for the image visualizing detections and valdiation boxes for given class,
                        ### supplement with validation boxes:
                        # draw rectangles for validation boxes
                        cv2.rectangle(bbox_overlap, 
                                      (int(row['x_center'] - int(row['width']*0.5)), int(row['y_center'] + int(row['height']*0.5))), 
                                      (int(row['x_center'] + int(row['width']*0.5)), int(row['y_center'] - int(row['height']*0.5))),
                                      (0, 255, 255), 2)
                        # Define class name for box
                        text = classes[id]
                        # Input text
                        cv2.putText(bbox_overlap, text, (int(row['x_center'] - int(row['width']*0.5)), int(row['y_center'] + int(row['height']*0.5))), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                    (0, 255, 255), thickness=2)                    
                        #
                        ### For the validation mask: 
                        # draw bounding boxes in validation, but filled
                        cv2.rectangle(valid_area, 
                                      (int(row['x_center'] - int(row['width']*0.5)), int(row['y_center'] + int(row['height']*0.5))), 
                                      (int(row['x_center'] + int(row['width']*0.5)), int(row['y_center'] - int(row['height']*0.5))),(0,0,0),-1) 
                        # Convert to binary mask
                        valid_gray = cv2.cvtColor(valid_area, cv2.COLOR_BGR2GRAY)
                        vh, valid_mask = cv2.threshold(valid_gray, 0, 1, cv2.THRESH_BINARY)

                        #
                        ### TP FP evaluation 
                        x1 = int(row['x_center'] - int(row['width']*0.5))
                        y1 = int(row['y_center'] + int(row['height']*0.5))
                        x2 = int(row['x_center'] + int(row['width']*0.5))
                        y2 = int(row['y_center'] - int(row['height']*0.5))
                        # append bounding box to the validation/groundtruthing list
                        ground.append([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
                    
                    else:
                        #if class not in valid.txt, create a blank mask
                        valid_gray = cv2.cvtColor(valid_area, cv2.COLOR_BGR2GRAY)
                        vh, valid_mask = cv2.threshold(valid_gray, 0, 1, cv2.THRESH_BINARY)

            #
            #Create IoU mask using validation and detection masks
            IoU_mask = cv2.bitwise_or(test_mask, valid_mask)

            #######################
            ### Calcualte TP FP FN 
            #
            iou_threshhold = iou
            truth_boxes = ground
            pred_boxes = pred
            
            ###########################
            ### Count false positive
            ### Iterate prediction bounding boxes 
            prob1=[]
            for i in range(len(pred_boxes)):
                #define shapely polygon for the box
              f1=shapely.geometry.Polygon(pred_boxes[i])
              # determine polygon radius
              f1_radius=np.sqrt((f1.area)/np.pi)
              #buffer the polygon fromt the centroid
              f1_buffered=shapely.geometry.Point(f1.centroid).buffer(f1_radius*500)
              cont=[]
              for i in range(len(truth_boxes)):
                ff=shapely.geometry.Polygon(np.squeeze(truth_boxes[i]))
                if f1_buffered.contains(ff)== True:
                  iou=(ff.intersection(f1).area)/(ff.union(f1).area)  
                  cont.append((iou))
              prob1.append(cont)
            #
            # Function for checking if iou is less than threshhold, return all
            def CheckLess(list1,val):
                return(all(x<=val for x in list1))
            #
            # Count FP using checker
            fp=0
            for t in prob1:
                if CheckLess(t,iou_threshhold)==True:
                  fp=fp+1
            
            #########################
            ### Count true positives
            #
            # itereate validation/groundtruthing boxes
            prob2=[]
            for i in range(len(truth_boxes)):
                #define shapely polygon for the box
              f1=shapely.geometry.Polygon(truth_boxes[i])
              #find radius
              f1_radius=np.sqrt((f1.area)/np.pi)
              #buffer the polygon from the centroid
              f1_buffered=shapely.geometry.Point(f1.centroid).buffer(f1_radius*500)
              cont=[]
              # merge up the ground truth instance against prediction
              # to determine the IoU
              for i in range(len(pred_boxes)):
                ff=shapely.geometry.Polygon(np.squeeze(pred_boxes[i]))
                if f1_buffered.contains(ff)== True:
                  #calculate IoU
                  iou=(ff.intersection(f1).area)/(ff.union(f1).area)
                  cont.append((iou))
              # probability of a given prediction to be contained in a
              # ground truth instance
              prob2.append(cont)
            
            # Count TP and FN using checker
            fn=0
            tp=0
            for t in prob2:
                if np.sum(t)==0:
                  fn=fn+1
                elif CheckLess(t,iou_threshhold)==False:
                    tp=tp+1
            #
            # if no true positives are present, return null measures
            if tp == 0:
                precision_tp_fp = 0
                recall = 0
                f1 = 0
            # calculate precision recall and F1 when at least one TP is present
            else:
                precision_tp_fp=round(tp/(tp+fp),3) 
                recall=round(tp/(tp+fn),3)
                f1= round(2*((precision_tp_fp*recall)/(precision_tp_fp+recall)),3)

            ############################      
            ### Calculate mask coverages 
            #
            val_cover = ((img_width*img_height)-np.count_nonzero(valid_mask))*spatial_res/10000 
            #
            det_cover = ((img_width*img_height)-np.count_nonzero(test_mask))*spatial_res/10000 
            #
            IoU_cover = ((img_width*img_height)-np.count_nonzero(IoU_mask))*spatial_res/10000
            
            ################################
            ### Calculate coverage measures
            #
            # Get detector underestimate area
            det_underestimate_area = val_cover - IoU_cover #missed validation area
            # Get detector overestimate area
            det_overestimate_area = det_cover - IoU_cover #incorrect detection area
            # Calculate measures
            accuracy = (IoU_cover/val_cover)*100
            undershoot = 100 - accuracy 
            precision = (IoU_cover/(val_cover+det_overestimate_area))*100
            overshoot = 100 - (IoU_cover/det_cover)*100

            ########################################################
            ### Print F1 evaluation, coverages and coverage measures
            print('\033[1mClass:',classes[id],'\033[0m')
            #
            # Confusion matrix
            print('>>Confusion matrix<<')
            print("TP (IoU > ",NMS_lvl,"):",tp,"\t FP (IoU < ",NMS_lvl,"):",fp,"\nFN (Groundtruthing outside detections):",fn,)
            print("Precision:",precision_tp_fp,"\t Recall:",recall, "\t F1 score:",f1)
            #
            # Coverages
            print('\n>>Coverages<<')
            print('Groundtruth coverage: ',round(val_cover,2),'m2.')
            print('Detection coverage: ',round(det_cover,2),'m2.')
            print('Intersecting coverage/IoU: ',round(IoU_cover,2),'m2.')
            print('Groundtruthing coverage missed: ',round(det_underestimate_area,2),'m2')
            print('Error/overestimated coverage: ',round(det_overestimate_area,2),'m2 of detection not part of the IoU.')
            #
            # Coverage measures
            print('\n>>Evaluation metrics on coverage<<')
            print('Accuracy, fraction of IoU coverage relative to groundtruthing: ',round(accuracy,1),'%')
            print('Underestimate, fraction of invalid detection relative to IoU coverage: ',round(undershoot,1),'%')
            print('Precision, fraction of IoU coverage relative to the sum of invalid- and groundtruthing coverages: ', round(precision,1),'%')
            print('Overestimate, fraction of the detection not part of the IoU coverage: ',round(overshoot,1),'%')
            
            #######################
            ### Visualize imagery
            #
            # Visualizing masks for IoU, detector over- and underestimate
            print('\n>>Visualizing IoU assesment<<\n[Green == Accuracy, Valid detection coverage(IoU between groundtruthing- & detection mask)')
            print('[Orange == Underestimate, Missed groundtruthing (Groundtruthing mask)] \n[Red == Overestimate, Invalid detection coverage (Detection mask)]')
            RGB_overlap = img.copy()
            RGB_overlap[valid_mask==0] = (0,143,255)
            RGB_overlap[test_mask==0] = (0,0,255)
            RGB_overlap[IoU_mask==0] = (0,255,0)
            plt.figure(figsize=(fig,fig))
            plt.imshow(RGB_overlap[:,:,::-1])
            plt.show()
            #
            # Visualize validation and detection bounding boxes
            print('\nDetections marked blue, validation marked yellow')
            plt.figure(figsize=(fig,fig))
            plt.imshow(bbox_overlap[:,:,::-1])
            plt.show()
            #
            # If prompted to visualize detections and validation individually
            if print_bboxes == 'yes':
                print('Detector bounding boxes:')
                plt.figure(figsize=(fig,fig))
                plt.imshow(detections[:,:,::-1])
                plt.show()        
                #
                print('Validation bounding boxes:')
                plt.figure(figsize=(fig,fig))
                plt.imshow(valid[:,:,::-1])
                plt.show()        
            #
            # If prompted to visualize binary masks individually 
            if print_mask == 'yes':            
                print('Masks used..')
                print('IoU mask:')
                plt.figure(figsize=(fig,fig))
                plt.imshow(IoU_mask, cmap='gray')
                plt.show()        
                #
                print('Detection mask:')
                plt.figure(figsize=(fig,fig))
                plt.imshow(test_mask, cmap='gray')
                plt.show()        
                #
                print('Validation mask:')
                plt.figure(figsize=(fig,fig))
                plt.imshow(valid_mask, cmap='gray')
                plt.show()        
    
    #####################################################
    # If no objects were detected initially, output null
    else:
        print('No classes were detected within imagery named: ',img_name)