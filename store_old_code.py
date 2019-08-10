#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 20:12:42 2019

@author: vlsilab
"""

def match(truths      = None, 
          labels     = None, 
          priors     = None, 
          variance   = None, 
          threshold  = 0.5,
#          idx        = None,
#          loc_t      = None,
#          conf_t     = None
        ):
    """
    Match each prior (or anchor) box with the ground truth box of the ighest jaccard overlap, 
    encode the bounding boxes, then return the matched indices correspoding to both confidence 
    and location predictions.
    
    Both Truth and Priors are in the form (cx, cy, w, h)
    Convert to form (xmin, ymin,xmax, ymax) before getting IOU
    
    Arguments:
        threshold: (float) The overlap threshold used when matching boxes
        truth    : (tensor) Ground truth boxes, sahep [num_obj, num_priors]
        priors   : (tensor)  Prior boxes from prior boxes layers, shape [num_prioirs, 4]
        variance : (tensor) Variance corresponding to each prioir coordinate, shape [num_priors, 4]
        
        labels   : (tensor) All the class label for the image, shape : [num_obj]
        
        loc_t    : (tensor) Tensor to be filled with encoded location targets
        conf_t   : (tensor) Tensor to be filled with ematched indices for conf preds.
        idx     : (int) current batch index
        
    Returns:
        The match indices corresponding to 
            1) location 
            2) cofidence predcition
    """
    # Ground truth are  in form (xc, yc, w, h)
    # convert it to form (xmin, ymin, xmax, ymax)
    truths = point_form(truths)
    
    iou = jaccard(truths, point_form(priors))
    
    best_prior_overlap = np.amax(iou, axis=-1).astype(np.float32)
    best_prior_idx     = np.argmax(iou, axis =-1)
    
#    print(best_prior_overlap.shape)
#    print(best_prior_idx.shape)

    best_truth_overlap = np.amax(iou, axis=0).astype(np.float32)
    best_truth_idx     = np.argmax(iou, axis = 0)
    best_truth_overlap = index_fill(best_truth_overlap, best_prior_idx, axis=0, value=2)
#    print(best_truth_overlap.shape)
#    print(best_truth_idx.shape)

    for j in range(best_prior_idx.shape[0]):
        best_truth_idx[best_prior_idx[j]] = j
    
    matches = truths[best_truth_idx]
    conf    = labels[best_truth_idx]

    conf[best_truth_overlap < threshold] = 0
    
    loc       = encode(matched=matches, priors=priors, variances=variance)
     
    return loc, conf


def nms(boxes, overlapThresh):
    """ Calcalte non maximum supression
    Args:
        boxes : Predicetd bounding boxes
        overlapThresh : IOU threshold
        
    Returns:
        supressed boxes
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    
    # intialize picked boxes
    pick  = []
    
    # grab coordinates of bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    # Compite the ares of the boundingboxes and sort the bounding 
    # boxes by the bottom-rigth y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    # keep looping while some indexes still remian in the indexes list
    
    while len(idxs) > 0:
        # Grab the last index in the indexes list, add the index value ot the 
        # list of picked index, the initialize the suppression list (i.e. indexes that will be deleted)
        # using the last index
       
        last  = len(idxs) -1
        i     = idxs[last]
        pick.append(i)
        suppress = [last]
        
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            
            # grab the current index
            j = idxs[pos]
            
            # Find the largest (x,y) coordinates for the start of 
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            
            # compute the width and height of the bounding box
            w = max(0, xx2- xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            
            # Compute the ratio of overlap betwen the computed 
            # bounding box and the bounding box in the areas lsit
            
            overlap = float( w * h) / area[j]
            
            # if there is sufficient overlap, supress the 
            # current bouding box
            if overlap > overlapThresh:
                suppress.append(pos)
                
                
        idxs = np.delete(idxs, suppress)
        
        
        # return only the bounding boxes that were picked
        
    return boxes[pick]

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# if probabilities are provided, sort on them instead
	if probs is not None:
		idxs = probs

	# sort the indexes
	idxs = np.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int")
            
 
if __name__ == "__main__":
   boxes =  np.array([
                    	(12, 84, 140, 212),
                    	(24, 84, 152, 212),
                    	(36, 84, 164, 212),
                    	(12, 96, 140, 224),
                    	(24, 96, 152, 224),
                    	(24, 108, 152, 236)])      

   picked_boxes = nms(boxes = boxes, overlapThresh = 0.5)     
