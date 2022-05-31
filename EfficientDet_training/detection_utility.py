import numpy as np
from PIL import Image
import tensorflow as tf

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]

  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
  return output_dict

def filter_detections(output_dict, threshold = 0.5):
  indices = [output_dict['detection_scores'] >= threshold]
  raw_detections = {
      'scores': output_dict['detection_scores'][tuple(indices)],
      'boxes': output_dict['detection_boxes'][tuple(indices)],
      'classes': output_dict['detection_classes'][tuple(indices)]
  }
  detections = []
  for i in range(len(raw_detections['scores'])):
    detections.append({
        'score': raw_detections['scores'][i],
        'box': raw_detections['boxes'][i],
        'class': raw_detections['classes'][i],
    })
  return detections

def get_actual_size_detections(detections, patch_size):
  for detection in detections:
    box = (detection['box'] * patch_size).astype(np.int32)
    box = [box[1], box[0], box[3], box[2]] ### very important (x and y directions are changed)
    detection['box'] = box
  return detections

def fix_offset_of_the_detections(detections, x_offset, y_offset):
  for detection in detections:
    box = detection['box']
    detection['box'] = [box[0] + x_offset, box[1] + y_offset, box[2] + x_offset,  box[3] + y_offset]
  return detections

def get_initial_detections(model, image_np, x_offset, y_offset, patch_size, threshold = 0.5):
  output_dict = run_inference_for_single_image(model, image_np)
  detections = filter_detections(output_dict, threshold)
  detections = get_actual_size_detections(detections, patch_size)
  detections = fix_offset_of_the_detections(detections, x_offset, y_offset)
  return detections

def split_detections_x_left(detections, x):
  left = []
  right = []
  for d in detections:
    if (d['box'][2] >= x):
      right.append(d)
    else:
      left.append(d)
  return left, right

def split_detections_x_right(detections, x):
  left = []
  right = []
  for d in detections:
    if (d['box'][0] <= x):
      left.append(d)
    else:
      right.append(d)
  return left, right

def split_detections_y_top(detections, y):
  top = []
  bottom = []
  for d in detections:
    if (d['box'][3] >= y):
      bottom.append(d)
    else:
      top.append(d)
  return top, bottom

def split_detections_y_bottom(detections, y):
  top = []
  bottom = []
  for d in detections:
    if (d['box'][1] <= y):
      top.append(d)
    else:
      bottom.append(d)
  return top, bottom

def get_oeverlapping_detections(boxA, boxB, threshold = 0.8):
  interArea = float(max((min(boxA[2], boxB[2]) - max(boxA[0], boxB[0]), 0)) * max((min(boxA[3], boxB[3]) - max(boxA[1], boxB[1])), 0))
  boxAArea = float((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
  boxBArea = float((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
  if interArea <= 0:
    return 0 # do nothing
  boxARatio = interArea / boxAArea
  boxBRatio = interArea / boxBArea
  if boxARatio < threshold and boxBRatio < threshold:
    return 0 # do nothing
  elif boxARatio < boxBRatio:
    return 2 # remove box B
  else:
    return 1 # remove box A

def combine_detections(detections_a, detections_b, threshold = 0.8):
  removed_a = []
  removed_b = []
  for idx_a, d_a in enumerate(detections_a):
    temp_detections_b = []
    for idx_b, d_b in enumerate(detections_b):
      r = get_oeverlapping_detections(d_a['box'], d_b['box'], threshold)
      if r == 1:
        removed_a.append(idx_a)
      if r != 2:
        temp_detections_b.append(d_b)
    detections_b = temp_detections_b
  temp_detections_a = []
  for idx_a, d_a in enumerate(detections_a):
    if idx_a not in removed_a:
      temp_detections_a.append(d_a)
  return temp_detections_a, detections_b

def remove_overlap_detections(detections, overlp_threshold = 0.8):
  removed = []
  for i in range(len(detections)):
    for j in range(i + 1, len(detections)):
      boxA = detections[i]['box']
      boxB = detections[j]['box']
      interArea = float(max((min(boxA[2], boxB[2]) - max(boxA[0], boxB[0]), 0)) * max((min(boxA[3], boxB[3]) - max(boxA[1], boxB[1])), 0))
      boxAArea = float((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
      boxBArea = float((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
      boxARatio = interArea / boxAArea
      boxBRatio = interArea / boxBArea

      if boxARatio >= overlp_threshold or boxBRatio >= overlp_threshold:
        r = j if detections[i]['score'] > detections[j]['score'] else i
        removed.append(r)
  
  final_detections = []
  for idx, detection in enumerate(detections):
    if idx not in removed:
      final_detections.append(detection)

  print('original : ', len(detections), ' after removal: ', len(final_detections))

  return final_detections

def get_detections(model, img, patch_size, overlap, threshold = 0.5, combine_threshold = 0.8, remove_overlap = True, overlap_threshold = 0.8):  
  width, height = img.size
  h = 0
  all_detections = []

  while h < height:
    w = 0
    row_detections = []
    h_s = h if (h + patch_size < height) else (height - patch_size)
    h_overlap = overlap if h == h_s else (patch_size - (height - h))

    while w < width:
      w_s = w if (w + patch_size < width) else (width - patch_size)
      w_overlap = overlap  if w == w_s else (patch_size - (width - w))

      print('h_s: ', h_s, ' h_overlap: ', h_overlap, ' w_s: ', w_s, ' w_overlap: ', w_overlap)

      patch = img.crop((w_s, h_s, w_s + patch_size, h_s + patch_size))
      patch_np = np.array(patch.getdata()).reshape((patch_size, patch_size, 3)).astype(np.uint8)
      current_patch_detections = get_initial_detections(model, patch_np, w_s, h_s, patch_size, threshold)
      print('current patch detections: ', len(current_patch_detections))

      if len(row_detections) > 0:
        previous_patch_detections = row_detections[len(row_detections) - 1]
        print('previous patch detections: ', len(previous_patch_detections))
        p_p_l, p_p_r = split_detections_x_left(previous_patch_detections, w_s)
        c_p_l, c_p_r = split_detections_x_right(current_patch_detections, w_s + w_overlap)
        p_p_r_f, c_p_l_f = combine_detections(p_p_r, c_p_l, combine_threshold)
        previous_patch_detections = p_p_l + p_p_r_f
        print('previous patch detections after w-merge: ', len(previous_patch_detections))
        current_patch_detections = c_p_l_f + c_p_r
        print('current patch detections after w-merge: ', len(current_patch_detections))

        if len(all_detections) > 0:
          previous_row_previous_patch_detections = all_detections[len(all_detections) - 1][len(row_detections) - 1]
          print('previous row previous patch detections: ', len(previous_row_previous_patch_detections))
          p_r_p_p_d_t, p_r_p_p_d_b = split_detections_y_top(previous_row_previous_patch_detections, h_s)
          p_p_d_t, p_p_d_b = split_detections_y_bottom(previous_patch_detections, h_s + h_overlap)
          p_r_p_p_d_b_f, p_p_d_t_f = combine_detections(p_r_p_p_d_b, p_p_d_t, combine_threshold)
          previous_row_previous_patch_detections = p_r_p_p_d_t + p_r_p_p_d_b_f
          print('previous row previous patch detections after h-merge: ', len(previous_row_previous_patch_detections))
          previous_patch_detections = p_p_d_t_f + p_p_d_b
          print('previous patch detections after h-merge: ', len(previous_patch_detections))

          all_detections[len(all_detections) - 1][len(row_detections) - 1] = previous_row_previous_patch_detections

          if (w_s + patch_size) == w:
            previous_row_current_patch_detections = all_detections[len(all_detections) - 1][len(row_detections)]
            print('previous row current patch detections: ', len(previous_row_current_patch_detections))
            p_r_c_p_d_t, p_r_c_p_d_b = split_detections_y_top(previous_row_current_patch_detections, h_s)
            c_p_d_t, c_p_d_b = split_detections_y_bottom(current_patch_detections, h_s + h_overlap)
            p_r_c_p_d_b_f, c_p_d_t_f = combine_detections(p_r_c_p_d_b, c_p_d_t, combine_threshold)
            previous_row_current_patch_detections = p_r_c_p_d_t + p_r_c_p_d_b_f
            print('previous row current patch detections after h-merge: ', len(previous_row_current_patch_detections))
            current_patch_detections = c_p_d_t_f + c_p_d_b
            print('current patch detections after h-merge: ', len(current_patch_detections))

        row_detections[len(row_detections) - 1] = previous_patch_detections
        
      row_detections.append(current_patch_detections)
      
      w = w + patch_size - overlap

    all_detections.append(row_detections)
      
    h = h + patch_size - overlap

  detections = []
  for i in range(len(all_detections)):
    for j in range(len(all_detections[i])):
      cleaned_detections = all_detections[i][j]
      if remove_overlap:
        cleaned_detections = remove_overlap_detections(cleaned_detections, overlap_threshold)
      detections = detections +  cleaned_detections

  print('total detections: ', len(detections))

  return detections

def remove_anomalous_boxes(detections, remove_factor = 10):
  areas = []
  for detection in detections:
    box = detection['box']
    detection['area'] = float((box[2] - box[0]) * (box[3] - box[1]))
    areas.append(detection['area'])

  areas.sort()
  offset = int(len(areas) * 0.2)
  areas = areas[offset:-offset]
  avg_area = sum(areas) / len(areas)

  final_detections = []
  for detection in detections:
    if detection['area'] <= (4 * avg_area):
      final_detections.append(detection)

  print('total detections: ', len(detections), ' after removing anomalous boxes: ', len(final_detections))

  return final_detections