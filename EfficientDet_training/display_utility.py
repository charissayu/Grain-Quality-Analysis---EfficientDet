from PIL import Image, ImageDraw, ImageFont

def draw_detections(img, detections, colors, letters, font_path, font_size = 30, dispaly_format = None):
  imd = ImageDraw.Draw(img)
  offset = font_size // 2
  font = ImageFont.truetype(font_path, font_size)

  for d in detections:
    if dispaly_format == 'ellipse':
      imd.ellipse(d['box'], outline=colors[d['class']-1], width=2)
    elif  dispaly_format == 'letter':
      x = int((d['box'][0] + d['box'][2]) / 2) - offset
      y = int((d['box'][1] + d['box'][3]) / 2) - offset
      imd.text((x, y), letters[d['class']-1], fill=colors[d['class']-1], align ="left", font=font)
    else:
      imd.rectangle(d['box'], outline=colors[d['class']-1], width=2)
  return img

def group_by_label(detections):
  group = {}
  for detection in detections:
    if str(detection['class']) not in group.keys():
      group[str(detection['class'])] = 0      
    group[str(detection['class'])] = group[str(detection['class'])] + 1
  return group