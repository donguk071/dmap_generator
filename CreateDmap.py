import torch
import cv2
import os
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
import struct
import argparse


def loadDMAP(dmap_path):
  with open(dmap_path, 'rb') as dmap:
    file_type = dmap.read(2).decode()
    content_type = np.frombuffer(dmap.read(1), dtype=np.dtype('B'))
    reserve = np.frombuffer(dmap.read(1), dtype=np.dtype('B'))
    
    has_depth = content_type > 0
    has_normal = content_type in [3, 7, 11, 15]
    has_conf = content_type in [5, 7, 13, 15]
    has_views = content_type in [9, 11, 13, 15]
    
    image_width, image_height = np.frombuffer(dmap.read(8), dtype=np.dtype('I'))
    depth_width, depth_height = np.frombuffer(dmap.read(8), dtype=np.dtype('I'))
    
    if (file_type != 'DR' or has_depth == False or depth_width <= 0 or depth_height <= 0 or image_width < depth_width or image_height < depth_height):
      print('error: opening file \'{}\' for reading depth-data'.format(dmap_path))
      return
    
    depth_min, depth_max = np.frombuffer(dmap.read(8), dtype=np.dtype('f'))
    
    file_name_size = np.frombuffer(dmap.read(2), dtype=np.dtype('H'))[0]
    file_name = dmap.read(file_name_size).decode()
    
    view_ids_size = np.frombuffer(dmap.read(4), dtype=np.dtype('I'))[0]
    reference_view_id, *neighbor_view_ids = np.frombuffer(dmap.read(4 * view_ids_size), dtype=np.dtype('I'))
    
    K = np.frombuffer(dmap.read(72), dtype=np.dtype('d')).reshape(3, 3)
    R = np.frombuffer(dmap.read(72), dtype=np.dtype('d')).reshape(3, 3)
    C = np.frombuffer(dmap.read(24), dtype=np.dtype('d'))
    
    print(content_type)

    data = {
      'has_normal': has_normal,
      'has_conf': has_conf,
      'has_views': has_views,
      'image_width': image_width,
      'image_height': image_height,
      'depth_width': depth_width,
      'depth_height': depth_height,
      'depth_min': depth_min,
      'depth_max': depth_max,
      'file_name': file_name,
      'reference_view_id': reference_view_id,
      'neighbor_view_ids': neighbor_view_ids,
      'K': K,
      'R': R,
      'C': C
    }
    
    map_size = depth_width * depth_height
    depth_map = np.frombuffer(dmap.read(4 * map_size), dtype=np.dtype('f')).reshape(depth_height, depth_width)
    data.update({'depth_map': depth_map})
    if has_normal:
      normal_map = np.frombuffer(dmap.read(4 * map_size * 3), dtype=np.dtype('f')).reshape(depth_height, depth_width, 3)
      data.update({'normal_map': normal_map})
    if has_conf:
      confidence_map = np.frombuffer(dmap.read(4 * map_size), dtype=np.dtype('f')).reshape(depth_height, depth_width)
      data.update({'confidence_map': confidence_map})
    if has_views:
      views_map = np.frombuffer(dmap.read(map_size * 4), dtype=np.dtype('B')).reshape(depth_height, depth_width, 4)
      data.update({'views_map': views_map})
  
  return data

def loadMVSInterface(archive_path):
  with open(archive_path, 'rb') as mvs:
    archive_type = mvs.read(4).decode()
    version = np.frombuffer(mvs.read(4), dtype=np.dtype('I')).tolist()[0]
    reserve = np.frombuffer(mvs.read(4), dtype=np.dtype('I'))
    
    if archive_type != 'MVSI':
      print('error: opening file \'{}\''.format(archive_path))
      return
    
    data = {
      'project_stream': archive_type,
      'project_stream_version': version,
      'platforms': [],
      'images': [],
      'vertices': [],
      'vertices_normal': [],
      'vertices_color': []
    }
    
    platforms_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
    for platform_index in range(platforms_size):
      platform_name_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
      platform_name = mvs.read(platform_name_size).decode()
      data['platforms'].append({'name': platform_name, 'cameras': []})
      cameras_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
      for camera_index in range(cameras_size):
        camera_name_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
        camera_name = mvs.read(camera_name_size).decode()
        data['platforms'][platform_index]['cameras'].append({'name': camera_name})
        if version > 3:
          band_name_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
          band_name = mvs.read(band_name_size).decode()
          data['platforms'][platform_index]['cameras'][camera_index].update({'band_name': band_name})
        if version > 0:
          width, height = np.frombuffer(mvs.read(8), dtype=np.dtype('I')).tolist()
          data['platforms'][platform_index]['cameras'][camera_index].update({'width': width, 'height': height})
        K = np.asarray(np.frombuffer(mvs.read(72), dtype=np.dtype('d'))).reshape(3, 3).tolist()
        data['platforms'][platform_index]['cameras'][camera_index].update({'K': K, 'poses': []})
        identity_matrix = np.asarray(np.frombuffer(mvs.read(96), dtype=np.dtype('d'))).reshape(4, 3)
        poses_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
        for _ in range(poses_size):
          R = np.asarray(np.frombuffer(mvs.read(72), dtype=np.dtype('d'))).reshape(3, 3).tolist()
          C = np.asarray(np.frombuffer(mvs.read(24), dtype=np.dtype('d'))).tolist()
          data['platforms'][platform_index]['cameras'][camera_index]['poses'].append({'R': R, 'C': C})
    
    images_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
    for image_index in range(images_size):
      name_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
      name = mvs.read(name_size).decode()
      data['images'].append({'name': name})
      if version > 4:
        mask_name_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
        mask_name = mvs.read(mask_name_size).decode()
        data['images'][image_index].update({'mask_name': mask_name})
      platform_id, camera_id, pose_id = np.frombuffer(mvs.read(12), dtype=np.dtype('I')).tolist()
      data['images'][image_index].update({'platform_id': platform_id, 'camera_id': camera_id, 'pose_id': pose_id})
      if version > 2:
        id = np.frombuffer(mvs.read(4), dtype=np.dtype('I')).tolist()[0]
        data['images'][image_index].update({'id': id})
      if version > 6:
        min_depth, avg_depth, max_depth = np.frombuffer(mvs.read(12), dtype=np.dtype('f')).tolist()
        data['images'][image_index].update({'min_depth': min_depth, 'avg_depth': avg_depth, 'max_depth': max_depth, 'view_scores': []})
        view_score_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
        for _ in range(view_score_size):
          id, points = np.frombuffer(mvs.read(8), dtype=np.dtype('I')).tolist()
          scale, angle, area, score = np.frombuffer(mvs.read(16), dtype=np.dtype('f')).tolist()
          data['images'][image_index]['view_scores'].append({'id': id, 'points': points, 'scale': scale, 'angle': angle, 'area': area, 'score': score})
    
    vertices_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
    for vertex_index in range(vertices_size):
      X = np.frombuffer(mvs.read(12), dtype=np.dtype('f')).tolist()
      data['vertices'].append({'X': X, 'views': []})
      views_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
      for _ in range(views_size):
        image_id = np.frombuffer(mvs.read(4), dtype=np.dtype('I')).tolist()[0]
        confidence = np.frombuffer(mvs.read(4), dtype=np.dtype('f')).tolist()[0]
        data['vertices'][vertex_index]['views'].append({'image_id': image_id, 'confidence': confidence})
    
    vertices_normal_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
    for _ in range(vertices_normal_size):
      normal = np.frombuffer(mvs.read(12), dtype=np.dtype('f')).tolist()
      data['vertices_normal'].append(normal)
    
    vertices_color_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
    for _ in range(vertices_color_size):
      color = np.frombuffer(mvs.read(3), dtype=np.dtype('B')).tolist()
      data['vertices_color'].append(color)
    
    if version > 0:
      data.update({'lines': [], 'lines_normal': [], 'lines_color': []})
      lines_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
      for line_index in range(lines_size):
        pt1 = np.frombuffer(mvs.read(12), dtype=np.dtype('f')).tolist()
        pt2 = np.frombuffer(mvs.read(12), dtype=np.dtype('f')).tolist()
        data['lines'].append({'pt1': pt1, 'pt2': pt2, 'views': []})
        views_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
        for _ in range(views_size):
          image_id = np.frombuffer(mvs.read(4), dtype=np.dtype('I')).tolist()[0]
          confidence = np.frombuffer(mvs.read(4), dtype=np.dtype('f')).tolist()[0]
          data['lines'][line_index]['views'].append({'image_id': image_id, 'confidence': confidence})
      lines_normal_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
      for _ in range(lines_normal_size):
        normal = np.frombuffer(mvs.read(12), dtype=np.dtype('f')).tolist()
        data['lines_normal'].append(normal)
      lines_color_size = np.frombuffer(mvs.read(8), dtype=np.dtype('Q'))[0]
      for _ in range(lines_color_size):
        color = np.frombuffer(mvs.read(3), dtype=np.dtype('B')).tolist()
        data['lines_color'].append(color)
      if version > 1:
        transform = np.frombuffer(mvs.read(128), dtype=np.dtype('d')).reshape(4, 4).tolist()
        data.update({'transform': transform})
        if version > 5:
          rot = np.frombuffer(mvs.read(72), dtype=np.dtype('d')).reshape(3, 3).tolist()
          pt_min = np.frombuffer(mvs.read(24), dtype=np.dtype('d')).tolist()
          pt_max = np.frombuffer(mvs.read(24), dtype=np.dtype('d')).tolist()
          data.update({'obb': {'rot': rot, 'pt_min': pt_min, 'pt_max': pt_max}})
  
  return data

def saveDMAP(depth_map, dmap_path, file_name, reference_view_id=0, neighbor_view_ids=None, 
             K=np.eye(3), R=np.eye(3), C=np.zeros(3), depth_min=None, depth_max=None):
    """
    Save depth map data in DMAP format.
    """
    if neighbor_view_ids is None:
        neighbor_view_ids = []
    
    depth_height, depth_width = depth_map.shape
    image_height, image_width = depth_height, depth_width
    
    if depth_min is None:
        depth_min = float(np.min(depth_map))
    if depth_max is None:
        depth_max = float(np.max(depth_map))
    
 
    with open(dmap_path, 'wb') as dmap:
        # Write file type ('DR')
        dmap.write(b'DR')
        
        # Write content type (depth information only, 1)
        content_type = 1
        dmap.write(struct.pack('B', content_type))
        
        # Write reserved byte
        dmap.write(struct.pack('B', 0))
        
        # Write image dimensions
        dmap.write(struct.pack('II', image_width, image_height))
        
        # Write depth map dimensions
        dmap.write(struct.pack('II', depth_width, depth_height))
        
        # Write depth map range
        dmap.write(struct.pack('ff', depth_min, depth_max))
        
        # Write file name
        file_name_bytes = file_name.encode('utf-8')
        file_name_size = len(file_name_bytes)
        dmap.write(struct.pack('H', file_name_size))
        dmap.write(file_name_bytes)
        
        # Write view IDs
        view_ids_size = len(neighbor_view_ids) + 1  # Including reference_view_id
        dmap.write(struct.pack('I', view_ids_size))
        dmap.write(struct.pack('I', reference_view_id))
        for view_id in neighbor_view_ids:
            dmap.write(struct.pack('I', view_id))
        
        # Write camera matrix
        K = np.array(K)/2
        R = np.array(R)
        C = np.array(C)
        dmap.write(K.astype('d').tobytes())
        dmap.write(R.astype('d').tobytes())
        dmap.write(C.astype('d').tobytes())
        
        # Write depth map
        dmap.write(depth_map.astype('f').tobytes())

def load_midas_model(model_type="DPT_Large"):
    # MiDaS model load
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return midas, transform

def generate_depth_map(image_path, model, transform, device):

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original_height, original_width = img.shape[:2]

    img = cv2.resize(img, (int(original_width/2), int(original_height/2)), interpolation=cv2.INTER_LINEAR)
    height, width = img.shape[:2]

    input_batch = transform(img).unsqueeze(0)

    if input_batch.dim() == 5:
        input_batch = input_batch.squeeze(0)

    if input_batch.dim() != 4:
        raise ValueError(f"Expected input batch to be 4D, but got {input_batch.dim()}D.")

    input_batch = input_batch.to(device)

    with torch.no_grad():
        prediction = model(input_batch)

    depth_map = prediction.squeeze().cpu().numpy()

    # depth_map = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_map_resized = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)

    return img, depth_map_resized

def display_images(original_image, depth_map):
    """
    Display the original image and depth map.
    """
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Depth Map', depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_images_in_folder(image_path_list, file_name_list, K , R, C, model_type="DPT_Large"):
    """
    Process all images in the given list to generate and save depth maps.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model, transform = load_midas_model(model_type)
    except RuntimeError as e:
        # del cache file and retry not neccessary
        cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
        model_file = os.path.join(cache_dir, 'dpt_large_384.pt')
        if os.path.exists(model_file):
            os.remove(model_file)
        print(f"Deleted cached model file: {model_file}")
        model, transform = load_midas_model(model_type)
    
    model.to(device)
    
    # seqence of images must be sorted
    for i,image_path in enumerate(image_path_list):

        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue
        
        original_image, depth_map = generate_depth_map(image_path, model, transform, device)
        
        image_number = os.path.splitext(os.path.split(image_path)[-1])[0][1:]
        dmap_path = os.path.split(os.path.split(image_path)[0])[0] + f'/depth{image_number}.dmap'
        
        print(f"Saving depth map to {dmap_path}")

        reference_view_id = i
        neighbor_view_ids = [1, 2, 3]
        K_element = K
        R_element = R[i]
        C_element = C[i]

        saveDMAP(depth_map, dmap_path, file_name_list[i], reference_view_id, neighbor_view_ids, K_element, R_element, C_element)

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process and save depth maps from images.")
    parser.add_argument('-mvs_path', default="sample2/scene.mvs",type=str, help='Path to the MVS file')
    args = parser.parse_args()

    #Get information from mvs file 
    mvs_path = args.mvs_path
    mvs_origin = loadMVSInterface(mvs_path)  

    print("read mvs file : %s" % mvs_path)

    K = mvs_origin['platforms'][0]['cameras'][0]['K']
    R , C  = [] , []
    for id in range(len(mvs_origin['platforms'][0]['cameras'][0]['poses'])):
        R.append(mvs_origin['platforms'][0]['cameras'][0]['poses'][id]['R'])
        C.append(mvs_origin['platforms'][0]['cameras'][0]['poses'][id]['C'])

    image_list = [] 
    file_name_list = []
    base_dir = os.path.dirname(mvs_path) 
    
    for info in mvs_origin['images']:
        image_path = info['name']
        image_list.append(os.path.join(base_dir, image_path))
        file_name_list.append(image_path)
    print('%i image read' %len(image_list)) 

    process_images_in_folder(image_list, file_name_list,  K , R , C)

    print("Processing complete")
    