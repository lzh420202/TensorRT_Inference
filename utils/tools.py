import os


def parse_label_file(file):
    with open(file, 'r') as f:
        labels = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    return labels


def is_image(path):
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions


def print_cfg(cfg: dict):
    print(f'{"-"*10 + " configure " + "-"*10}')

    assert cfg['mode'].lower() in ['whole', 'fix', 'server']
    print(f"Mode: {cfg['mode'].lower()}")
    if cfg['mode'].lower() == 'server':
        port = cfg.get('port')
        if port:
            print(f"ZeroMQ Server Port: {port}")
        else:
            raise AttributeError('Can not find server port configure.')
    print(f"Model file: {cfg['model']['engine_file']}")
    if cfg['model']['labels']:
        ALL_LABEL = parse_label_file(cfg['model']['labels'])
        label_str = '[' + ', '.join(ALL_LABEL) + ']'
        print(f"Labels: {label_str}")
    if cfg['mode'].lower() in ['whole', 'fix']:
        if cfg.get('io'):
            print(f"Input directory: {os.path.abspath(cfg['io']['input_dir'])}")
            print(f"Output directory: {os.path.abspath(cfg['io']['output_dir'])}")
        else:
            raise AttributeError('Can not find IO configure, please check configure file.')
    elif cfg['mode'].lower() in ['server']:
        if cfg.get('io'):
            print(f"Warning, Server mode does not support local filesystem IO.")
    cfg_preprocess = cfg['preprocess']
    print(f"Preprocessor: {cfg_preprocess['num_process']}")
    print(f"Preprocessor queue: {cfg_preprocess['queue_length']}")
    cfg_norm = cfg_preprocess['normalization']
    if cfg_norm['enable']:
        print("Normalization: Enable")
        print(f"Normalization mean: [{', '.join([str(m) for m in cfg_norm['mean']])}]")
        print(f"Normalization std: [{', '.join([str(s) for s in cfg_norm['std']])}]")
    else:
        print("Normalization: Disable")
    if cfg['mode'].lower() in ['whole', 'server']:
        cfg_split = cfg_preprocess.get('split')
        if cfg_split:
            print(f"Split size: {cfg_split['subsize']}")
            print(f"Split gap: {cfg_split['gap']}")
        else:
            raise AttributeError("Can not find split configure, please check configure file.")

    cfg_postprocess = cfg['postprocess']
    print(f"Postprocessor: {cfg_postprocess['num_process']}")
    print(f"Postprocessor queue: {cfg_postprocess['queue_length']}")
    print(f"Postprocess score threshold: {cfg_postprocess['score_threshold']}")
    print(f"Postprocess nms threshold: {cfg_postprocess['nms_threshold']}")
    print(f"Single picture max detection number: {cfg_postprocess['max_det_num']}")
    if cfg['mode'].lower() in ['whole', 'fix']:
        if cfg_postprocess['draw_image']['enable']:
            print("Draw result: Enable")
            print(f"Draw number: {cfg_postprocess['draw_image']['num']}")
        else:
            print("Draw result: Disable")
    elif cfg['mode'].lower() in ['server']:
        if cfg_postprocess.get('draw_image'):
            print('Warning, Server mode does not support result visualization.')
    print(f'{"-" * 31}')


def print_log(name, log):
    print(f'{name}: size {log["shape"][0]} x {log["shape"][1]}, \tpatch {log["patch_num"]}, \tdet {log["det_num"]}, \ttime {log["time"]: .2f} Sec.')


def generate_split_box(image_shape, split_size, gap):
    height, width = image_shape
    stride_length = split_size - gap
    n_h = max(1, height // stride_length)
    n_w = max(1, width // stride_length)
    if n_h * stride_length + gap < height:
        n_h += 1
    if n_w * stride_length + gap < width:
        n_w += 1
    boxes = []
    for i in range(n_h):
        for j in range(n_w):
            offset_h = i * stride_length
            offset_w = j * stride_length
            if offset_h + split_size > height:
                offset_h = max(0, height - split_size)
            if offset_w + split_size > width:
                offset_w = max(0, width - split_size)
            boxes.append([offset_h, min(height, offset_h + split_size),
                          offset_w, min(width, offset_w + split_size)])
    return boxes